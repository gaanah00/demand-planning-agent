"""Promotion lift estimation.

Panel = weekly demand per series (product x location x channel). Model:
    log(1 + units) = sum_k beta_k * [on promotion of type k] + gamma * [week after a promotion]
                     + series fixed effect + Fourier week-of-year seasonality + trend + calendar-event flag + error
Lift of type k = exp(beta_k) - 1 (the % change in demand during the promotion, versus the same series in a
comparable non-promotion week). gamma measures the post-promotion dip (pull-forward). 95% CIs are cluster-robust
by product. The same model is also fitted with discount-depth buckets instead of promotion types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .analytics import ToolError, apply_filters, enrich
from .elasticity import _ols_cluster
from .semantic import get_semantics
from .util import to_jsonable


def estimate_promo_lift(session, filters: list[dict] | None = None, by: str = "promo_type", group: str | None = None) -> dict:
    sem = get_semantics(session)
    dm = sem.get("demand")
    if not dm or not dm.get("promo_col"):
        raise ToolError("Promotion lift needs a weekly sales table with a promotion column.")
    if by not in ("promo_type", "discount"):
        raise ToolError("by must be promo_type or discount")
    key = ("promo_lift", str(filters), by, group)
    if key in session.cache:
        return session.cache[key]
    df = session.df(dm["table"])
    p, d, m, pr = dm["product_col"], dm["date_col"], dm["measure"], dm["promo_col"]
    series_keys = [k for k in dm["keys"] if k != d]
    needed = [c for c in [group] + [f["column"] for f in (filters or [])] if c and c not in df.columns]
    df = enrich(session, dm["table"], df, needed, [])
    df = apply_filters(df, filters)
    if df.empty:
        raise ToolError("No rows match the filters.")
    pt = sem.get("promotion_table")
    if by == "promo_type":
        if not pt or "promo_type" not in session.df(pt).columns:
            raise ToolError("No promotion table with a promo_type column; use by='discount'.")
        types = session.df(pt).set_index(session.tables[pt].pk[0])["promo_type"]
        df = df.assign(_cls=df[pr].map(types))
    else:
        dc = dm.get("discount_col")
        if not dc:
            raise ToolError("No discount column found.")
        bins = [-0.1, 0, 12, 20, 30, 100]
        labels = ["none", "up to 12%", "13-20%", "21-30%", "over 30%"]
        df = df.assign(_cls=pd.cut(df[dc], bins=bins, labels=labels).astype(str).where(df[pr].notna()))
    df = df.sort_values(series_keys + [d])
    sid = df[series_keys].astype(str).agg("|".join, axis=1)
    df = df.assign(_sid=sid.values)
    on = df["_cls"].notna()
    df = df.assign(_post=(on.groupby(df["_sid"]).shift(1, fill_value=False) & ~on).astype(float))
    cal = sem.get("calendar_table")
    ev = pd.Series(0.0, index=df.index)
    if cal and "shopping_event" in session.df(cal).columns:
        c = session.df(cal)
        evw = set(c.loc[c["shopping_event"].notna(), session.tables[cal].pk[0]])
        ev = df[d].isin(evw).astype(float)
    df = df.assign(_ev=ev.values)
    classes = sorted(df["_cls"].dropna().unique())
    classes = [c for c in classes if c != "none"]

    def fit(sub: pd.DataFrame, label) -> dict:
        y = np.log1p(sub[m].astype(float).clip(lower=0))
        X = pd.DataFrame(index=sub.index)
        for c in classes:
            X[f"promo:{c}"] = (sub["_cls"] == c).astype(float)
        X["post_promo_week"] = sub["_post"]
        woy = sub[d].dt.isocalendar().week.astype(float)
        for k in (1, 2, 3):
            X[f"s{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
            X[f"c{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
        X["trend"] = (sub[d] - sub[d].min()).dt.days / 365.25
        X["event"] = sub["_ev"]
        cols = [c for c in X.columns if X[c].std() > 1e-12]
        Xd = X[cols] - X[cols].groupby(sub["_sid"]).transform("mean")
        yd = y - y.groupby(sub["_sid"]).transform("mean")
        beta, V, r2, G = _ols_cluster(Xd.values, yd.values, sub[p].astype(str).values)
        tcrit = float(stats.t.ppf(0.975, max(G - 1, 1)))
        rows = []
        for i, c in enumerate(cols):
            if not (c.startswith("promo:") or c == "post_promo_week"):
                continue
            b, se = float(beta[i]), float(np.sqrt(max(V[i, i], 0)))
            n_weeks = int(X[c].sum())
            rows.append({"effect": c.replace("promo:", ""), "kind": "post-promotion dip" if c == "post_promo_week" else "promotion lift",
                         "lift_pct": (np.exp(b) - 1) * 100, "ci95_pct": [(np.exp(b - tcrit * se) - 1) * 100, (np.exp(b + tcrit * se) - 1) * 100],
                         "series_weeks": n_weeks, "significant": bool(abs(b) > tcrit * se),
                         "reliable": bool(n_weeks >= 30 and abs(b) > tcrit * se)})
        return {"group": label, "effects": sorted(rows, key=lambda r: -r["lift_pct"]), "r2_within": r2,
                "observations": int(len(sub)), "series": int(sub["_sid"].nunique()), "products": int(G)}

    results = [fit(df, "all")] if not group else [fit(g, k) for k, g in df.groupby(group) if len(g) > 200]
    out = to_jsonable({
        "by": by, "group": group, "measure": m, "results": results,
        "method": __doc__.split("Panel =")[1].strip(),
        "caveats": ["Lift includes everything that came with the promotion (display, advertising, retailer support).",
                    "Estimates are averages across products; individual SKUs can differ.",
                    "A promotion type with few promoted weeks gives a wide interval - treat it as indicative only.",
                    "Lift is measured on unconstrained demand; stock must be planned to serve it."],
    })
    session.cache[key] = out
    return out


def promo_lift_chart(session, res: dict) -> dict:
    from .charts import SERIES, _store, base_layout
    r = res["results"][0]
    eff = [e for e in r["effects"]]
    names = [e["effect"] for e in eff]
    y = [e["lift_pct"] for e in eff]
    lo = [e["lift_pct"] - e["ci95_pct"][0] for e in eff]
    hi = [e["ci95_pct"][1] - e["lift_pct"] for e in eff]
    tr = [{"type": "scatter", "mode": "markers", "x": names, "y": y, "name": "Estimated lift",
           "marker": {"size": 10, "color": [SERIES[7] if e["kind"] != "promotion lift" else (SERIES[0] if e["reliable"] else "#9a9993") for e in eff]},
           "error_y": {"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "thickness": 1.5, "width": 6},
           "hovertemplate": "%{x}: %{y:.1f}%<extra></extra>"}]
    lay = base_layout(f"Demand lift by {res['by'].replace('_', ' ')} with 95% confidence intervals (grey = not reliable)",
                      res["by"].replace("_", " ").capitalize(), "Change in weekly demand (%)", legend=False)
    lay["hovermode"] = "closest"
    lay["yaxis"]["rangemode"] = "normal"
    lay["shapes"].append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0, "line": {"color": "rgba(0,0,0,0.35)", "width": 1}})
    return {"chart_id": _store(session, {"data": tr, "layout": lay}, {"tool": "promo_lift"})}
