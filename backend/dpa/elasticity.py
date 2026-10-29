"""Price elasticity estimation (log-log fixed-effects regression) with confidence intervals.

Model, per SKU-week (summed over locations/channels):
    log(units) = beta * log(price) + SKU fixed effect + week-of-year seasonality (Fourier terms)
                 + linear trend per SKU-year + calendar event flag + error
`beta` is the price elasticity: a 1% price increase changes units by beta %.

Two variants are reported because they answer different questions:
* "all price variation" (promotional discounts + list-price changes): well identified but promotions also
  bring display/advertising lift, so it overstates the response to a permanent price rise.
* "list-price changes only" (non-promo weeks): closer to a permanent price change such as tariff
  pass-through, but usually based on few price changes, so wider intervals.
Standard errors are cluster-robust by SKU. An estimate is marked usable only when: the 95% CI excludes 0,
the sign is negative, there are at least 20 price changes, and the CI half-width is below 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import ToolError, enrich, apply_filters
from .semantic import get_semantics
from .util import to_jsonable


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(groups)
    for g in ug:
        m = groups == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    G, n, k = len(ug), len(y), X.shape[1]
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = corr * XtX_inv @ meat @ XtX_inv
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, V, 1 - ss_res / ss_tot if ss_tot else None, G


def _demean(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return df[cols] - df.groupby(by)[cols].transform("mean")


def _fit(panel: pd.DataFrame, label: str) -> dict:
    from scipy import stats
    d = panel.copy()
    d = d[(d["units"] > 0) & (d["price"] > 0)]
    if d["sku"].nunique() < 1 or len(d) < 30:
        return {"variant": label, "usable": False, "reason": "not enough observations with positive units and price"}
    d["ly"] = np.log(d["units"])
    d["lp"] = np.log(d["price"])
    woy = d["week"].dt.isocalendar().week.astype(float).values
    for k in (1, 2, 3):
        d[f"s{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        d[f"c{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    d["trend"] = (d["week"] - d["week"].min()).dt.days / 365.25
    cols = ["ly", "lp", "trend", "event"] + [f"s{k}" for k in (1, 2, 3)] + [f"c{k}" for k in (1, 2, 3)]
    dm = _demean(d, cols, "sku")  # SKU fixed effects
    # price changes: count of week-over-week changes in log price > 1%
    changes = int((d.sort_values("week").groupby("sku")["lp"].diff().abs() > 0.01).sum())
    xcols = [c for c in cols if c != "ly" and dm[c].std() > 1e-9]
    if "lp" not in xcols:
        return {"variant": label, "usable": False, "reason": "no price variation", "price_changes": changes}
    X = dm[xcols].values
    y = dm["ly"].values
    beta, V, r2, G = _ols_cluster(X, y, d["sku"].values)
    i = xcols.index("lp")
    b, se = float(beta[i]), float(np.sqrt(max(V[i, i], 0)))
    dfree = max(G - 1, 1)
    tcrit = float(stats.t.ppf(0.975, dfree))
    lo, hi = b - tcrit * se, b + tcrit * se
    reasons = []
    if not (hi < 0):
        reasons.append("95% CI includes zero or positive values")
    if changes < 20:
        reasons.append(f"only {changes} price changes")
    if (hi - lo) / 2 > 1.0:
        reasons.append("interval too wide (half-width > 1.0)")
    return {"variant": label, "elasticity": b, "ci95": [lo, hi], "std_error": se, "r2_within": r2,
            "observations": int(len(d)), "skus": int(d["sku"].nunique()), "price_changes": changes,
            "price_range": [float(d["price"].min()), float(d["price"].max())],
            "usable": not reasons, "reason": "; ".join(reasons) if reasons else "CI excludes zero, negative sign, enough price changes",
            "interpretation": f"a 1% price increase is associated with a {b:.2f}% change in units (95% CI {lo:.2f} to {hi:.2f})"}


def estimate_price_elasticity(session, level: str = "category", values: list[str] | None = None,
                              filters: list[dict] | None = None) -> dict:
    sem = get_semantics(session)
    dm = sem.get("demand")
    if not dm or not dm.get("price_col"):
        raise ToolError("Price elasticity needs a weekly sales table with a selling price column.")
    key = ("elasticity", level, tuple(values or []), str(filters))
    if key in session.cache:
        return session.cache[key]
    df = session.df(dm["table"])
    p, d = dm["product_col"], dm["date_col"]
    group_col = None if level in ("total", "sku") else level
    needed = [c for c in [group_col] + [f["column"] for f in (filters or [])] if c and c not in df.columns]
    df = enrich(session, dm["table"], df, needed, [])
    df = apply_filters(df, filters)
    if group_col and values:
        df = df[df[group_col].astype(str).isin([str(v) for v in values])]
    if level == "sku" and values:
        df = df[df[p].astype(str).isin([str(v) for v in values])]
    if df.empty:
        raise ToolError("No rows match the requested scope.")
    # Units sold respond to price; use unconstrained demand when present (stockouts would bias toward zero)
    units_col = dm["measure"]
    df = df.assign(_rev=df[units_col] * df[dm["price_col"]])
    promo = dm.get("promo_col")
    df = df.assign(_promo=df[promo].notna() if promo else False)
    cal = sem.get("calendar_table")
    ev = pd.Series(0.0, index=df.index)
    if cal and "shopping_event" in session.df(cal).columns:
        c = session.df(cal)
        ev_weeks = set(c.loc[c["shopping_event"].notna(), session.tables[cal].pk[0]])
        ev = df[d].isin(ev_weeks).astype(float)
    df = df.assign(_ev=ev)
    grp = [p, d] + ([group_col] if group_col else [])
    panel = df.groupby(grp).agg(units=(units_col, "sum"), rev=("_rev", "sum"), promo=("_promo", "mean"), event=("_ev", "max")).reset_index()
    panel["price"] = panel["rev"] / panel["units"].replace(0, np.nan)
    panel = panel.rename(columns={p: "sku", d: "week"}).dropna(subset=["price"])
    results = []
    groups = [(None, panel)] if not group_col else list(panel.groupby(group_col))
    for g, pn in groups:
        allv = _fit(pn, "all price variation (promotions + list-price changes)")
        lst = _fit(pn[pn["promo"] == 0], "list-price changes only (non-promotion weeks)")
        rec = lst if lst.get("usable") else (allv if allv.get("usable") else None)
        results.append({"group": g if g is not None else ("all SKUs" if not values else ", ".join(values)),
                        "all_variation": allv, "list_price_only": lst,
                        "recommended_for_permanent_price_change": (lst["variant"] if lst.get("usable") else
                                                                   (allv["variant"] + " (likely overstates response)" if allv.get("usable") else None)),
                        "recommended_elasticity": rec.get("elasticity") if rec else None,
                        "recommended_ci95": rec.get("ci95") if rec else None})
    out = to_jsonable({
        "level": level, "results": results, "measure": units_col, "price": f"weekly average {dm['price_col']} (revenue-weighted)",
        "method": "log-log regression with SKU fixed effects, Fourier week-of-year seasonality, trend and event flag; "
                  "cluster-robust (by SKU) 95% confidence intervals.",
        "caveats": ["Promotional price cuts come with display/advertising support, so 'all price variation' tends to overstate how "
                    "demand responds to a permanent price increase.",
                    "Elasticities are averages across the SKUs in each group; individual SKUs can differ.",
                    "Estimated from historical price ranges; applying them far outside those ranges is extrapolation."],
    })
    session.cache[key] = out
    return out
