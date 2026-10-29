"""Data quality report: table checks, referential integrity, time-series checks, outliers, stockouts.

Nothing is ever deleted. Every issue is reported with a count, examples and the default treatment.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .events import events_overlapping
from .semantic import get_semantics
from .timeseries import ROBUST_Z_THRESHOLD, detect_level_shifts, regularise_weekly, seasonal_decompose_robust
from .util import to_jsonable

MIN_ABS_DEVIATION = 5.0
NONNEG_PAT = re.compile(r"(units|qty|quantity|price|revenue|cost|demand|sold|on_hand|receipts|spend|capacity|refund|amount|freight)")
SIGNED_PAT = re.compile(r"(^adj_|adjust|change|delta|anomaly|diff|variance|margin|net_)")


def _status(issues: list[dict]) -> str:
    if any(i["severity"] == "error" for i in issues):
        return "error"
    if any(i["severity"] == "warning" for i in issues):
        return "warning"
    return "ok"


STRUCTURAL_BLANK = re.compile(r"(end_date|_end$|_to$|discontinue|promo_id|promotion_id|event_ids?$|shopping_event|holiday_name|^spf$|notes?$|comment|reason$)")


def date_pk_of(t):
    return next((c for c in (t.pk or []) if pd.api.types.is_datetime64_any_dtype(t.df[c])), None)


def table_checks(session) -> dict[str, dict]:
    out = {}
    sem = get_semantics(session)
    last_actual = None
    if sem.get("demand"):
        last_actual = session.df(sem["demand"]["table"])[sem["demand"]["date_col"]].max()
    for name, t in session.tables.items():
        df = t.df
        issues: list[dict] = []
        if t.pk:
            dup = int(df.duplicated(t.pk).sum())
            if dup:
                issues.append({"check": "duplicate keys", "severity": "error", "count": dup,
                               "detail": f"{dup:,} rows share a key {t.pk} with another row",
                               "treatment": "not removed; confirm which row is correct"})
        else:
            issues.append({"check": "no primary key", "severity": "warning", "count": 0,
                           "detail": "no combination of id/date columns is unique", "treatment": "joins to this table are checked for fan-out"})
        exact = int(df.duplicated().sum())
        if exact:
            issues.append({"check": "exact duplicate rows", "severity": "warning", "count": exact,
                           "detail": f"{exact:,} rows are exact copies of another row", "treatment": "kept; aggregate results would double count them"})
        miss = df.isna().mean() * 100
        for c, p in miss[miss > 0].items():
            sev, why = ("warning" if p >= 5 else "info"), ""
            if STRUCTURAL_BLANK.search(c):
                sev, why = "info", " - likely structural: blank means 'none / not applicable'"
            elif date_pk_of(t) and last_actual is not None:
                dpk = date_pk_of(t)
                miss_dates = df.loc[df[c].isna(), dpk]
                if len(miss_dates) and miss_dates.min() > last_actual:
                    sev, why = "info", f" - only in future weeks after {last_actual:%Y-%m-%d} (expected: no actuals yet)"
            issues.append({"check": "missing values", "severity": sev, "column": c, "count": int(df[c].isna().sum()),
                           "detail": f"{c}: {p:.1f}% missing{why}", "treatment": "left as missing (not imputed)"})
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and NONNEG_PAT.search(c) and not SIGNED_PAT.search(c):
                neg = int((s < 0).sum())
                if neg:
                    issues.append({"check": "negative values", "severity": "warning", "column": c, "count": neg,
                                   "detail": f"{c}: {neg:,} negative values in a quantity/price column",
                                   "treatment": "kept; excluded from demand forecasting inputs", "examples": to_jsonable(s[s < 0].head(3).tolist())})
        # Date gaps for weekly time-indexed tables
        date_pk = next((c for c in (t.pk or []) if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        if date_pk is not None:
            u = pd.Series(sorted(df[date_pk].dropna().unique()))
            if len(u) > 3:
                gap = u.diff().dt.days.dropna()
                med = gap.median()
                if med == 7:
                    missing_weeks = int(((gap / 7) - 1).clip(lower=0).sum())
                    if missing_weeks:
                        issues.append({"check": "missing weeks (table level)", "severity": "warning", "count": missing_weeks,
                                       "detail": f"{missing_weeks} calendar weeks have no rows at all", "treatment": "reported; series are re-indexed weekly when forecasting"})
        out[name] = {"status": _status(issues), "issues": issues, "rows": len(df)}
    # Referential integrity from inferred relationships
    for r in session.relationships:
        if r.orphan_count:
            out[r.child_table]["issues"].append({
                "check": "orphan keys", "severity": "warning", "count": r.orphan_count,
                "detail": f"{r.orphan_count:,} rows of {r.child_table}{r.child_columns} have no match in {r.parent_table}",
                "examples": r.orphan_examples, "treatment": "kept; lost in inner joins"})
            out[r.child_table]["status"] = _status(out[r.child_table]["issues"])
    return out


def _consistency_checks(df: pd.DataFrame, dm: dict) -> list[dict]:
    res = []
    cols = df.columns
    if dm.get("sold_col") and dm.get("lost_col") and dm["measure"] != dm.get("sold_col"):
        ok = (df[dm["sold_col"]] + df[dm["lost_col"]] == df[dm["measure"]]).mean() * 100
        res.append({"check": f"{dm['sold_col']} + {dm['lost_col']} = {dm['measure']}", "pass_pct": round(float(ok), 2)})
    if dm.get("revenue_col") and dm.get("price_col") and dm.get("sold_col"):
        ok = np.isclose(df[dm["revenue_col"]], df[dm["sold_col"]] * df[dm["price_col"]], atol=0.02).mean() * 100
        res.append({"check": f"{dm['revenue_col']} = {dm['sold_col']} x {dm['price_col']}", "pass_pct": round(float(ok), 2)})
    if dm.get("list_price_col") and dm.get("price_col") and dm.get("discount_col"):
        ok = np.isclose(df[dm["price_col"]], df[dm["list_price_col"]] * (1 - df[dm["discount_col"]] / 100), atol=0.02).mean() * 100
        res.append({"check": f"{dm['price_col']} = {dm['list_price_col']} x (1 - {dm['discount_col']}/100)", "pass_pct": round(float(ok), 2)})
    return res


def _attribution(session, dm: dict, week: pd.Timestamp, product: str | None, category: str | None) -> list[str]:
    """Explain a flagged week using promotions, calendar events and external events."""
    reasons = []
    df = session.df(dm["table"])
    d = dm["date_col"]
    if dm.get("promo_col"):
        w = df[df[d] == week]
        if product is not None:
            w = w[w[dm["product_col"]] == product]
        elif category is not None and "_category" in w.columns:
            w = w[w["_category"] == category]
        if len(w):
            share = w[dm["promo_col"]].notna().mean()
            if share > 0.2:
                ids = w[dm["promo_col"]].dropna().unique()[:3]
                reasons.append(f"promotion active on {share:.0%} of rows ({', '.join(map(str, ids))})")
    sem = get_semantics(session)
    cal = sem.get("calendar_table")
    if cal:
        c = session.df(cal)
        row = c[c[c.columns[0]] == week] if pd.api.types.is_datetime64_any_dtype(c[c.columns[0]]) else c.iloc[0:0]
        for col in c.columns:
            if ("event" in col or "holiday" in col) and len(row):
                v = row[col].iloc[0]
                if isinstance(v, str) and v:
                    reasons.append(f"calendar: {v}")
                elif col.startswith("is_") and v == 1:
                    reasons.append(f"calendar: {col.replace('is_', '').replace('_', ' ')}")
    for e in events_overlapping(session, week, week + pd.Timedelta(days=6), sku=product)[:3]:
        reasons.append(f"event {e['event_id']} ({e['side']}): {e['name']}")
    return reasons


def outlier_scan(session, level: str = "sku", max_examples: int = 25) -> dict:
    """Seasonally adjusted robust-z outlier detection on weekly demand at total/category/sku level."""
    sem = get_semantics(session)
    dm = sem["demand"]
    if not dm:
        return {"error": "no weekly demand table identified"}
    df = session.df(dm["table"])
    d, p, m = dm["date_col"], dm["product_col"], dm["measure"]
    groups: dict[str, pd.Series] = {}
    cat_of = {}
    if level == "total":
        groups["Total"] = df.groupby(d)[m].sum()
    elif level == "category":
        pt = sem.get("product_table")
        if not pt or "category" not in session.df(pt).columns:
            return {"error": "no product category available"}
        prod = session.df(pt)[[p, "category"]]
        x = df[[d, p, m]].merge(prod, on=p, how="left")
        for k, g in x.groupby("category"):
            groups[k] = g.groupby(d)[m].sum()
    else:
        for k, g in df.groupby(p):
            groups[k] = g.groupby(d)[m].sum()
    flagged = []
    series_no_season = 0
    for k, s in groups.items():
        s = regularise_weekly(s)
        if len(s) < 26:
            continue
        dec = seasonal_decompose_robust(s, log=True)
        if not dec["seasonal_used"]:
            series_no_season += 1
        z = dec["robust_z"]
        hits = z[(z.abs() > ROBUST_Z_THRESHOLD) & ((s - dec["expected"]).abs() >= MIN_ABS_DEVIATION)]
        for wk, zz in hits.items():
            expected = float(dec["expected"][wk])
            flagged.append({"series": str(k), "week": wk.strftime("%Y-%m-%d"), "value": float(s[wk]),
                            "expected": round(expected, 1), "robust_z": round(float(zz), 2),
                            "direction": "spike" if zz > 0 else "dip"})
    flagged.sort(key=lambda r: -abs(r["robust_z"]))
    for r in flagged[:max_examples]:
        r["possible_causes"] = _attribution(session, dm, pd.Timestamp(r["week"]),
                                            r["series"] if level == "sku" else None, None)
    n_series = len(groups)
    explained = sum(1 for r in flagged[:max_examples] if r.get("possible_causes"))
    return {
        "level": level, "measure": m, "series_checked": n_series, "flagged_weeks": len(flagged),
        "series_with_flags": len({r["series"] for r in flagged}),
        "examples": flagged[:max_examples],
        "examples_with_known_cause": explained,
        "series_without_seasonal_adjustment": series_no_season,
        "method": ("Robust z-score on the seasonally adjusted weekly series, computed on log(1 + demand) so each series is "
                   "judged relative to its own level: residual = value - 13-week rolling median - median week-of-year effect; "
                   "z = (residual - median) / (MAD / 0.6745); flag |z| > 3.5 and a deviation of at least 5 units. "
                   "Series with < 2 years of history are adjusted for trend only."),
        "treatment": "kept (default). Flagged weeks are not deleted; most coincide with promotions or events, which the "
                     "forecast models as features. Cap/impute options are available and their effect is reported.",
    }


def treat_outliers(y: pd.Series, method: str = "keep") -> tuple[pd.Series, dict]:
    """Apply an outlier treatment to one weekly series and report the effect. method: keep|cap|impute."""
    s = regularise_weekly(y)
    dec = seasonal_decompose_robust(s, log=True)
    z = dec["robust_z"]
    expected = dec["expected"]
    mask = (z.abs() > ROBUST_Z_THRESHOLD) & ((s - expected).abs() >= MIN_ABS_DEVIATION)
    out = s.copy()
    if method == "impute":
        out[mask] = expected[mask].round()
    elif method == "cap":
        fitted_log = dec["trend"] + dec["seasonal"] + dec["resid"].median()
        bound = ROBUST_Z_THRESHOLD * dec["scale"]
        upper = np.expm1(fitted_log + bound)
        lower = np.expm1(fitted_log - bound).clip(lower=0)
        out[mask] = s[mask].clip(lower=lower[mask], upper=upper[mask]).round()
    elif method != "keep":
        raise ValueError("method must be keep, cap or impute")
    effect = {"method": method, "weeks_flagged": int(mask.sum()),
              "total_before": float(s.sum()), "total_after": float(out.sum()),
              "total_change_pct": float((out.sum() - s.sum()) / s.sum() * 100) if s.sum() else 0.0,
              "std_before": float(s.std()), "std_after": float(out.std())}
    return out, effect


def timeseries_checks(session) -> dict:
    sem = get_semantics(session)
    dm = sem["demand"]
    if not dm:
        return {"available": False, "reason": "No weekly fact table with a product key and a demand/units measure was found."}
    df = session.df(dm["table"])
    d, p, m = dm["date_col"], dm["product_col"], dm["measure"]
    series_keys = [k for k in dm["keys"] if k != d]
    g = df.groupby(series_keys)[d].agg(["min", "max", "count"])
    span = ((g["max"] - g["min"]).dt.days // 7 + 1)
    gaps = (span - g["count"]).clip(lower=0)
    gmin, gmax = df[d].min(), df[d].max()
    sku = df.groupby(p)[d].agg(["min", "max", "nunique"]).rename(columns={"nunique": "count"})
    new = sku[sku["min"] > gmin]
    disc = sku[sku["max"] < gmax]
    pt = sem.get("product_table")
    launch_info = None
    if pt and "lifecycle_status" in session.df(pt).columns:
        launch_info = session.df(pt).set_index(p)["lifecycle_status"].to_dict()
    y = df[m]
    zeros = int((y == 0).sum())
    neg = int((y < 0).sum())
    total = df.groupby(d)[m].sum()
    total_r = regularise_weekly(total)
    shifts_total = detect_level_shifts(total_r)
    shifts_cat = {}
    if pt and "category" in session.df(pt).columns:
        x = df[[d, p, m]].merge(session.df(pt)[[p, "category"]], on=p, how="left")
        for k, gg in x.groupby("category"):
            s = regularise_weekly(gg.groupby(d)[m].sum())
            br = detect_level_shifts(s)
            if br:
                shifts_cat[k] = br
    # SKU-level structural breaks (on SKU totals)
    sku_breaks = []
    for k, gg in df.groupby(p):
        s = regularise_weekly(gg.groupby(d)[m].sum())
        br = detect_level_shifts(s, min_rel_change=0.4)
        for b in br:
            sku_breaks.append({"sku": k, **b})
    sku_breaks.sort(key=lambda r: -abs(r["change_pct"]))
    # Stockouts / censored demand
    stock = {"available": False}
    if dm.get("stockout_col"):
        so = df[dm["stockout_col"]] == 1
        stock = {"available": True, "method": f"explicit flag column '{dm['stockout_col']}'",
                 "rows_flagged": int(so.sum()), "rows_flagged_pct": round(float(so.mean() * 100), 2)}
        if dm.get("sold_col"):
            z0 = df[dm["sold_col"]] == 0
            stock["zero_sales_rows"] = int(z0.sum())
            stock["zero_sales_with_stockout"] = int((z0 & so).sum())
            stock["zero_sales_true_zero_demand"] = int((z0 & ~so).sum())
        if dm.get("lost_col"):
            lost = df[dm["lost_col"]].sum()
            stock["lost_units"] = int(lost)
            stock["lost_share_of_demand_pct"] = round(float(lost / df[m].sum() * 100), 2) if dm["measure"] != dm.get("sold_col") else None
        stock["treatment"] = (f"forecast target is '{m}' (unconstrained demand), so stockout weeks do not drag the forecast down"
                              if m != dm.get("sold_col") else
                              "stockout weeks are flagged; their sales understate demand (censored)")
    elif sem.get("inventory"):
        inv = session.df(sem["inventory"])
        avail = next((c for c in ("available_to_sell", "opening_on_hand", "closing_on_hand") if c in inv.columns), None)
        if avail and dm.get("location_col") in inv.columns:
            agg = df.groupby([p, dm["location_col"], d])[m].sum().reset_index()
            j = agg.merge(inv[[p, dm["location_col"], d, avail]], on=[p, dm["location_col"], d], how="left")
            likely = (j[m] == 0) & (j[avail] <= 0)
            stock = {"available": True, "method": f"zero sales while '{avail}' <= 0 in {sem['inventory']}",
                     "zero_sales_rows": int((j[m] == 0).sum()), "likely_stockouts": int(likely.sum()),
                     "treatment": "zero-sales weeks with no stock are treated as censored, not as zero demand"}
    hist = g["count"]
    return to_jsonable({
        "available": True,
        "table": dm["table"], "measure": m, "series_keys": series_keys,
        "date_range": {"min": gmin, "max": gmax, "weeks": int(df[d].nunique())},
        "series": int(len(g)),
        "series_with_internal_gaps": int((gaps > 0).sum()), "missing_weeks_total": int(gaps.sum()),
        "gap_examples": [{"series": list(i) if isinstance(i, tuple) else [i], "missing_weeks": int(v)} for i, v in gaps[gaps > 0].sort_values(ascending=False).head(5).items()],
        "zero_demand_rows": zeros, "zero_demand_pct": round(zeros / len(df) * 100, 2), "negative_demand_rows": neg,
        "history_weeks": {"min": int(hist.min()), "median": float(hist.median()), "max": int(hist.max())},
        "short_series_lt52": int((hist < 52).sum()), "short_series_lt104": int((hist < 104).sum()),
        "new_skus": [{"sku": k, "first_week": r["min"], "weeks": int(r["count"]), "status": (launch_info or {}).get(k)} for k, r in new.iterrows()],
        "discontinued_skus": [{"sku": k, "last_week": r["max"], "weeks": int(r["count"]), "status": (launch_info or {}).get(k)} for k, r in disc.iterrows()],
        "level_shifts_total": shifts_total,
        "level_shifts_by_category": shifts_cat,
        "sku_structural_breaks": sku_breaks[:10], "sku_structural_breaks_count": len(sku_breaks),
        "partial_years": _partial_years(df[d]),
        "stockouts": stock,
        "consistency": _consistency_checks(df, dm),
    })


def _partial_years(dates: pd.Series) -> list[dict]:
    u = pd.Series(dates.dropna().unique())
    by = u.groupby(u.dt.year).nunique()
    return [{"year": int(y), "weeks": int(n)} for y, n in by.items() if n < 52]


def data_quality_report(session, force: bool = False) -> dict:
    if session.quality is not None and not force:
        return session.quality
    tables = table_checks(session)
    ts = timeseries_checks(session)
    outl = outlier_scan(session, level="sku") if ts.get("available") else None
    outl_total = outlier_scan(session, level="total") if ts.get("available") else None
    counts = {"ok": 0, "warning": 0, "error": 0}
    for v in tables.values():
        counts[v["status"]] += 1
    rep = to_jsonable({
        "tables": tables, "table_status_counts": counts, "timeseries": ts,
        "outliers_sku": outl, "outliers_total": outl_total,
        "semantics": {k: v for k, v in get_semantics(session).items()},
    })
    session.quality = rep
    return rep
