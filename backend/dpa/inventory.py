"""Inventory & reorder recommendations (periodic-review, order-up-to policy).

Per SKU x location:
  weekly forecast      = SKU-level forecast (recommended model) x the location's share of the SKU's demand
                         over the last 13 weeks (top-down allocation)
  forecast error (sd)  = SKU backtest RMSE relative to actual demand, x the location forecast, x sqrt(1/share)
                         (a location is noisier than the SKU total; independent-noise assumption)
  protection period    = lead time L (sourcing-share-weighted planned lead time) + review period R
  safety stock         = z(service level) x sd x sqrt(L + R)
  order-up-to level S  = forecast over (L + R) + safety stock
  inventory position   = on hand (latest week) + on order (open purchase orders)
  recommended order    = max(0, S - inventory position), rounded UP to the case pack, at least the MOQ
Projected on hand week by week (on hand + scheduled PO receipts - forecast) shows the first week stock runs
out and the first week it falls below safety stock; a stockout inside the lead time cannot be fixed by a new
standard order (expedite instead).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .analytics import ToolError, store_result
from .semantic import get_semantics
from .util import df_records, to_jsonable


def _policy(session) -> pd.DataFrame | None:
    name = next((n for n, t in session.tables.items() if {"case_pack_units", "order_min_qty_units"} & set(t.df.columns)
                 and "dc_id" in t.df.columns), None)
    return session.df(name) if name else None


def _lead_times(session) -> pd.Series:
    sem = get_semantics(session)
    ps = session.df(sem["product_supplier"]) if sem.get("product_supplier") else None
    if ps is None:
        return pd.Series(dtype=float)
    lt_col = next((c for c in ("planned_lead_time_weeks", "lead_time_weeks") if c in ps.columns), None)
    if lt_col is None and sem.get("supplier_table"):
        sup = session.df(sem["supplier_table"])
        if "standard_lead_time_weeks" in sup.columns:
            ps = ps.merge(sup[["supplier_id", "standard_lead_time_weeks"]], on="supplier_id", how="left")
            lt_col = "standard_lead_time_weeks"
    if lt_col is None:
        return pd.Series(dtype=float)
    share = ps["sourcing_share"] if "sourcing_share" in ps.columns else 1.0
    w = ps.assign(_w=share, _lt=ps[lt_col] * share)
    g = w.groupby("sku_id")[["_lt", "_w"]].sum()
    return g["_lt"] / g["_w"]


def inventory_plan(session, service_level: float = 0.95, horizon_weeks: int = 13, skus: list[str] | None = None,
                   dcs: list[str] | None = None, review_period_weeks: float | None = None, top: int = 12) -> dict:
    if not 0.5 <= service_level < 1:
        raise ToolError("service_level must be between 0.5 and 0.999 (e.g. 0.95 for 95%)")
    sem = get_semantics(session)
    dm, inv_t = sem.get("demand"), sem.get("inventory")
    if not dm or not inv_t:
        raise ToolError("Reorder planning needs a weekly demand table and an inventory table (on hand by product, location and week).")
    loc = dm.get("location_col")
    p, d, m = dm["product_col"], dm["date_col"], dm["measure"]
    inv = session.df(inv_t)
    oh_col = next((c for c in ("closing_on_hand", "on_hand", "on_hand_units", "available_to_sell") if c in inv.columns), None)
    oo_col = next((c for c in ("on_order_units", "on_order") if c in inv.columns), None)
    if oh_col is None or not loc or loc not in inv.columns:
        raise ToolError(f"Inventory table '{inv_t}' needs an on-hand column and a location column.")
    # ---- SKU forecast (cached) and its backtest error
    from .forecasting import forecast_weekly, get_forecast
    horizon = max(13, int(horizon_weeks))
    fsum = forecast_weekly(session, level="sku", horizon=horizon, top_n=None)
    f = get_forecast(session, fsum["forecast_id"])
    rec = fsum["model_choice"]["recommended"]
    fc = f["forecast"].rename(columns={"series": p})
    bt = f["backtest"]
    b = bt[(bt["model"] == rec)].dropna(subset=["forecast", "actual"])
    rel_err = b.groupby("series").apply(lambda g: float(np.sqrt(((g["forecast"] - g["actual"]) ** 2).mean()) / max(g["actual"].mean(), 1e-9)),
                                        include_groups=False)
    overall_rel = float(np.sqrt(((b["forecast"] - b["actual"]) ** 2).mean()) / max(b["actual"].mean(), 1e-9)) if len(b) else 0.3
    # ---- location shares (last 13 weeks)
    sales = session.df(dm["table"])
    last = sales[d].max()
    recent = sales[sales[d] > last - pd.Timedelta(weeks=13)]
    sh = recent.groupby([p, loc])[m].sum()
    sh = (sh / sh.groupby(level=0).transform("sum")).rename("share").reset_index()
    # ---- current position
    inv_last = inv[d if d in inv.columns else "week_start"].max()
    cur = inv[inv[d if d in inv.columns else "week_start"] == inv_last]
    pos = cur.groupby([p, loc]).agg(on_hand=(oh_col, "sum"), **({"on_order_inv": (oo_col, "sum")} if oo_col else {})).reset_index()
    # open POs -> scheduled receipts by week
    receipts = pd.DataFrame(columns=[p, loc, "week", "qty"])
    on_order = None
    po_t = sem.get("purchase_orders")
    if po_t:
        po = session.df(po_t)
        if {"planned_receipt_date", "qty_ordered"} <= set(po.columns) and loc in po.columns:
            open_po = po[po["actual_receipt_date"].isna()] if "actual_receipt_date" in po.columns else po.iloc[0:0]
            if len(open_po):
                wk = open_po["planned_receipt_date"].dt.to_period("W-SUN").dt.start_time
                late = wk <= inv_last  # past-due POs: assume they land in the first forecast week
                wk = wk.where(~late, inv_last + pd.Timedelta(weeks=1))
                receipts = open_po.assign(week=wk).groupby([p, loc, "week"])["qty_ordered"].sum().rename("qty").reset_index()
                on_order = open_po.groupby([p, loc])["qty_ordered"].sum().rename("on_order").reset_index()
    pol = _policy(session)
    lts = _lead_times(session)
    # ---- assemble
    plan = sh.merge(pos, on=[p, loc], how="left")
    if on_order is not None:
        plan = plan.merge(on_order, on=[p, loc], how="left")
        plan["on_order"] = plan["on_order"].fillna(0.0)
        on_order_source = f"open purchase orders in {po_t}"
    else:
        plan["on_order"] = plan.get("on_order_inv", 0.0)
        on_order_source = f"{inv_t}.{oo_col}" if oo_col else "none available (assumed 0)"
    plan["on_hand"] = plan["on_hand"].fillna(0.0)
    plan = plan[plan[p].isin(fc[p].unique())]  # discontinued / not forecast -> excluded
    if skus:
        plan = plan[plan[p].astype(str).isin([str(s) for s in skus])]
    if dcs:
        plan = plan[plan[loc].astype(str).isin([str(s) for s in dcs])]
    if plan.empty:
        raise ToolError("No active SKU/location combinations match.")
    if pol is not None:
        cols = [c for c in ("review_period_weeks", "safety_stock_weeks", "order_min_qty_units", "case_pack_units") if c in pol.columns]
        plan = plan.merge(pol[[p, loc] + cols].drop_duplicates([p, loc]), on=[p, loc], how="left")
    for c, dflt in (("review_period_weeks", 1.0), ("order_min_qty_units", 0.0), ("case_pack_units", 1.0)):
        if c not in plan.columns:
            plan[c] = dflt
        plan[c] = plan[c].fillna(dflt)
    if review_period_weeks is not None:
        plan["review_period_weeks"] = float(review_period_weeks)
    plan["lead_time_weeks"] = plan[p].map(lts).fillna(lts.median() if len(lts) else 4.0)
    z = float(norm.ppf(service_level))
    fcw = fc.pivot_table(index=p, columns="week", values="forecast", aggfunc="sum")
    weeks = list(fcw.columns)
    rows, proj_rows = [], []
    for r in plan.itertuples(index=False):
        sku, dc = getattr(r, p), getattr(r, loc)
        share = float(r.share) if r.share == r.share else 0.0
        if sku not in fcw.index or share <= 0:
            continue
        f_week = fcw.loc[sku].values * share
        mean_w = float(np.nanmean(f_week[:13])) if len(f_week) else 0.0
        L, R = float(r.lead_time_weeks), float(r.review_period_weeks)
        prot = L + R
        n_full = int(math.floor(prot))
        demand_prot = float(np.nansum(f_week[:n_full]) + (prot - n_full) * (f_week[n_full] if n_full < len(f_week) else mean_w))
        rel = float(rel_err.get(str(sku), overall_rel))
        sd_w = rel * mean_w * math.sqrt(1 / max(share, 0.05))
        ss = z * sd_w * math.sqrt(prot)
        S = demand_prot + ss
        ip = float(r.on_hand) + float(r.on_order)
        raw = max(0.0, S - ip)
        cp = max(float(r.case_pack_units), 1.0)
        qty = 0.0 if raw <= 0 else max(math.ceil(raw / cp) * cp, float(r.order_min_qty_units))
        # projected on hand with scheduled receipts, no new order
        rc = receipts[(receipts[p] == sku) & (receipts[loc] == dc)].set_index("week")["qty"] if len(receipts) else pd.Series(dtype=float)
        oh = float(r.on_hand)
        first_out, first_below = None, None
        for k, wk in enumerate(weeks[: max(int(horizon_weeks), 13)]):
            oh = oh + float(rc.get(wk, 0.0)) - float(f_week[k])
            proj_rows.append({p: sku, loc: dc, "week": wk, "projected_on_hand": oh})
            if first_below is None and oh < ss:
                first_below = wk
            if first_out is None and oh < 0:
                first_out = wk
        lt_end = weeks[min(int(math.ceil(L)), len(weeks) - 1)] if weeks else None
        urgency = ("stockout within lead time - expedite" if first_out is not None and lt_end is not None and first_out <= lt_end
                   else "reorder this cycle" if qty > 0 else "ok")
        policy_ss = float(r.safety_stock_weeks) * mean_w if hasattr(r, "safety_stock_weeks") and r.safety_stock_weeks == r.safety_stock_weeks else None
        rows.append({p: sku, loc: dc, "share_of_sku_demand": share, "avg_weekly_forecast": mean_w,
                     "on_hand": float(r.on_hand), "on_order": float(r.on_order), "weeks_of_cover": float(r.on_hand) / mean_w if mean_w > 0 else None,
                     "lead_time_weeks": L, "review_period_weeks": R, "forecast_error_sd_week": sd_w,
                     "safety_stock": ss, "policy_safety_stock": policy_ss, "order_up_to": S, "inventory_position": ip,
                     "recommended_order_units": qty, "case_pack": cp, "moq": float(r.order_min_qty_units),
                     "first_stockout_week": first_out, "first_below_safety_week": first_below, "status": urgency})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ToolError("No forecastable SKU/location combinations.")
    # value the orders at blended landed cost when cost data exists
    try:
        from .scenarios import cost_basis
        cb = cost_basis(session)
        cost = (cb.assign(w=cb["landed_cost"] * cb["sourcing_share"]).groupby("sku_id")["w"].sum() /
                cb.groupby("sku_id")["sourcing_share"].sum())
        out["unit_landed_cost"] = out[p].map(cost)
        out["order_value"] = out["recommended_order_units"] * out["unit_landed_cost"]
    except ToolError:
        pass
    order_rank = {"stockout within lead time - expedite": 0, "reorder this cycle": 1, "ok": 2}
    out = out.sort_values(["status", "first_stockout_week", "recommended_order_units"],
                          key=lambda s: s.map(order_rank) if s.name == "status" else s, ascending=[True, True, False], na_position="last")
    proj = pd.DataFrame(proj_rows)
    pid = f"inv_{len(session.cache.get('inventory_plans', {})) + 1}"
    session.cache.setdefault("inventory_plans", {})[pid] = {"plan": out, "projection": proj, "params": {
        "service_level": service_level, "horizon_weeks": horizon_weeks}}
    rid = store_result(session, out, {"tool": "inventory_plan", "plan_id": pid, "table": inv_t})
    counts = out["status"].value_counts().to_dict()
    checks = []
    if on_order is not None and oo_col:
        inv_oo = float(cur[oo_col].sum())
        po_oo = float(on_order["on_order"].sum())
        checks.append({"check": f"open PO quantity vs {inv_t}.{oo_col}", "open_po_units": po_oo, "inventory_on_order_units": inv_oo,
                       "difference_pct": (po_oo - inv_oo) / inv_oo * 100 if inv_oo else None})
    summ = {
        "plan_id": pid, "result_id": rid, "as_of_inventory_week": inv_last,
        "sku_locations": int(len(out)), "status_counts": counts,
        "total_recommended_units": float(out["recommended_order_units"].sum()),
        "total_order_value": float(out["order_value"].sum()) if "order_value" in out.columns else None,
        "service_level": service_level, "z": z, "forecast_model": rec,
        "forecast_backtest_wape_pct": fsum["backtest"]["metrics"][rec]["wape_pct"],
        "top_actions": df_records(out.head(min(int(top), 30))[[c for c in (p, loc, "status", "on_hand", "on_order", "avg_weekly_forecast",
                                                           "safety_stock", "recommended_order_units", "first_stockout_week",
                                                           "order_value") if c in out.columns]].round(
            {c: 1 for c in out.columns if pd.api.types.is_float_dtype(out[c])})),
        "full_plan": "all rows are in the plan's Excel/CSV download (result_id above)",
        "consistency_checks": checks,
        "statistical_vs_policy_safety_stock_median_ratio": float((out["safety_stock"] / out["policy_safety_stock"]).median())
        if "policy_safety_stock" in out.columns and out["policy_safety_stock"].notna().any() else None,
        "excluded": "discontinued SKUs and locations with no demand in the last 13 weeks",
        "inputs": {"on_hand": f"{inv_t}.{oh_col} at {inv_last:%Y-%m-%d}", "on_order": on_order_source,
                   "lead_time": "sourcing-share-weighted planned lead time per SKU (product-supplier table)",
                   "review_period, case pack, MOQ, policy safety stock": "assortment policy table" if pol is not None else "defaults (1 week, 1 unit, 0)",
                   "allocation": "SKU forecast split by each location's share of the last 13 weeks of demand"},
        "method": __doc__.split("Per SKU x location:")[1].strip(),
        "assumptions": [f"service level {service_level:.0%} (z = {z:.2f}) - a choice, not a fact; change it per SKU class if needed",
                        "location forecast error = SKU relative error x sqrt(1 / location share) (independent noise)",
                        "past-due open POs are assumed to arrive in the first forecast week"],
        "source_labels": {"on_hand, on_order, lead times, case packs, MOQ": "computed",
                          "forecast, safety stock, recommended orders, stockout weeks": "estimated",
                          "service level": "assumed"},
    }
    return to_jsonable(summ)


def get_plan(session, pid: str) -> dict:
    p = session.cache.get("inventory_plans", {}).get(pid)
    if p is None:
        raise ToolError(f"Unknown plan_id '{pid}'")
    return p


def inventory_chart(session, pid: str, sku: str | None = None, dc: str | None = None) -> dict:
    """Projected on hand vs safety stock for one SKU-location (default: the most urgent one)."""
    from .charts import SERIES, _store, base_layout
    pl = get_plan(session, pid)
    plan, proj = pl["plan"], pl["projection"]
    pc, lc = plan.columns[0], plan.columns[1]
    row = plan.iloc[0] if sku is None else plan[(plan[pc].astype(str) == str(sku)) & ((plan[lc].astype(str) == str(dc)) if dc else True)].iloc[0]
    s, l = row[pc], row[lc]
    pr = proj[(proj[pc] == s) & (proj[lc] == l)].sort_values("week")
    lay = base_layout(f"{s} at {l}: projected on hand without a new order", "Week", "Units on hand (end of week)")
    tr = [{"type": "scatter", "mode": "lines+markers", "x": list(pr["week"]), "y": list(pr["projected_on_hand"]), "name": "Projected on hand",
           "line": {"color": SERIES[0], "width": 2}, "marker": {"size": 5}},
          {"type": "scatter", "mode": "lines", "x": list(pr["week"]), "y": [row["safety_stock"]] * len(pr), "name": "Safety stock",
           "line": {"color": SERIES[3], "width": 1.5, "dash": "dash"}}]
    lay["yaxis"]["rangemode"] = "normal"
    lay["shapes"].append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0, "line": {"color": "rgba(179,54,47,0.6)", "width": 1}})
    cid = _store(session, {"data": tr, "layout": lay}, {"plan_id": pid, "sku": str(s), "dc": str(l)})
    return {"chart_id": cid, "sku": s, "location": l, "status": row["status"]}
