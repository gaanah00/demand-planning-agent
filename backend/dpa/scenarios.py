"""Scenario & impact analysis: structured scenario objects, direct cost effects, demand effects only when
defensible, baseline-vs-scenario comparison, supplier exposure, mitigations, sensitivity and comparisons.

Every output number carries a source label:
  computed  - straight from the data and the user-stated event parameters (e.g. landed cost per unit)
  estimated - depends on a model (the demand forecast, a fitted elasticity)
  assumed   - depends on a user-supplied or hypothetical assumption (demand change, holding cost rate)
"""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd

from .analytics import ToolError
from .semantic import get_semantics
from .session import now_iso
from .util import to_jsonable

TODAY = pd.Timestamp(date.today())
EVENT_TYPES = ["tariff", "supplier_cost_increase", "supplier_failure", "natural_disaster", "logistics_disruption",
               "economic_shift", "competitor_move", "promotion", "damaged_inventory", "demand_shock", "other"]
SCOPE_KEYS = ["supplier_ids", "supplier_countries", "skus", "categories", "subcategories", "brands", "dcs", "regions"]
DEFAULT_HOLDING_RATE = 0.25


def _today(session) -> pd.Timestamp:
    t = session.settings.get("today")
    return pd.Timestamp(t) if t else TODAY


# ============================================================================ reference data

def cost_basis(session, window_weeks: int = 26) -> pd.DataFrame:
    """Per SKU x supplier: unit cost, current tariff rate, freight per unit, landed cost, sourcing share,
    lead time, supplier country/capacity. All computed from purchase orders + contracts."""
    if "cost_basis" in session.cache:
        return session.cache["cost_basis"]
    sem = get_semantics(session)
    po_t, ps_t, sup_t = sem.get("purchase_orders"), sem.get("product_supplier"), sem.get("supplier_table")
    if not (po_t and ps_t and sup_t):
        raise ToolError("Cost analysis needs purchase orders, a product-supplier table and a supplier table.")
    po, ps, sup = session.df(po_t), session.df(ps_t), session.df(sup_t)
    date_col = "order_date" if "order_date" in po.columns else next(c for c in po.columns if pd.api.types.is_datetime64_any_dtype(po[c]))
    cutoff = po[date_col].max() - pd.Timedelta(weeks=window_weeks)
    recent = po[po[date_col] >= cutoff]
    qty = "qty_ordered" if "qty_ordered" in po.columns else "qty"
    g = recent.assign(_val=recent["unit_cost_usd"] * recent[qty]).groupby(["sku_id", "supplier_id"]).agg(
        val=("_val", "sum"), qty=(qty, "sum"), freight=("freight_cost_usd", "sum"), orders=(qty, "size"))
    g["po_unit_cost"] = g["val"] / g["qty"]
    g["freight_per_unit"] = g["freight"] / g["qty"]
    # latest tariff rate per supplier (by order date) - tariffs apply by country of origin
    last_tariff = po.sort_values(date_col).groupby("supplier_id")["tariff_rate_pct"].last()
    sup_freight = recent.groupby("supplier_id").apply(lambda x: x["freight_cost_usd"].sum() / x[qty].sum(), include_groups=False)
    b = ps.merge(sup, on="supplier_id", how="left", suffixes=("", "_supplier"))
    b = b.merge(g[["po_unit_cost", "freight_per_unit", "qty", "orders"]].reset_index(), on=["sku_id", "supplier_id"], how="left")
    b["unit_cost"] = b["po_unit_cost"].fillna(b.get("contract_unit_cost_usd"))
    b["unit_cost_source"] = np.where(b["po_unit_cost"].notna(), f"POs last {window_weeks} weeks (qty-weighted)", "contract cost (no recent POs)")
    b["freight_per_unit"] = b["freight_per_unit"].fillna(b["supplier_id"].map(sup_freight)).fillna(0.0)
    b["tariff_pct"] = b["supplier_id"].map(last_tariff).fillna(0.0)
    b["landed_cost"] = b["unit_cost"] * (1 + b["tariff_pct"] / 100) + b["freight_per_unit"]
    b["lead_time_weeks"] = b.get("planned_lead_time_weeks", b.get("standard_lead_time_weeks"))
    b = b.rename(columns={"qty": "recent_qty_ordered"})
    # supplier weekly volume (all SKUs) over the window -> capacity headroom
    vol = recent.groupby("supplier_id")[qty].sum() / window_weeks
    b["supplier_recent_weekly_units"] = b["supplier_id"].map(vol).fillna(0.0)
    session.cache["cost_basis"] = b
    return b


def price_basis(session, weeks: int = 13) -> pd.Series:
    sem = get_semantics(session)
    dm = sem["demand"]
    df = session.df(dm["table"])
    last = df[dm["date_col"]].max()
    r = df[df[dm["date_col"]] > last - pd.Timedelta(weeks=weeks)]
    sold = dm.get("sold_col") or dm["measure"]
    rev = r.groupby(dm["product_col"])[dm["revenue_col"]].sum() if dm.get("revenue_col") else \
        (r[sold] * r[dm["price_col"]]).groupby(r[dm["product_col"]]).sum()
    units = r.groupby(dm["product_col"])[sold].sum()
    return (rev / units.replace(0, np.nan)).rename("price")


def sku_attributes(session) -> pd.DataFrame:
    sem = get_semantics(session)
    prod = session.df(sem["product_table"]).copy()
    brand = next((n for n, t in session.tables.items() if t.role == "dimension" and t.pk == ["brand_id"]), None)
    if brand and "brand_id" in prod.columns:
        prod = prod.merge(session.df(brand)[["brand_id"] + [c for c in ("brand_name",) if c in session.df(brand).columns]],
                          on="brand_id", how="left")
    return prod


# ============================================================================ scenario objects

def _norm_list(v):
    if v is None:
        return []
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def resolve_scope(session, scope: dict) -> dict:
    """Turn a scope (suppliers, countries, SKUs, categories...) into concrete SKU x supplier pairs."""
    cb = cost_basis(session)
    attrs = sku_attributes(session)
    sup_ids = set(_norm_list(scope.get("supplier_ids")))
    countries = {c.lower() for c in _norm_list(scope.get("supplier_countries"))}
    sup_mask = pd.Series(True, index=cb.index)
    if sup_ids or countries:
        sup_mask = cb["supplier_id"].isin(sup_ids) | cb["country"].str.lower().isin(countries)
    sku_mask_attr = pd.Series(True, index=attrs.index)
    for key, col in (("skus", "sku_id"), ("categories", "category"), ("subcategories", "subcategory"), ("brands", "brand_name")):
        vals = {v.lower() for v in _norm_list(scope.get(key))}
        if vals and col in attrs.columns:
            m = attrs[col].astype(str).str.lower().isin(vals)
            if key == "brands" and "brand_id" in attrs.columns:
                m = m | attrs["brand_id"].astype(str).str.lower().isin(vals)
            sku_mask_attr &= m
    skus = set(attrs.loc[sku_mask_attr, "sku_id"].astype(str))
    pairs = cb[sup_mask & cb["sku_id"].astype(str).isin(skus)]
    unknown = []
    if countries and not cb["country"].str.lower().isin(countries).any():
        unknown.append(f"no supplier is located in {sorted(countries)} (supplier countries in data: {sorted(cb['country'].dropna().unique())})")
    if sup_ids - set(cb["supplier_id"]):
        unknown.append(f"unknown supplier ids {sorted(sup_ids - set(cb['supplier_id']))}")
    return {"pairs": pairs, "skus": sorted(pairs["sku_id"].unique(), key=lambda x: (len(x), x)),
            "suppliers": sorted(pairs["supplier_id"].unique()), "unmatched": unknown,
            "dcs": _norm_list(scope.get("dcs")), "regions": _norm_list(scope.get("regions"))}


def create_scenario(session, name: str, event_type: str, description: str = "", scope: dict | None = None,
                    start_date: str | None = None, end_date: str | None = None,
                    tariff_pct: float | None = None, tariff_basis: str = "additive_points",
                    unit_cost_change_pct: float | None = None, supply_capacity_change_pct: float | None = None,
                    lead_time_change_weeks: float | None = None, price_pass_through_pct: float | None = None,
                    demand_effect_pct: float | None = None, demand_effect_source: str = "none",
                    elasticity: float | None = None, horizon_weeks: int = 26, cost_timing: str = "after_current_stock",
                    holding_cost_rate: float | None = None, damaged_units: dict | None = None) -> dict:
    """Build (or replace) a draft scenario. Returns the structured object for user confirmation."""
    if event_type not in EVENT_TYPES:
        raise ToolError(f"event_type must be one of {EVENT_TYPES}")
    if not name or len(name) > 60:
        raise ToolError("Give the scenario a short name (<= 60 characters).")
    if demand_effect_source not in ("none", "assumed", "estimated"):
        raise ToolError("demand_effect_source must be none, assumed or estimated")
    if demand_effect_source == "none" and (demand_effect_pct is not None or elasticity is not None):
        raise ToolError("A demand effect was given but demand_effect_source is 'none'. Use 'assumed' (user-provided) or "
                        "'estimated' (from estimate_price_elasticity).")
    if tariff_basis not in ("additive_points", "replace_rate"):
        raise ToolError("tariff_basis must be additive_points or replace_rate")
    if cost_timing not in ("after_current_stock", "immediate"):
        raise ToolError("cost_timing must be after_current_stock or immediate")
    scope = {k: _norm_list((scope or {}).get(k)) for k in SCOPE_KEYS}
    res = resolve_scope(session, scope)
    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    if start is not None and end is not None and end < start:
        raise ToolError("end_date is before start_date")
    assumptions, questions = [], []
    if event_type == "tariff":
        if tariff_pct is None:
            questions.append("What tariff rate applies (in %)?")
        assumptions.append({"field": "tariff_basis", "value": tariff_basis, "source": "default" if tariff_basis == "additive_points" else "user",
                            "text": ("New tariff is ADDED to each supplier's current rate (percentage points)" if tariff_basis == "additive_points"
                                     else "New tariff REPLACES the current rate")})
        assumptions.append({"field": "tariff_applies_to", "value": "supplier unit cost", "source": "default",
                            "text": "Tariff is charged on the supplier unit cost (customs value); freight is not tariffed"})
    if start is None:
        questions.append("When does the event start?")
    if price_pass_through_pct is None and event_type in ("tariff", "supplier_cost_increase"):
        price_pass_through_pct = 0.0
        assumptions.append({"field": "price_pass_through_pct", "value": 0, "source": "default",
                            "text": "No price pass-through (cost increase absorbed in margin) until you choose a pass-through %"})
    elif price_pass_through_pct is not None:
        assumptions.append({"field": "price_pass_through_pct", "value": price_pass_through_pct, "source": "user",
                            "text": f"{price_pass_through_pct:g}% of the landed-cost increase is passed into selling price"})
    assumptions.append({"field": "cost_timing", "value": cost_timing, "source": "default" if cost_timing == "after_current_stock" else "user",
                        "text": ("Cost changes hit units sold after stock already on hand or arriving before the start date is used up (FIFO)"
                                 if cost_timing == "after_current_stock" else "Cost changes hit every unit sold from the start date")})
    if demand_effect_source == "none":
        assumptions.append({"field": "demand_effect", "value": None, "source": "default",
                            "text": "No demand effect applied (none estimated or provided yet)"})
    elif demand_effect_source == "assumed":
        assumptions.append({"field": "demand_effect", "value": demand_effect_pct if demand_effect_pct is not None else elasticity,
                            "source": "user", "text": (f"Demand changes by {demand_effect_pct:+g}% for affected SKUs (user assumption)"
                                                       if demand_effect_pct is not None else f"Price elasticity {elasticity:g} (user assumption)")})
    else:
        assumptions.append({"field": "demand_effect", "value": elasticity, "source": "estimated",
                            "text": f"Price elasticity {elasticity:g} estimated from historical price variation"})
    hr = holding_cost_rate if holding_cost_rate is not None else DEFAULT_HOLDING_RATE
    assumptions.append({"field": "holding_cost_rate", "value": hr, "source": "user" if holding_cost_rate is not None else "default",
                        "text": f"Inventory holding cost {hr:.0%} of unit cost per year (used only for pre-buy / safety stock)"})
    if not res["skus"]:
        questions.append("No SKU-supplier pairs match this scope: " + ("; ".join(res["unmatched"]) or "check the scope") + ". Which suppliers or products should it apply to?")
    spec = {
        "name": name, "event_type": event_type, "description": description, "scope": scope,
        "start_date": start, "end_date": end, "tariff_pct": tariff_pct, "tariff_basis": tariff_basis,
        "unit_cost_change_pct": unit_cost_change_pct, "supply_capacity_change_pct": supply_capacity_change_pct,
        "lead_time_change_weeks": lead_time_change_weeks, "price_pass_through_pct": price_pass_through_pct,
        "demand_effect": {"source": demand_effect_source, "pct": demand_effect_pct, "elasticity": elasticity},
        "horizon_weeks": int(horizon_weeks), "cost_timing": cost_timing, "holding_cost_rate": hr,
        "damaged_units": damaged_units,
    }
    session.scenarios[name] = {"spec": spec, "status": "draft", "created": now_iso(), "result": None,
                               "assumptions": assumptions}
    return to_jsonable({
        "scenario": spec, "status": "draft - needs your confirmation before it is applied",
        "resolved_scope": {"skus_affected": len(res["skus"]), "skus": res["skus"][:30], "suppliers": res["suppliers"],
                           "sku_supplier_pairs": int(len(res["pairs"])), "unmatched": res["unmatched"]},
        "assumptions": assumptions, "open_questions": questions,
    })


def confirm_scenario(session, name: str) -> dict:
    sc = session.scenarios.get(name)
    if not sc:
        raise ToolError(f"No scenario named '{name}'. Existing: {list(session.scenarios)}")
    if sc["spec"].get("start_date") is None:
        raise ToolError("Scenario has no start date yet; ask the user for it before confirming.")
    if sc["spec"]["event_type"] == "tariff" and sc["spec"].get("tariff_pct") is None:
        raise ToolError("Scenario has no tariff rate yet; ask the user before confirming.")
    sc["status"] = "confirmed"
    sc["confirmed_at"] = now_iso()
    for a in sc["assumptions"]:
        session.add_assumption(a["text"], a["source"], a.get("value"), scenario=name)
    return {"name": name, "status": "confirmed", "assumptions_logged": len(sc["assumptions"])}


# ============================================================================ apply

def _baseline_units(session, horizon: int) -> pd.DataFrame:
    """SKU x week baseline demand from the recommended SKU-level forecast (cached)."""
    from .forecasting import forecast_weekly, get_forecast
    s = forecast_weekly(session, level="sku", horizon=max(13, horizon), top_n=None)
    f = get_forecast(session, s["forecast_id"])
    fc = f["forecast"].rename(columns={"series": "sku_id"})[["sku_id", "week", "forecast", "lower", "upper"]]
    return fc[fc["week"] <= fc["week"].min() + pd.Timedelta(weeks=horizon - 1)], s


def _protected_units(session, eff: pd.Timestamp, fc: pd.DataFrame) -> pd.Series:
    """Units per SKU that will still be on hand at the effective date and were bought at the old cost:
    on hand now + open PO receipts planned before the effective date - demand until the effective date."""
    sem = get_semantics(session)
    inv_t, po_t = sem.get("inventory"), sem.get("purchase_orders")
    out = pd.Series(0.0, index=fc["sku_id"].unique())
    if not inv_t:
        return out
    inv = session.df(inv_t)
    last = inv["week_start"].max()
    on_hand = inv[inv["week_start"] == last].groupby("sku_id")["closing_on_hand"].sum()
    arriving = pd.Series(0.0)
    if po_t:
        po = session.df(po_t)
        open_po = po[po["actual_receipt_date"].isna()] if "actual_receipt_date" in po.columns else po.iloc[0:0]
        arriving = open_po[open_po["planned_receipt_date"] < eff].groupby("sku_id")["qty_ordered"].sum()
    before = fc[fc["week"] < eff].groupby("sku_id")["forecast"].sum()
    prot = on_hand.reindex(out.index).fillna(0) + arriving.reindex(out.index).fillna(0) - before.reindex(out.index).fillna(0)
    return prot.clip(lower=0)


def _run(session, spec: dict) -> dict:
    horizon = spec["horizon_weeks"]
    fc, fsum = _baseline_units(session, horizon)
    cb = cost_basis(session)
    price = price_basis(session)
    attrs = sku_attributes(session).set_index("sku_id")
    res = resolve_scope(session, spec["scope"])
    pairs = res["pairs"]
    eff = spec["start_date"]
    end = spec["end_date"] if spec["end_date"] is not None else pd.Timestamp("2100-01-01")
    # --- per SKU x supplier cost under baseline and scenario
    cbs = cb[["sku_id", "supplier_id", "supplier_name", "country", "sourcing_share", "unit_cost", "tariff_pct",
              "freight_per_unit", "landed_cost", "lead_time_weeks"]].copy()
    cbs["affected"] = cbs.set_index(["sku_id", "supplier_id"]).index.isin(pairs.set_index(["sku_id", "supplier_id"]).index)
    cbs["new_tariff_pct"] = cbs["tariff_pct"]
    cbs["new_unit_cost"] = cbs["unit_cost"]
    if spec.get("tariff_pct") is not None:
        add = float(spec["tariff_pct"])
        cbs.loc[cbs["affected"], "new_tariff_pct"] = (cbs.loc[cbs["affected"], "tariff_pct"] + add) if spec["tariff_basis"] == "additive_points" else add
    if spec.get("unit_cost_change_pct") is not None:
        cbs.loc[cbs["affected"], "new_unit_cost"] = cbs.loc[cbs["affected"], "unit_cost"] * (1 + spec["unit_cost_change_pct"] / 100)
    cbs["new_landed_cost"] = cbs["new_unit_cost"] * (1 + cbs["new_tariff_pct"] / 100) + cbs["freight_per_unit"]
    cbs["landed_cost_delta"] = cbs["new_landed_cost"] - cbs["landed_cost"]
    # blended per SKU (by sourcing share)
    blend = cbs.assign(w_old=cbs["landed_cost"] * cbs["sourcing_share"], w_new=cbs["new_landed_cost"] * cbs["sourcing_share"],
                       w_aff=cbs["affected"] * cbs["sourcing_share"]).groupby("sku_id")[["w_old", "w_new", "w_aff", "sourcing_share"]].sum()
    blend["cost_old"] = blend["w_old"] / blend["sourcing_share"]
    blend["cost_new"] = blend["w_new"] / blend["sourcing_share"]
    blend["affected_share"] = blend["w_aff"] / blend["sourcing_share"]
    # --- weekly SKU panel
    p = fc.merge(blend[["cost_old", "cost_new", "affected_share"]], left_on="sku_id", right_index=True, how="left")
    p["price"] = p["sku_id"].map(price)
    missing_cost = p["cost_old"].isna().groupby(p["sku_id"]).any()
    p = p.dropna(subset=["cost_old", "price"])
    in_window = (p["week"] >= eff) & (p["week"] <= end)
    p["is_affected_sku"] = p["affected_share"] > 0
    # cost timing: FIFO protection of stock bought at the old cost
    prot = _protected_units(session, eff, fc) if spec["cost_timing"] == "after_current_stock" else pd.Series(0.0, index=p["sku_id"].unique())
    p = p.sort_values(["sku_id", "week"])
    p["cum_in_window"] = p["forecast"].where(in_window, 0).groupby(p["sku_id"]).cumsum()
    p["prot"] = p["sku_id"].map(prot).fillna(0.0)
    p["new_cost_units_share"] = np.where(in_window, ((p["cum_in_window"] - p["prot"]).clip(lower=0).clip(upper=p["forecast"]) /
                                                     p["forecast"].replace(0, np.nan)).fillna(0), 0.0)
    # baseline
    p["units_base"] = p["forecast"]
    p["rev_base"] = p["units_base"] * p["price"]
    p["cogs_base"] = p["units_base"] * p["cost_old"]
    # scenario price & demand
    unit_cost_now = p["cost_old"] + (p["cost_new"] - p["cost_old"]) * p["new_cost_units_share"]
    pt = (spec.get("price_pass_through_pct") or 0) / 100
    p["price_scen"] = np.where(in_window & p["is_affected_sku"], p["price"] + pt * (p["cost_new"] - p["cost_old"]), p["price"])
    de = spec["demand_effect"]
    mult = np.ones(len(p))
    if de["source"] != "none":
        if de.get("pct") is not None:
            mult = np.where(in_window & p["is_affected_sku"], 1 + de["pct"] / 100, 1.0)
        elif de.get("elasticity") is not None:
            mult = np.where(in_window & p["is_affected_sku"], (p["price_scen"] / p["price"]) ** de["elasticity"], 1.0)
    cap = spec.get("supply_capacity_change_pct")
    supply_mult = np.ones(len(p))
    if cap is not None and cap < 0:
        # affected supply share loses capacity -> units unavailable unless mitigated
        supply_mult = np.where(in_window, 1 + p["affected_share"] * cap / 100, 1.0)
    dmg = spec.get("damaged_units") or {}
    p["units_scen"] = p["units_base"] * mult * supply_mult
    p["rev_scen"] = p["units_scen"] * p["price_scen"]
    p["cogs_scen"] = p["units_scen"] * unit_cost_now
    for c in ("base", "scen"):
        p[f"gm_{c}"] = p[f"rev_{c}"] - p[f"cogs_{c}"]
    write_off = 0.0
    if dmg:
        for sku, units in dmg.items():
            write_off += float(units) * float(blend["cost_old"].get(sku, np.nan) if sku in blend.index else 0.0)
    # --- aggregates
    def tot(df):
        return {"units": df["units_base"].sum(), "revenue": df["rev_base"].sum(), "cogs": df["cogs_base"].sum(), "gross_margin": df["gm_base"].sum(),
                "units_s": df["units_scen"].sum(), "revenue_s": df["rev_scen"].sum(), "cogs_s": df["cogs_scen"].sum(), "gross_margin_s": df["gm_scen"].sum() - write_off}
    T = tot(p)
    sku = p.groupby("sku_id").agg(units_base=("units_base", "sum"), units_scen=("units_scen", "sum"), rev_base=("rev_base", "sum"),
                                  rev_scen=("rev_scen", "sum"), cogs_base=("cogs_base", "sum"), cogs_scen=("cogs_scen", "sum"),
                                  gm_base=("gm_base", "sum"), gm_scen=("gm_scen", "sum"), price=("price", "first"),
                                  cost_old=("cost_old", "first"), cost_new=("cost_new", "first"), affected_share=("affected_share", "first"))
    sku["gm_delta"] = sku["gm_scen"] - sku["gm_base"]
    sku["margin_pct_base"] = sku["gm_base"] / sku["rev_base"] * 100
    sku["margin_pct_scen"] = sku["gm_scen"] / sku["rev_scen"].replace(0, np.nan) * 100
    sku["unit_margin_old"] = sku["price"] - sku["cost_old"]
    sku["unit_margin_new_full"] = sku["price"] - sku["cost_new"]
    sku["protected_units"] = sku.index.map(prot).fillna(0.0) if len(prot) else 0.0
    for c in ("product_name", "category", "subcategory"):
        if c in attrs.columns:
            sku[c] = sku.index.map(attrs[c])
    sku = sku.sort_values("gm_delta")
    # per supplier: COGS attributable by sourcing share
    ps = p[["sku_id", "units_base", "units_scen", "new_cost_units_share", "week"]].merge(
        cbs[["sku_id", "supplier_id", "supplier_name", "country", "sourcing_share", "landed_cost", "new_landed_cost", "tariff_pct", "new_tariff_pct", "affected", "unit_cost"]],
        on="sku_id", how="left")
    ps["units_b"] = ps["units_base"] * ps["sourcing_share"]
    ps["units_s"] = ps["units_scen"] * ps["sourcing_share"]
    ps["cogs_b"] = ps["units_b"] * ps["landed_cost"]
    ps["cogs_s"] = ps["units_s"] * (ps["landed_cost"] + (ps["new_landed_cost"] - ps["landed_cost"]) * ps["new_cost_units_share"])
    ps["tariff_b"] = ps["units_b"] * ps["unit_cost"] * ps["tariff_pct"] / 100
    ps["tariff_s"] = ps["units_s"] * ps["unit_cost"] * (ps["tariff_pct"] + (ps["new_tariff_pct"] - ps["tariff_pct"]) * ps["new_cost_units_share"]) / 100
    sup = ps.groupby(["supplier_id", "supplier_name", "country"]).agg(
        units=("units_b", "sum"), cogs_base=("cogs_b", "sum"), cogs_scen=("cogs_s", "sum"), tariff_base=("tariff_b", "sum"),
        tariff_scen=("tariff_s", "sum"), affected=("affected", "max"), skus=("sku_id", "nunique")).reset_index()
    sup["cogs_delta"] = sup["cogs_scen"] - sup["cogs_base"]
    sup["avg_landed_cost_base"] = sup["cogs_base"] / sup["units"].replace(0, np.nan)
    sup["avg_landed_cost_scen"] = sup["cogs_scen"] / sup["units"].replace(0, np.nan)
    sup = sup.sort_values("cogs_delta", ascending=False)
    country = sup.groupby("country").agg(units=("units", "sum"), cogs_base=("cogs_base", "sum"), cogs_scen=("cogs_scen", "sum"),
                                         cogs_delta=("cogs_delta", "sum"), suppliers=("supplier_id", "nunique")).reset_index().sort_values("cogs_delta", ascending=False)
    weekly = p.groupby("week")[["units_base", "units_scen", "rev_base", "rev_scen", "cogs_base", "cogs_scen", "gm_base", "gm_scen"]].sum().reset_index()
    # Full run-rate (all units at new cost, demand as scenario) for affected SKUs, per 52 weeks
    aff = p[p["is_affected_sku"]]
    wk_units = aff.groupby("week")["units_base"].sum().mean() if len(aff) else 0.0
    unit_delta = (aff.groupby("sku_id")["cost_new"].first() - aff.groupby("sku_id")["cost_old"].first()) if len(aff) else pd.Series(dtype=float)
    weekly_units_sku = aff.groupby("sku_id")["units_base"].mean() if len(aff) else pd.Series(dtype=float)
    annual_run_rate = float((unit_delta * weekly_units_sku).sum() * 52) if len(aff) else 0.0
    return {"panel": p, "sku": sku, "supplier": sup, "country": country, "weekly": weekly, "totals": T,
            "cost_pairs": cbs, "forecast_summary": fsum, "protected": prot, "annual_cost_run_rate": annual_run_rate,
            "weekly_units_affected": wk_units, "missing_cost_skus": list(missing_cost[missing_cost].index), "write_off": write_off,
            "resolved": res}


def apply_scenario(session, name: str) -> dict:
    sc = session.scenarios.get(name)
    if not sc:
        raise ToolError(f"No scenario named '{name}'. Existing: {list(session.scenarios)}")
    if sc["status"] != "confirmed":
        raise ToolError(f"Scenario '{name}' is still a draft. Show it to the user and get explicit confirmation, then call confirm_scenario.")
    spec = sc["spec"]
    r = _run(session, spec)
    sc["result"] = r
    T = r["totals"]
    d = lambda a, b: T[b] - T[a]
    de = spec["demand_effect"]
    demand_label = "estimated" if de["source"] == "estimated" else ("assumed" if de["source"] == "assumed" else None)
    vol_src = demand_label or "estimated"   # horizon totals depend on the forecast
    kpis = [
        {"kpi": "Gross margin (baseline)", "value": T["gross_margin"], "unit": "USD", "source": "estimated", "note": "forecast units x current price - landed cost"},
        {"kpi": "Gross margin (scenario)", "value": T["gross_margin_s"], "unit": "USD", "source": vol_src},
        {"kpi": "Gross margin change", "value": T["gross_margin_s"] - T["gross_margin"], "unit": "USD", "source": vol_src},
        {"kpi": "Gross margin % (baseline)", "value": T["gross_margin"] / T["revenue"] * 100, "unit": "%", "source": "estimated"},
        {"kpi": "Gross margin % (scenario)", "value": T["gross_margin_s"] / T["revenue_s"] * 100 if T["revenue_s"] else None, "unit": "%", "source": vol_src},
        {"kpi": "COGS change (landed)", "value": d("cogs", "cogs_s"), "unit": "USD", "source": vol_src},
        {"kpi": "Revenue change", "value": d("revenue", "revenue_s"), "unit": "USD", "source": vol_src},
        {"kpi": "Units change", "value": d("units", "units_s"), "unit": "units", "source": demand_label or "computed",
         "note": "no demand effect applied" if not demand_label else ""},
        {"kpi": "Annualised landed-cost increase at full run-rate (affected SKUs)", "value": r["annual_cost_run_rate"], "unit": "USD/yr",
         "source": "estimated", "note": "per-unit cost increase (computed) x average forecast weekly units x 52"},
    ]
    aff_pairs = r["cost_pairs"][r["cost_pairs"]["affected"]]
    unit_table = aff_pairs[["sku_id", "supplier_id", "supplier_name", "country", "sourcing_share", "unit_cost", "tariff_pct", "new_tariff_pct",
                            "freight_per_unit", "landed_cost", "new_landed_cost", "landed_cost_delta"]].sort_values("landed_cost_delta", ascending=False)
    top = r["sku"].head(15).reset_index()
    out = {
        "scenario": spec["name"], "window": {"start": spec["start_date"], "end": spec["end_date"],
                                             "horizon_weeks": spec["horizon_weeks"],
                                             "forecast_weeks": [r["panel"]["week"].min(), r["panel"]["week"].max()]},
        "1_expected_under_normal_conditions": {k: T[k] for k in ("units", "revenue", "cogs", "gross_margin")} | {
            "gross_margin_pct": T["gross_margin"] / T["revenue"] * 100, "source": "estimated (forecast) x computed (price, landed cost)",
            "forecast_model": r["forecast_summary"]["model_choice"]["recommended"],
            "forecast_backtest_wape_pct": r["forecast_summary"]["backtest"]["metrics"][r["forecast_summary"]["model_choice"]["recommended"]]["wape_pct"]},
        "2_what_changes": {"kpis": kpis,
                           "per_unit_cost_changes_computed": unit_table.head(25).to_dict("records"),
                           "affected_sku_supplier_pairs": int(len(aff_pairs))},
        "3_what_is_affected": {
            "skus_ranked_by_margin_impact": top[["sku_id", "product_name", "category", "units_base", "gm_base", "gm_scen", "gm_delta",
                                                 "cost_old", "cost_new", "protected_units", "margin_pct_base", "margin_pct_scen"]].to_dict("records")
            if "product_name" in top.columns else top.to_dict("records"),
            "suppliers": r["supplier"].to_dict("records"),
            "countries": r["country"].to_dict("records"),
        },
        "timing": {"cost_timing": spec["cost_timing"],
                   "units_protected_by_existing_stock": float(r["protected"].reindex(r["resolved"]["skus"]).fillna(0).sum()) if len(r["protected"]) else 0.0,
                   "explanation": "Stock on hand plus open POs arriving before the start date were bought at the old cost and are sold first (FIFO)."
                   if spec["cost_timing"] == "after_current_stock" else "All units sold from the start date carry the new cost."},
        "demand_effect": {"source": de["source"], "pct": de.get("pct"), "elasticity": de.get("elasticity"),
                          "note": "No demand effect applied. Use sensitivity_analysis to see hypothetical demand changes, then ask the user which to apply."
                          if de["source"] == "none" else ""},
        "data_notes": ([f"SKUs without cost data (excluded): {r['missing_cost_skus']}"] if r["missing_cost_skus"] else []) +
                      [f"Scope note: {u}" for u in r["resolved"]["unmatched"]],
        "assumptions": sc["assumptions"],
    }
    sc["summary"] = to_jsonable(out)
    sc["applied_at"] = now_iso()
    return sc["summary"]


# ============================================================================ sensitivity / compare

def sensitivity_analysis(session, name: str, variable: str, values: list[float]) -> dict:
    """Re-run a confirmed scenario with one parameter varied. All results are hypothetical (assumed)."""
    sc = session.scenarios.get(name)
    if not sc or sc["status"] != "confirmed":
        raise ToolError(f"Scenario '{name}' must exist and be confirmed first.")
    allowed = {"demand_pct", "tariff_pct", "price_pass_through_pct", "elasticity", "unit_cost_change_pct"}
    if variable not in allowed:
        raise ToolError(f"variable must be one of {sorted(allowed)}")
    if not values or len(values) > 8:
        raise ToolError("Provide 1-8 values.")
    base = sc.get("result") or _run(session, sc["spec"])
    rows = []
    for v in values:
        spec = deepcopy(sc["spec"])
        if variable == "demand_pct":
            spec["demand_effect"] = {"source": "assumed", "pct": float(v), "elasticity": None}
        elif variable == "elasticity":
            spec["demand_effect"] = {"source": "assumed", "pct": None, "elasticity": float(v)}
        else:
            spec[variable] = float(v)
        r = _run(session, spec)
        T = r["totals"]
        rows.append({variable: v, "units": T["units_s"], "revenue": T["revenue_s"], "cogs": T["cogs_s"],
                     "gross_margin": T["gross_margin_s"], "gross_margin_change_vs_baseline": T["gross_margin_s"] - T["gross_margin"],
                     "gross_margin_pct": T["gross_margin_s"] / T["revenue_s"] * 100 if T["revenue_s"] else None})
    out = {"scenario": name, "variable": variable, "label": "Assumed (hypothetical) - not estimated from data",
           "baseline_gross_margin": base["totals"]["gross_margin"], "rows": rows,
           "applies_to": "affected SKUs, within the event window"}
    return to_jsonable(out)


def compare_scenarios(session, names: list[str]) -> dict:
    rows = []
    base_T = None
    for n in names:
        sc = session.scenarios.get(n)
        if not sc or sc["status"] != "confirmed":
            raise ToolError(f"Scenario '{n}' must exist and be confirmed.")
        r = sc.get("result") or _run(session, sc["spec"])
        sc["result"] = r
        T = r["totals"]
        base_T = base_T or T
        rows.append({"scenario": n, "units": T["units_s"], "revenue": T["revenue_s"], "cogs": T["cogs_s"], "gross_margin": T["gross_margin_s"],
                     "gross_margin_pct": T["gross_margin_s"] / T["revenue_s"] * 100 if T["revenue_s"] else None,
                     "gross_margin_change": T["gross_margin_s"] - T["gross_margin"],
                     "demand_effect_source": sc["spec"]["demand_effect"]["source"], "horizon_weeks": sc["spec"]["horizon_weeks"]})
    horizons = {r["horizon_weeks"] for r in rows}
    T = base_T
    rows.insert(0, {"scenario": "Baseline", "units": T["units"], "revenue": T["revenue"], "cogs": T["cogs"], "gross_margin": T["gross_margin"],
                    "gross_margin_pct": T["gross_margin"] / T["revenue"] * 100, "gross_margin_change": 0.0,
                    "demand_effect_source": "-", "horizon_weeks": rows[0]["horizon_weeks"]})
    note = "" if len(horizons) == 1 else "Scenarios use different horizons; totals are not directly comparable."
    return to_jsonable({"rows": rows, "note": note})


# ============================================================================ mitigations

def evaluate_mitigations(session, name: str, holding_cost_rate: float | None = None) -> dict:
    sc = session.scenarios.get(name)
    if not sc or sc["status"] != "confirmed":
        raise ToolError(f"Scenario '{name}' must exist and be confirmed first.")
    r = sc.get("result") or _run(session, sc["spec"])
    sc["result"] = r
    spec = sc["spec"]
    hr = holding_cost_rate if holding_cost_rate is not None else spec["holding_cost_rate"]
    cbs = r["cost_pairs"]
    p = r["panel"]
    eff = spec["start_date"]
    today = _today(session)
    days_to_eff = (eff - today).days
    affected = cbs[cbs["affected"]]
    mits = []
    # ---------------- 1. Shift volume to qualified alternate suppliers
    shift_rows, no_alt = [], []
    cb_full = cost_basis(session)
    weekly_after = p[p["week"] >= eff].groupby("sku_id")["units_base"].mean()
    sup_cap = cb_full.groupby("supplier_id").agg(cap=("weekly_capacity_units", "first"), used=("supplier_recent_weekly_units", "first"))
    sup_cap["headroom"] = (sup_cap["cap"] - sup_cap["used"]).clip(lower=0)
    used_headroom: dict[str, float] = {}
    moq = None
    sem = get_semantics(session)
    ap = next((n for n, t in session.tables.items() if "order_min_qty_units" in t.df.columns), None)
    if ap:
        moq = session.df(ap).groupby("sku_id")["order_min_qty_units"].median()
    for _, a in affected.iterrows():
        alts = cbs[(cbs["sku_id"] == a["sku_id"]) & (~cbs["affected"])]
        wk_units = float(weekly_after.get(a["sku_id"], 0.0)) * a["sourcing_share"]
        if alts.empty:
            no_alt.append({"sku_id": a["sku_id"], "supplier_id": a["supplier_id"], "weekly_units": wk_units,
                           "cost_increase_per_unit": a["landed_cost_delta"]})
            continue
        best = alts.sort_values("new_landed_cost").iloc[0]
        saving_unit = a["new_landed_cost"] - best["new_landed_cost"]
        head = max(0.0, float(sup_cap.loc[best["supplier_id"], "headroom"]) - used_headroom.get(best["supplier_id"], 0.0))
        shift = min(wk_units, head)
        used_headroom[best["supplier_id"]] = used_headroom.get(best["supplier_id"], 0.0) + shift
        lt = float(best["lead_time_weeks"]) if pd.notna(best["lead_time_weeks"]) else None
        start_shift = max(eff, today + pd.Timedelta(weeks=lt or 0))
        weeks_eff = max(0, p[(p["week"] >= start_shift)]["week"].nunique())
        shift_rows.append({
            "sku_id": a["sku_id"], "from_supplier": a["supplier_id"], "from_country": a["country"],
            "to_supplier": best["supplier_id"], "to_supplier_name": best["supplier_name"], "to_country": best["country"],
            "affected_landed_cost": a["new_landed_cost"], "alternate_landed_cost": best["new_landed_cost"],
            "saving_per_unit": saving_unit, "weekly_units_to_shift": wk_units, "shiftable_within_capacity": shift,
            "alternate_lead_time_weeks": lt, "first_shifted_receipt": start_shift, "weeks_of_benefit_in_horizon": weeks_eff,
            "estimated_saving_in_horizon": max(0.0, saving_unit) * shift * weeks_eff,
            "annualised_saving": max(0.0, saving_unit) * shift * 52,
            "moq_units": float(moq.get(a["sku_id"])) if moq is not None and a["sku_id"] in moq.index else None,
            "worth_it": bool(saving_unit > 0),
        })
    sh = pd.DataFrame(shift_rows)
    if len(sh):
        good = sh[sh["worth_it"]]
        mits.append({
            "mitigation": "Shift volume to qualified alternate suppliers", "evaluable": True,
            "estimated_benefit_in_horizon": float(good["estimated_saving_in_horizon"].sum()),
            "annualised_benefit": float(good["annualised_saving"].sum()),
            "source": "estimated (forecast volume) x computed (landed-cost difference)",
            "skus_with_cheaper_alternate": int(len(good)), "skus_where_alternate_costs_more": int((~sh["worth_it"]).sum()),
            "details": sh.sort_values("estimated_saving_in_horizon", ascending=False).head(20).to_dict("records"),
            "constraints": ["alternate supplier weekly capacity minus its recent weekly volume (from POs)",
                            "alternate lead time delays the first lower-cost receipt",
                            "minimum order quantities per SKU-DC apply"],
            "trade_offs": ["concentration risk moves to the alternate supplier", "possible quality/spec differences; check contracts",
                           "only suppliers already contracted for the SKU are considered"],
        })
    if no_alt:
        na = pd.DataFrame(no_alt)
        mits.append({"mitigation": "Qualify a new supplier (SKUs with no existing alternate)", "evaluable": False,
                     "reason": "No alternate supplier is contracted for these SKUs in the data, so cost, capacity and lead time of a new source are unknown.",
                     "skus": na.sort_values("cost_increase_per_unit", ascending=False).head(20).to_dict("records"),
                     "skus_count": int(len(na))})
    # ---------------- 2. Pre-buy before the effective date
    pre = []
    po_t = sem.get("purchase_orders")
    air = None
    if po_t:
        po = session.df(po_t)
        if "transport_mode" in po.columns and "actual_receipt_date" in po.columns:
            a_ = po[po["transport_mode"].str.lower() == "air"].dropna(subset=["actual_receipt_date"])
            if len(a_):
                air = {"median_transit_days": float((a_["actual_receipt_date"] - a_["order_date"]).dt.days.median()),
                       "freight_per_unit": float(a_["freight_cost_usd"].sum() / a_["qty_ordered"].sum()),
                       "orders": int(len(a_))}
    for (sid, sname, country), g in affected.groupby(["supplier_id", "supplier_name", "country"]):
        lt = float(g["lead_time_weeks"].median()) if g["lead_time_weeks"].notna().any() else None
        feasible = lt is not None and today + pd.Timedelta(weeks=lt) <= eff
        weeks_cover = 8
        units = sum(float(weekly_after.get(s, 0)) * sh_ for s, sh_ in zip(g["sku_id"], g["sourcing_share"])) * weeks_cover
        tariff_saved_unit = float((g["unit_cost"] * (g["new_tariff_pct"] - g["tariff_pct"]) / 100 * g["sourcing_share"]).sum() /
                                  max(g["sourcing_share"].sum(), 1e-9))
        avg_cost = float(g["unit_cost"].mean())
        carry = units * avg_cost * hr * (weeks_cover / 2) / 52
        row = {"supplier_id": sid, "supplier_name": sname, "country": country, "lead_time_weeks": lt,
               "days_until_effective": days_to_eff, "feasible_by_sea_or_standard_lead_time": feasible,
               "prebuy_units_for_8_weeks": units, "tariff_avoided": units * tariff_saved_unit, "holding_cost": carry,
               "net_benefit": units * tariff_saved_unit - carry if feasible else None}
        if not feasible and air:
            row["air_expedite_feasible"] = air["median_transit_days"] <= days_to_eff
            extra_freight = units * max(0.0, air["freight_per_unit"] - float(g["freight_per_unit"].mean()))
            row["air_extra_freight"] = extra_freight
            row["air_net_benefit"] = units * tariff_saved_unit - carry - extra_freight if row["air_expedite_feasible"] else None
        pre.append(row)
    if pre:
        feas = [x for x in pre if x["feasible_by_sea_or_standard_lead_time"]]
        airf = [x for x in pre if x.get("air_expedite_feasible")]
        mits.append({
            "mitigation": "Pre-buy before the effective date", "evaluable": True,
            "estimated_benefit_in_horizon": float(sum((x["net_benefit"] or 0) for x in feas) + sum((x.get("air_net_benefit") or 0) for x in airf if (x.get("air_net_benefit") or 0) > 0)),
            "source": "computed (lead times, tariff delta) + estimated (forecast units) + assumed (holding cost rate)",
            "summary": (f"{len(feas)} of {len(pre)} affected suppliers can deliver before {eff:%Y-%m-%d} at standard lead time "
                        f"({days_to_eff} days away)."),
            "details": pre, "air_freight_reference": air,
            "constraints": ["goods must clear customs before the effective date to avoid the new rate", "supplier capacity",
                            "shelf life limits how far ahead you can buy", "warehouse space"],
            "trade_offs": [f"ties up cash (holding cost assumed {hr:.0%}/yr)", "obsolescence risk if demand falls",
                           "air expedite adds freight cost"],
        })
    # ---------------- 3. Re-price (pass-through)
    aff_skus = r["resolved"]["skus"]
    if aff_skus:
        delta = (r["sku"].loc[r["sku"].index.isin(aff_skus), "cost_new"] - r["sku"].loc[r["sku"].index.isin(aff_skus), "cost_old"])
        units_aff = r["sku"].loc[r["sku"].index.isin(aff_skus), "units_base"]
        rec_full = float((delta * units_aff).sum())
        price_aff = r["sku"].loc[r["sku"].index.isin(aff_skus), "price"]
        avg_price_increase_pct = float((delta / price_aff).mean() * 100)
        mits.append({
            "mitigation": "Re-price: pass the cost increase into selling price", "evaluable": "partially",
            "margin_recovered_at_100pct_pass_through_if_volume_unchanged": rec_full,
            "average_price_increase_needed_pct": avg_price_increase_pct,
            "source": "computed (cost delta) x estimated (forecast units); volume response is NOT known",
            "why_partial": "The data cannot reliably estimate how demand responds to a permanent price increase (list-price-only "
                           "elasticity is not statistically distinguishable from zero). Run sensitivity_analysis on 'elasticity' or "
                           "'demand_pct' and agree an assumption with the user.",
            "constraints": ["competitor price index", "retail partner price agreements"],
            "trade_offs": ["volume and share loss", "possible retailer pushback"],
        })
    # ---------------- 4. Safety stock / lead time
    if spec.get("lead_time_change_weeks"):
        lt_change = float(spec["lead_time_change_weeks"])
        ss_rows = []
        pol = None
        if ap and "safety_stock_weeks" in session.df(ap).columns:
            pol = session.df(ap).groupby("sku_id")["safety_stock_weeks"].mean()
        for _, a in affected.iterrows():
            L = float(a["lead_time_weeks"]) if pd.notna(a["lead_time_weeks"]) else None
            if not L or pol is None or a["sku_id"] not in pol.index:
                continue
            ssw = float(pol[a["sku_id"]])
            ssw_new = ssw * math.sqrt((L + lt_change) / L)
            wk = float(weekly_after.get(a["sku_id"], 0)) * a["sourcing_share"]
            extra_units = (ssw_new - ssw) * wk
            ss_rows.append({"sku_id": a["sku_id"], "supplier_id": a["supplier_id"], "lead_time_weeks": L, "new_lead_time_weeks": L + lt_change,
                            "safety_stock_weeks": ssw, "new_safety_stock_weeks": ssw_new, "extra_units": extra_units,
                            "extra_inventory_value": extra_units * float(a["unit_cost"]),
                            "annual_holding_cost": extra_units * float(a["unit_cost"]) * hr})
        if ss_rows:
            ssd = pd.DataFrame(ss_rows)
            mits.append({"mitigation": "Raise safety stock for longer lead times", "evaluable": True,
                         "estimated_cost": float(ssd["annual_holding_cost"].sum()), "extra_inventory_value": float(ssd["extra_inventory_value"].sum()),
                         "method": "safety stock weeks scale with sqrt(lead time): new = current x sqrt((L + change) / L)",
                         "source": "computed (policy, lead times) + estimated (forecast) + assumed (holding cost rate)",
                         "details": ssd.sort_values("extra_inventory_value", ascending=False).head(15).to_dict("records"),
                         "trade_offs": ["more cash in inventory", "lower stockout risk during the disruption"]})
    return to_jsonable({"scenario": name, "today": today, "effective_date": eff, "mitigations": mits,
                        "assumptions": [f"holding cost rate {hr:.0%}/yr ({'user' if holding_cost_rate is not None else 'default'})",
                                        "pre-buy sized at 8 weeks of post-effective-date demand"]})


def scenario_charts(session, name: str) -> list[dict]:
    """Baseline vs scenario weekly gross margin, and per-supplier cost comparison."""
    from .charts import BASELINE, SCENARIO, SERIES, _store, base_layout
    sc = session.scenarios.get(name)
    if not sc or not sc.get("result"):
        raise ToolError("Apply the scenario first.")
    r = sc["result"]
    w = r["weekly"]
    out = []
    lay = base_layout(f"Weekly gross margin: baseline vs '{name}'", "Week", "Gross margin (USD)")
    tr = [{"type": "scatter", "mode": "lines", "x": w["week"], "y": w["gm_base"], "name": "Baseline", "line": {"color": BASELINE, "width": 2}},
          {"type": "scatter", "mode": "lines", "x": w["week"], "y": w["gm_scen"], "name": f"Scenario: {name}", "line": {"color": SCENARIO, "width": 2}}]
    eff = sc["spec"]["start_date"]
    lay["shapes"].append({"type": "line", "x0": eff, "x1": eff, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": "rgba(0,0,0,0.4)", "dash": "dot"}})
    lay["annotations"].append({"x": eff, "yref": "paper", "y": 1.02, "text": f"event start {eff:%Y-%m-%d}", "showarrow": False, "font": {"size": 10}})
    out.append({"chart_id": _store(session, {"data": tr, "layout": lay}, {"scenario": name}), "title": lay["title"]["text"]})
    s = r["supplier"]
    s = s[s["units"] > 0].sort_values("cogs_delta", ascending=False).head(12)
    lab = s["supplier_name"] + " (" + s["country"] + ")"
    lay2 = base_layout(f"Average landed cost per unit by supplier: baseline vs '{name}'", "Supplier", "USD per unit")
    lay2["barmode"] = "group"
    tr2 = [{"type": "bar", "x": lab, "y": s["avg_landed_cost_base"], "name": "Baseline", "marker": {"color": BASELINE}},
           {"type": "bar", "x": lab, "y": s["avg_landed_cost_scen"], "name": f"Scenario: {name}", "marker": {"color": SCENARIO}}]
    out.append({"chart_id": _store(session, {"data": tr2, "layout": lay2}, {"scenario": name}), "title": lay2["title"]["text"]})
    sku = r["sku"].head(12)
    lay3 = base_layout(f"Gross margin change by SKU (most exposed first)", "Gross margin change (USD)", "SKU", legend=False)
    lay3["hovermode"] = "closest"
    labels = [f"{i} {n}" if isinstance(n, str) else str(i) for i, n in zip(sku.index, sku.get("product_name", pd.Series(index=sku.index)))]
    tr3 = [{"type": "bar", "orientation": "h", "y": labels, "x": sku["gm_delta"], "marker": {"color": SERIES[7]}}]
    lay3["yaxis"] = {"autorange": "reversed", "automargin": True}
    lay3["xaxis"] = {"title": {"text": "Gross margin change in horizon (USD)"}}
    out.append({"chart_id": _store(session, {"data": tr3, "layout": lay3}, {"scenario": name}), "title": lay3["title"]["text"]})
    return to_jsonable(out)
