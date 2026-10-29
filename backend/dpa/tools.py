"""Tool library exposed to the LLM: typed JSON schemas + deterministic Python implementations.

The LLM only chooses tools and explains their results. Each call is recorded in the audit trail with its
parameters, duration and (truncated) output; KPI usage and charts produced are collected for the response.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import analytics, charts, elasticity, forecasting, inventory, promotions, quality, scenarios
from .analytics import ToolError
from .kpis import record_kpi
from .semantic import get_semantics
from .util import to_jsonable

MAX_RESULT_CHARS = 14000


@dataclass
class ToolContext:
    session: Any
    message_id: str
    charts: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    exports: list[dict] = field(default_factory=list)   # data behind the answer, offered as an Excel download

    def kpi(self, kid: str, inputs: str, period: str, example: str | None = None):
        record_kpi(self.session, kid, inputs, period, example, self.message_id)
        if kid not in self.kpis:
            self.kpis.append(kid)

    def chart(self, cid: str):
        if cid not in self.charts:
            self.charts.append(cid)


FILTER_SCHEMA = {"type": "array", "description": "Row filters applied before aggregation.",
                 "items": {"type": "object", "properties": {
                     "column": {"type": "string"}, "op": {"type": "string", "enum": sorted(analytics.OPS)},
                     "value": {"description": "value, list for in/not_in, [low, high] for between; dates as YYYY-MM-DD"}},
                     "required": ["column", "op"]}}
SCOPE_SCHEMA = {"type": "object", "description": "What the event affects. Lists are combined: supplier filters AND product filters.",
                "properties": {k: {"type": "array", "items": {"type": "string"}} for k in scenarios.SCOPE_KEYS}}


# ============================================================================ implementations

def t_list_tables(ctx: ToolContext, **_) -> dict:
    s = ctx.session
    if not s.tables:
        return {"tables": [], "message": "No data loaded yet. Ask the user to upload files or click 'Load demo data'."}
    rows = []
    for n in sorted(s.tables):
        p = s.table_profile(n)
        rows.append({"table": n, "rows": p["rows"], "columns": p["columns"], "role": p["role"], "primary_key": p["primary_key"],
                     "date_range": p["date_range"]})
    return {"tables": rows, "relationships_confirmed": s.relationships_confirmed, "semantics": _sem_brief(s)}


def _sem_brief(s) -> dict:
    sem = get_semantics(s)
    dm = sem.get("demand") or {}
    return {"demand_table": dm.get("table"), "demand_measure": dm.get("measure"), "product_table": sem.get("product_table"),
            "purchase_orders": sem.get("purchase_orders"), "inventory": sem.get("inventory"), "notes": sem.get("notes")}


def t_profile_tables(ctx: ToolContext, tables: list[str] | None = None, include_columns: bool | None = None, **_) -> dict:
    s = ctx.session
    names = tables or sorted(s.tables)
    if include_columns is None:
        include_columns = len(names) <= 3   # column detail for a few tables at a time keeps results readable
    out = []
    dd = s.dictionary()
    for n in names:
        if n not in s.tables:
            raise ToolError(f"Unknown table '{n}'")
        p = s.table_profile(n)
        if not include_columns:
            p = {k: p[k] for k in ("table", "rows", "columns", "role", "primary_key", "date_range", "missing_cells_pct", "source_file")}
            p["column_names"] = list(s.tables[n].df.columns)
        else:
            p["columns_detail"] = [{k: r[k] for k in ("column", "original_name", "type", "unit", "description", "missing_pct", "distinct", "key", "examples")}
                                   for r in dd if r["table"] == n]
        out.append(p)
    note = None if include_columns else "Column details omitted for many tables; call again with up to 3 tables for full detail."
    return {"profiles": out, "note": note}


def t_infer_relationships(ctx: ToolContext, **_) -> dict:
    s = ctx.session
    rels = []
    for r in s.relationships:
        d = {"id": r.id, "from": f"{r.child_table}.{'+'.join(r.child_columns)}", "to": f"{r.parent_table}.{'+'.join(r.parent_columns)}",
             "kind": r.kind, "coverage_pct": r.coverage_pct, "status": r.status}
        if r.orphan_count:
            d["orphans"], d["orphan_examples"] = r.orphan_count, r.orphan_examples
        if r.note:
            d["note"] = r.note
        rels.append(d)
    return {"relationships": rels, "all_confirmed": s.relationships_confirmed,
            "note": "Show these to the user and ask them to confirm (they can also confirm in the upload summary panel)."}


def t_confirm_relationships(ctx: ToolContext, relationship_ids: list[str] | None = None, reject_ids: list[str] | None = None,
                            confirm_all: bool = False, **_) -> dict:
    s = ctx.session
    for r in s.relationships:
        if confirm_all or (relationship_ids and r.id in relationship_ids):
            r.status = "confirmed"
        if reject_ids and r.id in reject_ids:
            r.status = "rejected"
    s.relationships_confirmed = all(r.status != "inferred" for r in s.relationships)
    return {"confirmed": sum(r.status == "confirmed" for r in s.relationships), "rejected": sum(r.status == "rejected" for r in s.relationships),
            "pending": sum(r.status == "inferred" for r in s.relationships)}


def t_join_tables(ctx: ToolContext, left: str, right: str, on: Any = None, how: str = "left", **_) -> dict:
    return analytics.join_tables(ctx.session, left, right, on, how)


def t_data_quality_report(ctx: ToolContext, table: str | None = None, **_) -> dict:
    s = ctx.session
    rep = quality.data_quality_report(s)
    ts = rep["timeseries"]
    if ts.get("stockouts", {}).get("available"):
        so = ts["stockouts"]
        ctx.kpi("stockout_rate", f"{ts['table']}.stockout_flag", f"all {ts['date_range']['weeks']} weeks, row level",
                f"{so['rows_flagged']:,} flagged rows / {s.tables[ts['table']].df.shape[0]:,} rows = {so['rows_flagged_pct']:.2f}%")
        if so.get("lost_share_of_demand_pct") is not None:
            ctx.kpi("lost_sales", f"{ts['table']}.lost_sales_units, {ts['measure']}", "all history, total",
                    f"{so['lost_units']:,} lost units / total demand = {so['lost_share_of_demand_pct']:.2f}%")
    if rep.get("outliers_sku"):
        o = rep["outliers_sku"]
        ctx.kpi("robust_z", f"{ts['table']}.{ts['measure']}", "weekly, SKU level (all locations/channels)",
                f"{o['flagged_weeks']} weeks flagged across {o['series_with_flags']} of {o['series_checked']} SKUs")
    if table:
        if table not in rep["tables"]:
            raise ToolError(f"Unknown table '{table}'")
        return {"table": table, **rep["tables"][table]}
    tables = {n: {"status": v["status"], "rows": v["rows"],
                  "issues": [i for i in v["issues"] if i["severity"] != "info"][:6],
                  "info_items": len([i for i in v["issues"] if i["severity"] == "info"])} for n, v in rep["tables"].items()}
    ts_brief = {k: v for k, v in ts.items() if k not in ("new_skus", "discontinued_skus", "sku_structural_breaks")}
    ts_brief["new_skus"] = ts.get("new_skus", [])[:10]
    ts_brief["discontinued_skus"] = ts.get("discontinued_skus", [])[:10]
    ts_brief["sku_structural_breaks"] = ts.get("sku_structural_breaks", [])[:5]
    o = rep.get("outliers_sku") or {}
    return {"table_status_counts": rep["table_status_counts"], "tables": tables, "timeseries": ts_brief,
            "outliers": {k: o.get(k) for k in ("level", "series_checked", "flagged_weeks", "series_with_flags", "method", "treatment",
                                               "examples_with_known_cause")} | {"top_examples": (o.get("examples") or [])[:8]},
            "outliers_total_level": {k: (rep.get("outliers_total") or {}).get(k) for k in ("flagged_weeks", "examples")}}


def t_aggregate(ctx: ToolContext, **kw) -> dict:
    r = analytics.aggregate(ctx.session, **kw)
    for m in kw.get("measures", []):
        if m.get("numerator") and "revenue" in m["numerator"] and ("unit" in m.get("denominator", "") or "sold" in m.get("denominator", "")):
            ctx.kpi("average_selling_price", f"{kw['table']}.{m['numerator']}, {m['denominator']}",
                    f"{kw.get('time_bucket') or 'whole period'}, grouped by {kw.get('group_by') or 'total'}")
        if m.get("column") == "revenue":
            ctx.kpi("revenue", f"{kw['table']}.revenue", f"{kw.get('time_bucket') or 'whole period'}, by {kw.get('group_by') or 'total'}",
                    f"total in result = {r['column_totals'].get(m.get('name') or 'revenue', 0):,.0f}")
    return r


def t_descriptive_stats(ctx: ToolContext, **kw) -> dict:
    r = analytics.descriptive_stats(ctx.session, **kw)
    ctx.kpi("cv", f"{kw.get('table') or kw.get('result_id')}: {kw.get('columns') or 'numeric columns'}", f"grouped by {kw.get('group_by') or 'none'}")
    return r


def t_make_chart(ctx: ToolContext, **kw) -> dict:
    r = charts.make_chart(ctx.session, **kw)
    ctx.chart(r["chart_id"])
    return r


def t_detect_outliers(ctx: ToolContext, level: str = "sku", treatment: str = "keep", series: str | None = None, **_) -> dict:
    s = ctx.session
    if level not in ("total", "category", "sku"):
        raise ToolError("level must be total, category or sku")
    r = quality.outlier_scan(s, level=level, max_examples=15)
    if "error" in r:
        raise ToolError(r["error"])
    out = {k: v for k, v in r.items()}
    if treatment != "keep":
        from .forecasting import build_series
        ss = build_series(s, level=level if level != "total" else "total")
        effects = []
        for c in ([series] if series else list(ss.y.columns)[:60]):
            if c not in ss.y.columns:
                raise ToolError(f"Series '{c}' not found")
            _, eff = quality.treat_outliers(forecasting._active(ss.y[c]), treatment)
            if eff["weeks_flagged"]:
                effects.append({"series": c, **eff})
        tot_b = sum(e["total_before"] for e in effects)
        tot_a = sum(e["total_after"] for e in effects)
        out["treatment_effect"] = {"method": treatment, "series_changed": len(effects),
                                   "total_before": tot_b, "total_after": tot_a,
                                   "total_change_pct": (tot_a - tot_b) / tot_b * 100 if tot_b else 0.0,
                                   "examples": effects[:10],
                                   "note": "Preview only - the data itself is never modified. Pass outlier_treatment to forecast_weekly to use it."}
    ctx.kpi("robust_z", f"{get_semantics(s)['demand']['table']}.{r['measure']}", f"weekly, {level} level",
            f"{r['flagged_weeks']} weeks flagged in {r['series_with_flags']} of {r['series_checked']} series")
    return out


def _forecast_kpis(ctx: ToolContext, summ: dict):
    f = forecasting.get_forecast(ctx.session, summ["forecast_id"])
    rec = summ["model_choice"]["recommended"]
    bt = f["backtest"]
    g = bt[(bt["model"] == rec)].dropna(subset=["forecast", "actual"])
    ae = float((g["forecast"] - g["actual"]).abs().sum())
    act = float(g["actual"].sum())
    err = float((g["forecast"] - g["actual"]).sum())
    m = summ["backtest"]["metrics"][rec]
    inputs = f"{summ['table']}.{summ['measure']} (weekly, summed to {summ['level']})"
    period = (f"rolling-origin backtest, {len(summ['backtest']['origins'])} origins in the last {summ['backtest']['holdout_weeks']} weeks, "
              f"horizon {summ['backtest']['horizon']} weeks, {summ['level']} level, model {rec}")
    ctx.kpi("wape", inputs, period, f"sum|F-A| = {ae:,.0f}; sum A = {act:,.0f}; WAPE = {ae:,.0f} / {act:,.0f} = {m['wape_pct']:.1f}%")
    ctx.kpi("mae", inputs, period, f"{ae:,.0f} / {m['n']} forecast weeks = {m['mae']:,.1f} units")
    ctx.kpi("rmse", inputs, period, f"RMSE = {m['rmse']:,.1f} units (vs MAE {m['mae']:,.1f})")
    ctx.kpi("bias", inputs, period, f"sum(F-A) = {err:,.0f}; / {act:,.0f} = {m['bias_pct']:+.1f}%")
    if m.get("mase") is not None:
        ctx.kpi("mase", inputs, period, f"MASE = {m['mase']:.2f} (<1 beats the seasonal-naive benchmark)")
    if m.get("mape_pct") is not None:
        ctx.kpi("mape", inputs, period, f"MAPE = {m['mape_pct']:.1f}% ({m.get('mape_excluded_zero_weeks', 0)} zero-demand weeks excluded)")
    q = summ["interval"]["quantiles_by_band"]
    if q:
        b = next(iter(q))
        ctx.kpi("prediction_interval", inputs, f"{int(summ['interval']['level'] * 100)}% interval from backtest errors",
                f"weeks {b}: actual/forecast - 1 ranged {q[b][0] * 100:+.1f}% to {q[b][1] * 100:+.1f}% (n={q[b][2]})")


def t_forecast_weekly(ctx: ToolContext, chart: bool = True, chart_series: str | None = None, **kw) -> dict:
    s = ctx.session
    summ = forecasting.forecast_weekly(s, **kw)
    _forecast_kpis(ctx, summ)
    out = dict(summ)
    if chart:
        c = forecasting.forecast_chart(s, summ["forecast_id"], series=chart_series)
        ctx.chart(c["chart_id"])
        out["chart_id"] = c["chart_id"]
    out["forecast_by_week_total"] = summ["forecast_by_week_total"][:26]
    return out


def t_backtest(ctx: ToolContext, forecast_id: str, **_) -> dict:
    f = forecasting.get_forecast(ctx.session, forecast_id)
    bt = f["backtest"].dropna(subset=["forecast", "actual"])
    by_origin = []
    for (m, o), g in bt.groupby(["model", "origin"]):
        mm = forecasting.metrics(g["actual"].values, g["forecast"].values)
        by_origin.append({"model": m, "origin_week": f["history"]["week"].sort_values().unique()[o] if o < f["history"]["week"].nunique() else o,
                          "wape_pct": mm.get("wape_pct"), "bias_pct": mm.get("bias_pct")})
    by_h = []
    for (m, h), g in bt.groupby(["model", "h"]):
        mm = forecasting.metrics(g["actual"].values, g["forecast"].values)
        by_h.append({"model": m, "weeks_ahead": int(h), "wape_pct": mm.get("wape_pct")})
    _forecast_kpis(ctx, f["summary"])
    c = forecasting.backtest_chart(ctx.session, forecast_id)
    ctx.chart(c["chart_id"])
    return {"forecast_id": forecast_id, "metrics_by_model": f["summary"]["backtest"]["metrics"], "by_origin": by_origin,
            "wape_by_weeks_ahead": [x for x in by_h if x["weeks_ahead"] in (1, 2, 4, 8, 13)], "chart_id": c["chart_id"],
            "worst_series": f["summary"]["worst_series_backtest"], "method": forecasting.__doc__.split("Method (also in the README):")[1].strip()[:1500]}


def t_compare_models(ctx: ToolContext, forecast_id: str, **_) -> dict:
    s = forecasting.get_forecast(ctx.session, forecast_id)["summary"]
    mets = s["backtest"]["metrics"]
    rows = sorted([{"model": m, "type": "baseline" if m in forecasting.BASELINES else "candidate", **{k: v.get(k) for k in ("wape_pct", "mae", "rmse", "bias_pct", "mase", "mape_pct")}}
                   for m, v in mets.items()], key=lambda r: r["wape_pct"] if r["wape_pct"] is not None else 1e9)
    c = forecasting.backtest_chart(ctx.session, forecast_id)
    ctx.chart(c["chart_id"])
    _forecast_kpis(ctx, s)
    return {"ranking": rows, "choice": s["model_choice"], "chart_id": c["chart_id"], "ml_implementation": s["backtest"]["ml_implementation"]}


def t_forecast_chart(ctx: ToolContext, forecast_id: str, series: str | None = None, show_models: bool = False, **_) -> dict:
    c = forecasting.forecast_chart(ctx.session, forecast_id, series=series, show_models=show_models)
    ctx.chart(c["chart_id"])
    return c


def t_estimate_price_elasticity(ctx: ToolContext, **kw) -> dict:
    r = dict(elasticity.estimate_price_elasticity(ctx.session, **kw))
    rows = [{"group": g["group"], "variant": v["variant"], "elasticity": v.get("elasticity"), "ci95_low": (v.get("ci95") or [None, None])[0],
             "ci95_high": (v.get("ci95") or [None, None])[1], "usable": v.get("usable"), "reason": v.get("reason"),
             "price_changes": v.get("price_changes"), "observations": v.get("observations")}
            for g in r["results"] for v in (g["all_variation"], g["list_price_only"])]
    r["result_id"] = analytics.store_result(ctx.session, pd.DataFrame(rows), {"tool": "estimate_price_elasticity"})
    g = r["results"][0]
    a = g["all_variation"]
    if a.get("elasticity") is not None:
        ctx.kpi("price_elasticity", f"{get_semantics(ctx.session)['demand']['table']}: {r['measure']}, {r['price']}",
                f"weekly, SKU panel, level={r['level']}",
                f"{g['group']}: all variation beta = {a['elasticity']:.2f} (95% CI {a['ci95'][0]:.2f} to {a['ci95'][1]:.2f}), "
                f"list-price only usable = {g['list_price_only'].get('usable')}")
    return r


def _scenario_kpis(ctx: ToolContext, name: str, summ: dict):
    pu = summ["2_what_changes"]["per_unit_cost_changes_computed"]
    if pu:
        x = pu[0]
        ctx.kpi("landed_cost", "purchase orders: unit_cost_usd, tariff_rate_pct, freight_cost_usd, qty_ordered",
                f"last 26 weeks of POs, SKU x supplier, scenario '{name}'",
                f"{x['sku_id']} from {x['supplier_id']}: {x['unit_cost']:.2f} x (1 + {x['new_tariff_pct']:.0f}%) + {x['freight_per_unit']:.2f} = {x['new_landed_cost']:.2f} "
                f"(was {x['landed_cost']:.2f})")
    b = summ["1_expected_under_normal_conditions"]
    ctx.kpi("gross_margin", "forecast units x recent selling price - landed cost", f"{summ['window']['horizon_weeks']}-week horizon, SKU level",
            f"baseline: {b['revenue']:,.0f} - {b['cogs']:,.0f} = {b['gross_margin']:,.0f}")
    ctx.kpi("gross_margin_pct", "gross margin / revenue", f"{summ['window']['horizon_weeks']}-week horizon",
            f"baseline: {b['gross_margin']:,.0f} / {b['revenue']:,.0f} = {b['gross_margin_pct']:.1f}%")
    ctx.kpi("cogs", "units x sourcing-share-weighted landed cost", f"{summ['window']['horizon_weeks']}-week horizon", f"baseline COGS = {b['cogs']:,.0f}")


def t_create_scenario(ctx: ToolContext, **kw) -> dict:
    return scenarios.create_scenario(ctx.session, **kw)


def t_confirm_scenario(ctx: ToolContext, name: str, **_) -> dict:
    return scenarios.confirm_scenario(ctx.session, name)


def t_apply_scenario(ctx: ToolContext, name: str, charts_: bool = True, **_) -> dict:
    summ = scenarios.apply_scenario(ctx.session, name)
    _scenario_kpis(ctx, name, summ)
    out = dict(summ)
    cs = scenarios.scenario_charts(ctx.session, name)
    for c in cs:
        ctx.chart(c["chart_id"])
    out["chart_ids"] = [c["chart_id"] for c in cs]
    out["2_what_changes"] = dict(out["2_what_changes"])
    pu = out["2_what_changes"]["per_unit_cost_changes_computed"]
    out["2_what_changes"]["per_unit_cost_changes_computed"] = pu[:12]
    if len(pu) > 12:
        out["2_what_changes"]["per_unit_note"] = f"showing 12 of {len(pu)} SKU-supplier pairs; all are in the Excel download"
    out["3_what_is_affected"] = dict(out["3_what_is_affected"])
    out["3_what_is_affected"]["skus_ranked_by_margin_impact"] = out["3_what_is_affected"]["skus_ranked_by_margin_impact"][:10]
    out["3_what_is_affected"]["suppliers"] = [x for x in out["3_what_is_affected"]["suppliers"] if x.get("affected")] + \
        [{"other_unaffected_suppliers": sum(1 for x in out["3_what_is_affected"]["suppliers"] if not x.get("affected"))}]
    return out


def t_compare_scenarios(ctx: ToolContext, names: list[str], **_) -> dict:
    r = scenarios.compare_scenarios(ctx.session, names)
    from .charts import SERIES, _store, base_layout
    lay = base_layout("Gross margin: baseline vs scenarios", "Scenario", "Gross margin (USD)", legend=False)
    lay["hovermode"] = "closest"
    tr = [{"type": "bar", "x": [x["scenario"] for x in r["rows"]], "y": [x["gross_margin"] for x in r["rows"]],
           "marker": {"color": [SERIES[0]] + [SERIES[1 + i % 7] for i in range(len(r["rows"]) - 1)]}}]
    cid = _store(ctx.session, {"data": tr, "layout": lay}, {"compare": names})
    ctx.chart(cid)
    ctx.kpi("gross_margin", "scenario results", "scenario horizon", None)
    r["chart_id"] = cid
    return r


def t_sensitivity_analysis(ctx: ToolContext, name: str, variable: str, values: list[float], **_) -> dict:
    r = dict(scenarios.sensitivity_analysis(ctx.session, name, variable, values))
    r["result_id"] = analytics.store_result(ctx.session, pd.DataFrame(r["rows"]), {"tool": "sensitivity_analysis", "scenario": name})
    ctx.kpi("gross_margin", f"scenario '{name}' re-run with {variable} varied", "scenario horizon", None)
    return r


def t_supplier_cost_impact(ctx: ToolContext, name: str | None = None, **_) -> dict:
    s = ctx.session
    if name:
        sc = s.scenarios.get(name)
        if not sc or not sc.get("summary"):
            raise ToolError("Apply the scenario first (apply_scenario).")
        return {"scenario": name, "suppliers": sc["summary"]["3_what_is_affected"]["suppliers"],
                "countries": sc["summary"]["3_what_is_affected"]["countries"]}
    cb = scenarios.cost_basis(s)
    g = cb.groupby(["supplier_id", "supplier_name", "country"]).agg(
        skus=("sku_id", "nunique"), avg_unit_cost=("unit_cost", "mean"), current_tariff_pct=("tariff_pct", "max"),
        avg_freight_per_unit=("freight_per_unit", "mean"), avg_landed_cost=("landed_cost", "mean"),
        lead_time_weeks=("lead_time_weeks", "median"), weekly_capacity=("weekly_capacity_units", "first"),
        recent_weekly_units=("supplier_recent_weekly_units", "first")).reset_index()
    g["capacity_headroom_units_per_week"] = g["weekly_capacity"] - g["recent_weekly_units"]
    ctx.kpi("landed_cost", "purchase orders + product_supplier + supplier", "last 26 weeks of POs, supplier level")
    return {"suppliers": g.sort_values("avg_landed_cost", ascending=False).to_dict("records"),
            "note": "Current state (no scenario). Averages are simple means across each supplier's SKUs."}


def t_financial_impact(ctx: ToolContext, name: str, **_) -> dict:
    sc = ctx.session.scenarios.get(name)
    if not sc or not sc.get("summary"):
        raise ToolError("Apply the scenario first (apply_scenario).")
    s = sc["summary"]
    _scenario_kpis(ctx, name, s)
    return {"scenario": name, "baseline": s["1_expected_under_normal_conditions"], "changes": s["2_what_changes"]["kpis"], "window": s["window"]}


def t_evaluate_mitigations(ctx: ToolContext, name: str, holding_cost_rate: float | None = None, **_) -> dict:
    r = scenarios.evaluate_mitigations(ctx.session, name, holding_cost_rate)
    ctx.kpi("holding_cost", "unit cost x holding rate", "pre-buy / safety stock evaluation",
            f"rate {holding_cost_rate if holding_cost_rate is not None else scenarios.DEFAULT_HOLDING_RATE:.0%} per year")
    if any("safety stock" in m["mitigation"].lower() for m in r["mitigations"]):
        ctx.kpi("safety_stock", "assortment policy safety_stock_weeks, lead times", "SKU x supplier")
    return r


def t_inventory_plan(ctx: ToolContext, chart: bool = True, **kw) -> dict:
    r = inventory.inventory_plan(ctx.session, **kw)
    ctx.kpi("safety_stock", "SKU forecast backtest error, lead time, review period, service level",
            f"SKU x location, {r['service_level']:.0%} service level, as of {r['as_of_inventory_week']}",
            f"z = {r['z']:.2f}; safety stock = z x weekly error sd x sqrt(lead time + review period)")
    ctx.kpi("weeks_of_cover", "on hand / average weekly forecast", f"SKU x location, as of {r['as_of_inventory_week']}")
    ctx.kpi("reorder_quantity", "order-up-to level - (on hand + on order), case pack, MOQ", "SKU x location, this review cycle")
    if chart:
        c = inventory.inventory_chart(ctx.session, r["plan_id"])
        ctx.chart(c["chart_id"])
        r["chart_id"] = c["chart_id"]
    return r


def t_estimate_promo_lift(ctx: ToolContext, chart: bool = True, **kw) -> dict:
    r = promotions.estimate_promo_lift(ctx.session, **kw)
    top = r["results"][0]["effects"][0] if r["results"] and r["results"][0]["effects"] else None
    ctx.kpi("promo_lift", f"{get_semantics(ctx.session)['demand']['table']}: {r['measure']}, promotion id/type, discount",
            f"weekly, product x location x channel panel, by {r['by']}" + (f", per {r['group']}" if r.get("group") else ""),
            f"{top['effect']}: exp(beta) - 1 = {top['lift_pct']:+.1f}% (95% CI {top['ci95_pct'][0]:+.1f}% to {top['ci95_pct'][1]:+.1f}%)" if top else None)
    rows = [{"group": res["group"], **{k: v for k, v in e.items() if k != "ci95_pct"}, "ci95_low_pct": e["ci95_pct"][0],
             "ci95_high_pct": e["ci95_pct"][1]} for res in r["results"] for e in res["effects"]]
    r = dict(r)
    r["result_id"] = analytics.store_result(ctx.session, pd.DataFrame(rows), {"tool": "estimate_promo_lift"})
    if chart and r["results"]:
        c = promotions.promo_lift_chart(ctx.session, r)
        ctx.chart(c["chart_id"])
        r = dict(r)
        r["chart_id"] = c["chart_id"]
    return r


def t_inventory_chart(ctx: ToolContext, plan_id: str, sku: str | None = None, location: str | None = None, **_) -> dict:
    c = inventory.inventory_chart(ctx.session, plan_id, sku, location)
    ctx.chart(c["chart_id"])
    return c


# ---------------------------------------------------------------------------- calculator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def _eval(node, names):
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ToolError(f"Unknown name '{node.id}'; pass it in 'values'")
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a, names) for a in node.args])
    raise ToolError("Only arithmetic (+ - * / **), numbers, named values and abs/round/min/max/sqrt/log/exp are allowed.")


def t_calculate(ctx: ToolContext, expression: str, values: dict | None = None, label: str = "", **_) -> dict:
    if len(expression) > 300:
        raise ToolError("Expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
        v = _eval(tree, {k: float(x) for k, x in (values or {}).items()})
    except ZeroDivisionError:
        raise ToolError("Division by zero")
    except SyntaxError:
        raise ToolError("Invalid expression")
    return {"label": label, "expression": expression, "values": values or {}, "result": float(v)}


# ============================================================================ registry

@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOLS: list[Tool] = [
    Tool("list_tables", "List loaded tables with rows, role, key and date range, plus which table/measure is used for demand.", _obj({}), t_list_tables),
    Tool("profile_tables", "Profile tables: rows, columns, types, units, keys, date ranges, granularity, % missing and column descriptions.",
         _obj({"tables": {"type": "array", "items": {"type": "string"}}, "include_columns": {"type": "boolean"}}), t_profile_tables),
    Tool("infer_relationships", "Return inferred primary/foreign-key relationships with coverage, orphans and confirmation status.", _obj({}), t_infer_relationships),
    Tool("confirm_relationships", "Record the user's confirmation (or rejection) of inferred relationships. Only call after the user explicitly confirms.",
         _obj({"relationship_ids": {"type": "array", "items": {"type": "string"}}, "reject_ids": {"type": "array", "items": {"type": "string"}},
               "confirm_all": {"type": "boolean"}}), t_confirm_relationships),
    Tool("join_tables", "Validate a join between two tables: row counts before/after, key uniqueness, orphan keys, fan-out and grain-mismatch warnings.",
         _obj({"left": {"type": "string"}, "right": {"type": "string"},
               "on": {"description": "list of shared columns or {left_col: right_col}; omit to use the inferred relationship"},
               "how": {"type": "string", "enum": ["left", "inner", "right", "outer"]}}, ["left", "right"]), t_join_tables),
    Tool("data_quality_report", "Data quality report: per-table status, missing values, duplicates, orphans, time-series checks (gaps, zeros, short "
         "histories, new/discontinued SKUs, level shifts, structural breaks), stockouts/censored demand and outliers with likely causes.",
         _obj({"table": {"type": "string", "description": "optional: details for one table"}}), t_data_quality_report),
    Tool("aggregate", "Group, filter and time-bucket any table. Columns from related dimension tables (e.g. category, brand_name, country) are "
         "joined automatically through confirmed many-to-one relationships; qualify ambiguous names as table.column. Returns a result_id for charts.",
         _obj({"table": {"type": "string"},
               "measures": {"type": "array", "items": {"type": "object", "properties": {
                   "column": {"type": "string"}, "agg": {"type": "string", "enum": sorted(analytics.AGGS)}, "name": {"type": "string"},
                   "numerator": {"type": "string"}, "denominator": {"type": "string"}}},
                   "description": "e.g. {column: revenue, agg: sum} or a ratio of sums {name: avg_price, numerator: revenue, denominator: units_sold}"},
               "group_by": {"type": "array", "items": {"type": "string"}}, "filters": FILTER_SCHEMA,
               "time_bucket": {"type": "string", "enum": list(analytics.TIME_BUCKETS)}, "date_col": {"type": "string"},
               "sort_by": {"type": "string"}, "descending": {"type": "boolean"}, "limit": {"type": "integer"}},
              ["table", "measures"]), t_aggregate),
    Tool("descriptive_stats", "Count, sum, mean, std, quartiles, coefficient of variation and % zero for numeric columns of a table or a stored result.",
         _obj({"result_id": {"type": "string"}, "table": {"type": "string"}, "columns": {"type": "array", "items": {"type": "string"}},
               "group_by": {"type": "string"}, "filters": FILTER_SCHEMA}), t_descriptive_stats),
    Tool("make_chart", "Interactive chart from a stored result (result_id). One y-axis only; up to 8 coloured series (rest folded into Other). "
         "Time charts are annotated with external events; mark_outliers flags robust-z outliers.",
         _obj({"result_id": {"type": "string"}, "chart_type": {"type": "string", "enum": sorted(charts.CHART_TYPES)},
               "x": {"type": "string"}, "y": {"type": ["string", "array"], "items": {"type": "string"}}, "color": {"type": "string"},
               "title": {"type": "string"}, "x_title": {"type": "string"}, "y_title": {"type": "string", "description": "include units"},
               "annotate_events": {"type": "boolean"}, "mark_outliers": {"type": "boolean"}, "top_n": {"type": "integer"}},
              ["result_id", "chart_type", "x", "y", "title", "y_title"]), t_make_chart),
    Tool("detect_outliers", "Outlier detection on weekly demand (seasonally adjusted robust z-score) at total, category or SKU level, with likely "
         "causes; optionally preview the effect of a treatment (cap/impute). Never modifies the data.",
         _obj({"level": {"type": "string", "enum": ["total", "category", "sku"]}, "treatment": {"type": "string", "enum": ["keep", "cap", "impute"]},
               "series": {"type": "string"}}), t_detect_outliers),
    Tool("forecast_weekly", "Validated weekly forecast of ANY additive planning measure (default: customer demand) at total, category, SKU or any "
         "dimension level. Always computes naive / seasonal-naive / moving-average baselines plus ETS and LightGBM, runs a rolling-origin backtest, "
         "recommends a model only if it beats the baselines, and returns prediction intervals and a chart.",
         _obj({"measure": {"type": "string", "description": "column to forecast, e.g. customer_demand, units_sold, revenue, qty_ordered"},
               "table": {"type": "string", "description": "defaults to the weekly demand table"},
               "level": {"type": "string", "description": "total | category | sku | any column such as brand_name, channel_id, dc_id, subcategory"},
               "values": {"type": "array", "items": {"type": "string"}, "description": "restrict to these level values (e.g. ['SKU8'])"},
               "filters": FILTER_SCHEMA, "horizon": {"type": "integer", "minimum": 1, "maximum": 52},
               "models": {"type": "array", "items": {"type": "string", "enum": forecasting.BASELINES + forecasting.CANDIDATES}},
               "holdout_weeks": {"type": "integer"}, "interval": {"type": "number"},
               "outlier_treatment": {"type": "string", "enum": ["keep", "cap", "impute"]}, "top_n": {"type": "integer"},
               "agg": {"type": "string", "enum": ["sum", "mean"]}, "chart_series": {"type": "string"}, "chart": {"type": "boolean"}}), t_forecast_weekly),
    Tool("backtest", "Backtest details for a forecast_id: metrics by model, by origin and by weeks ahead, worst series, plus a WAPE-by-model chart.",
         _obj({"forecast_id": {"type": "string"}}, ["forecast_id"]), t_backtest),
    Tool("compare_models", "Rank all models of a forecast_id on WAPE/MAE/RMSE/bias/MASE/MAPE versus the baselines, with the recommendation rule result.",
         _obj({"forecast_id": {"type": "string"}}, ["forecast_id"]), t_compare_models),
    Tool("forecast_chart", "Chart actuals + forecast + interval for a forecast_id, total or a single series; show_models overlays all models.",
         _obj({"forecast_id": {"type": "string"}, "series": {"type": "string"}, "show_models": {"type": "boolean"}}, ["forecast_id"]), t_forecast_chart),
    Tool("estimate_price_elasticity", "Estimate price elasticity from historical price variation (log-log, SKU fixed effects, seasonality), separately for "
         "all variation (promotions) and list-price changes only, with 95% CIs and a usable/not-usable verdict.",
         _obj({"level": {"type": "string", "description": "total | category | sku | subcategory | brand_name"},
               "values": {"type": "array", "items": {"type": "string"}}, "filters": FILTER_SCHEMA}), t_estimate_price_elasticity),
    Tool("create_scenario", "Create or replace a DRAFT structured scenario from the user's event description. Returns the parsed object, the SKUs and "
         "suppliers it resolves to, default assumptions and open questions. Must be shown to the user for confirmation before applying.",
         _obj({"name": {"type": "string"}, "event_type": {"type": "string", "enum": scenarios.EVENT_TYPES}, "description": {"type": "string"},
               "scope": SCOPE_SCHEMA, "start_date": {"type": "string", "description": "YYYY-MM-DD"}, "end_date": {"type": "string"},
               "tariff_pct": {"type": "number"}, "tariff_basis": {"type": "string", "enum": ["additive_points", "replace_rate"]},
               "unit_cost_change_pct": {"type": "number"}, "supply_capacity_change_pct": {"type": "number", "description": "e.g. -50 for half capacity"},
               "lead_time_change_weeks": {"type": "number"}, "price_pass_through_pct": {"type": "number"},
               "demand_effect_source": {"type": "string", "enum": ["none", "assumed", "estimated"]},
               "demand_effect_pct": {"type": "number"}, "elasticity": {"type": "number"},
               "horizon_weeks": {"type": "integer", "minimum": 4, "maximum": 52},
               "cost_timing": {"type": "string", "enum": ["after_current_stock", "immediate"]}, "holding_cost_rate": {"type": "number"},
               "damaged_units": {"type": "object", "description": "{sku_id: units written off}"}},
              ["name", "event_type"]), t_create_scenario),
    Tool("confirm_scenario", "Mark a draft scenario as confirmed. ONLY call when the user has explicitly confirmed the parsed scenario.",
         _obj({"name": {"type": "string"}}, ["name"]), t_confirm_scenario),
    Tool("apply_scenario", "Apply a confirmed scenario: baseline vs scenario (units, revenue, COGS, landed cost, gross margin), per-unit cost changes, "
         "ranked affected SKUs/suppliers/countries, timing, and charts. Every KPI carries a source label.",
         _obj({"name": {"type": "string"}}, ["name"]), t_apply_scenario),
    Tool("compare_scenarios", "Compare confirmed scenarios side by side against the baseline (with chart).",
         _obj({"names": {"type": "array", "items": {"type": "string"}}}, ["names"]), t_compare_scenarios),
    Tool("sensitivity_analysis", "Re-run a confirmed scenario with one parameter varied (demand_pct, tariff_pct, price_pass_through_pct, elasticity, "
         "unit_cost_change_pct). Results are hypothetical (Assumed).",
         _obj({"name": {"type": "string"}, "variable": {"type": "string", "enum": ["demand_pct", "tariff_pct", "price_pass_through_pct", "elasticity", "unit_cost_change_pct"]},
               "values": {"type": "array", "items": {"type": "number"}}}, ["name", "variable", "values"]), t_sensitivity_analysis),
    Tool("supplier_cost_impact", "Per-supplier cost comparison: current landed cost, tariff, freight, lead time and capacity headroom; or, with a scenario "
         "name, the scenario's per-supplier and per-country cost impact.", _obj({"name": {"type": "string"}}), t_supplier_cost_impact),
    Tool("financial_impact", "Baseline vs scenario financial KPIs of an applied scenario.", _obj({"name": {"type": "string"}}, ["name"]), t_financial_impact),
    Tool("evaluate_mitigations", "Evaluate mitigations for an applied scenario: shift volume to qualified alternate suppliers (capacity, lead time, MOQ), "
         "pre-buy before the effective date (lead-time feasibility, air expedite), re-price, safety stock. Marks what the data cannot evaluate.",
         _obj({"name": {"type": "string"}, "holding_cost_rate": {"type": "number"}}, ["name"]), t_evaluate_mitigations),
    Tool("inventory_plan", "Reorder and stockout plan per SKU x location: safety stock sized from the forecast's measured error, order-up-to "
         "level over lead time + review period, recommended order rounded to case pack and MOQ, projected on hand with scheduled PO "
         "receipts, first stockout week, and which items need expediting. Service level is an assumption the user should confirm.",
         _obj({"service_level": {"type": "number", "description": "e.g. 0.95"}, "horizon_weeks": {"type": "integer"},
               "skus": {"type": "array", "items": {"type": "string"}}, "dcs": {"type": "array", "items": {"type": "string"}},
               "review_period_weeks": {"type": "number"}, "top": {"type": "integer"}, "chart": {"type": "boolean"}}), t_inventory_plan),
    Tool("estimate_promo_lift", "Average demand lift during promotions by promotion type (or discount depth), plus the post-promotion dip, "
         "with 95% confidence intervals; optionally per group (e.g. category). Use for promotion planning questions.",
         _obj({"by": {"type": "string", "enum": ["promo_type", "discount"]}, "group": {"type": "string", "description": "e.g. category"},
               "filters": FILTER_SCHEMA, "chart": {"type": "boolean"}}), t_estimate_promo_lift),
    Tool("inventory_chart", "Chart projected on hand vs safety stock for one SKU at one location from an inventory plan.",
         _obj({"plan_id": {"type": "string"}, "sku": {"type": "string"}, "location": {"type": "string"}}, ["plan_id"]), t_inventory_chart),
    Tool("calculate", "Exact arithmetic on numbers taken from tool results (e.g. a % change). Use instead of doing maths yourself.",
         _obj({"expression": {"type": "string", "description": "e.g. (b - a) / a * 100"}, "values": {"type": "object"}, "label": {"type": "string"}},
              ["expression"]), t_calculate),
]

FINAL_ANSWER = Tool("final_answer", "Deliver the final structured answer to the user. Call exactly once, at the end.", _obj({
    "answer": {"type": "string", "description": "2-3 sentence direct answer with the key number(s)"},
    "key_results": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string"}, "value": {"type": "string", "description": "as displayed, copied from a tool result"},
        "source": {"type": "string", "enum": ["computed", "estimated", "assumed"]}, "note": {"type": "string"}},
        "required": ["label", "value", "source"]}},
    "chart_ids": {"type": "array", "items": {"type": "string"}},
    "how_computed": {"type": "string", "description": "the method in plain English"},
    "data_notes": {"type": "array", "items": {"type": "string"}},
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "recommendations": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"}, "based_on": {"type": "string", "description": "the key result label or number it relies on"}},
        "required": ["text"]}},
    "next_questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
    "kpis_used": {"type": "array", "items": {"type": "string"}},
    "needs_user_input": {"type": "boolean", "description": "true when you are asking the user to confirm or provide something"},
}, ["answer", "next_questions"]), None)

TOOL_MAP = {t.name: t for t in TOOLS}


def anthropic_tools() -> list[dict]:
    out = [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in TOOLS + [FINAL_ANSWER]]
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def run_tool(ctx: ToolContext, name: str, args: dict) -> tuple[dict, bool]:
    """Execute a tool. Returns (result, is_error). Never raises."""
    t0 = time.time()
    tool = TOOL_MAP.get(name)
    if tool is None:
        res, err = {"error": f"Unknown tool '{name}'"}, True
    else:
        try:
            res = to_jsonable(tool.fn(ctx, **(args or {})))
            err = False
        except ToolError as e:
            res, err = {"error": str(e)}, True
        except KeyError as e:
            res, err = {"error": f"Not found: {e}"}, True
        except TypeError as e:
            res, err = {"error": f"Bad arguments for {name}: {e}"}, True
        except Exception as e:  # never crash the agent loop
            res, err = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-800:]}, True
    if not err and isinstance(res, dict):
        for k, kind in (("result_id", "result"), ("forecast_id", "forecast"), ("plan_id", "inventory_plan")):
            if res.get(k) and not any(e["id"] == res[k] for e in ctx.exports):
                ctx.exports.append({"kind": kind, "id": res[k], "tool": name})
        if name == "apply_scenario" and res.get("scenario"):
            ctx.exports.append({"kind": "scenario", "id": res["scenario"], "tool": name})
    ctx.calls.append({"tool": name, "args": to_jsonable(args), "seconds": round(time.time() - t0, 2), "error": err,
                      "result_preview": json.dumps(res, default=str)[:3000]})
    return res, err


def result_for_llm(res: dict) -> str:
    s = json.dumps(res, default=str)
    if len(s) <= MAX_RESULT_CHARS:
        return s
    return s[:MAX_RESULT_CHARS] + f'... [truncated {len(s) - MAX_RESULT_CHARS} chars; ask for a narrower result if needed]'
