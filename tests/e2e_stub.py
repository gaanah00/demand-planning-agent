"""Offline stand-in for Claude used ONLY by the end-to-end script / browser walkthrough when no API key is
available. It routes on keywords and calls the real tools, then writes final answers from the tool results,
so every layer except the language model itself is exercised. Enable with DPA_LLM_STUB=e2e_stub:make."""
from __future__ import annotations

import json
import re

from fake_llm import last_results, tool


def _f(x, d=1):
    if x is None:
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{d}f}"


def _all_results(messages):
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    try:
                        out.append(json.loads(b["content"].split("... [truncated")[0]))
                    except Exception:
                        pass
    return out


class StubClient:
    def __init__(self):
        from types import SimpleNamespace
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        from types import SimpleNamespace
        msgs = kw["messages"]
        first_user = next(m["content"] for m in reversed(msgs) if m["role"] == "user" and isinstance(m["content"], str))
        n_rounds = sum(1 for m in msgs if m["role"] == "assistant")
        prior = sum(1 for m in msgs[:-1] if m["role"] == "user" and isinstance(m["content"], str))
        rounds_this_turn = sum(1 for m in msgs[msgs.index(next(m for m in reversed(msgs) if m["role"] == "user" and isinstance(m["content"], str))):] if m["role"] == "assistant")
        self.raw = first_user
        blocks = self.plan(first_user.lower(), rounds_this_turn, msgs, kw)
        return SimpleNamespace(content=[SimpleNamespace(**b) for b in blocks], stop_reason="tool_use")

    # ------------------------------------------------------------------
    def plan(self, q, r, msgs, kw):
        res = _all_results(msgs[-(2 * r + 1):]) if r else []
        if "reorder" in q:
            if r == 0:
                return [tool("inventory_plan", service_level=0.95)]
            d = res[-1]
            c = d["status_counts"]
            top = d["top_actions"][0]
            return [tool("final_answer",
                         answer=f"{c.get('stockout within lead time - expedite', 0)} SKU-locations will run out before a standard order can arrive and "
                                f"need expediting; {c.get('reorder this cycle', 0)} need a reorder this cycle, {_f(d['total_recommended_units'])} units in total.",
                         key_results=[{"label": "Expedite", "value": str(c.get('stockout within lead time - expedite', 0)), "source": "estimated"},
                                      {"label": "Reorder this cycle", "value": str(c.get('reorder this cycle', 0)), "source": "estimated"},
                                      {"label": "Units to order", "value": _f(d['total_recommended_units']), "source": "estimated"},
                                      {"label": "Order value", "value": _f(d['total_order_value']), "source": "estimated"},
                                      {"label": "Service level", "value": "95%", "source": "assumed"}],
                         chart_ids=[d["chart_id"]], how_computed=d["method"][:400],
                         assumptions=d["assumptions"],
                         recommendations=[{"text": f"Expedite {top['sku_id']} at {top['dc_id']} first", "based_on": "Expedite"}],
                         next_questions=[f"Show the projected stock for {top['sku_id']} at {top['dc_id']}", "Rerun the plan at a 98% service level"],
                         kpis_used=["safety_stock", "weeks_of_cover", "reorder_quantity"])]
        if "promotion" in q and "lift" in q:
            if r == 0:
                return [tool("estimate_promo_lift", by="promo_type")]
            d = res[-1]
            e = d["results"][0]["effects"]
            best = e[0]
            return [tool("final_answer",
                         answer=f"{best['effect']} promotions lift weekly demand the most, by about {best['lift_pct']:.0f}% on average.",
                         key_results=[{"label": f"{x['effect']} lift", "value": f"{x['lift_pct']:.0f}%", "source": "estimated"} for x in e[:4]],
                         chart_ids=[d["chart_id"]], how_computed=d["method"][:300], data_notes=d["caveats"],
                         next_questions=[f"Plan a {best['effect']} promotion for skincare in December", "Show the lift by category"],
                         kpis_used=["promo_lift"])]
        if "quality" in q:
            if r == 0:
                return [tool("data_quality_report")]
            d = res[-1]
            ts = d["timeseries"]
            o = d["outliers"]
            return [tool("final_answer",
                         answer=f"All {sum(d['table_status_counts'].values())} tables passed the structural checks ({d['table_status_counts']['ok']} OK). "
                                f"The demand history has {ts['missing_weeks_total']} missing weeks, {_f(ts['zero_demand_rows'])} zero-demand rows and "
                                f"{ts['short_series_lt52']} series shorter than 52 weeks; {o['flagged_weeks']} SKU-weeks are flagged as outliers and kept.",
                         key_results=[{"label": "Tables OK", "value": str(d['table_status_counts']['ok']), "source": "computed"},
                                      {"label": "Stockout rows", "value": f"{ts['stockouts']['rows_flagged_pct']:.2f}%", "source": "computed"},
                                      {"label": "Outlier SKU-weeks", "value": str(o['flagged_weeks']), "source": "computed"}],
                         how_computed="Ran the data quality report: key uniqueness, missing values, orphans, weekly gaps, zero demand, level shifts and robust z-score outliers on seasonally adjusted demand.",
                         data_notes=[f"Outlier method: {o['method']}", f"Treatment: {o['treatment']}"],
                         recommendations=[{"text": "Keep flagged weeks; the forecast models promotions explicitly.", "based_on": "Outlier SKU-weeks"}],
                         next_questions=[f"Show the {o['flagged_weeks']} flagged outlier weeks for the top SKU",
                                         f"How do the {ts['short_series_lt52']} short-history series affect the forecast?"],
                         kpis_used=["robust_z", "stockout_rate", "lost_sales"])]
        if "category" in q and "demand" in q and "forecast" not in q:
            if r == 0:
                return [tool("aggregate", table="fact_weekly_sales", measures=[{"column": "customer_demand", "agg": "sum"}],
                             group_by=["category"], time_bucket="week")]
            if r == 1:
                return [tool("make_chart", result_id=res[-1]["result_id"], chart_type="line", x="week", y="customer_demand",
                             color="category", title="Weekly customer demand by category", y_title="Units per week")]
            agg = _all_results(msgs)[-2]
            tot = agg["column_totals"]["customer_demand"]
            return [tool("final_answer", answer=f"Total customer demand over the history is {_f(tot)} units, split across three categories.",
                         key_results=[{"label": "Total demand", "value": _f(tot), "source": "computed"}],
                         chart_ids=[res[-1]["chart_id"]], how_computed="Summed weekly customer demand by category (auto-joined from dim_product).",
                         next_questions=["Forecast skincare demand for the next 13 weeks", "Which haircare SKUs drove the 2024 level shift?"],
                         kpis_used=[])]
        if "forecast" in q:
            if r == 0:
                return [tool("forecast_weekly", level="total", horizon=13)]
            f = res[-1]
            rec = f["model_choice"]["recommended"]
            m = f["backtest"]["metrics"][rec]
            b = f["model_choice"]["best_baseline"]
            return [tool("final_answer",
                         answer=f"Total demand for the next 13 weeks is forecast at {_f(f['forecast_total']['sum_forecast'])} units using {rec}, "
                                f"which beat the best baseline ({b}) in the backtest with a WAPE of {m['wape_pct']:.1f}%.",
                         key_results=[{"label": "13-week forecast", "value": _f(f['forecast_total']['sum_forecast']), "source": "estimated"},
                                      {"label": f"WAPE ({rec})", "value": f"{m['wape_pct']:.1f}%", "source": "estimated"},
                                      {"label": f"WAPE ({b})", "value": f"{f['backtest']['metrics'][b]['wape_pct']:.1f}%", "source": "estimated"},
                                      {"label": "Bias", "value": f"{m['bias_pct']:+.1f}%", "source": "estimated"}],
                         chart_ids=[f["chart_id"]], how_computed=f"Rolling-origin backtest at origins {', '.join(f['backtest']['origins'])}.",
                         next_questions=[f"Which SKUs have the worst forecast error?", "Show the forecast for skincare only"],
                         kpis_used=["wape", "mae", "rmse", "bias", "mase", "prediction_interval"])]
        if "tariff" in q and "confirm" not in q:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
            pct = float(m.group(1)) if m else 25.0
            country = "South Korea" if "korea" in q else ("India" if "india" in q else "China")
            name = f"{country} {pct:g}% tariff"
            if r == 0:
                return [tool("create_scenario", name=name, event_type="tariff", scope={"supplier_countries": [country]},
                             start_date="2026-10-29", tariff_pct=pct, description=q)]
            d = res[-1]
            rs = d["resolved_scope"]
            if rs["skus_affected"] == 0:
                return [tool("final_answer", answer=f"No supplier in the data is located in {country}, so a tariff on {country} imports affects 0 SKUs. "
                                                    f"{rs['unmatched'][0] if rs['unmatched'] else ''}",
                             key_results=[{"label": "SKUs exposed", "value": "0", "source": "computed"}], needs_user_input=True,
                             next_questions=["Run the same 25% tariff on South Korea instead", "Which countries do we import from?"])]
            return [tool("final_answer", answer=f"I parsed this as a {pct:g}% tariff on imports from {country} starting 2026-10-29, affecting {rs['skus_affected']} SKUs "
                                                f"from suppliers {', '.join(rs['suppliers'])}. Please confirm the scenario and the default assumptions before I apply it.",
                         key_results=[{"label": "SKUs affected", "value": str(rs['skus_affected']), "source": "computed"},
                                      {"label": "Tariff", "value": f"+{pct:g} pts", "source": "assumed"}],
                         assumptions=[a["text"] for a in d["assumptions"]], needs_user_input=True,
                         next_questions=[f"Confirm and apply '{name}'", "Use 50% price pass-through instead"])]
        if "confirm" in q and "scenario" in q:
            name = re.search(r"'([^']+)'", self.raw).group(1)
            if r == 0:
                return [tool("apply_scenario", name=name)]
            if r == 1:
                return [tool("sensitivity_analysis", name=name, variable="demand_pct", values=[0, -5, -10, -20]),
                        tool("evaluate_mitigations", name=name)]
            allr = _all_results(msgs)
            ap = next(x for x in reversed(allr) if "2_what_changes" in x)
            mit = next(x for x in reversed(allr) if "mitigations" in x)
            k = {x["kpi"]: x for x in ap["2_what_changes"]["kpis"]}
            gm = k["Gross margin change"]["value"]
            rr = k["Annualised landed-cost increase at full run-rate (affected SKUs)"]["value"]
            top = ap["3_what_is_affected"]["skus_ranked_by_margin_impact"][0]
            shift = next((m for m in mit["mitigations"] if m["mitigation"].startswith("Shift")), None)
            return [tool("final_answer",
                         answer=f"Over the next {ap['window']['horizon_weeks']} weeks the tariff reduces gross margin by {_f(-gm)} USD with no demand effect applied; "
                                f"at full run-rate it adds {_f(rr)} USD a year to landed cost. {top['sku_id']} ({top['product_name']}) is the most exposed SKU.",
                         key_results=[{"label": "Gross margin change", "value": _f(gm), "source": "estimated"},
                                      {"label": "Annual landed-cost increase", "value": _f(rr), "source": "estimated"},
                                      {"label": f"{top['sku_id']} margin change", "value": _f(top['gm_delta']), "source": "estimated"},
                                      {"label": "Shift-volume saving in horizon", "value": _f(shift['estimated_benefit_in_horizon']) if shift else "n/a", "source": "estimated"}],
                         chart_ids=ap["chart_ids"], how_computed="Landed cost per SKU-supplier from recent POs, tariff added to affected suppliers, applied to forecast units after current stock is used.",
                         assumptions=[a["text"] for a in ap["assumptions"]],
                         recommendations=[{"text": f"Shift volume for {shift['details'][0]['sku_id']} to {shift['details'][0]['to_supplier_name']}", "based_on": "Shift-volume saving in horizon"}] if shift else [],
                         next_questions=[f"Compare {shift['details'][0]['from_supplier']} vs {shift['details'][0]['to_supplier']} for {shift['details'][0]['sku_id']}" if shift else "Compare suppliers",
                                         "What if we pass 50% of the cost into price?"],
                         kpis_used=["gross_margin", "landed_cost", "cogs"])]
        if r == 0:
            return [tool("list_tables")]
        return [tool("final_answer", answer=f"There are {len(res[-1].get('tables', []))} tables loaded.", next_questions=["Show data quality", "Forecast total demand"])]


def make():
    return StubClient()
