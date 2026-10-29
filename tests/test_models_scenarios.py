"""Metrics, baselines, backtest (no look-ahead), model choice, elasticity recovery and scenario cost maths,
checked against hand-computed values."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from dpa import forecasting as F
from dpa.elasticity import _fit
from dpa.scenarios import apply_scenario, confirm_scenario, cost_basis, create_scenario, evaluate_mitigations, sensitivity_analysis
from dpa.analytics import ToolError


def test_metrics_hand_computed():
    a = np.array([10, 20, 0, 30.0])
    f = np.array([12, 18, 1, 33.0])
    m = F.metrics(a, f)
    # |e| = 2, 2, 1, 3 -> sum 8; sum a = 60
    assert m["wape_pct"] == pytest.approx(8 / 60 * 100)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 1 + 9) / 4))
    assert m["bias_pct"] == pytest.approx((2 - 2 + 1 + 3) / 60 * 100)
    # MAPE excludes the zero week: (0.2 + 0.1 + 0.1) / 3
    assert m["mape_pct"] == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100)
    assert m["mape_excluded_zero_weeks"] == 1


def test_baselines():
    y = np.arange(1, 61, dtype=float)
    assert F.fc_naive(y, 3).tolist() == [60, 60, 60]
    assert F.fc_seasonal_naive(y, 2).tolist() == [9, 10]          # values 52 weeks earlier
    assert F.fc_moving_average(y, 1)[0] == pytest.approx(np.mean(y[-8:]))
    assert np.isnan(F.fc_seasonal_naive(y[:40], 1)).all()           # not enough history


def test_choose_model_requires_beating_baselines():
    mets = {"naive": {"wape_pct": 20}, "seasonal_naive": {"wape_pct": 10}, "moving_average": {"wape_pct": 15},
            "ets": {"wape_pct": 12}, "lightgbm": {"wape_pct": 11}}
    c = F.choose_model(mets)
    assert c["recommended"] == "seasonal_naive" and not c["beats_baselines"]
    mets["lightgbm"]["wape_pct"] = 8
    c = F.choose_model(mets)
    assert c["recommended"] == "lightgbm" and c["beats_baselines"]
    assert c["wape_improvement_vs_best_baseline_pct"] == pytest.approx(20.0)


def test_backtest_has_no_lookahead(mini_session):
    ss = F.build_series(mini_session, level="sku")
    bt1 = F.backtest(ss, holdout=26, step=13, horizon=13, models=["naive", "seasonal_naive", "moving_average", "ets", "lightgbm"])
    # Corrupt everything after the first origin's forecast window: forecasts made AT that origin must not change.
    o = ss.y.index.get_loc(pd.Timestamp(bt1["origins"][0]))
    ss2 = F.build_series(mini_session, level="sku")
    ss2.y.iloc[o:] = ss2.y.iloc[o:] * 5 + 1000
    bt2 = F.backtest(ss2, holdout=26, step=13, horizon=13, models=["naive", "seasonal_naive", "moving_average", "ets", "lightgbm"])
    r1 = bt1["records"][bt1["records"]["origin"] == o].sort_values(["model", "series", "h"])
    r2 = bt2["records"][bt2["records"]["origin"] == o].sort_values(["model", "series", "h"])
    assert np.allclose(r1["forecast"].values, r2["forecast"].values)


def test_forecast_tool_end_to_end(mini_session):
    s = F.forecast_weekly(mini_session, level="sku", horizon=8)
    assert s["series_forecast"] == 2
    assert set(s["backtest"]["metrics"]) >= {"naive", "seasonal_naive", "moving_average", "ets", "lightgbm"}
    f = F.get_forecast(mini_session, s["forecast_id"])["forecast"]
    assert (f["lower"] <= f["forecast"] + 1e-9).all() and (f["upper"] >= f["forecast"] - 1e-9).all()
    assert len(f) == 16


def test_elasticity_recovers_known_value():
    rng = np.random.default_rng(3)
    rows = []
    weeks = pd.date_range("2022-01-03", periods=156, freq="7D")
    for s in range(8):
        base = 100 * (1 + s)
        for i, w in enumerate(weeks):
            price = 10 * (1 + 0.15 * rng.choice([0, 0, 0, -1, 1]))
            season = 1 + 0.2 * np.sin(2 * np.pi * i / 52)
            units = base * season * price ** -1.5 * np.exp(rng.normal(0, 0.05))
            rows.append({"sku": f"S{s}", "week": w, "units": units, "price": price, "promo": 0.0, "event": 0.0})
    r = _fit(pd.DataFrame(rows), "test")
    assert r["elasticity"] == pytest.approx(-1.5, abs=0.1)
    assert r["ci95"][0] < -1.5 < r["ci95"][1]
    assert r["usable"]


@pytest.fixture
def supply_session(tmp_path):
    """One SKU sold 100/week at $20, two suppliers: S_KR (Korea, $10, tariff 0%, freight $100/100 units, 60% share)
    and S_US (US, $11, freight $50/100 units, 40% share)."""
    weeks = pd.date_range("2024-01-01", periods=110, freq="7D")
    sales = pd.DataFrame({"sku_id": "A", "dc_id": "D1", "week_start": weeks.strftime("%Y-%m-%d"), "customer_demand": 100,
                          "units_sold": 100, "lost_sales_units": 0, "selling_price": 20.0, "revenue": 2000.0, "stockout_flag": 0})
    prod = pd.DataFrame({"sku_id": ["A"], "category": ["skin"]})
    sup = pd.DataFrame({"supplier_id": ["S_KR", "S_US"], "supplier_name": ["Kor", "Usa"], "country": ["South Korea", "United States"],
                        "weekly_capacity_units": [1000, 1000], "standard_lead_time_weeks": [8, 2]})
    ps = pd.DataFrame({"sku_id": ["A", "A"], "supplier_id": ["S_KR", "S_US"], "sourcing_share": [0.6, 0.4],
                       "contract_unit_cost_usd": [10.0, 11.0], "planned_lead_time_weeks": [8, 2]})
    po = pd.DataFrame({"po_line_id": ["P1", "P2"], "sku_id": ["A", "A"], "dc_id": ["D1", "D1"], "supplier_id": ["S_KR", "S_US"],
                       "order_date": [weeks[-3].strftime("%Y-%m-%d")] * 2, "qty_ordered": [100, 100], "unit_cost_usd": [10.0, 11.0],
                       "tariff_rate_pct": [0.0, 0.0], "freight_cost_usd": [100.0, 50.0]})
    dc = pd.DataFrame({"dc_id": ["D1"]})
    p = tmp_path / "supply.zip"
    with zipfile.ZipFile(p, "w") as z:
        for n, d in {"fact_weekly_sales": sales, "dim_product": prod, "dim_supplier": sup, "product_supplier": ps,
                     "fact_purchase_orders": po, "dim_dc": dc}.items():
            z.writestr(f"{n}.csv", d.to_csv(index=False))
    from dpa.session import STORE
    s = STORE.create()
    s.add_upload(str(p), "supply.zip")
    s.settings["today"] = str(weeks[-1].date())
    return s, weeks


def test_scenario_landed_cost_and_margin_hand_computed(supply_session):
    s, weeks = supply_session
    cb = cost_basis(s).set_index("supplier_id")
    # landed = unit cost x (1 + tariff) + freight / units
    assert cb.loc["S_KR", "landed_cost"] == pytest.approx(10.0 + 1.0)
    assert cb.loc["S_US", "landed_cost"] == pytest.approx(11.0 + 0.5)
    start = (weeks[-1] + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    d = create_scenario(s, "KR25", "tariff", scope={"supplier_countries": ["South Korea"]}, start_date=start, tariff_pct=25,
                        horizon_weeks=8, cost_timing="immediate")
    assert d["resolved_scope"]["skus_affected"] == 1 and d["resolved_scope"]["suppliers"] == ["S_KR"]
    with pytest.raises(ToolError):
        apply_scenario(s, "KR25")                      # drafts cannot be applied
    confirm_scenario(s, "KR25")
    out = apply_scenario(s, "KR25")
    pu = out["2_what_changes"]["per_unit_cost_changes_computed"][0]
    # Korea landed cost: 10 x 1.25 + 1.0 = 13.5 (+2.5)
    assert pu["new_landed_cost"] == pytest.approx(13.5) and pu["landed_cost_delta"] == pytest.approx(2.5)
    base = out["1_expected_under_normal_conditions"]
    units = base["units"]
    # blended cost: 0.6 x 11 + 0.4 x 11.5 = 11.2 ; new 0.6 x 13.5 + 0.4 x 11.5 = 12.7 -> margin falls 1.5 per unit
    assert base["cogs"] == pytest.approx(units * 11.2, rel=1e-6)
    k = {x["kpi"]: x for x in out["2_what_changes"]["kpis"]}
    assert k["Gross margin change"]["value"] == pytest.approx(-1.5 * units, rel=1e-6)
    assert k["Gross margin change"]["source"] == "estimated"   # horizon total depends on the forecast
    assert k["Units change"]["value"] == pytest.approx(0)      # no demand effect by default
    # Sensitivity: demand -10% on the affected SKU
    sens = sensitivity_analysis(s, "KR25", "demand_pct", [0, -10])
    r0, r10 = sens["rows"]
    assert r10["units"] == pytest.approx(r0["units"] * 0.9, rel=1e-6)
    assert sens["label"].startswith("Assumed")
    # Mitigation: shifting Korea volume to the US supplier saves 13.5 - 11.5 = 2.0 per unit
    m = evaluate_mitigations(s, "KR25")
    shift = next(x for x in m["mitigations"] if x["mitigation"].startswith("Shift"))
    assert shift["details"][0]["saving_per_unit"] == pytest.approx(2.0)
    pre = next(x for x in m["mitigations"] if x["mitigation"].startswith("Pre-buy"))
    assert pre["details"][0]["feasible_by_sea_or_standard_lead_time"] is False   # 8-week lead time, starts in 1 week


def test_scenario_pass_through_and_elasticity(supply_session):
    s, weeks = supply_session
    start = (weeks[-1] + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    create_scenario(s, "PT", "tariff", scope={"supplier_countries": ["South Korea"]}, start_date=start, tariff_pct=25,
                    horizon_weeks=8, cost_timing="immediate", price_pass_through_pct=100, demand_effect_source="assumed", elasticity=-1.0)
    confirm_scenario(s, "PT")
    out = apply_scenario(s, "PT")
    k = {x["kpi"]: x for x in out["2_what_changes"]["kpis"]}
    # price rises by the full blended cost increase (1.5): 20 -> 21.5; units scale by (21.5/20)^-1
    assert k["Units change"]["source"] == "assumed"
    base_units = out["1_expected_under_normal_conditions"]["units"]
    assert k["Units change"]["value"] == pytest.approx(base_units * ((21.5 / 20) ** -1 - 1), rel=1e-6)


def test_unknown_country_scope_reports_no_exposure(supply_session):
    s, _ = supply_session
    d = create_scenario(s, "IN", "tariff", scope={"supplier_countries": ["India"]}, start_date="2026-10-29", tariff_pct=25)
    assert d["resolved_scope"]["skus_affected"] == 0
    assert "India".lower() in d["resolved_scope"]["unmatched"][0].lower()
    assert d["open_questions"]
