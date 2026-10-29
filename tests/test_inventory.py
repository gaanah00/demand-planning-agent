"""Reorder planning checked against a hand-computed example.

One SKU at one DC sells exactly 100 units/week. Lead time 2 weeks, review period 1 week, case pack 24, MOQ 48.
With a flat history the forecast is 100/week and the backtest error ~0, so safety stock ~0 and the
order-up-to level = 100 x (2 + 1) = 300.
"""
import zipfile

import pandas as pd
import pytest

from dpa.inventory import get_plan, inventory_plan


def _session(tmp_path, on_hand, open_po_qty, po_weeks_ahead):
    weeks = pd.date_range("2024-01-01", periods=110, freq="7D")
    last = weeks[-1]
    sales = pd.DataFrame({"sku_id": "A", "dc_id": "D1", "week_start": weeks.strftime("%Y-%m-%d"), "customer_demand": 100,
                          "units_sold": 100, "selling_price": 20.0, "revenue": 2000.0, "stockout_flag": 0})
    inv = pd.DataFrame({"sku_id": "A", "dc_id": "D1", "week_start": weeks.strftime("%Y-%m-%d"), "closing_on_hand": 500,
                        "on_order_units": 0, "units_sold": 100})
    inv.loc[inv.index[-1], "closing_on_hand"] = on_hand
    inv.loc[inv.index[-1], "on_order_units"] = open_po_qty
    prod = pd.DataFrame({"sku_id": ["A"], "category": ["skin"]})
    dc = pd.DataFrame({"dc_id": ["D1"], "dc_name": ["Main"]})
    sup = pd.DataFrame({"supplier_id": ["S1"], "supplier_name": ["One"], "country": ["United States"], "weekly_capacity_units": [10000]})
    ps = pd.DataFrame({"sku_id": ["A"], "supplier_id": ["S1"], "sourcing_share": [1.0], "contract_unit_cost_usd": [5.0],
                       "planned_lead_time_weeks": [2]})
    po_rows = [{"po_line_id": "P0", "sku_id": "A", "dc_id": "D1", "supplier_id": "S1", "order_date": weeks[-6].strftime("%Y-%m-%d"),
                "planned_receipt_date": weeks[-4].strftime("%Y-%m-%d"), "actual_receipt_date": weeks[-4].strftime("%Y-%m-%d"),
                "qty_ordered": 200, "unit_cost_usd": 5.0, "tariff_rate_pct": 0.0, "freight_cost_usd": 20.0}]
    if open_po_qty:
        po_rows.append({"po_line_id": "P1", "sku_id": "A", "dc_id": "D1", "supplier_id": "S1", "order_date": weeks[-1].strftime("%Y-%m-%d"),
                        "planned_receipt_date": (last + pd.Timedelta(weeks=po_weeks_ahead)).strftime("%Y-%m-%d"), "actual_receipt_date": "",
                        "qty_ordered": open_po_qty, "unit_cost_usd": 5.0, "tariff_rate_pct": 0.0, "freight_cost_usd": 20.0})
    po = pd.DataFrame(po_rows)
    pol = pd.DataFrame({"sku_id": ["A"], "dc_id": ["D1"], "review_period_weeks": [1], "safety_stock_weeks": [1.0],
                        "order_min_qty_units": [48], "case_pack_units": [24]})
    p = tmp_path / "inv.zip"
    with zipfile.ZipFile(p, "w") as z:
        for n, d in {"fact_weekly_sales": sales, "fact_weekly_inventory": inv, "dim_product": prod, "dim_dc": dc, "dim_supplier": sup,
                     "product_supplier": ps, "fact_purchase_orders": po, "assortment_policy": pol}.items():
            z.writestr(f"{n}.csv", d.to_csv(index=False))
    from dpa.session import STORE
    s = STORE.create()
    s.add_upload(str(p), "inv.zip")
    return s


def test_order_rounded_to_case_pack(tmp_path):
    s = _session(tmp_path, on_hand=50, open_po_qty=0, po_weeks_ahead=0)
    r = inventory_plan(s, service_level=0.95)
    row = get_plan(s, r["plan_id"])["plan"].iloc[0]
    assert row["lead_time_weeks"] == 2 and row["review_period_weeks"] == 1
    assert row["order_up_to"] == pytest.approx(300, abs=6)          # 100/week x 3 weeks (+ ~0 safety stock)
    assert row["inventory_position"] == 50
    assert row["recommended_order_units"] == 264                    # ceil(250 / 24) x 24
    assert row["recommended_order_units"] % 24 == 0
    # 50 on hand, 100/week demand, nothing arriving -> out in the first week, inside the 2-week lead time
    assert row["status"] == "stockout within lead time - expedite"


def test_open_po_counts_and_no_order_needed(tmp_path):
    s = _session(tmp_path, on_hand=150, open_po_qty=300, po_weeks_ahead=2)
    r = inventory_plan(s)
    row = get_plan(s, r["plan_id"])["plan"].iloc[0]
    assert row["on_order"] == 300 and row["inventory_position"] == 450
    assert row["recommended_order_units"] == 0                      # 450 > 300 order-up-to level
    assert row["status"] == "ok"                                    # 150 -> 50 -> 250 -> ... never negative in lead time
    assert r["consistency_checks"][0]["difference_pct"] == pytest.approx(0.0)


def test_moq_applies(tmp_path):
    s = _session(tmp_path, on_hand=290, open_po_qty=0, po_weeks_ahead=0)
    r = inventory_plan(s)
    row = get_plan(s, r["plan_id"])["plan"].iloc[0]
    # need ~10 units -> one case of 24 is below the MOQ of 48, so 48
    assert row["recommended_order_units"] == 48
    with pytest.raises(Exception):
        inventory_plan(s, service_level=1.5)
