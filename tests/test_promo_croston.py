"""Croston/SBA, demand classes and promotion-lift recovery on synthetic data with a known effect."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from dpa.forecasting import demand_class, fc_croston
from dpa.promotions import estimate_promo_lift


def test_croston_hand_computed():
    # demand 4 every 2nd week: sizes stay 4, intervals settle at 2 -> SBA = (1 - 0.05) x 4 / 2 = 1.9
    y = np.array([0, 4] * 30, float)
    f = fc_croston(y, 3)
    assert f == pytest.approx([1.9, 1.9, 1.9], abs=0.02)
    assert fc_croston(np.zeros(10), 2).tolist() == [0, 0]


def test_demand_classes():
    assert demand_class(np.full(52, 10.0)) == "smooth"
    assert demand_class(np.array([0, 0, 5, 0, 0, 5] * 10, float)) == "intermittent"
    assert demand_class(np.array([0, 0, 1, 0, 0, 40] * 10, float)) == "lumpy"


def test_promo_lift_recovers_known_effect(tmp_path):
    rng = np.random.default_rng(7)
    weeks = pd.date_range("2023-01-02", periods=104, freq="7D")
    rows, promos, scope = [], [], []
    for s in range(6):
        base = 50 * (s + 1)
        for i, w in enumerate(weeks):
            on = (i % 9 == 4)
            pid = f"P{s}_{i}" if on else ""
            mult = 1.5 if on else 1.0                       # true lift +50%
            d = int(round(base * mult * np.exp(rng.normal(0, 0.05))))
            rows.append({"sku_id": f"S{s}", "dc_id": "D1", "week_start": w.date().isoformat(), "customer_demand": d,
                         "units_sold": d, "promo_id": pid, "discount_pct": 20 if on else 0})
            if on:
                promos.append({"promo_id": pid, "promo_type": "TPR", "start_week": w.date().isoformat(), "end_week": w.date().isoformat()})
    p = tmp_path / "promo.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("fact_weekly_sales.csv", pd.DataFrame(rows).to_csv(index=False))
        z.writestr("dim_promotion.csv", pd.DataFrame(promos).to_csv(index=False))
        z.writestr("dim_product.csv", pd.DataFrame({"sku_id": [f"S{s}" for s in range(6)], "category": "x"}).to_csv(index=False))
        z.writestr("dim_dc.csv", pd.DataFrame({"dc_id": ["D1"], "dc_name": ["Main"]}).to_csv(index=False))
    from dpa.session import STORE
    sess = STORE.create()
    sess.add_upload(str(p), "promo.zip")
    r = estimate_promo_lift(sess)
    eff = {e["effect"]: e for e in r["results"][0]["effects"]}
    assert eff["TPR"]["lift_pct"] == pytest.approx(50, abs=6)
    assert eff["TPR"]["ci95_pct"][0] < 50 < eff["TPR"]["ci95_pct"][1]
    assert abs(eff["post_promo_week"]["lift_pct"]) < 6     # no true dip
