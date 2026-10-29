"""Key inference, join validation (fan-out, grain mismatch) and data-quality checks."""
import numpy as np
import pandas as pd
import pytest

from dpa.profiling import infer_primary_key
from dpa.quality import data_quality_report, treat_outliers
from dpa.relationships import validated_join
from dpa.timeseries import seasonal_decompose_robust


def test_primary_key_inference():
    df = pd.DataFrame({"sku_id": ["a", "a", "b"], "dc_id": ["x", "y", "x"], "units": [1, 2, 3]})
    assert infer_primary_key(df)[0] == ["dc_id", "sku_id"] or infer_primary_key(df)[0] == ["sku_id", "dc_id"]
    df2 = pd.DataFrame({"po_line_id": ["1", "2"], "sku_id": ["a", "a"]})
    assert infer_primary_key(df2)[0] == ["po_line_id"]


def test_relationships_inferred(mini_session):
    rels = {(r.child_table, tuple(r.child_columns), r.parent_table) for r in mini_session.relationships}
    assert ("fact_weekly_sales", ("sku_id",), "dim_product") in rels
    assert ("fact_weekly_sales", ("dc_id",), "dim_dc") in rels
    assert ("dim_product", ("brand_id",), "dim_brand") in rels
    for r in mini_session.relationships:
        assert r.coverage_pct == 100.0 and r.orphan_count == 0


def test_join_fanout_detected():
    left = pd.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
    right = pd.DataFrame({"k": [1, 1, 2], "w": ["a", "b", "c"]})
    out, rep = validated_join(left, right, ["k"], "left", "L", "R")
    # hand count: key 1 matches 2 rows, key 2 matches 1, key 3 none -> 2 + 1 + 1 = 4 rows
    assert rep.result_rows == 4 and len(out) == 4
    assert rep.right_key_unique is False and rep.fanout_max == 2
    assert rep.left_unmatched_rows == 1
    assert any("FAN-OUT" in w for w in rep.warnings)
    assert any("Row count changed" in w for w in rep.warnings)


def test_join_grain_mismatch_warning():
    sales = pd.DataFrame({"sku": ["a", "a"], "channel": ["RET", "ECOM"], "units": [5, 7]})
    inv = pd.DataFrame({"sku": ["a"], "on_hand": [100]})
    out, rep = validated_join(sales, inv, ["sku"], "left", "sales", "inventory", right_role="fact")
    assert len(out) == 2 and rep.right_key_unique
    assert any("GRAIN MISMATCH" in w for w in rep.warnings)
    # The naive sum of on_hand would be 200, double the true 100 - which is why we warn.
    assert out["on_hand"].sum() == 200


def test_join_inner_drops_reported():
    left = pd.DataFrame({"k": [1, 2]})
    right = pd.DataFrame({"k": [1], "x": [1]})
    out, rep = validated_join(left, right, ["k"], "inner", "L", "R")
    assert len(out) == 1 and any("dropped by the inner join" in w for w in rep.warnings)


def test_quality_report_mini(mini_session):
    r = data_quality_report(mini_session)
    ts = r["timeseries"]
    assert ts["series"] == 2 and ts["missing_weeks_total"] == 0
    assert ts["date_range"]["weeks"] == 120
    assert ts["consistency"][0]["pass_pct"] == 100.0
    assert r["table_status_counts"]["error"] == 0


def test_outlier_detection_finds_injected_spike():
    idx = pd.date_range("2022-01-03", periods=156, freq="7D")
    rng = np.random.default_rng(1)
    y = pd.Series(100 + 20 * np.sin(2 * np.pi * np.arange(156) / 52) + rng.normal(0, 2, 156), index=idx).round()
    y.iloc[80] = 400  # injected spike
    dec = seasonal_decompose_robust(y, log=True)
    flagged = dec["robust_z"].abs() > 3.5
    assert flagged.iloc[80]
    assert flagged.sum() <= 3  # nothing else materially flagged
    capped, eff = treat_outliers(y, "impute")
    assert eff["weeks_flagged"] >= 1
    assert abs(capped.iloc[80] - y.drop(y.index[80]).iloc[78:82].mean()) < 40
    assert eff["total_after"] < eff["total_before"]
    kept, eff_keep = treat_outliers(y, "keep")
    assert eff_keep["total_after"] == eff_keep["total_before"]
