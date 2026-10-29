import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("DPA_SESSION_DIR", str(ROOT / ".pytest_sessions"))


def make_mini_dataset(tmp_path: Path) -> Path:
    """A tiny relational dataset (hand-checkable) written as a zip of CSVs."""
    import zipfile
    weeks = pd.date_range("2023-01-02", periods=120, freq="7D")
    prod = pd.DataFrame({"sku_id": ["S1", "S2"], "category": ["skin", "hair"], "brand_id": ["B1", "B1"]})
    brand = pd.DataFrame({"brand_id": ["B1"], "brand_name": ["Brand"]})
    rows = []
    rng = np.random.default_rng(0)
    for s, base in (("S1", 100), ("S2", 40)):
        for i, w in enumerate(weeks):
            season = 1 + 0.3 * np.sin(2 * np.pi * i / 52)
            d = int(round(base * season + rng.normal(0, 3)))
            rows.append({"sku_id": s, "dc_id": "D1", "week_start": w.date().isoformat(), "customer_demand": d,
                         "units_sold": d, "lost_sales_units": 0, "stockout_flag": 0})
    sales = pd.DataFrame(rows)
    dc = pd.DataFrame({"dc_id": ["D1"], "dc_name": ["Main"]})
    p = tmp_path / "mini.zip"
    with zipfile.ZipFile(p, "w") as z:
        for name, df in {"dim_product": prod, "dim_brand": brand, "fact_weekly_sales": sales, "dim_dc": dc}.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
    return p


@pytest.fixture
def mini_zip(tmp_path):
    return make_mini_dataset(tmp_path)


@pytest.fixture
def mini_session(mini_zip):
    from dpa.session import STORE
    s = STORE.create()
    s.add_upload(str(mini_zip), "mini.zip")
    return s
