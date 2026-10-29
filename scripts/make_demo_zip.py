"""Build data/demo/demo_data.zip: one Excel workbook per table (one sheet each), from a folder of CSV files.

Usage: uv run python scripts/make_demo_zip.py <folder-with-csvs>
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "data" / "demo" / "demo_data.zip"
tmp = out.parent / "_xlsx"
tmp.mkdir(parents=True, exist_ok=True)
for csv in sorted(src.glob("*.csv")):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    # keep values as text so dates/ids look exactly as in the source; the loader re-infers types
    with pd.ExcelWriter(tmp / f"{csv.stem}.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name=csv.stem[:31], index=False)
    print("wrote", csv.stem, df.shape)
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(tmp.glob("*.xlsx")):
        z.write(f, f"demo_data/{f.name}")
for f in tmp.glob("*.xlsx"):
    f.unlink()
tmp.rmdir()
print("->", out, round(out.stat().st_size / 1e6, 1), "MB")
