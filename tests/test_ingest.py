"""Zip / CSV / Excel loading tests, including guardrails."""
import io
import zipfile

import pandas as pd
import pytest

from dpa.ingest import UploadError, ingest_upload


def _zip(tmp_path, members: dict[str, bytes], name="t.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        for n, data in members.items():
            z.writestr(n, data)
    return p


def _xlsx_bytes(sheets: dict[str, pd.DataFrame], title_rows: int = 0) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for sh, df in sheets.items():
            df.to_excel(w, sheet_name=sh, index=False, startrow=title_rows)
            if title_rows:
                w.sheets[sh].cell(row=1, column=1, value="Report title")
    return buf.getvalue()


CSV = b"SKU ID,Week Start,Units Sold,Unit Cost ($)\nA,2024-01-01,5,\"$1,200.50\"\nB,2024-01-08,0,3\n"


def test_zip_basic_and_system_files(tmp_path):
    p = _zip(tmp_path, {
        "data/sales.csv": CSV,
        "__MACOSX/data/._sales.csv": b"junk",
        "data/.DS_Store": b"junk",
        "data/~$products.xlsx": b"lock",
        "readme.txt": b"hello",
    })
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    assert [t.name for t in r.tables] == ["sales"]
    t = r.tables[0]
    assert list(t.df.columns) == ["sku_id", "week_start", "units_sold", "unit_cost"]
    assert t.column_map["unit_cost"] == "Unit Cost ($)"
    assert t.df["unit_cost"].tolist() == [1200.5, 3.0]
    assert pd.api.types.is_datetime64_any_dtype(t.df["week_start"])
    reasons = {s["file"]: s["reason"] for s in r.skipped}
    assert "__MACOSX" in reasons["__MACOSX/data/._sales.csv"]
    assert ".DS_Store" in reasons["data/.DS_Store"]
    assert "lock" in reasons["data/~$products.xlsx"]
    assert "not a table file" in reasons["readme.txt"]


def test_path_traversal_rejected(tmp_path):
    p = tmp_path / "evil.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("../../evil.csv", CSV)
        z.writestr("ok.csv", CSV)
    r = ingest_upload(str(p), "evil.zip", str(tmp_path / "x"))
    assert [t.name for t in r.tables] == ["ok"]
    assert any("unsafe path" in s["reason"] for s in r.skipped)
    assert not (tmp_path / "evil.csv").exists()


def test_bad_zip_raises_clear_error(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"this is not a zip")
    with pytest.raises(UploadError, match="not a valid ZIP"):
        ingest_upload(str(p), "bad.zip", str(tmp_path / "x"))


def test_corrupt_member_does_not_fail_upload(tmp_path):
    p = _zip(tmp_path, {"good.csv": CSV, "broken.xlsx": b"PK\x03\x04garbage", "empty.csv": b""})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    assert [t.name for t in r.tables] == ["good"]
    files = {s["file"] for s in r.skipped}
    assert {"broken.xlsx", "empty.csv"} <= files


def test_password_protected_xlsx_reported(tmp_path):
    # An encrypted .xlsx is an OLE2 container (magic D0 CF 11 E0)
    p = _zip(tmp_path, {"secret.xlsx": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600, "good.csv": CSV})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    reason = next(s["reason"] for s in r.skipped if s["file"] == "secret.xlsx")
    assert "password-protected" in reason


def test_header_row_detection_and_blank_trim(tmp_path):
    df = pd.DataFrame({"Store Code": ["S1", "S2"], "Region Name": ["N", "S"]})
    data = _xlsx_bytes({"Stores": df}, title_rows=2)
    p = _zip(tmp_path, {"Stores Master.xlsx": data})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    t = r.tables[0]
    assert t.name == "stores_master"
    assert list(t.df.columns) == ["store_code", "region_name"]
    assert len(t.df) == 2
    assert any("header found on row" in n for n in t.notes)


def test_multisheet_workbook_needs_decision(tmp_path):
    a = pd.DataFrame({"id": [1, 2]})
    b = pd.DataFrame({"code": ["x", "y"]})
    data = _xlsx_bytes({"First": a, "Second": b})
    p = _zip(tmp_path, {"book.xlsx": data})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    assert r.tables == [] and r.pending_workbooks and r.pending_workbooks[0].sheets == ["First", "Second"]
    wb = r.pending_workbooks[0]
    r_all = ingest_upload(wb.file_path, "book.xlsx", str(tmp_path / "y"), sheet_mode={"book.xlsx": "all"})
    assert sorted(t.name for t in r_all.tables) == ["first", "second"]
    r_first = ingest_upload(wb.file_path, "book.xlsx", str(tmp_path / "z"), sheet_mode={"book.xlsx": "first"})
    assert [t.name for t in r_first.tables] == ["book"]


def test_duplicate_table_csv_preferred_and_compared(tmp_path):
    df = pd.read_csv(io.BytesIO(CSV))
    same = _xlsx_bytes({"s": df})
    df2 = df.copy()
    df2.loc[0, "Units Sold"] = 99
    diff = _xlsx_bytes({"s": df2})
    p = _zip(tmp_path, {"sales.csv": CSV, "sales.xlsx": same})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    assert len(r.tables) == 1 and r.tables[0].source_file == "sales.csv"
    assert r.duplicates[0]["identical"] is True
    p2 = _zip(tmp_path, {"sales.csv": CSV, "sales.xlsx": diff}, name="t2.zip")
    r2 = ingest_upload(str(p2), "t2.zip", str(tmp_path / "y"))
    assert r2.duplicates[0]["identical"] is False and r2.warnings


def test_leading_zero_ids_stay_text(tmp_path):
    p = _zip(tmp_path, {"stores.csv": b"store_id,sales\n001,5\n002,7\n"})
    r = ingest_upload(str(p), "t.zip", str(tmp_path / "x"))
    assert r.tables[0].df["store_id"].tolist() == ["001", "002"]


def test_single_csv_and_unsupported(tmp_path):
    f = tmp_path / "Sales Data.csv"
    f.write_bytes(CSV)
    r = ingest_upload(str(f), "Sales Data.csv", str(tmp_path / "x"))
    assert r.tables[0].name == "sales_data"
    g = tmp_path / "notes.pdf"
    g.write_bytes(b"%PDF")
    r2 = ingest_upload(str(g), "notes.pdf", str(tmp_path / "y"))
    assert not r2.tables and "unsupported" in r2.skipped[0]["reason"]
