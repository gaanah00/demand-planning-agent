"""Upload handling: CSV, Excel and ZIP archives (one table per file / per sheet).

Design rules (all reported to the user, never silent):
* ZIP members are extracted into a per-session folder after a path-traversal check.
* System files (__MACOSX/, .DS_Store, ~$ lock files, hidden files) are skipped with a reason.
* Each file becomes one table named after the file (snake_case). Multi-sheet workbooks are
  held back until the user decides: first sheet only, or one table per sheet.
* Header rows are detected, blank rows/columns trimmed, column names standardised; the
  mapping to the original names is kept for the data dictionary.
* If two files produce the same table name (e.g. sales.csv and sales.xlsx), one copy is
  kept (CSV preferred) and the other is reported as a duplicate, with a content comparison.
* Unreadable / password-protected / corrupt files are reported individually; they never
  fail the whole upload.
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .util import DATE_NAME, ISO_DATE, ISO_MONTH, US_DATE, dedupe_names, snake_case

TABLE_EXTS = {".csv", ".xlsx", ".xls", ".xlsm"}
MAX_ZIP_MEMBERS = 500
MAX_UNCOMPRESSED_BYTES = 1_000_000_000  # 1 GB zip-bomb guard
MAX_FILE_BYTES = 400_000_000
EXT_PREFERENCE = {".csv": 0, ".xlsx": 1, ".xlsm": 2, ".xls": 3}


class UploadError(Exception):
    """A user-facing error about an uploaded file."""


@dataclass
class LoadedTable:
    name: str
    df: pd.DataFrame
    source_file: str
    sheet: str | None = None
    header_row: int = 0
    column_map: dict[str, str] = field(default_factory=dict)  # clean -> original
    notes: list[str] = field(default_factory=list)


@dataclass
class PendingWorkbook:
    file_path: str
    display_name: str
    table_name: str
    sheets: list[str]


@dataclass
class IngestResult:
    tables: list[LoadedTable] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)       # {file, reason}
    duplicates: list[dict] = field(default_factory=list)    # {table, kept, skipped, identical, detail}
    pending_workbooks: list[PendingWorkbook] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers

def _is_system_file(member: str) -> str | None:
    p = PurePosixPath(member.replace("\\", "/"))
    parts = p.parts
    if any(part == "__MACOSX" for part in parts):
        return "macOS metadata folder (__MACOSX)"
    base = p.name
    if base == ".DS_Store":
        return "macOS system file (.DS_Store)"
    if base.startswith("~$"):
        return "Office lock/temp file (~$)"
    if base.lower() in {"thumbs.db", "desktop.ini"}:
        return "Windows system file"
    if base.startswith("."):
        return "hidden file"
    return None


def _unsafe_member(member: str) -> bool:
    m = member.replace("\\", "/")
    if m.startswith("/") or (len(m) > 1 and m[1] == ":"):
        return True
    return any(part == ".." for part in PurePosixPath(m).parts)


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 20) -> int:
    """Return index of the header row in a header=None frame.

    The header is the first row (within the first rows) that is mostly non-empty text and is
    followed by at least one row of data. Works for files with title rows above the table.
    """
    if raw.empty:
        return 0
    width = raw.shape[1]
    best = 0
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i]
        nonnull = row.dropna()
        if len(nonnull) == 0:
            continue
        str_share = np.mean([isinstance(v, str) and not _looks_numeric(v) for v in nonnull])
        fill = len(nonnull) / width
        if fill >= 0.6 and str_share >= 0.8:
            return i
        if i == 0:
            best = 0
    return best


def _looks_numeric(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def _standardise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    originals = [str(c) if not str(c).startswith("Unnamed:") else f"column_{i+1}" for i, c in enumerate(df.columns)]
    clean = dedupe_names([snake_case(c) for c in originals])
    df = df.copy()
    df.columns = clean
    return df, dict(zip(clean, originals))


def _coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse date-like and numeric-like text columns. Returns frame and notes."""
    notes: list[str] = []
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        if pd.api.types.is_object_dtype(s) and s.dropna().map(lambda v: isinstance(v, (pd.Timestamp,))).all() and s.notna().any():
            out[col] = pd.to_datetime(s, errors="coerce")
            continue
        nonnull = s.dropna().astype(str).str.strip()
        nonnull = nonnull[nonnull != ""]
        if nonnull.empty:
            continue
        sample = nonnull.head(500)
        # Dates ---------------------------------------------------------------
        if sample.str.match(ISO_DATE).mean() > 0.95:
            parsed = pd.to_datetime(s, errors="coerce")
            bad = parsed.isna().sum() - s.isna().sum()
            out[col] = parsed
            if bad > 0:
                notes.append(f"{col}: {bad} value(s) could not be parsed as dates and were set to missing")
            continue
        if sample.str.match(ISO_MONTH).mean() > 0.95 and DATE_NAME.search(col):
            # Monthly period ("2024-03") -> first day of month; granularity recorded as monthly.
            out[col] = pd.to_datetime(s.astype("string") + "-01", errors="coerce")
            notes.append(f"{col}: monthly values (YYYY-MM) stored as the first day of each month")
            continue
        if sample.str.match(US_DATE).mean() > 0.95:
            out[col] = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
            notes.append(f"{col}: US-style dates (M/D/Y) parsed")
            continue
        # Numbers with currency symbols / thousands separators / percent signs ----------
        cleaned = nonnull.str.replace(r"[,$€£\s]", "", regex=True)
        is_pct = cleaned.str.endswith("%").mean() > 0.95
        cleaned = cleaned.str.rstrip("%")
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().all() and not nonnull.str.match(r"^0\d+$").any():
            full = pd.to_numeric(s.astype("string").str.strip().replace("", pd.NA).str.replace(r"[,$€£\s%]", "", regex=True), errors="coerce")
            full = full.astype("float64")  # plain numpy dtype (avoid nullable Float64/Int64)
            if full.notna().all() and (full % 1 == 0).all():
                full = full.astype("int64")
            out[col] = full
            if is_pct:
                notes.append(f"{col}: percent signs removed; values are in percent units")
            elif (nonnull.str.replace(r"\s", "", regex=True) != cleaned).any():
                notes.append(f"{col}: currency symbols / thousands separators removed")
    return out, notes


def _finalise(raw: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, str], list[str]]:
    raw = _trim(raw)
    if raw.empty:
        raise UploadError("the file contains no data")
    hdr_pos = _detect_header_row(raw.reset_index(drop=True))
    raw = raw.reset_index(drop=True)
    header = raw.iloc[hdr_pos].tolist()
    body = raw.iloc[hdr_pos + 1:].reset_index(drop=True)
    body.columns = [h if (h is not None and not (isinstance(h, float) and np.isnan(h))) else f"Unnamed: {i}" for i, h in enumerate(header)]
    body = _trim(body)
    if body.empty:
        raise UploadError("a header row was found but no data rows")
    body, cmap = _standardise_columns(body)
    # header=None reads everything as object; re-infer dtypes before targeted coercion
    body = body.infer_objects()
    body, notes = _coerce_types(body)
    notes = list(notes)
    if hdr_pos > 0:
        notes.insert(0, f"header found on row {hdr_pos + 1}; {hdr_pos} title row(s) above it ignored")
    return body, hdr_pos, cmap, notes


def _sniff_sep(path: str, enc: str) -> str:
    import csv
    with open(path, "r", encoding=enc, errors="strict") as fh:
        sample = fh.read(64_000)
    if not sample.strip():
        raise UploadError("the file is empty")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_file(path: str) -> tuple[pd.DataFrame, int, dict[str, str], list[str]]:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            sep = _sniff_sep(path, enc)
            # dtype=str keeps leading zeros and lets header detection see raw text
            raw = pd.read_csv(path, header=None, dtype=str, encoding=enc, sep=sep, skip_blank_lines=True)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except UploadError:
            raise
        except pd.errors.EmptyDataError:
            raise UploadError("the file is empty")
        except Exception as e:  # malformed CSV
            raise UploadError(f"could not parse as CSV ({type(e).__name__}: {str(e)[:120]})")
    else:
        raise UploadError(f"unknown text encoding ({last_err})")
    return _finalise(raw)


def _excel_engine(path: str) -> str | None:
    """Prefer the fast Rust-based calamine reader; fall back to openpyxl/xlrd."""
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return "xlrd" if path.lower().endswith(".xls") else "openpyxl"


def list_sheets(path: str) -> list[str]:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
        if head.startswith(b"\xd0\xcf\x11\xe0") and not path.lower().endswith(".xls"):
            # OLE container with an .xlsx extension => encrypted/password-protected workbook
            raise UploadError("the workbook is password-protected or encrypted")
        xf = pd.ExcelFile(path, engine=_excel_engine(path))
        names = []
        for sh in xf.sheet_names:
            try:
                peek = xf.parse(sh, header=None, nrows=5)
                if not peek.dropna(how="all").empty:
                    names.append(sh)
            except Exception:
                continue
        return names
    except UploadError:
        raise
    except ImportError as e:
        raise UploadError(f"reading this Excel format needs an extra package ({e})")
    except Exception as e:
        msg = str(e).lower()
        if "encrypt" in msg or "password" in msg:
            raise UploadError("the workbook is password-protected or encrypted")
        raise UploadError(f"not a readable Excel workbook ({type(e).__name__})")


def read_excel_sheet(path: str, sheet: str) -> tuple[pd.DataFrame, int, dict[str, str], list[str]]:
    try:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, engine=_excel_engine(path))
    except Exception as e:
        raise UploadError(f"could not read sheet '{sheet}' ({type(e).__name__})")
    return _finalise(raw)


# --------------------------------------------------------------------------- main entry points

def _collect_files(upload_path: str, display_name: str, extract_dir: str, result: IngestResult) -> list[tuple[str, str]]:
    """Return list of (local_path, display_name) table files, extracting ZIPs safely."""
    ext = Path(display_name).suffix.lower()
    if ext != ".zip":
        if ext not in TABLE_EXTS:
            result.skipped.append({"file": display_name, "reason": f"unsupported file type '{ext or '?'}' (use .csv, .xlsx, .xls or .zip)"})
            return []
        return [(upload_path, display_name)]

    try:
        zf = zipfile.ZipFile(upload_path)
    except zipfile.BadZipFile:
        raise UploadError(f"'{display_name}' is not a valid ZIP archive (it may be corrupt or incomplete)")
    files: list[tuple[str, str]] = []
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            raise UploadError(f"the ZIP has {len(infos)} entries; the limit is {MAX_ZIP_MEMBERS}")
        total = sum(i.file_size for i in infos)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise UploadError("the ZIP expands to more than 1 GB; please upload a smaller archive")
        for info in infos:
            name = info.filename
            if info.is_dir():
                continue
            if _unsafe_member(name):
                result.skipped.append({"file": name, "reason": "rejected: unsafe path (path traversal or absolute path)"})
                continue
            sysr = _is_system_file(name)
            if sysr:
                result.skipped.append({"file": name, "reason": f"ignored: {sysr}"})
                continue
            mext = Path(name).suffix.lower()
            if mext == ".zip":
                result.skipped.append({"file": name, "reason": "ignored: nested ZIP archives are not opened"})
                continue
            if mext not in TABLE_EXTS:
                result.skipped.append({"file": name, "reason": f"ignored: not a table file ('{mext or 'no extension'}')"})
                continue
            if info.flag_bits & 0x1:
                result.skipped.append({"file": name, "reason": "could not read: the file is password-protected inside the ZIP"})
                continue
            if info.file_size > MAX_FILE_BYTES:
                result.skipped.append({"file": name, "reason": "skipped: file larger than 400 MB"})
                continue
            target = Path(extract_dir) / PurePosixPath(name.replace("\\", "/"))
            # Final guard: resolved path must stay inside extract_dir
            if not str(target.resolve()).startswith(str(Path(extract_dir).resolve())):
                result.skipped.append({"file": name, "reason": "rejected: unsafe path"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            except RuntimeError as e:  # encrypted member
                result.skipped.append({"file": name, "reason": f"could not read: {e}"})
                continue
            except (zipfile.BadZipFile, OSError, EOFError) as e:
                result.skipped.append({"file": name, "reason": f"could not extract: {type(e).__name__}"})
                continue
            files.append((str(target), name))
    if not files:
        result.warnings.append(f"No table files (.csv/.xlsx/.xls) were found in '{display_name}'.")
    return files


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shapes differ ({a.shape[0]}x{a.shape[1]} vs {b.shape[0]}x{b.shape[1]})"
    if list(a.columns) != list(b.columns):
        return False, "column names differ"
    diffs = 0
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            neq = ~np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-9)
        elif pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_datetime64_any_dtype(y):
            neq = pd.to_datetime(x, errors="coerce").ne(pd.to_datetime(y, errors="coerce")) & ~(x.isna() & y.isna())
        else:
            neq = x.astype("string").fillna("<NA>").ne(y.astype("string").fillna("<NA>"))
        diffs += int(np.asarray(neq).sum())
    if diffs:
        return False, f"{diffs} cell(s) differ"
    return True, "identical content"


def ingest_upload(upload_path: str, display_name: str, extract_dir: str,
                  sheet_mode: dict[str, str] | None = None,
                  existing_tables: set[str] | None = None) -> IngestResult:
    """Load an uploaded file (CSV / Excel / ZIP) into tables.

    sheet_mode maps a workbook display name -> "first" | "all" for multi-sheet workbooks.
    """
    sheet_mode = sheet_mode or {}
    result = IngestResult()
    os.makedirs(extract_dir, exist_ok=True)
    files = _collect_files(upload_path, display_name, extract_dir, result)

    candidates: dict[str, list[tuple[int, LoadedTable]]] = {}
    for path, shown in files:
        ext = Path(shown).suffix.lower()
        stem = snake_case(Path(shown).stem)
        try:
            if ext == ".csv":
                df, hdr, cmap, notes = read_csv_file(path)
                lt = LoadedTable(stem, df, shown, None, hdr, cmap, notes)
                candidates.setdefault(stem, []).append((EXT_PREFERENCE[ext], lt))
            else:
                sheets = list_sheets(path)
                if not sheets:
                    result.skipped.append({"file": shown, "reason": "workbook has no non-empty sheets"})
                    continue
                mode = sheet_mode.get(shown)
                if len(sheets) > 1 and mode is None:
                    result.pending_workbooks.append(PendingWorkbook(path, shown, stem, sheets))
                    continue
                use = sheets if mode == "all" else sheets[:1]
                for sh in use:
                    df, hdr, cmap, notes = read_excel_sheet(path, sh)
                    tname = stem if (len(use) == 1) else snake_case(sh)
                    lt = LoadedTable(tname, df, shown, sh, hdr, cmap, notes)
                    if len(sheets) > 1 and mode == "first":
                        lt.notes.append(f"multi-sheet workbook: only the first sheet ('{sh}') was loaded, as chosen")
                    candidates.setdefault(tname, []).append((EXT_PREFERENCE.get(ext, 9), lt))
        except UploadError as e:
            result.skipped.append({"file": shown, "reason": f"could not read: {e}"})
        except Exception as e:  # never fail the whole upload because of one file
            result.skipped.append({"file": shown, "reason": f"could not read: unexpected {type(e).__name__}: {str(e)[:150]}"})

    for tname, items in candidates.items():
        items.sort(key=lambda t: t[0])
        kept = items[0][1]
        for _, other in items[1:]:
            same, detail = _frames_equal(kept.df, other.df)
            result.duplicates.append({
                "table": tname, "kept": kept.source_file, "skipped": other.source_file,
                "identical": same, "detail": detail,
            })
            if not same:
                result.warnings.append(
                    f"'{other.source_file}' and '{kept.source_file}' both map to table '{tname}' but their content differs "
                    f"({detail}). Kept '{kept.source_file}' (CSV preferred). Tell me if the other file should be used instead.")
        if existing_tables and tname in existing_tables:
            kept.notes.append("replaced a table with the same name from an earlier upload")
        result.tables.append(kept)
    result.tables.sort(key=lambda t: t.name)
    return result
