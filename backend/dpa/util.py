"""Small shared helpers: name normalisation, JSON-safe conversion, number formatting."""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def snake_case(name: Any) -> str:
    """Normalise a table or column name to snake_case ("Unit Cost ($)" -> "unit_cost")."""
    s = str(name).strip()
    s = re.sub(r"\(.*?\)", " ", s)                      # drop parenthetical units
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)       # camelCase -> camel_Case
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "col"
    if s[0].isdigit():
        s = "c_" + s
    return s


def dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to plain JSON-serialisable Python."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        if pd.isna(obj):
            return None
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, date):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return df_records(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def df_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict]:
    d = df if max_rows is None else df.head(max_rows)
    return [to_jsonable(r) for r in d.to_dict(orient="records")]


def round_sig(x: float | None, digits: int = 4) -> float | None:
    if x is None:
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
    except TypeError:
        return x
    if x == 0:
        return 0.0
    return float(round(x, digits - int(math.floor(math.log10(abs(x)))) - 1))


def pct(a: float, b: float) -> float | None:
    return None if not b else 100.0 * a / b


ID_LIKE = re.compile(r"(^id$|_id$|_key$|_code$|^sku$)")
DATE_NAME = re.compile(r"(date|week|month|^period$|_start$|_end$|^ds$|^day$|^year_month$|_from$|_to$)")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
US_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
