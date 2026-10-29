"""Table profiling, primary-key inference and the auto-generated data dictionary."""
from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd

from .util import DATE_NAME, ID_LIKE, to_jsonable

# ---------------------------------------------------------------------------- roles & keys

def table_role(name: str, df: pd.DataFrame, pk: list[str] | None) -> str:
    """Classify a table as dimension / fact / bridge / reference / external."""
    n = name.lower()
    if n.startswith(("dim_", "d_")):
        return "dimension"
    if n.startswith(("fact_", "f_", "fct_")):
        return "fact"
    if n.startswith(("ref_", "lkp_", "lookup_")):
        return "reference"
    if n.endswith(("_scope", "_map", "_bridge", "_link", "_policy")) or (pk and len(pk) >= 2 and all(ID_LIKE.search(c) or c.endswith("_type") or c.endswith("_value") for c in pk)):
        return "bridge"
    if pk and any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in pk if c in df):
        return "fact"
    if pk and len(pk) == 1:
        return "dimension"
    return "other"


def _key_candidates(df: pd.DataFrame) -> list[str]:
    cands = []
    for c in df.columns:
        s = df[c]
        if ID_LIKE.search(c) or c in ("category", "subcategory", "month", "scope_type", "scope_value", "region") \
                or pd.api.types.is_datetime64_any_dtype(s) and DATE_NAME.search(c):
            cands.append(c)
    return cands


def infer_primary_key(df: pd.DataFrame, max_cols: int = 4) -> tuple[list[str] | None, str]:
    """Smallest set of key-like columns that uniquely identifies rows (no nulls).

    Preference order: single id-like column; then combinations of id/date columns.
    Returns (key, method description).
    """
    if df.empty:
        return None, "empty table"
    n = len(df)
    cands = _key_candidates(df)
    # Prefer "<table-ish>_id" single columns that are unique and non-null
    for c in cands:
        s = df[c]
        if s.notna().all() and s.nunique(dropna=False) == n and not pd.api.types.is_float_dtype(s):
            return [c], "single column unique and non-null"
    nonnull = [c for c in cands if df[c].notna().all()]
    # order: ids first, then dates, then others; limit search space
    nonnull.sort(key=lambda c: (0 if ID_LIKE.search(c) else (1 if pd.api.types.is_datetime64_any_dtype(df[c]) else 2), df[c].nunique()))
    nonnull = nonnull[:8]
    for k in range(2, max_cols + 1):
        for combo in itertools.combinations(nonnull, k):
            if not df.duplicated(list(combo)).any():
                return list(combo), f"{k}-column combination unique and non-null"
    # fall back: any single column unique
    for c in df.columns:
        if df[c].notna().all() and df[c].nunique() == n:
            return [c], "single non-id column unique (weak key)"
    return None, "no unique key found among id/date columns (duplicates present?)"


def infer_granularity(s: pd.Series) -> str | None:
    if not pd.api.types.is_datetime64_any_dtype(s):
        return None
    u = pd.Series(s.dropna().unique()).sort_values()
    if len(u) < 3:
        return "single date" if len(u) == 1 else "few dates"
    diffs = u.diff().dropna().dt.days
    med = float(diffs.median())
    if med == 1:
        return "daily"
    if med == 7:
        wd = pd.to_datetime(u).dt.day_name().value_counts()
        return f"weekly (weeks start on {wd.index[0]})" if wd.iloc[0] / len(u) > 0.95 else "weekly"
    if 28 <= med <= 31:
        return "monthly"
    if 89 <= med <= 92:
        return "quarterly"
    if 364 <= med <= 366:
        return "yearly"
    return f"irregular (median gap {med:.0f} days)"


# ---------------------------------------------------------------------------- units & descriptions

def guess_unit(col: str, s: pd.Series) -> str | None:
    c = col.lower()
    if not pd.api.types.is_numeric_dtype(s):
        return None
    if c.endswith("_usd") or any(k in c for k in ("revenue", "cost", "price", "spend", "refund", "amount", "margin")):
        if "index" in c:
            return "index"
        return "USD"
    if c.endswith("_pct") or "rate_pct" in c or c.endswith("_percent") or "share" in c and s.max() <= 1.0:
        return "percent (0-100)" if c.endswith("_pct") else "fraction (0-1)"
    if c.endswith("_rate") and s.dropna().between(0, 1).all():
        return "fraction (0-1)"
    if c.endswith("_f") or "temp" in c:
        return "°F"
    if c.endswith("_ml"):
        return "ml"
    if "weeks" in c:
        return "weeks"
    if c.endswith("_days") or "days" in c:
        return "days"
    if c.endswith("_months"):
        return "months"
    if "index" in c:
        return "index"
    if any(k in c for k in ("units", "qty", "demand", "on_hand", "receipts", "sold", "capacity")):
        return "units"
    if c.startswith(("is_", "has_")) or c.endswith("_flag") or set(s.dropna().unique()) <= {0, 1}:
        return "flag (0/1)"
    return None


_DESC_RULES: list[tuple[str, str]] = [
    (r"^sku_id$", "Product (SKU) identifier"),
    (r"^dc_id$", "Distribution centre identifier"),
    (r"^channel_id$", "Sales channel identifier"),
    (r"^brand_id$", "Brand identifier"),
    (r"^supplier_id$", "Supplier identifier"),
    (r"^promo_id$", "Promotion identifier"),
    (r"^event_id$", "External event identifier"),
    (r"^week_start$", "First day (Monday) of the week"),
    (r"^week_end$", "Last day of the week"),
    (r"customer_demand", "Unconstrained customer demand in units (sold + lost to stockouts)"),
    (r"^units_sold$", "Units actually sold / shipped"),
    (r"lost_sales", "Demand not fulfilled because of stockouts (units)"),
    (r"^list_price$", "Regular shelf/list price per unit"),
    (r"^selling_price$", "Actual price paid per unit after discount"),
    (r"^discount_pct$", "Discount off list price, in percent"),
    (r"^revenue$", "Sales revenue = units sold x selling price"),
    (r"stockout_flag", "1 if the item was out of stock during the week"),
    (r"opening_on_hand", "Inventory on hand at the start of the week"),
    (r"closing_on_hand", "Inventory on hand at the end of the week"),
    (r"receipts", "Units received into inventory"),
    (r"available_to_sell", "Units available to sell during the week"),
    (r"on_order", "Units on open purchase orders"),
    (r"weeks_of_cover", "Weeks of demand the current stock would cover"),
    (r"unit_cost", "Purchase cost per unit"),
    (r"tariff_rate", "Import tariff rate applied, in percent"),
    (r"freight_cost", "Freight cost for the order line"),
    (r"lead_time", "Lead time from order to receipt"),
    (r"qty_ordered", "Units ordered"),
    (r"qty_received", "Units received"),
    (r"delay_days", "Days late versus planned receipt"),
    (r"sourcing_share", "Share of the SKU's volume sourced from this supplier"),
    (r"capacity", "Supplier production capacity"),
    (r"order_min_qty|moq", "Minimum order quantity"),
    (r"case_pack", "Units per case"),
    (r"safety_stock", "Safety stock target"),
    (r"abc_class", "ABC classification by value"),
    (r"country", "Country"),
    (r"region", "Region"),
    (r"category", "Product category"),
    (r"subcategory", "Product subcategory"),
    (r"_name$", "Display name"),
    (r"description", "Free-text description"),
    (r"(start|from)$|_start_|start_", "Start date"),
    (r"(end|to)$|end_", "End date"),
    (r"announced", "Date the item was announced/known (use to avoid look-ahead)"),
    (r"release_date", "Publication date of the statistic (use to avoid look-ahead)"),
    (r"spend", "Marketing spend"),
    (r"temp", "Temperature"),
    (r"humidity", "Relative humidity"),
    (r"uv_index", "UV index"),
    (r"index", "Index value"),
    (r"^is_|_flag$", "Yes/no flag"),
]


def describe_column(table: str, col: str, s: pd.Series) -> str:
    for pat, desc in _DESC_RULES:
        if re.search(pat, col):
            return desc
    words = col.replace("_", " ")
    return words[:1].upper() + words[1:]


def _examples(s: pd.Series, k: int = 3) -> list[Any]:
    vals = s.dropna().unique()[:k]
    return [to_jsonable(v) for v in vals]


def profile_table(name: str, df: pd.DataFrame, pk: list[str] | None) -> dict:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_ranges = {}
    for c in date_cols:
        s = df[c].dropna()
        if len(s):
            date_ranges[c] = {"min": to_jsonable(s.min()), "max": to_jsonable(s.max()),
                              "distinct": int(s.nunique()), "granularity": infer_granularity(df[c])}
    main_date = None
    if pk:
        for c in pk:
            if c in date_ranges:
                main_date = c
                break
    if main_date is None and date_ranges:
        main_date = max(date_ranges, key=lambda c: date_ranges[c]["distinct"])
    grain = (" x ".join(pk) if pk else "unknown")
    return {
        "table": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "role": table_role(name, df, pk),
        "primary_key": pk,
        "grain": grain,
        "date_columns": date_ranges,
        "main_date": main_date,
        "date_range": date_ranges.get(main_date) if main_date else None,
        "numeric_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "missing_cells_pct": round(float(df.isna().mean().mean() * 100), 2) if df.size else 0.0,
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 2),
    }


def dtype_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "decimal"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def data_dictionary(name: str, df: pd.DataFrame, column_map: dict[str, str], pk: list[str] | None,
                    fk_cols: dict[str, str]) -> list[dict]:
    rows = []
    for c in df.columns:
        s = df[c]
        key = None
        if pk and c in pk:
            key = "PK" if len(pk) == 1 else "PK (part)"
        if c in fk_cols:
            key = (key + ", " if key else "") + f"FK -> {fk_cols[c]}"
        rows.append({
            "table": name,
            "column": c,
            "original_name": column_map.get(c, c),
            "type": dtype_label(s),
            "unit": guess_unit(c, s),
            "description": describe_column(name, c, s),
            "description_status": "inferred",
            "examples": _examples(s),
            "missing_pct": round(float(s.isna().mean() * 100), 1),
            "distinct": int(s.nunique()),
            "key": key,
        })
    return rows
