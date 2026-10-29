"""Semantic layer: works out which uploaded tables/columns play which planning role.

Everything here is inferred from table roles, keys and column names, then shown to the user
(it is part of the upload summary) so that a wrong guess can be corrected. No dataset-specific
names are hard-coded beyond common naming conventions.
"""
from __future__ import annotations

import pandas as pd

DEMAND_MEASURES = ["customer_demand", "demand", "demand_units", "unconstrained_demand", "units_demanded",
                   "units_sold", "sales_units", "qty_sold", "quantity", "qty", "units"]
PRODUCT_KEYS = ["sku_id", "sku", "product_id", "item_id"]
LOCATION_KEYS = ["dc_id", "store_id", "location_id", "warehouse_id", "site_id"]
CHANNEL_KEYS = ["channel_id", "channel"]
SUPPLIER_KEYS = ["supplier_id", "vendor_id"]


def _first(cols, options):
    for o in options:
        if o in cols:
            return o
    return None


def infer_semantics(session) -> dict:
    tables = session.tables
    sem: dict = {"demand": None, "product_table": None, "calendar_table": None, "inventory": None,
                 "purchase_orders": None, "product_supplier": None, "supplier_table": None, "price_list": None,
                 "promotion_table": None, "events": None, "event_scope": None, "notes": []}
    # Demand table: weekly fact with product key + date key + demand-like measure; prefer more rows
    best = None
    for name, t in tables.items():
        df = t.df
        if t.role != "fact":
            continue
        pk = t.pk or []
        date_col = next((c for c in pk if pd.api.types.is_datetime64_any_dtype(df[c])), None) or \
            next((c for c in ("week_start", "week", "date", "ds") if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])), None)
        prod = _first(df.columns, PRODUCT_KEYS)
        meas = _first(df.columns, DEMAND_MEASURES)
        if not (date_col and prod and meas):
            continue
        score = ("sale" in name) * 2 + ("demand" in " ".join(df.columns)) + len(df) / 1e7
        if best is None or score > best[0]:
            best = (score, name, date_col, prod, meas)
    if best:
        _, name, date_col, prod, meas = best
        df = tables[name].df
        cols = list(df.columns)
        sem["demand"] = {
            "table": name, "date_col": date_col, "product_col": prod,
            "location_col": _first(cols, LOCATION_KEYS), "channel_col": _first(cols, CHANNEL_KEYS),
            "measure": meas,
            "sold_col": "units_sold" if "units_sold" in cols else None,
            "lost_col": next((c for c in cols if "lost" in c), None),
            "stockout_col": next((c for c in cols if "stockout" in c), None),
            "price_col": _first(cols, ["selling_price", "net_price", "price", "unit_price"]),
            "list_price_col": _first(cols, ["list_price", "regular_price", "base_price"]),
            "discount_col": _first(cols, ["discount_pct", "discount", "markdown_pct"]),
            "promo_col": _first(cols, ["promo_id", "promotion_id", "on_promo", "promo_flag"]),
            "revenue_col": _first(cols, ["revenue", "net_sales", "sales_usd", "sales_value"]),
            # series grain = product/location/channel columns present + date (independent of the minimal unique key)
            "keys": [c for c in (prod, _first(cols, LOCATION_KEYS), _first(cols, CHANNEL_KEYS)) if c] + [date_col],
        }
        if meas != "customer_demand" and "units_sold" in meas:
            sem["notes"].append("No unconstrained-demand column found; forecasting units sold (censored by stockouts).")
    for name, t in tables.items():
        cols = set(t.df.columns)
        if t.role == "dimension" and t.pk and t.pk[0] in PRODUCT_KEYS and sem["product_table"] is None:
            sem["product_table"] = name
        if t.role == "dimension" and t.pk and pd.api.types.is_datetime64_any_dtype(t.df[t.pk[0]]) and sem["calendar_table"] is None:
            sem["calendar_table"] = name
        if t.pk and t.pk[0] in SUPPLIER_KEYS and not _first(cols, PRODUCT_KEYS) and t.role == "dimension":
            sem["supplier_table"] = name
        if t.role == "fact" and ("on_hand" in " ".join(cols) or "inventory" in name) and _first(cols, PRODUCT_KEYS):
            sem["inventory"] = name
        if "po" in name or "purchase" in name:
            if {"qty_ordered"} & cols or "unit_cost_usd" in cols:
                sem["purchase_orders"] = name
        if t.role != "fact" and _first(cols, PRODUCT_KEYS) and _first(cols, SUPPLIER_KEYS) and \
                not any(pd.api.types.is_datetime64_any_dtype(t.df[c]) for c in (t.pk or [])):
            sem["product_supplier"] = name
        if "price_list" in name or ("list_price" in cols and "effective" in " ".join(cols)):
            sem["price_list"] = name
        if t.role == "dimension" and t.pk and t.pk[0] in ("promo_id", "promotion_id"):
            sem["promotion_table"] = name
        if t.role == "dimension" and t.pk and t.pk[0] == "event_id":
            sem["events"] = name
        if t.role == "bridge" and "event_id" in cols and any(c.endswith("_type") for c in cols):
            sem["event_scope"] = name
    return sem


def get_semantics(session) -> dict:
    if "semantics" not in session.cache:
        session.cache["semantics"] = infer_semantics(session)
    return session.cache["semantics"]
