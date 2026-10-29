"""Resolve which external events apply to which SKUs, using the polymorphic event-scope table.

scope_type values are mapped to product attributes:
  sku -> product key, subcategory/category -> product table columns, brand -> brand_id,
  supplier -> SKUs sourced from that supplier, supplier_country -> SKUs sourced from that country,
  dc -> location (affects every SKU stocked there), all -> everything.
Supplier-side scopes are labelled "supply-side"; the rest are "demand-side / location".
"""
from __future__ import annotations

import pandas as pd

from .semantic import get_semantics

SUPPLY_SCOPES = {"supplier", "supplier_country", "country", "port"}


def _sku_attributes(session) -> pd.DataFrame | None:
    sem = get_semantics(session)
    pt = sem.get("product_table")
    if not pt:
        return None
    prod = session.df(pt).copy()
    pk = session.tables[pt].pk[0]
    prod = prod.rename(columns={pk: "_sku"})
    ps, st = sem.get("product_supplier"), sem.get("supplier_table")
    if ps and st:
        link = session.df(ps)
        sup = session.df(st)
        skey = session.tables[st].pk[0]
        pkey = next(c for c in link.columns if c in (pk, "sku_id", "product_id"))
        m = link[[pkey, skey]].merge(sup[[skey] + [c for c in ("country",) if c in sup.columns]], on=skey, how="left")
        agg = m.groupby(pkey).agg(_suppliers=(skey, lambda x: set(x)),
                                  _countries=("country", lambda x: set(x)) if "country" in m.columns else (skey, lambda x: set()))
        prod = prod.merge(agg, left_on="_sku", right_index=True, how="left")
    return prod


def scope_matches(scope_type: str, scope_value: str, sku_row: pd.Series | None) -> bool:
    st = str(scope_type).lower()
    if st == "all":
        return True
    if sku_row is None:
        return st in ("dc",)  # location-wide events affect totals
    if st == "sku":
        return str(sku_row["_sku"]) == str(scope_value)
    if st in ("subcategory", "category", "brand_id", "brand"):
        col = "brand_id" if st.startswith("brand") else st
        return col in sku_row and str(sku_row[col]) == str(scope_value)
    if st == "supplier":
        return isinstance(sku_row.get("_suppliers"), set) and scope_value in sku_row["_suppliers"]
    if st in ("supplier_country", "country"):
        return isinstance(sku_row.get("_countries"), set) and scope_value in sku_row["_countries"]
    if st == "dc":
        return True  # location events touch every SKU stocked in that DC
    return False


def events_overlapping(session, start: pd.Timestamp, end: pd.Timestamp, sku: str | None = None) -> list[dict]:
    sem = get_semantics(session)
    ev = sem.get("events")
    if not ev:
        return []
    e = session.df(ev)
    if "start_date" not in e.columns:
        return []
    eend = (e["end_date"] if "end_date" in e.columns else e["start_date"]).fillna(pd.Timestamp("2100-01-01"))
    hit = e[(e["start_date"] <= end) & (eend >= start)]
    if hit.empty:
        return []
    scope = session.df(sem["event_scope"]) if sem.get("event_scope") else None
    attrs = _sku_attributes(session)
    sku_row = None
    if sku is not None and attrs is not None:
        r = attrs[attrs["_sku"].astype(str) == str(sku)]
        sku_row = r.iloc[0] if len(r) else None
    out = []
    for _, r in hit.iterrows():
        eid = r.get("event_id")
        if scope is not None and "event_id" in scope.columns:
            sc = scope[scope["event_id"] == eid]
            stype_col = next((c for c in sc.columns if c.endswith("_type")), None)
            sval_col = next((c for c in sc.columns if c.endswith("_value")), None)
            if stype_col and sval_col and len(sc):
                matched = [(t, v) for t, v in zip(sc[stype_col], sc[sval_col]) if scope_matches(t, v, sku_row)]
                if not matched:
                    continue
                kinds = {str(t).lower() for t, _ in matched}
                side = "supply-side" if kinds <= SUPPLY_SCOPES else ("location" if kinds == {"dc"} else "demand-side")
                out.append({"event_id": eid, "name": r.get("event_name"), "type": r.get("event_type"), "side": side,
                            "scope": [f"{t}={v}" for t, v in matched]})
                continue
        out.append({"event_id": eid, "name": r.get("event_name"), "type": r.get("event_type"), "side": "unknown scope", "scope": []})
    return out


def skus_in_scope(session, scope_type: str, scope_value: str) -> list[str]:
    attrs = _sku_attributes(session)
    if attrs is None:
        return []
    return [str(r["_sku"]) for _, r in attrs.iterrows() if scope_matches(scope_type, scope_value, r)]
