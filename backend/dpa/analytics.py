"""Descriptive analytics tools: auto-joined aggregation, descriptive statistics, validated joins.

Every result is stored in the session under a result id (e.g. "res_3") so charts and follow-up tools
can use the full data without the LLM ever re-typing numbers.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from .relationships import validated_join
from .util import df_records, to_jsonable

AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std"}
OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "between", "contains", "is_null", "not_null"}
TIME_BUCKETS = {"week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS"}
MAX_ROWS_TO_LLM = 60


class ToolError(Exception):
    """User-facing error from an analytics tool (bad column, bad parameter...)."""


# ---------------------------------------------------------------------------- result store

def store_result(session, df: pd.DataFrame, meta: dict) -> str:
    res = session.cache.setdefault("results", {})
    rid = f"res_{len(res) + 1}"
    res[rid] = {"df": df, "meta": meta}
    return rid


def get_result(session, rid: str) -> tuple[pd.DataFrame, dict]:
    res = session.cache.get("results", {})
    if rid not in res:
        raise ToolError(f"Unknown result_id '{rid}'. Available: {list(res)[-10:]}")
    return res[rid]["df"], res[rid]["meta"]


# ---------------------------------------------------------------------------- auto-join

def _join_path(session, base: str, target_col: str, target_table: str | None = None) -> list | None:
    """Breadth-first search over many-to-one relationships for the nearest table that has target_col.

    Raises ToolError when two different tables at the same distance have the column (ambiguous, e.g.
    'region' on both the DC and the supplier table) - the caller must then qualify it as 'table.column'.
    """
    rels = [r for r in session.relationships if r.status != "rejected" and r.kind == "many-to-one"]
    frontier = [(base, [])]
    seen = {base}
    for _depth in range(4):
        hits = [(t, p) for t, p in frontier if t != base and target_col in session.df(t).columns
                and (target_table is None or t == target_table)]
        if len(hits) == 1:
            return hits[0][1]
        if len(hits) > 1:
            opts = ", ".join(f"{t}.{target_col}" for t, _ in hits)
            raise ToolError(f"'{target_col}' is ambiguous from '{base}' (found in several related tables). Use one of: {opts}")
        nxt = []
        for t, path in frontier:
            for r in rels:
                if r.child_table == t and r.parent_table not in seen:
                    seen.add(r.parent_table)
                    nxt.append((r.parent_table, path + [r]))
        if not nxt:
            break
        frontier = nxt
    return None


def enrich(session, base: str, df: pd.DataFrame, needed: list[str], joins: list[dict]) -> pd.DataFrame:
    """Add columns from parent tables (many-to-one) to df via validated joins."""
    for qcol in needed:
        ttable, col = (qcol.split(".", 1) if "." in qcol else (None, qcol))
        if ttable is None and col in df.columns:
            continue
        if ttable is not None and ttable not in session.tables:
            raise ToolError(f"Unknown table '{ttable}' in '{qcol}'.")
        path = _join_path(session, base, col, ttable)
        if path is None:
            raise ToolError(f"Column '{qcol}' is not in '{base}' and no relationship path leads to a table that has it.")
        cur = df
        for r in path:
            parent = session.df(r.parent_table)
            cols_needed = list(dict.fromkeys(r.parent_columns + ([col] if col in parent.columns else []) +
                                             [c for c in parent.columns if c in _next_keys(path, r)]))
            right = parent[cols_needed].drop_duplicates()
            on = dict(zip(r.child_columns, r.parent_columns))
            if any(c not in cur.columns for c in on):
                raise ToolError(f"Join key missing while joining {r.parent_table}")
            new_cols = [c for c in right.columns if c not in r.parent_columns and c in cur.columns]
            right = right.drop(columns=new_cols)
            before = len(cur)
            cur, rep = validated_join(cur, right, on, "left", r.child_table, r.parent_table,
                                      right_role=session.tables[r.parent_table].role)
            rd = rep.to_dict()
            rd["purpose"] = f"add '{col}'"
            joins.append(rd)
            session.log_join(rd)
            if len(cur) != before:
                raise ToolError(f"Join to {r.parent_table} changed row count {before} -> {len(cur)}; refusing (fan-out).")
        if ttable is not None and qcol not in cur.columns:
            cur = cur.rename(columns={col: qcol}) if col not in df.columns else cur.rename(columns={f"{col}__{ttable}": qcol})
        df = cur
    return df


def _next_keys(path, r):
    i = path.index(r)
    return path[i + 1].child_columns if i + 1 < len(path) else []


# ---------------------------------------------------------------------------- filters

def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    for f in filters or []:
        col, op, val = f.get("column"), f.get("op", "=="), f.get("value")
        if col not in df.columns:
            raise ToolError(f"Filter column '{col}' not found.")
        if op not in OPS:
            raise ToolError(f"Unsupported filter op '{op}'. Use one of {sorted(OPS)}")
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) and val is not None and op not in ("in", "not_in", "is_null", "not_null"):
            val = [pd.Timestamp(v) for v in val] if isinstance(val, list) else pd.Timestamp(val)
        if op == "==":
            m = s == val
        elif op == "!=":
            m = s != val
        elif op == ">":
            m = s > val
        elif op == ">=":
            m = s >= val
        elif op == "<":
            m = s < val
        elif op == "<=":
            m = s <= val
        elif op == "in":
            m = s.isin(val if isinstance(val, list) else [val])
        elif op == "not_in":
            m = ~s.isin(val if isinstance(val, list) else [val])
        elif op == "between":
            m = (s >= val[0]) & (s <= val[1])
        elif op == "contains":
            m = s.astype(str).str.contains(str(val), case=False, na=False)
        elif op == "is_null":
            m = s.isna()
        else:
            m = s.notna()
        df = df[m]
    return df


# ---------------------------------------------------------------------------- aggregate

def aggregate(session, table: str, measures: list[dict], group_by: list[str] | None = None,
              filters: list[dict] | None = None, time_bucket: str | None = None, date_col: str | None = None,
              sort_by: str | None = None, descending: bool = True, limit: int | None = None) -> dict:
    """Group / filter / time-bucket a table, auto-joining dimension attributes (many-to-one only).

    measures: [{"column": "revenue", "agg": "sum", "name": "revenue"}]
              or ratio of sums: {"name": "avg_price", "numerator": "revenue", "denominator": "units_sold"}
    """
    base = session.df(table)
    group_by = list(group_by or [])
    if not measures:
        raise ToolError("At least one measure is required.")
    needed = list(group_by)
    needed += [f["column"] for f in (filters or []) if f.get("column")]
    for m in measures:
        for k in ("column", "numerator", "denominator"):
            if m.get(k):
                needed.append(m[k])
    tinfo = session.tables[table]
    if time_bucket:
        if time_bucket not in TIME_BUCKETS:
            raise ToolError(f"time_bucket must be one of {list(TIME_BUCKETS)}")
        date_col = date_col or next((c for c in (tinfo.pk or []) if pd.api.types.is_datetime64_any_dtype(base[c])), None) \
            or next((c for c in base.columns if pd.api.types.is_datetime64_any_dtype(base[c])), None)
        if not date_col:
            raise ToolError(f"Table '{table}' has no date column for time bucketing.")
        needed.append(date_col)
    joins: list[dict] = []
    df = enrich(session, table, base, [c for c in dict.fromkeys(needed) if c not in base.columns], joins)
    rows_in = len(df)
    df = apply_filters(df, filters)
    rows_after_filter = len(df)
    keys = list(group_by)
    if time_bucket:
        bcol = f"{time_bucket}"
        if time_bucket == "week":
            df = df.assign(**{bcol: df[date_col]})
        else:
            df = df.assign(**{bcol: df[date_col].dt.to_period({"month": "M", "quarter": "Q", "year": "Y"}[time_bucket]).dt.start_time})
        keys = [bcol] + keys
    agg_spec: dict[str, tuple[str, str]] = {}
    ratios = []
    for m in measures:
        if m.get("numerator"):
            name = m.get("name") or f"{m['numerator']}_per_{m['denominator']}"
            agg_spec[f"__n_{name}"] = (m["numerator"], "sum")
            agg_spec[f"__d_{name}"] = (m["denominator"], "sum")
            ratios.append(name)
        else:
            col, how = m["column"], m.get("agg", "sum")
            if how not in AGGS:
                raise ToolError(f"Unsupported agg '{how}'. Use one of {sorted(AGGS)}")
            if how in ("sum", "mean", "median", "std") and not pd.api.types.is_numeric_dtype(df[col]):
                raise ToolError(f"Cannot {how} non-numeric column '{col}'.")
            agg_spec[m.get("name") or (f"{col}" if how == "sum" else f"{col}_{how}")] = (col, how)
    if keys:
        out = df.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
    else:
        out = pd.DataFrame({k: [getattr(df[c], h)()] for k, (c, h) in agg_spec.items()})
    for name in ratios:
        out[name] = out[f"__n_{name}"] / out[f"__d_{name}"].replace(0, np.nan)
        out = out.drop(columns=[f"__n_{name}", f"__d_{name}"])
    if sort_by:
        if sort_by not in out.columns:
            raise ToolError(f"sort_by '{sort_by}' is not a result column: {list(out.columns)}")
        out = out.sort_values(sort_by, ascending=not descending)
    elif time_bucket:
        out = out.sort_values(keys)
    total_groups = len(out)
    if limit:
        out = out.head(int(limit))
    meta = {"tool": "aggregate", "table": table, "group_by": group_by, "time_bucket": time_bucket,
            "date_col": date_col, "measures": measures, "filters": filters or [], "rows_in": rows_in,
            "rows_after_filter": rows_after_filter, "groups": total_groups, "joins": joins,
            "columns": list(out.columns)}
    rid = store_result(session, out, meta)
    totals = {}
    for c in out.columns:
        if c not in keys and pd.api.types.is_numeric_dtype(out[c]) and c not in ratios and not c.endswith(("_mean", "_median", "_min", "_max", "_std")):
            totals[c] = float(out[c].sum())
    warn = [w for j in joins for w in j.get("warnings", [])]
    return to_jsonable({
        "result_id": rid, "rows": df_records(out, MAX_ROWS_TO_LLM), "row_count": len(out),
        "truncated": len(out) > MAX_ROWS_TO_LLM, "column_totals": totals,
        "source_rows": rows_in, "rows_after_filter": rows_after_filter,
        "joins": [{"left": j["left"], "right": j["right"], "on": j["on"], "rows_before": j["left_rows"],
                   "rows_after": j["result_rows"], "warnings": j["warnings"]} for j in joins],
        "warnings": warn,
        "date_range": ({"min": df[date_col].min(), "max": df[date_col].max()} if date_col and len(df) else None),
    })


def descriptive_stats(session, result_id: str | None = None, table: str | None = None, columns: list[str] | None = None,
                      group_by: str | None = None, filters: list[dict] | None = None) -> dict:
    """count, mean, std, quantiles, CV, % zero for numeric columns of a table or stored result."""
    if result_id:
        df, _ = get_result(session, result_id)
    elif table:
        df = session.df(table)
        needed = ([group_by] if group_by else []) + [f["column"] for f in (filters or [])] + list(columns or [])
        joins: list[dict] = []
        df = enrich(session, table, df, [c for c in needed if c not in df.columns], joins)
        df = apply_filters(df, filters)
    else:
        raise ToolError("Provide result_id or table.")
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != group_by]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ToolError(f"Columns not found: {missing}")

    def stats(x: pd.Series) -> dict:
        x = x.dropna().astype(float)
        if x.empty:
            return {"count": 0}
        mean = x.mean()
        return {"count": int(len(x)), "sum": float(x.sum()), "mean": float(mean), "std": float(x.std()),
                "min": float(x.min()), "p25": float(x.quantile(.25)), "median": float(x.median()),
                "p75": float(x.quantile(.75)), "max": float(x.max()),
                "cv": float(x.std() / mean) if mean else None, "pct_zero": float((x == 0).mean() * 100)}

    if group_by:
        out = {str(k): {c: stats(g[c]) for c in cols} for k, g in df.groupby(group_by)}
    else:
        out = {c: stats(df[c]) for c in cols}
    return to_jsonable({"stats": out, "rows": len(df), "definitions": {
        "cv": "coefficient of variation = std / mean (volatility relative to level)",
        "pct_zero": "share of rows equal to zero"}})


def join_tables(session, left: str, right: str, on: list[str] | dict | None = None, how: str = "left") -> dict:
    """Validate a join between two loaded tables and report row counts, key uniqueness, orphans and fan-out."""
    l, r = session.df(left), session.df(right)
    if on is None:
        rel = next((x for x in session.relationships if x.child_table == left and x.parent_table == right), None)
        if rel is None:
            shared = [c for c in l.columns if c in r.columns]
            if not shared:
                raise ToolError(f"No shared columns or known relationship between {left} and {right}.")
            on = shared
        else:
            on = dict(zip(rel.child_columns, rel.parent_columns))
    out, rep = validated_join(l, r, on, how, left, right, right_role=session.tables[right].role)
    rd = rep.to_dict()
    session.log_join(rd)
    rid = store_result(session, out, {"tool": "join_tables", "left": left, "right": right, "on": on, "how": how})
    rd["result_id"] = rid
    rd["preview"] = df_records(out, 5)
    return to_jsonable(rd)
