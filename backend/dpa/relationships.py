"""Foreign-key inference and validated joins.

Inference is deliberately conservative and explainable:
1. Single-column FK: column in table A has the same name as (or ends with) the single-column
   primary key of table B, types are compatible, and >= 95% of A's non-null values exist in B.
2. Composite FK: table A contains every column of B's composite key (B is a bridge/reference,
   i.e. its key has no date column) and >= 95% of A's key tuples exist in B.
3. Attribute link (many-to-many): a key column of A shares its name with a non-unique column of a
   dimension B (e.g. competitor price by `category` vs product.category). These cannot be joined
   row-to-row; they are flagged "aggregate first".
Each relationship carries evidence (coverage, orphans, cardinality) and a status the user confirms.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .util import to_jsonable


@dataclass
class Relationship:
    id: str
    child_table: str
    child_columns: list[str]
    parent_table: str
    parent_columns: list[str]
    kind: str                    # "many-to-one" | "attribute (many-to-many)"
    coverage_pct: float          # % of child non-null keys found in parent
    orphan_count: int
    orphan_examples: list[Any]
    child_null_pct: float
    status: str = "inferred"     # inferred | confirmed | rejected
    note: str = ""

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))


def _compatible(a: pd.Series, b: pd.Series) -> bool:
    num = pd.api.types.is_numeric_dtype
    dt = pd.api.types.is_datetime64_any_dtype
    if dt(a) or dt(b):
        return dt(a) and dt(b)
    if num(a) and num(b):
        return True
    if num(a) != num(b):
        return False
    return True


def _coverage(child: pd.DataFrame, ccols: list[str], parent: pd.DataFrame, pcols: list[str]) -> tuple[float, int, list, float]:
    sub = child[ccols]
    nonnull_mask = sub.notna().all(axis=1)
    null_pct = float((~nonnull_mask).mean() * 100)
    keys = sub[nonnull_mask].drop_duplicates()
    if keys.empty:
        return 0.0, 0, [], null_pct
    pk = parent[pcols].drop_duplicates()
    pk.columns = ccols
    m = keys.merge(pk, how="left", on=ccols, indicator=True)
    orph_keys = m[m["_merge"] == "left_only"][ccols]
    # count orphan *rows* in child
    if len(orph_keys):
        mm = sub[nonnull_mask].merge(orph_keys, on=ccols, how="inner")
        orphan_rows = len(mm)
    else:
        orphan_rows = 0
    cov = 100.0 * (1 - orphan_rows / max(int(nonnull_mask.sum()), 1))
    ex = [to_jsonable(tuple(r) if len(ccols) > 1 else r[0]) for r in orph_keys.head(5).itertuples(index=False)]
    return round(cov, 2), int(orphan_rows), ex, round(null_pct, 2)


def infer_relationships(tables: dict[str, pd.DataFrame], pks: dict[str, list[str] | None],
                        roles: dict[str, str], min_coverage: float = 95.0) -> list[Relationship]:
    rels: list[Relationship] = []
    n = 0
    singles = {t: k[0] for t, k in pks.items() if k and len(k) == 1}
    composites = {t: k for t, k in pks.items() if k and len(k) > 1}

    for child, cdf in tables.items():
        used_cols: set[str] = set()
        # 2. composite FKs first (more specific)
        for parent, pkey in composites.items():
            if parent == child:
                continue
            pdf = tables[parent]
            if any(pd.api.types.is_datetime64_any_dtype(pdf[c]) for c in pkey):
                continue  # don't link fact->fact on time keys
            if roles.get(parent) not in ("bridge", "reference", "dimension", "other"):
                continue
            if not all(c in cdf.columns for c in pkey):
                continue
            if set(pkey) == set(pks.get(child) or []):
                continue  # same grain: peer tables, not a parent
            if not all(_compatible(cdf[c], pdf[c]) for c in pkey):
                continue
            cov, orph, ex, nullp = _coverage(cdf, pkey, pdf, pkey)
            if cov >= min_coverage:
                n += 1
                rels.append(Relationship(f"R{n}", child, list(pkey), parent, list(pkey), "many-to-one", cov, orph, ex, nullp,
                                         note="composite key"))
        # 1. single-column FKs
        for col in cdf.columns:
            for parent, pcol in singles.items():
                if parent == child:
                    continue
                if not (col == pcol or col.endswith("_" + pcol)):
                    continue
                if roles.get(parent) == "fact" and not roles.get(child) == "fact":
                    pass
                pdf = tables[parent]
                if not _compatible(cdf[col], pdf[pcol]):
                    continue
                if cdf[col].notna().sum() == 0:
                    continue
                cov, orph, ex, nullp = _coverage(cdf, [col], pdf, [pcol])
                if cov >= min_coverage:
                    n += 1
                    note = "" if col == pcol else f"name match by suffix ('{col}' -> '{pcol}')"
                    if nullp > 0:
                        note = (note + "; " if note else "") + f"{nullp:.1f}% of child rows have no value (optional link)"
                    rels.append(Relationship(f"R{n}", child, [col], parent, [pcol], "many-to-one", cov, orph, ex, nullp, note=note))
                    used_cols.add(col)
        # 3. attribute links: a key column of child that matches a non-unique dimension attribute
        ckey = pks.get(child) or []
        for col in ckey:
            if col in used_cols or col.endswith("_id"):
                continue
            if pd.api.types.is_datetime64_any_dtype(cdf[col]):
                continue
            for parent, pdf in tables.items():
                if parent == child or roles.get(parent) != "dimension" or col not in pdf.columns:
                    continue
                if pdf[col].is_unique:
                    continue
                if not _compatible(cdf[col], pdf[col]):
                    continue
                cov, orph, ex, nullp = _coverage(cdf, [col], pdf.drop_duplicates(col), [col])
                if cov >= min_coverage:
                    n += 1
                    rels.append(Relationship(
                        f"R{n}", child, [col], parent, [col], "attribute (many-to-many)", cov, orph, ex, nullp,
                        note=f"'{col}' is not a key of {parent}: aggregate {parent} to '{col}' level before joining"))
    return rels


# ---------------------------------------------------------------------------- validated join

@dataclass
class JoinReport:
    left: str
    right: str
    on: list[str]
    how: str
    left_rows: int
    right_rows: int
    result_rows: int
    right_key_unique: bool
    left_unmatched_rows: int
    right_unmatched_keys: int
    fanout_max: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))


def validated_join(left: pd.DataFrame, right: pd.DataFrame, on: list[str] | dict[str, str], how: str = "left",
                   left_name: str = "left", right_name: str = "right",
                   right_role: str | None = None, left_key: list[str] | None = None,
                   right_key: list[str] | None = None) -> tuple[pd.DataFrame, JoinReport]:
    """Join two tables and report row counts, key uniqueness, orphans and fan-out.

    `on` is a list of shared column names or a mapping {left_col: right_col}.
    """
    if isinstance(on, dict):
        lcols, rcols = list(on.keys()), list(on.values())
    else:
        lcols = rcols = list(on)
    missing_l = [c for c in lcols if c not in left.columns]
    missing_r = [c for c in rcols if c not in right.columns]
    if missing_l or missing_r:
        raise ValueError(f"join columns not found: left {missing_l}, right {missing_r}")
    if how not in ("left", "inner", "right", "outer"):
        raise ValueError("how must be one of left, inner, right, outer")
    warnings: list[str] = []
    r = right.copy()
    if lcols != rcols:
        r = r.rename(columns=dict(zip(rcols, lcols)))
    # Avoid silent column collisions
    overlap = [c for c in r.columns if c in left.columns and c not in lcols]
    if overlap:
        r = r.rename(columns={c: f"{c}__{right_name}" for c in overlap})
        warnings.append(f"columns present in both tables were suffixed with '__{right_name}': {overlap}")

    rk_counts = r.groupby(lcols, dropna=False).size()
    right_unique = bool(rk_counts.max() <= 1) if len(rk_counts) else True
    fanout = int(rk_counts.max()) if len(rk_counts) else 0
    lkeys = left[lcols].drop_duplicates()
    m = lkeys.merge(r[lcols].drop_duplicates(), how="outer", on=lcols, indicator=True)
    left_only_keys = m[m["_merge"] == "left_only"][lcols]
    right_only_keys = int((m["_merge"] == "right_only").sum())
    left_unmatched_rows = len(left.merge(left_only_keys, on=lcols, how="inner")) if len(left_only_keys) else 0

    out = left.merge(r, how=how, on=lcols)
    rep = JoinReport(left_name, right_name, lcols if lcols == rcols else [f"{a}={b}" for a, b in zip(lcols, rcols)], how,
                     len(left), len(right), len(out), right_unique, left_unmatched_rows, right_only_keys, fanout, warnings)

    if not right_unique:
        warnings.append(
            f"FAN-OUT: '{right_name}' is not unique on {lcols} (up to {fanout} rows per key). The join multiplied rows "
            f"from {len(left):,} to {len(out):,}. Sums of '{left_name}' measures would be overstated; aggregate "
            f"'{right_name}' to the join key first.")
    if how == "left" and len(out) != len(left):
        warnings.append(f"Row count changed on a left join ({len(left):,} -> {len(out):,}).")
    if how == "inner" and left_unmatched_rows:
        warnings.append(f"{left_unmatched_rows:,} '{left_name}' rows had no match and were dropped by the inner join.")
    if how == "left" and left_unmatched_rows:
        warnings.append(f"{left_unmatched_rows:,} '{left_name}' rows have no match in '{right_name}' (their new columns are empty).")
    # Measure duplication: right is unique on the key but the left has several rows per key and
    # right carries additive measures (fact grain coarser than left), e.g. inventory (sku, dc, week)
    # joined onto sales (sku, dc, channel, week): stock levels repeat per channel row.
    if right_unique:
        left_per_key = left.groupby(lcols, dropna=False).size()
        if len(left_per_key) and left_per_key.max() > 1 and (right_role == "fact"):
            measures = [c for c in r.columns if c not in lcols and pd.api.types.is_numeric_dtype(r[c])]
            if measures:
                warnings.append(
                    f"GRAIN MISMATCH: '{right_name}' is at a coarser grain than '{left_name}' ({int(left_per_key.max())} "
                    f"'{left_name}' rows per key). Its measures {measures[:5]}{'...' if len(measures) > 5 else ''} now repeat on "
                    f"each of those rows - do NOT sum them. Aggregate '{left_name}' to {lcols} before joining.")
    return out, rep
