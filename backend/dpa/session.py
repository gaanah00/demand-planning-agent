"""In-memory session store.

A Session holds everything one user (browser tab) has done: loaded tables, inferred and confirmed
relationships, the data dictionary (with user edits), quality results, scenarios, assumptions,
KPI usage, chat messages and the audit trail. Expensive results are cached per session.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import IngestResult, LoadedTable, PendingWorkbook, ingest_upload
from .profiling import data_dictionary, infer_primary_key, profile_table, table_role
from .relationships import Relationship, infer_relationships
from .util import to_jsonable

SESSION_ROOT = Path(os.environ.get("DPA_SESSION_DIR", Path(__file__).resolve().parents[2] / ".sessions"))


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TableInfo:
    name: str
    df: pd.DataFrame
    source_file: str
    sheet: str | None
    column_map: dict[str, str]
    load_notes: list[str]
    pk: list[str] | None
    pk_method: str
    role: str
    pk_status: str = "inferred"          # inferred | confirmed | edited


@dataclass
class Session:
    id: str
    created: str = field(default_factory=now_iso)
    name: str = ""
    updated: str = ""
    tables: dict[str, TableInfo] = field(default_factory=dict)
    uploads: list[dict] = field(default_factory=list)            # upload summaries
    pending_workbooks: list[PendingWorkbook] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    relationships_confirmed: bool = False
    dictionary_edits: dict[str, dict] = field(default_factory=dict)   # "table.col" -> {description, unit}
    joins_log: list[dict] = field(default_factory=list)
    quality: dict | None = None
    scenarios: dict[str, dict] = field(default_factory=dict)
    assumptions: list[dict] = field(default_factory=list)
    kpi_usage: dict[str, dict] = field(default_factory=dict)          # kpi_id -> {entry, used_in: [msg ids]}
    messages: list[dict] = field(default_factory=list)
    llm_history: list[dict] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    jobs: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ paths
    @property
    def folder(self) -> Path:
        p = SESSION_ROOT / self.id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def df(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Unknown table '{name}'. Loaded tables: {sorted(self.tables)}")
        return self.tables[name].df

    # ------------------------------------------------------------------ upload
    def add_upload(self, path: str, display_name: str, sheet_mode: dict[str, str] | None = None) -> dict:
        t0 = time.time()
        extract = self.folder / f"upload_{len(self.uploads) + 1}"
        res: IngestResult = ingest_upload(path, display_name, str(extract), sheet_mode=sheet_mode,
                                          existing_tables=set(self.tables))
        replaced = []
        for lt in res.tables:
            if lt.name in self.tables:
                replaced.append(lt.name)
            self._register(lt)
        self.pending_workbooks.extend(res.pending_workbooks)
        if res.tables and (not self.name or self.name.startswith("Workspace")):
            self.name = os.path.splitext(os.path.basename(display_name))[0].replace("_", " ")[:60] or self.name
        if res.tables:
            self.refresh_relationships()
            self.cache.clear()
            self.quality = None
        summary = self.upload_summary(res, display_name, time.time() - t0, replaced)
        self.uploads.append(summary)
        return summary

    def resolve_workbook(self, display_name: str, mode: str) -> dict:
        wb = next((w for w in self.pending_workbooks if w.display_name == display_name), None)
        if wb is None:
            raise KeyError(f"No pending workbook named '{display_name}'")
        if mode not in ("first", "all"):
            raise ValueError("mode must be 'first' or 'all'")
        self.pending_workbooks = [w for w in self.pending_workbooks if w is not wb]
        # Re-ingest just that workbook file with the chosen mode
        return self.add_upload(wb.file_path, os.path.basename(wb.display_name), sheet_mode={os.path.basename(wb.display_name): mode})

    def _register(self, lt: LoadedTable) -> None:
        pk, method = infer_primary_key(lt.df)
        role = table_role(lt.name, lt.df, pk)
        self.tables[lt.name] = TableInfo(lt.name, lt.df, lt.source_file, lt.sheet, lt.column_map, lt.notes, pk, method, role)

    def refresh_relationships(self) -> None:
        old = {(r.child_table, tuple(r.child_columns), r.parent_table): r.status for r in self.relationships}
        rels = infer_relationships({n: t.df for n, t in self.tables.items()},
                                   {n: t.pk for n, t in self.tables.items()},
                                   {n: t.role for n, t in self.tables.items()})
        for r in rels:
            st = old.get((r.child_table, tuple(r.child_columns), r.parent_table))
            if st:
                r.status = st
        self.relationships = rels
        self.relationships_confirmed = bool(rels) and all(r.status != "inferred" for r in rels)

    # ------------------------------------------------------------------ summaries
    def table_profile(self, name: str) -> dict:
        t = self.tables[name]
        p = profile_table(name, t.df, t.pk)
        p.update({"source_file": t.source_file, "sheet": t.sheet, "load_notes": t.load_notes,
                  "pk_method": t.pk_method, "pk_status": t.pk_status})
        return p

    def upload_summary(self, res: IngestResult, display_name: str, seconds: float, replaced: list[str]) -> dict:
        names = [lt.name for lt in res.tables]
        return to_jsonable({
            "upload": display_name,
            "loaded_at": now_iso(),
            "seconds": round(seconds, 1),
            "tables": [self.table_profile(n) for n in names],
            "skipped": res.skipped,
            "duplicates": res.duplicates,
            "pending_workbooks": [{"file": w.display_name, "sheets": w.sheets, "table": w.table_name} for w in res.pending_workbooks],
            "warnings": res.warnings,
            "replaced_tables": replaced,
            "relationships": [r.to_dict() for r in self.relationships],
        })

    def fk_columns(self, table: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in self.relationships:
            if r.child_table == table and r.status != "rejected":
                for c, p in zip(r.child_columns, r.parent_columns):
                    label = f"{r.parent_table}.{p}"
                    out[c] = label if c not in out else out[c] + f", {label}"
        return out

    def dictionary(self) -> list[dict]:
        rows: list[dict] = []
        for n in sorted(self.tables):
            t = self.tables[n]
            for row in data_dictionary(n, t.df, t.column_map, t.pk, self.fk_columns(n)):
                edit = self.dictionary_edits.get(f"{n}.{row['column']}")
                if edit:
                    row.update({k: v for k, v in edit.items() if v is not None})
                    row["description_status"] = "confirmed"
                rows.append(row)
        return rows

    def relationship_by_id(self, rid: str) -> Relationship:
        for r in self.relationships:
            if r.id == rid:
                return r
        raise KeyError(f"No relationship '{rid}'")

    def log_join(self, report: dict) -> None:
        report = dict(report)
        report["at"] = now_iso()
        self.joins_log.append(report)
        self.joins_log = self.joins_log[-200:]

    def add_assumption(self, text: str, source: str, value: Any = None, scenario: str | None = None) -> dict:
        a = {"id": f"A{len(self.assumptions) + 1}", "text": text, "source": source, "value": to_jsonable(value),
             "scenario": scenario, "added_at": now_iso()}
        self.assumptions.append(a)
        return a


class SessionStore:
    """Sessions ("workspaces") live in memory while in use and are saved to disk after every change."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, name: str | None = None) -> Session:
        with self._lock:
            s = Session(id=uuid.uuid4().hex[:12])
            s.name = name or f"Workspace {datetime.now():%b %d, %H:%M}"
            self._sessions[s.id] = s
            return s

    def get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if s is not None:
            return s
        from .persistence import STATE, load_session
        if sid and (SESSION_ROOT / sid / STATE).exists():
            with self._lock:
                if sid not in self._sessions:
                    self._sessions[sid] = load_session(sid, SESSION_ROOT)
                return self._sessions[sid]
        raise KeyError("Workspace not found. It may have been deleted; start a new one.")

    def get_or_create(self, sid: str | None) -> Session:
        try:
            return self.get(sid) if sid else self.create()
        except KeyError:
            return self.create()

    def save(self, s: Session) -> None:
        from .persistence import save_session
        try:
            with s.lock:
                save_session(s)
        except Exception as e:  # saving must never break the app
            s.settings["save_error"] = f"{type(e).__name__}: {e}"

    def list(self) -> list[dict]:
        from .persistence import list_workspaces
        return list_workspaces(SESSION_ROOT)

    def drop(self, sid: str) -> None:
        from .persistence import delete_workspace
        with self._lock:
            self._sessions.pop(sid, None)
        delete_workspace(SESSION_ROOT, sid)


STORE = SessionStore()
