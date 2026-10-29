"""Saved workspaces: each session is written to its own folder so it survives a restart.

  .sessions/<workspace id>/state.json      everything except table data and recomputable caches
  .sessions/<workspace id>/tables/*.pkl    one pandas pickle per loaded table (written by this app only)

Caches (forecasts, charts, cost basis) are not saved; they are recomputed on demand. Charts already shown in
answers are stored inside the messages, so past answers display exactly as before.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from .util import to_jsonable

STATE = "state.json"
SAFE = re.compile(r"^[a-z0-9_]{1,80}$")


def _scenario_to_json(sc: dict) -> dict:
    return to_jsonable({k: v for k, v in sc.items() if k != "result"})


def _scenario_from_json(sc: dict) -> dict:
    sp = sc.get("spec", {})
    for k in ("start_date", "end_date"):
        if sp.get(k):
            sp[k] = pd.Timestamp(sp[k])
    sc["result"] = None
    return sc


def save_session(session) -> None:
    from .session import now_iso
    folder = session.folder
    tdir = folder / "tables"
    tdir.mkdir(exist_ok=True)
    tables_meta = {}
    for name, t in session.tables.items():
        fname = name if SAFE.match(name) else re.sub(r"[^a-z0-9_]", "_", name.lower())[:80]
        path = tdir / f"{fname}.pkl"
        sig = (len(t.df), tuple(t.df.columns))
        if session.settings.get("_saved_sig", {}).get(name) != str(sig) or not path.exists():
            t.df.to_pickle(path)
            session.settings.setdefault("_saved_sig", {})[name] = str(sig)
        tables_meta[name] = {"file": f"{fname}.pkl", "source_file": t.source_file, "sheet": t.sheet, "column_map": t.column_map,
                             "load_notes": t.load_notes, "pk": t.pk, "pk_method": t.pk_method, "role": t.role, "pk_status": t.pk_status}
    for f in tdir.glob("*.pkl"):  # tables that were replaced/removed
        if f.name not in {m["file"] for m in tables_meta.values()}:
            f.unlink(missing_ok=True)
    session.updated = now_iso()
    state = {
        "version": 1, "id": session.id, "name": session.name, "created": session.created, "updated": session.updated,
        "tables": tables_meta, "uploads": session.uploads,
        "pending_workbooks": [{"file_path": w.file_path, "display_name": w.display_name, "table_name": w.table_name, "sheets": w.sheets}
                              for w in session.pending_workbooks],
        "relationships": [r.to_dict() for r in session.relationships], "relationships_confirmed": session.relationships_confirmed,
        "dictionary_edits": session.dictionary_edits, "joins_log": session.joins_log, "quality": session.quality,
        "scenarios": {n: _scenario_to_json(sc) for n, sc in session.scenarios.items()},
        "assumptions": session.assumptions, "kpi_usage": session.kpi_usage, "messages": session.messages,
        "llm_history": session.llm_history,
        "settings": {k: v for k, v in session.settings.items() if k not in ("warmup",)},
    }
    tmp = folder / (STATE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(state), fh)
    os.replace(tmp, folder / STATE)


def load_session(sid: str, root: Path):
    from .ingest import PendingWorkbook
    from .relationships import Relationship
    from .session import Session, TableInfo
    folder = root / sid
    with open(folder / STATE, encoding="utf-8") as fh:
        st = json.load(fh)
    s = Session(id=sid, created=st.get("created", ""))
    s.name = st.get("name") or "Workspace"
    s.updated = st.get("updated") or s.created
    for name, m in st.get("tables", {}).items():
        df = pd.read_pickle(folder / "tables" / m["file"])
        s.tables[name] = TableInfo(name, df, m["source_file"], m.get("sheet"), m.get("column_map", {}), m.get("load_notes", []),
                                   m.get("pk"), m.get("pk_method", ""), m.get("role", "other"), m.get("pk_status", "inferred"))
    s.uploads = st.get("uploads", [])
    s.pending_workbooks = [PendingWorkbook(**w) for w in st.get("pending_workbooks", [])]
    s.relationships = [Relationship(**r) for r in st.get("relationships", [])]
    s.relationships_confirmed = st.get("relationships_confirmed", False)
    s.dictionary_edits = st.get("dictionary_edits", {})
    s.joins_log = st.get("joins_log", [])
    s.quality = st.get("quality")
    s.scenarios = {n: _scenario_from_json(sc) for n, sc in st.get("scenarios", {}).items()}
    s.assumptions = st.get("assumptions", [])
    s.kpi_usage = st.get("kpi_usage", {})
    s.messages = st.get("messages", [])
    s.llm_history = st.get("llm_history", [])
    s.settings = st.get("settings", {})
    s.settings["_saved_sig"] = {n: str((len(t.df), tuple(t.df.columns))) for n, t in s.tables.items()}
    return s


def list_workspaces(root: Path) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for d in root.iterdir():
        f = d / STATE
        if not f.exists():
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": d.name, "name": st.get("name") or "Workspace", "created": st.get("created"), "updated": st.get("updated"),
                    "tables": len(st.get("tables", {})), "questions": sum(1 for m in st.get("messages", []) if m.get("role") == "user"),
                    "scenarios": len(st.get("scenarios", {}))})
    return sorted(out, key=lambda r: r.get("updated") or "", reverse=True)


def delete_workspace(root: Path, sid: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{12}", sid):
        raise ValueError("bad workspace id")
    shutil.rmtree(root / sid, ignore_errors=True)
