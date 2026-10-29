"""HTTP API + static frontend (Starlette ASGI app, served by uvicorn).

Long-running work (uploads, chat turns, background warm-up) runs in a thread pool as "jobs"; the UI polls
/api/jobs/{id} for progress, so every step shows a loading state and nothing blocks the server.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import agent as agent_mod
from .analytics import ToolError
from .charts import get_chart
from .ingest import UploadError
from .kpis import glossary
from .quality import data_quality_report
from .scenarios import confirm_scenario
from .session import STORE, now_iso
from .util import to_jsonable

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "frontend" / "dist"
DEMO_ZIP = ROOT / "data" / "demo" / "demo_data.zip"
MAX_UPLOAD_MB = int(os.environ.get("DPA_MAX_UPLOAD_MB", "300"))
JOB_TIMEOUT = int(os.environ.get("DPA_JOB_TIMEOUT", "300"))

EXEC = ThreadPoolExecutor(max_workers=int(os.environ.get("DPA_WORKERS", "4")))
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ============================================================================ helpers

def ok(data, status=200):
    return JSONResponse(to_jsonable(data), status_code=status)


def fail(msg: str, status=400, **extra):
    return JSONResponse({"error": msg, **to_jsonable(extra)}, status_code=status)


def _session(request: Request):
    sid = request.path_params.get("sid")
    return STORE.get(sid)


def new_job(kind: str, sid: str, fn, *args) -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "session": sid, "status": "running", "progress": [], "started": time.time(),
           "result": None, "error": None}
    with JOBS_LOCK:
        JOBS[jid] = job
        if len(JOBS) > 500:
            for k in sorted(JOBS, key=lambda k: JOBS[k]["started"])[:100]:
                JOBS.pop(k, None)

    def progress(msg: str):
        job["progress"].append({"t": round(time.time() - job["started"], 1), "msg": msg})
    progress.job = job

    def run():
        try:
            job["result"] = to_jsonable(fn(progress, *args))
            job["status"] = "done"
        except (ToolError, UploadError) as e:
            job["status"], job["error"] = "error", str(e)
        except KeyError as e:
            job["status"], job["error"] = "error", str(e).strip("'\"")
        except Exception as e:  # never let a job crash the server
            job["status"], job["error"] = "error", f"Something went wrong ({type(e).__name__}): {str(e)[:300]}"
            job["trace"] = traceback.format_exc()[-2000:]
        job["seconds"] = round(time.time() - job["started"], 1)

    EXEC.submit(run)
    return jid


def _api_status() -> dict:
    client, model, err = agent_mod.make_client()
    return {"llm_ready": client is not None, "model": model, "message": err}


# ============================================================================ state for the UI

def session_state(s) -> dict:
    q = s.quality
    qsum = None
    if q:
        qsum = {"table_status": {n: {"status": v["status"], "issues": len([i for i in v["issues"] if i["severity"] != "info"]),
                                     "rows": v["rows"]} for n, v in q["tables"].items()},
                "counts": q["table_status_counts"],
                "timeseries": {k: q["timeseries"].get(k) for k in ("series", "missing_weeks_total", "zero_demand_rows", "short_series_lt52",
                                                                   "stockouts", "date_range", "measure", "table")} if q["timeseries"].get("available") else None,
                "outliers": {k: (q.get("outliers_sku") or {}).get(k) for k in ("flagged_weeks", "series_with_flags", "series_checked", "method", "treatment")}}
    scen = []
    for n, sc in s.scenarios.items():
        scen.append({"name": n, "status": sc["status"], "applied": bool(sc.get("summary")), "created": sc["created"],
                     "spec": sc["spec"], "assumptions": sc["assumptions"]})
    return to_jsonable({
        "session_id": s.id, "created": s.created, "name": s.name, "updated": s.updated, "api": _api_status(),
        "save_error": s.settings.get("save_error"),
        "tables": [{"name": n, **{k: v for k, v in s.table_profile(n).items() if k in ("rows", "columns", "role", "primary_key", "date_range", "source_file", "pk_status", "load_notes")}}
                   for n in sorted(s.tables)],
        "uploads": s.uploads[-5:],
        "pending_workbooks": [{"file": w.display_name, "sheets": w.sheets} for w in s.pending_workbooks],
        "relationships": [r.to_dict() for r in s.relationships], "relationships_confirmed": s.relationships_confirmed,
        "dictionary": s.dictionary() if s.tables else [],
        "joins": s.joins_log[-40:],
        "quality": qsum,
        "scenarios": scen, "assumptions": s.assumptions,
        "glossary": glossary(s),
        "messages": s.messages[-60:],
        "warmup": s.settings.get("warmup"),
    })


# ============================================================================ jobs

def _upload_job(progress, sid: str, path: str, name: str, sheet_mode=None):
    s = STORE.get(sid)
    progress(f"Reading {name}")
    with s.lock:
        summ = s.add_upload(path, name, sheet_mode=sheet_mode)
        if summ["tables"]:
            progress("Profiling tables and inferring keys")
            progress("Running data quality checks")
            data_quality_report(s, force=True)
    msg = {"id": f"m{len(s.messages) + 1}", "role": "system", "kind": "upload_summary", "at": now_iso(), "summary": summ}
    s.messages.append(msg)
    STORE.save(s)
    if summ["tables"]:
        _start_warmup(s)
    return {"summary": summ, "message": msg}


def _start_warmup(s):
    """Pre-compute the SKU-level forecast used by scenarios so the first scenario answer is fast."""
    def run(progress):
        s.settings["warmup"] = {"status": "running", "started": now_iso()}
        try:
            from .forecasting import forecast_weekly
            forecast_weekly(s, level="sku", horizon=26, top_n=None)
            s.settings["warmup"] = {"status": "done", "finished": now_iso()}
        except Exception as e:
            s.settings["warmup"] = {"status": "error", "error": str(e)[:200]}
        return s.settings["warmup"]
    new_job("warmup", s.id, run)


def _chat_job(progress, sid: str, text: str):
    s = STORE.get(sid)
    user_msg = {"id": f"m{len(s.messages) + 1}", "role": "user", "text": text, "at": now_iso()}
    s.messages.append(user_msg)
    mid = f"m{len(s.messages) + 1}"
    client, model, err = agent_mod.make_client()
    if client is None:
        msg = {"id": mid, "role": "assistant", "at": now_iso(), "error": err,
               "final": {"answer": err, "next_questions": []}, "charts": [], "calls": [], "kpis": []}
        s.messages.append(msg)
        STORE.save(s)
        return {"message": msg}
    with s.lock:
        try:
            r = agent_mod.run_agent(s, text, client, model, mid, progress=progress,
                                    should_stop=lambda: bool(progress.job.get("cancel")))
        except Exception as e:
            friendly = _llm_error_text(e)
            msg = {"id": mid, "role": "assistant", "at": now_iso(), "error": friendly,
                   "final": {"answer": friendly, "next_questions": []}, "charts": [], "calls": [], "kpis": []}
            s.messages.append(msg)
            STORE.save(s)
            return {"message": msg}
    charts = []
    for cid in r.charts:
        try:
            charts.append({"id": cid, "figure": get_chart(s, cid)["figure"]})
        except ToolError:
            pass
    msg = {"id": mid, "role": "assistant", "at": now_iso(), "final": r.final, "charts": charts, "calls": r.calls,
           "kpis": r.kpis, "provenance": r.provenance, "seconds": r.seconds, "iterations": r.iterations, "error": r.error,
           "model": model, "exports": r.exports or []}
    s.messages.append(msg)
    STORE.save(s)
    return {"message": msg}


def _llm_error_text(e: Exception) -> str:
    name = type(e).__name__
    txt = str(e)
    if "authentication" in name.lower() or "401" in txt:
        return "The Anthropic API rejected the key (authentication error). Check ANTHROPIC_API_KEY in .env and restart."
    if "notfound" in name.lower() or "404" in txt and "model" in txt.lower():
        return f"The model name was not found. Check ANTHROPIC_MODEL in .env. ({txt[:120]})"
    if "ratelimit" in name.lower() or "429" in txt:
        return "The Anthropic API is rate-limiting requests. Wait a few seconds and try again."
    if "overloaded" in txt.lower() or "529" in txt:
        return "The Anthropic API is temporarily overloaded. Please try again in a moment."
    if "connection" in name.lower() or "timeout" in name.lower():
        return "Could not reach the Anthropic API (network or timeout). Check the internet connection and try again."
    return f"The AI service returned an error ({name}). The app is still running; please try again. Details: {txt[:200]}"


# ============================================================================ routes

async def health(request: Request):
    return ok({"ok": True, "time": now_iso(), **_api_status()})


async def create_session(request: Request):
    s = STORE.create()
    STORE.save(s)
    return ok({"session_id": s.id})


async def get_state(request: Request):
    try:
        return ok(session_state(_session(request)))
    except KeyError as e:
        return fail(str(e).strip("'\""), 404)


async def upload(request: Request):
    try:
        s = _session(request)
    except KeyError as e:
        return fail(str(e).strip("'\""), 404)
    try:
        form = await request.form(max_part_size=MAX_UPLOAD_MB * 1024 * 1024)
    except Exception as e:
        return fail(f"Could not read the upload ({type(e).__name__}). Files must be under {MAX_UPLOAD_MB} MB.")
    f = form.get("file")
    if f is None or not getattr(f, "filename", None):
        return fail("No file received. Choose a .zip, .csv, .xlsx or .xls file.")
    name = Path(f.filename).name  # strip any client-side path
    dest_dir = s.folder / "incoming"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:6]}_{name}"
    size = 0
    with open(dest, "wb") as out:
        while chunk := await f.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                return fail(f"File is larger than {MAX_UPLOAD_MB} MB.")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        return fail(f"'{name}' is empty.")
    jid = new_job("upload", s.id, _upload_job, s.id, str(dest), name)
    return ok({"job_id": jid})


async def load_demo(request: Request):
    try:
        s = _session(request)
    except KeyError as e:
        return fail(str(e).strip("'\""), 404)
    if not DEMO_ZIP.exists():
        return fail("The demo data file is missing (data/demo/demo_data.zip).", 500)
    dest = s.folder / "incoming" / "demo_data.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMO_ZIP, dest)
    jid = new_job("upload", s.id, _upload_job, s.id, str(dest), "demo_data.zip")
    return ok({"job_id": jid})


async def resolve_workbook(request: Request):
    s = _session(request)
    body = await request.json()
    mode = body.get("mode")
    file = body.get("file")
    wb = next((w for w in s.pending_workbooks if w.display_name == file), None)
    if wb is None:
        return fail(f"No pending workbook '{file}'")
    if mode not in ("first", "all"):
        return fail("mode must be 'first' or 'all'")
    s.pending_workbooks = [w for w in s.pending_workbooks if w is not wb]
    jid = new_job("upload", s.id, _upload_job, s.id, wb.file_path, Path(wb.display_name).name, {Path(wb.display_name).name: mode})
    return ok({"job_id": jid})


async def relationships(request: Request):
    s = _session(request)
    body = await request.json()
    ids, rej = set(body.get("confirm_ids") or []), set(body.get("reject_ids") or [])
    for r in s.relationships:
        if body.get("confirm_all") and r.status == "inferred":
            r.status = "confirmed"
        if r.id in ids:
            r.status = "confirmed"
        if r.id in rej:
            r.status = "rejected"
    s.relationships_confirmed = bool(s.relationships) and all(r.status != "inferred" for r in s.relationships)
    s.messages.append({"id": f"m{len(s.messages) + 1}", "role": "system", "kind": "note", "at": now_iso(),
                       "text": f"Relationships updated: {sum(r.status == 'confirmed' for r in s.relationships)} confirmed, "
                               f"{sum(r.status == 'rejected' for r in s.relationships)} rejected."})
    STORE.save(s)
    return ok({"relationships_confirmed": s.relationships_confirmed})


async def edit_dictionary(request: Request):
    s = _session(request)
    body = await request.json()
    t, c = body.get("table"), body.get("column")
    if t not in s.tables or c not in s.tables[t].df.columns:
        return fail("Unknown table/column")
    s.dictionary_edits[f"{t}.{c}"] = {"description": (body.get("description") or "")[:500] or None,
                                      "unit": (body.get("unit") or None)}
    STORE.save(s)
    return ok({"saved": True})


async def edit_pk(request: Request):
    s = _session(request)
    body = await request.json()
    t, cols = body.get("table"), body.get("columns") or []
    if t not in s.tables:
        return fail("Unknown table")
    df = s.tables[t].df
    missing = [c for c in cols if c not in df.columns]
    if missing or not cols:
        return fail(f"Columns not found: {missing or cols}")
    dups = int(df.duplicated(cols).sum())
    s.tables[t].pk = cols
    s.tables[t].pk_status = "edited"
    s.refresh_relationships()
    s.cache.clear()
    s.quality = None
    STORE.save(s)
    return ok({"saved": True, "duplicate_rows_on_key": dups,
               "warning": f"{dups} rows share this key - it is not unique" if dups else None})


async def confirm_scen(request: Request):
    s = _session(request)
    name = request.path_params["name"]
    try:
        r = confirm_scenario(s, name)
    except ToolError as e:
        return fail(str(e))
    s.messages.append({"id": f"m{len(s.messages) + 1}", "role": "system", "kind": "note", "at": now_iso(),
                       "text": f"Scenario '{name}' confirmed by the user."})
    STORE.save(s)
    return ok(r)


async def chat(request: Request):
    try:
        s = _session(request)
    except KeyError as e:
        return fail(str(e).strip("'\""), 404)
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return fail("Type a question first.")
    if len(text) > 4000:
        return fail("Message is too long (max 4000 characters).")
    jid = new_job("chat", s.id, _chat_job, s.id, text)
    return ok({"job_id": jid})


async def job_status(request: Request):
    j = JOBS.get(request.path_params["jid"])
    if j is None:
        return fail("Unknown job", 404)
    if j["status"] == "running" and time.time() - j["started"] > JOB_TIMEOUT:
        j["status"], j["error"] = "error", f"Timed out after {JOB_TIMEOUT} seconds. Try a narrower question."
    return ok({k: v for k, v in j.items() if k != "trace"})


async def chart(request: Request):
    s = _session(request)
    try:
        return ok(get_chart(s, request.path_params["cid"])["figure"])
    except ToolError as e:
        return fail(str(e), 404)


async def quality_full(request: Request):
    s = _session(request)
    if not s.tables:
        return fail("No data loaded")
    return ok(data_quality_report(s))


async def workspaces(request: Request):
    if request.method == "POST":
        body = await request.json() if (await request.body()) else {}
        s = STORE.create((body.get("name") or "").strip()[:60] or None)
        STORE.save(s)
        return ok({"session_id": s.id, "name": s.name})
    return ok({"workspaces": STORE.list()})


async def workspace_manage(request: Request):
    sid = request.path_params["sid"]
    if request.method == "DELETE":
        try:
            STORE.drop(sid)
        except ValueError as e:
            return fail(str(e))
        return ok({"deleted": sid})
    s = STORE.get(sid)
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return fail("Give the workspace a name.")
    s.name = name[:60]
    STORE.save(s)
    return ok({"name": s.name})


async def cancel_job(request: Request):
    j = JOBS.get(request.path_params["jid"])
    if j is None:
        return fail("Unknown job", 404)
    j["cancel"] = True
    j["progress"].append({"t": round(time.time() - j["started"], 1), "msg": "Stopping after the current step"})
    return ok({"cancelling": True})


async def export_answer(request: Request):
    from .exports import answer_workbook
    s = _session(request)
    mid = request.path_params["mid"]
    msg = next((m for m in s.messages if m.get("id") == mid and m.get("role") == "assistant"), None)
    if msg is None:
        return fail("Unknown answer", 404)
    data = answer_workbook(s, msg)
    fname = f"answer_{mid}_{s.name or 'workspace'}".replace(" ", "_")[:80] + ".xlsx"
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


async def export_csv(request: Request):
    from .exports import result_csv
    s = _session(request)
    kind, iid = request.path_params["kind"], request.path_params["iid"]
    if kind not in ("result", "forecast", "inventory_plan"):
        return fail("kind must be result, forecast or inventory_plan")
    try:
        data = result_csv(s, kind, iid)
    except ToolError as e:
        return fail(str(e), 404)
    return Response(data, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{kind}_{iid}.csv"'})


# ---------------------------------------------------------------------------- optional password

APP_PASSWORD = None
AUTH_COOKIE = "dpa_auth"


def _expected_token() -> str | None:
    import hashlib
    pw = os.environ.get("APP_PASSWORD", "").strip()
    if not pw:
        return None
    return hashlib.sha256(("dpa-v1:" + pw).encode()).hexdigest()


async def login(request: Request):
    import hmac
    tok = _expected_token()
    if tok is None:
        return ok({"ok": True, "auth": "disabled"})
    body = await request.json()
    import hashlib
    given = hashlib.sha256(("dpa-v1:" + str(body.get("password", ""))).encode()).hexdigest()
    if not hmac.compare_digest(given, tok):
        time.sleep(0.5)
        return fail("Wrong password.", 401)
    resp = ok({"ok": True})
    resp.set_cookie(AUTH_COOKIE, tok, httponly=True, samesite="strict", max_age=60 * 60 * 24 * 14)
    return resp


async def auth_status(request: Request):
    tok = _expected_token()
    return ok({"required": tok is not None, "authenticated": tok is None or request.cookies.get(AUTH_COOKIE) == tok})


async def plotly_js(request: Request):
    """Serve plotly.js from the installed Python plotly package (works offline)."""
    try:
        import plotly
        p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if p.exists():
            return FileResponse(p, media_type="application/javascript", headers={"Cache-Control": "max-age=86400"})
    except ImportError:
        pass
    return RedirectResponse("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js")


async def index(request: Request):
    f = STATIC / "index.html"
    if not f.exists():
        return PlainTextResponse("Frontend not built. Run: uv run python scripts/build_frontend.py", 500)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/session", create_session, methods=["POST"]),
    Route("/api/session/{sid}/state", get_state),
    Route("/api/session/{sid}/upload", upload, methods=["POST"]),
    Route("/api/session/{sid}/demo", load_demo, methods=["POST"]),
    Route("/api/session/{sid}/workbook", resolve_workbook, methods=["POST"]),
    Route("/api/session/{sid}/relationships", relationships, methods=["POST"]),
    Route("/api/session/{sid}/dictionary", edit_dictionary, methods=["POST"]),
    Route("/api/session/{sid}/primary_key", edit_pk, methods=["POST"]),
    Route("/api/session/{sid}/scenarios/{name}/confirm", confirm_scen, methods=["POST"]),
    Route("/api/session/{sid}/chat", chat, methods=["POST"]),
    Route("/api/session/{sid}/charts/{cid}", chart),
    Route("/api/session/{sid}/quality", quality_full),
    Route("/api/jobs/{jid}", job_status),
    Route("/api/jobs/{jid}/cancel", cancel_job, methods=["POST"]),
    Route("/api/workspaces", workspaces, methods=["GET", "POST"]),
    Route("/api/workspaces/{sid}", workspace_manage, methods=["POST", "DELETE"]),
    Route("/api/session/{sid}/export/answer/{mid}.xlsx", export_answer),
    Route("/api/session/{sid}/export/{kind}/{iid}.csv", export_csv),
    Route("/api/login", login, methods=["POST"]),
    Route("/api/auth", auth_status),
    Route("/vendor/plotly.min.js", plotly_js),
]
if STATIC.exists():
    routes.append(Mount("/assets", app=StaticFiles(directory=STATIC / "assets", check_dir=False), name="assets"))


async def _errors(request: Request, exc: Exception):
    if isinstance(exc, KeyError):
        return fail(str(exc).strip("'\""), 404)
    return fail(f"Server error ({type(exc).__name__}). The app is still running.", 500)


class PasswordGate:
    """When APP_PASSWORD is set, every /api route except login/auth/health needs the session cookie."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            tok = _expected_token()
            if tok and path.startswith("/api/") and path not in ("/api/login", "/api/auth", "/api/health"):
                from starlette.requests import Request as _R
                if _R(scope).cookies.get(AUTH_COOKIE) != tok:
                    await JSONResponse({"error": "Sign in required.", "auth": True}, status_code=401)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app = PasswordGate(Starlette(routes=routes, exception_handlers={Exception: _errors, KeyError: _errors}))
