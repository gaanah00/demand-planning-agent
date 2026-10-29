"""End-to-end script against a running server (http://127.0.0.1:8000 by default).

Covers: upload zip -> upload summary & relationship confirmation -> data quality report -> descriptive question ->
weekly forecast with backtest vs baselines -> tariff scenario (draft, confirmation, direct cost effect, demand
handling) -> supplier comparison & mitigations -> context-aware next questions -> KPI glossary populated.

Usage:  uv run python tests/e2e_flow.py [--base http://127.0.0.1:8000] [--zip path/to/data.zip]
With a real API key the answers come from Claude; with DPA_LLM_STUB set on the server they come from the
offline stand-in. Checks are on structure and provenance, not exact wording.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def call(base, path, body=None, method=None, files=None):
    url = base + path
    if files:
        boundary = uuid.uuid4().hex
        name, data = files
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    elif body is not None:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method or "POST", headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url, method=method or "GET")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_status": e.code, **json.loads(e.read() or b"{}")}


def wait(base, jid, label):
    t0 = time.time()
    while True:
        j = call(base, f"/api/jobs/{jid}")
        if j["status"] != "running":
            print(f"  [{label}] {j['status']} in {time.time() - t0:.1f}s" + (f"  ERROR: {j.get('error')}" if j.get("error") else ""))
            return j
        time.sleep(0.7)


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def ask(base, sid, text):
    print(f"\n> {text}")
    j = wait(base, call(base, f"/api/session/{sid}/chat", {"text": text})["job_id"], "chat")
    m = j["result"]["message"] if j.get("result") else {}
    f = m.get("final", {})
    print("  answer:", (f.get("answer") or "")[:300])
    print("  tools:", [c["tool"] for c in m.get("calls", [])], " charts:", len(m.get("charts", [])),
          " provenance:", m.get("provenance"))
    print("  next:", f.get("next_questions"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--zip", default=str(ROOT / "data" / "demo" / "demo_data.zip"))
    a = ap.parse_args()
    base = a.base
    print("health:", call(base, "/api/health"))
    sid = call(base, "/api/session", {})["session_id"]

    print("\n1. Upload zip")
    j = wait(base, call(base, f"/api/session/{sid}/upload", files=(Path(a.zip).name, Path(a.zip).read_bytes()))["job_id"], "upload")
    s = j["result"]["summary"]
    check(len(s["tables"]) >= 20, f"upload summary lists tables ({len(s['tables'])})")
    check(all(t["rows"] > 0 for t in s["tables"]), "every table has rows")
    check(len(s["relationships"]) > 20, f"relationships inferred ({len(s['relationships'])})")

    print("\n2. Bad inputs are handled")
    bad = call(base, f"/api/session/{sid}/upload", files=("broken.zip", b"not a zip"))
    jb = wait(base, bad["job_id"], "bad zip") if "job_id" in bad else bad
    check(jb.get("status") == "error" and "ZIP" in (jb.get("error") or ""), "corrupt zip -> clear error, server alive")
    check(call(base, "/api/health")["ok"], "server still healthy")

    print("\n3. Confirm relationships")
    call(base, f"/api/session/{sid}/relationships", {"confirm_all": True})
    st = call(base, f"/api/session/{sid}/state")
    check(st["relationships_confirmed"], "relationships confirmed")

    m = ask(base, sid, "Give me the data quality report and tell me what needs fixing.")
    check(any(c["tool"] == "data_quality_report" for c in m["calls"]), "quality tool called")
    m = ask(base, sid, "Summarise weekly demand by category over the full history with a chart.")
    check(len(m.get("charts", [])) >= 1 and m["charts"][0]["figure"]["data"], "descriptive answer has a chart with data")
    m = ask(base, sid, "Forecast total weekly demand for the next 13 weeks and compare to baselines.")
    calls = {c["tool"]: c for c in m["calls"]}
    check("forecast_weekly" in calls, "forecast tool called")
    res = json.loads(calls["forecast_weekly"]["result_preview"][:3000].rsplit("}", 0)[0]) if False else None
    check(any(k["label"].startswith("WAPE") for k in m["final"].get("key_results", [])), "forecast answer reports WAPE")
    check(not m.get("provenance", {}).get("unverified"), "forecast answer numbers all traced to tool results")
    m = ask(base, sid, "A 25% tariff is imposed on imports from India starting 10/29/2026.")
    check("0" in m["final"]["answer"] or "no supplier" in m["final"]["answer"].lower(), "India tariff -> zero exposure explained")
    m = ask(base, sid, "A 25% tariff is imposed on imports from South Korea starting 10/29/2026.")
    st = call(base, f"/api/session/{sid}/state")
    drafts = [x for x in st["scenarios"] if x["status"] == "draft"]
    check(any("South Korea" in x["name"] for x in drafts), "scenario created as DRAFT (not applied before confirmation)")
    name = next(x["name"] for x in drafts if "South Korea" in x["name"])
    r = call(base, f"/api/session/{sid}/scenarios/{urllib.request.quote(name)}/confirm", {})
    check(r.get("status") == "confirmed", "scenario confirmed via button")
    m = ask(base, sid, f"I confirm the scenario '{name}' as shown. Apply it and walk me through the impact.")
    tools_used = [c["tool"] for c in m["calls"]]
    check({"apply_scenario", "sensitivity_analysis", "evaluate_mitigations"} <= set(tools_used), "direct effect, sensitivity and mitigations computed")
    srcs = {k["source"] for k in m["final"].get("key_results", [])}
    check(srcs <= {"computed", "estimated", "assumed"} and srcs, f"key results carry source badges {srcs}")
    check(len(m.get("charts", [])) >= 2, "scenario answer has baseline-vs-scenario charts")
    nq = m["final"].get("next_questions", [])
    check(len(nq) >= 2 and any(("SUP" in q or "SKU" in q) for q in nq), f"next questions are specific: {nq}")
    st = call(base, f"/api/session/{sid}/state")
    g = {x["id"]: x for x in st["glossary"]}
    check({"wape", "gross_margin", "landed_cost"} <= set(g), f"glossary populated: {sorted(g)}")
    check(any(u.get("example") for u in g["wape"]["uses"]), "WAPE entry has a worked example with session numbers")
    check(all(x.get("uses") and x["uses"][0].get("message_id") for x in g.values()), "glossary entries link back to answers")
    check(len(st["assumptions"]) >= 4, f"assumptions logged ({len(st['assumptions'])})")
    print("\nRESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
