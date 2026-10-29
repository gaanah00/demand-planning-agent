"""Saved workspaces (survive a restart), HTTP API, Excel export, stop button and optional password."""
import io
import time

import pandas as pd
import pytest

from dpa.session import STORE


def test_workspace_round_trip(mini_zip):
    from dpa.scenarios import create_scenario  # noqa: F401  (scenario needs cost data; use spec only)
    s = STORE.create()  # default name -> renamed after the first upload
    s.add_upload(str(mini_zip), "mini.zip")
    s.relationships[0].status = "confirmed"
    s.dictionary_edits["dim_product.category"] = {"description": "Product family", "unit": None}
    s.messages.append({"id": "m1", "role": "user", "text": "hello"})
    s.scenarios["X"] = {"spec": {"name": "X", "start_date": pd.Timestamp("2026-10-29"), "end_date": None,
                                 "demand_effect": {"source": "none"}}, "status": "draft", "created": "now", "assumptions": []}
    STORE.save(s)
    sid = s.id
    STORE._sessions.pop(sid)                 # simulate an app restart
    s2 = STORE.get(sid)
    assert s2 is not s and s2.name == "mini"  # renamed after the first upload
    assert sorted(s2.tables) == sorted(s.tables)
    pd.testing.assert_frame_equal(s2.df("fact_weekly_sales"), s.df("fact_weekly_sales"))
    assert s2.relationships[0].status == "confirmed"
    assert s2.dictionary()[0]["table"]
    assert any(r["description_status"] == "confirmed" for r in s2.dictionary() if r["column"] == "category")
    assert s2.scenarios["X"]["spec"]["start_date"] == pd.Timestamp("2026-10-29")
    assert s2.messages[0]["text"] == "hello"
    assert any(w["id"] == sid for w in STORE.list())
    named = STORE.create("My plan")
    named.add_upload(str(mini_zip), "mini.zip")
    assert named.name == "My plan"           # a name the user chose is kept
    STORE.drop(named.id)
    STORE.drop(sid)
    assert not any(w["id"] == sid for w in STORE.list())
    with pytest.raises(KeyError):
        STORE.get(sid)


@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setenv("DPA_LLM_STUB", "e2e_stub:make")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    from dpa.server import app
    return TestClient(app)


def _wait(client, jid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_api_workspaces_and_excel_export(client, mini_zip):
    sid = client.post("/api/workspaces", json={}).json()["session_id"]
    j = _wait(client, client.post(f"/api/session/{sid}/upload", files={"file": ("mini.zip", mini_zip.read_bytes())}).json()["job_id"])
    assert j["status"] == "done"
    j = _wait(client, client.post(f"/api/session/{sid}/chat", json={"text": "Forecast total weekly demand for the next 13 weeks"}).json()["job_id"])
    msg = j["result"]["message"]
    assert msg["exports"] and msg["exports"][0]["kind"] == "forecast"
    r = client.get(f"/api/session/{sid}/export/answer/{msg['id']}.xlsx")
    assert r.status_code == 200
    book = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
    assert "Answer" in book and any(k.startswith("forecast") for k in book)
    fid = msg["exports"][0]["id"]
    r = client.get(f"/api/session/{sid}/export/forecast/{fid}.csv")
    assert r.status_code == 200 and b"forecast" in r.content.splitlines()[0]
    names = [w["name"] for w in client.get("/api/workspaces").json()["workspaces"]]
    assert "mini" in names
    assert client.post(f"/api/workspaces/{sid}", json={"name": "Renamed"}).json()["name"] == "Renamed"
    assert client.delete(f"/api/workspaces/{sid}").status_code == 200
    assert client.get(f"/api/session/{sid}/state").status_code == 404


def test_password_gate(client, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert client.get("/api/auth").json() == {"required": True, "authenticated": False}
    assert client.get("/api/workspaces").status_code == 401
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert client.get("/api/workspaces").status_code == 200          # cookie now set
    assert client.get("/api/health").status_code == 200


def test_cancel_job(client):
    from dpa import server
    jid = server.new_job("test", "x", lambda progress: (time.sleep(0.3), progress.job.get("cancel"))[1])
    assert client.post(f"/api/jobs/{jid}/cancel").json()["cancelling"]
    assert _wait(client, jid)["result"] is True
