"""One command to start everything:   uv run python start.py

Loads .env, checks the built frontend, starts the API + UI server on http://127.0.0.1:8000 and opens the browser.
Options: --port 8000  --no-browser  --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))


def load_env(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    load_env(ROOT / ".env")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        print("The built frontend is missing (frontend/dist). Rebuild it with: cd frontend && npm install && npm run build")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key to enable the chat.")
    import uvicorn
    url = f"http://{a.host}:{a.port}"
    if not a.no_browser:
        def open_when_ready():
            # Open the browser only once the server answers (first start after an update can take ~30-60 s).
            import urllib.request
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"{url}/api/health", timeout=2)
                    webbrowser.open(url)
                    return
                except Exception:
                    time.sleep(0.5)
            print(f"The server is taking unusually long to start. Open {url} yourself once it is ready.")
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"Starting Demand Planning Analyst at {url} ... the browser opens when it is ready (Ctrl+C to stop)")
    uvicorn.run("dpa.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
