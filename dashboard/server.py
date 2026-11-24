"""Call review dashboard: live call logs and caller-simulator results.

Mounted at /dashboard/ by bot.py, or run on its own:
    uv run python -m dashboard.server      -> http://localhost:8000
"""

import json
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse

from frontdesk import db

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT.parent / "data" / "sim_runs"
RECORDINGS_DIR = ROOT.parent / "data" / "recordings"

router = APIRouter()


def _row(r) -> dict:
    d = dict(r)
    for k in ("flags", "events", "metrics"):
        d[k] = json.loads(d[k] or "null")
    return d


@router.get("/")
def index():
    return FileResponse(ROOT / "index.html")


landing = APIRouter()


@landing.get("/", include_in_schema=False)
def landing_page():
    return FileResponse(ROOT / "landing.html")


@router.get("/api/calls")
def calls():
    conn = db.connect()
    rows = conn.execute("SELECT * FROM calls ORDER BY started_at DESC, rowid DESC LIMIT 200").fetchall()
    out = []
    for r in rows:
        d = _row(r)
        d.pop("events")
        out.append(d)
    return out


@router.get("/api/calls/{call_id}")
def call(call_id: str):
    r = db.connect().execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    if r is None:
        raise HTTPException(404)
    return _row(r)


@router.get("/recordings/{name}")
def recording(name: str):
    f = RECORDINGS_DIR / name
    if "/" in name or not name.endswith(".wav") or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/wav")


@router.get("/api/sim-runs")
def sim_runs():
    runs = []
    for f in sorted(RUNS_DIR.glob("run_*.json"), reverse=True):
        data = json.loads(f.read_text())
        runs.append({k: data.get(k) for k in ("run_at", "model", "total", "passed", "pass_rate",
                                               "avg_voice_style", "avg_empathy")} | {"file": f.stem})
    return runs


@router.get("/api/sim-runs/{name}")
def sim_run(name: str):
    f = RUNS_DIR / f"{name}.json"
    if not name.startswith("run_") or not f.exists():
        raise HTTPException(404)
    return json.loads(f.read_text())


if __name__ == "__main__":
    app = FastAPI(title="FrontDesk AI dashboard")
    app.include_router(router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
