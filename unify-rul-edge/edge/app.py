"""UNIFY-RUL edge service.

Ingests feature packets from the ESP32, assembles a sliding window, runs the
exported ONNX model (or a stub if the artifacts are not there yet), smooths,
evaluates the alarm rule, logs to SQLite, serves the dashboard.

    uvicorn app:app --host 0.0.0.0 --port 8000

Nothing here may import torch. If it does, the export step is incomplete.
"""
import json
import os
import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from inference import ArtifactsMissing, Predictor, Smoother
except ImportError:
    from edge.inference import ArtifactsMissing, Predictor, Smoother

STATIC = Path(__file__).parent / "static"
DB_PATH = Path(os.environ.get("UNIFY_DB", Path(__file__).parent / "unify.db"))

# ── model ─────────────────────────────────────────────────────────────────────
# Falls back to a stub so the pipeline is demonstrable before the model exists.
# Week 1 built this whole path against the stub on purpose: integration risk first.
try:
    predictor = Predictor()
    MODEL_NAME = predictor.name
    FEATURE_NAMES = predictor.channels
    L = predictor.window
    MODEL_NOTE = (f"ref={predictor.ref_dataset} scale={predictor.life_scale:.0f} "
                  f"cond={predictor.condition}")
except ArtifactsMissing as e:
    predictor = None
    MODEL_NAME = "stub-42"
    MODEL_NOTE = str(e)
    L = 30
    FEATURE_NAMES = [f"{a}_{f}" for a in ("ax", "ay", "az") for f in ("rms", "kurt", "crest")]
    print(f"[unify] no model artifacts, running stub: {e}")

S = len(FEATURE_NAMES)
ALARM = (predictor.alarm_cfg if predictor else {})
smoother = Smoother(alpha=ALARM.get("ewma_alpha", 0.3),
                    threshold=ALARM.get("threshold", 0.30),
                    consecutive=ALARM.get("consecutive", 5))

STALE_AFTER_S = 3.0
window: deque[list[float]] = deque(maxlen=L)


class State:
    last_seq = -1
    last_packet_t = 0.0
    gaps = 0
    rejected = 0
    started = time.time()
    latencies: deque[float] = deque(maxlen=500)
    alarm = False
    attention: list[float] = []


state = State()

# ── storage ───────────────────────────────────────────────────────────────────
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS predictions (
  id         INTEGER PRIMARY KEY,
  ts         REAL NOT NULL,
  seq        INTEGER,
  rul        REAL NOT NULL,
  rul_lo     REAL,
  rul_hi     REAL,
  health     REAL,
  alarm      INTEGER DEFAULT 0,
  latency_ms REAL,
  feat       TEXT,
  attn       TEXT,
  model      TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts);
""")
_uncommitted = 0


def log_prediction(row: dict):
    global _uncommitted
    db.execute(
        "INSERT INTO predictions(ts,seq,rul,rul_lo,rul_hi,health,alarm,latency_ms,"
        "feat,attn,model) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (row["ts"], row["seq"], row["rul"], row["rul_lo"], row["rul_hi"],
         row["health"], int(row["alarm"]), row["latency_ms"],
         json.dumps(row["feat"]), json.dumps(row["attn"]), MODEL_NAME))
    _uncommitted += 1
    if _uncommitted >= 10:      # batch — SD cards die of write amplification
        db.commit()
        _uncommitted = 0


def predict(win: deque) -> dict:
    if predictor is None:
        return {"rul": 42.0, "health": 1.0, "rul_lo": None, "rul_hi": None,
                "latency_ms": 0.0, "attention": [1.0 / S] * S}
    return predictor.predict(np.asarray(win, dtype=np.float32))


# ── api ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="UNIFY-RUL edge")


@app.exception_handler(RequestValidationError)
async def on_bad_packet(request: Request, exc: RequestValidationError):
    state.rejected += 1
    return JSONResponse(status_code=422, content={"detail": exc.errors()[:3]})


class Packet(BaseModel):
    node: str = Field(max_length=32)
    seq: int
    ts_ms: int
    feat: list[float] = Field(min_length=S, max_length=S)
    clip: int = 0
    jitter_us: int = 0


@app.post("/ingest")
def ingest(p: Packet):
    now = time.time()

    if state.last_seq >= 0 and p.seq > state.last_seq + 1:
        state.gaps += p.seq - state.last_seq - 1
    if p.seq <= state.last_seq:
        if p.seq == state.last_seq:
            return {"status": "duplicate"}
        # Node rebooted — everything before the restart belongs to a different run.
        window.clear()
    state.last_seq = p.seq
    state.last_packet_t = now

    if not all(np.isfinite(p.feat)):
        state.rejected += 1
        return JSONResponse(status_code=422, content={"detail": "non-finite feature"})

    window.append(p.feat)
    if len(window) < L:
        return {"status": "warming_up", "have": len(window), "need": L}

    out = predict(window)
    # With a quantile head, alarm on the LOWER bound rather than the point
    # estimate — conservative by construction, and the principled answer to P4.
    health_lo = (out["rul_lo"] / predictor.life_scale
                 if predictor is not None and out["rul_lo"] is not None
                 and predictor.life_scale > 0 else None)
    smoothed, alarm = smoother.update(out["health"], health_lo)
    state.alarm = alarm
    state.latencies.append(out["latency_ms"])
    state.attention = out["attention"]

    log_prediction({"ts": now, "seq": p.seq, "rul": out["rul"], "rul_lo": out["rul_lo"],
                    "rul_hi": out["rul_hi"], "health": smoothed, "alarm": alarm,
                    "latency_ms": out["latency_ms"], "feat": p.feat,
                    "attn": out["attention"]})

    # The node drives its own LED from this, so the machine shows status even if
    # the dashboard or the network dies.
    return {"status": "ok", "rul": out["rul"], "health": smoothed, "alarm": alarm}


@app.get("/health")
def health():
    age = time.time() - state.last_packet_t if state.last_packet_t else None
    return {
        "uptime_s": round(time.time() - state.started, 1),
        "last_packet_age_s": round(age, 2) if age is not None else None,
        "stale": age is None or age > STALE_AFTER_S,
        "window": f"{len(window)}/{L}",
        "gaps": state.gaps,
        "rejected": state.rejected,
        "alarm": state.alarm,
        "model": MODEL_NAME,
        "model_note": MODEL_NOTE,
        "features": FEATURE_NAMES,
        "attention": state.attention,
    }


@app.get("/metrics")
def metrics():
    lat = sorted(state.latencies)
    pct = lambda q: round(lat[int(q * (len(lat) - 1))], 3) if lat else None
    return {"n": len(lat), "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99)}


@app.get("/api/recent")
def recent(n: int = 120):
    rows = db.execute(
        "SELECT ts,seq,rul,rul_lo,rul_hi,health,alarm,latency_ms,feat,attn "
        "FROM predictions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    rows.reverse()
    keys = ["ts", "seq", "rul", "rul_lo", "rul_hi", "health", "alarm", "latency_ms"]
    return {
        "features": FEATURE_NAMES,
        "rows": [dict(zip(keys, r[:8]), feat=json.loads(r[8]), attn=json.loads(r[9]))
                 for r in rows],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
