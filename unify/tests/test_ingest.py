"""Week-1 check: the windowing, warm-up gating and gap detection actually work.

Run:  pip install pytest httpx && pytest tests/ -q
"""
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "edge"))


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Fresh DB per test, and an empty artifacts dir so the service always falls
    # back to the stub. Otherwise these tests would pass or fail depending on
    # whether someone had run `make export`, and on which datasets they trained.
    monkeypatch.setenv("UNIFY_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("UNIFY_ARTIFACTS", str(tmp_path / "empty"))
    import importlib
    import app as app_mod
    importlib.reload(app_mod)
    assert app_mod.MODEL_NAME == "stub-42"
    return TestClient(app_mod.app), app_mod


def packet(app_mod, seq, val=0.1):
    return {"node": "test", "seq": seq, "ts_ms": seq * 1000,
            "feat": [val] * app_mod.S, "clip": 0, "jitter_us": 0}


def test_warms_up_then_predicts(client):
    c, m = client
    for seq in range(m.L - 1):
        r = c.post("/ingest", json=packet(m, seq)).json()
        assert r["status"] == "warming_up", f"seq {seq} should still be warming up"
        assert r["have"] == seq + 1

    r = c.post("/ingest", json=packet(m, m.L - 1)).json()
    assert r["status"] == "ok"
    assert r["rul"] == 42.0          # the stub
    assert 0.0 <= r["health"] <= 1.0


def test_window_never_exceeds_L(client):
    c, m = client
    for seq in range(m.L * 3):
        c.post("/ingest", json=packet(m, seq))
    assert len(m.window) == m.L


def test_detects_gaps(client):
    c, m = client
    c.post("/ingest", json=packet(m, 0))
    c.post("/ingest", json=packet(m, 5))          # 4 packets lost
    assert c.get("/health").json()["gaps"] == 4


def test_duplicate_is_ignored(client):
    c, m = client
    c.post("/ingest", json=packet(m, 0))
    r = c.post("/ingest", json=packet(m, 0)).json()
    assert r["status"] == "duplicate"
    assert len(m.window) == 1


def test_node_restart_clears_window(client):
    c, m = client
    for seq in range(10):
        c.post("/ingest", json=packet(m, seq))
    assert len(m.window) == 10
    c.post("/ingest", json=packet(m, 0))          # ESP32 rebooted, seq restarts
    assert len(m.window) == 1, "stale pre-reboot samples must not stay in the window"


def test_rejects_wrong_feature_count(client):
    c, m = client
    bad = packet(m, 0)
    bad["feat"] = [0.1] * (m.S + 1)
    assert c.post("/ingest", json=bad).status_code == 422
    assert c.get("/health").json()["rejected"] == 1
