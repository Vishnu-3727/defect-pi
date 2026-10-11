"""The Pi normalised with the wrong statistics and still returned a confident RUL.

Measured 2026-08-28: the same C-MAPSS checkpoint, on the same FD001 engine at its
true end of life, returned RUL 0.0 offline and RUL 65.1 through edge/inference.py.
The cause was Predictor calling Normalizer.vector_for() without an operating
condition, which silently falls back to the global table. For C-MAPSS the global
and per-condition tables differ by up to 390 in the mean and a factor of 1105 in
the standard deviation, so the model receives garbage and reports it calmly.

These tests pin the guard: a per-condition dataset with no "condition" in the live
map must refuse to load, and when a condition IS given the statistics used must be
the per-condition ones.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "edge"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import ArtifactsMissing, Predictor          # noqa: E402
from unify.common.normalize import Normalizer             # noqa: E402

REAL_ONNX = Path(__file__).parent.parent / "edge" / "artifacts" / "unify_rul.onnx"


def _stats(per_condition: bool) -> dict:
    """Two sensors. Under condition "0" they are centred at 100; globally the mix
    with a second condition drags the mean to 500 and inflates the std."""
    entry = {
        "per_condition": per_condition,
        "global": {"2": {"mean": 500.0, "std": 400.0},
                   "3": {"mean": 500.0, "std": 400.0}},
        "cond": {"0": {"2": {"mean": 100.0, "std": 1.0},
                       "3": {"mean": 100.0, "std": 1.0}},
                 "1": {"2": {"mean": 900.0, "std": 1.0},
                       "3": {"mean": 900.0, "std": 1.0}}},
    }
    return {"window": 30, "scales": {"cmapss": 125.0}, "dataset_config": {},
            "datasets": {"cmapss": entry}}


def _artifacts(tmp_path: Path, per_condition: bool) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "norm_stats.json").write_text(json.dumps(_stats(per_condition)))
    (d / "sensor_vocab.json").write_text(json.dumps({"cmapss/s2": 2, "cmapss/s3": 3}))
    # The exported graph takes a dynamic sensor axis, so the real one loads happily
    # with S=2. Using it means the mean/std assertions below run against the actual
    # Predictor rather than being skipped, which is the point of the test.
    if REAL_ONNX.exists():
        (d / "unify_rul.onnx").write_bytes(REAL_ONNX.read_bytes())
    else:
        (d / "unify_rul.onnx").write_bytes(b"\x08\x07")     # forces the skip path
    return d


def _live_map(tmp_path: Path, condition=None) -> Path:
    cfg = {"ref_dataset": "cmapss", "domain_id": 0,
           "channels": ["cmapss/s2", "cmapss/s3"]}
    if condition is not None:
        cfg["condition"] = condition
    p = tmp_path / "live_map.json"
    p.write_text(json.dumps(cfg))
    return p


def test_per_condition_dataset_without_condition_is_refused(tmp_path):
    with pytest.raises(ArtifactsMissing) as e:
        Predictor(artifacts=_artifacts(tmp_path, True), live_map=_live_map(tmp_path))
    msg = str(e.value)
    assert "condition" in msg
    # The error has to name the valid IDs or the reader cannot act on it.
    assert "'0'" in msg and "'1'" in msg


def test_condition_selects_the_per_condition_statistics(tmp_path):
    """Not merely 'it loads' — the mean actually used must be the conditional one.
    Loading succeeds with the wrong table too; that was the whole bug."""
    art = _artifacts(tmp_path, True)
    try:
        p = Predictor(artifacts=art, live_map=_live_map(tmp_path, "0"))
    except Exception as e:                       # the stub .onnx cannot open
        assert "onnx" in str(e).lower() or "protobuf" in str(e).lower(), e
        pytest.skip("onnxruntime rejects the stub graph before the session is built")
    assert np.allclose(p.mean, [100.0, 100.0]), p.mean
    assert np.allclose(p.std, [1.0, 1.0]), p.std


def test_dataset_without_per_condition_stats_still_loads(tmp_path):
    """Synthetic and XJTU are not per-condition; the guard must not break them."""
    art = _artifacts(tmp_path, False)
    try:
        p = Predictor(artifacts=art, live_map=_live_map(tmp_path))
    except Exception as e:
        assert "onnx" in str(e).lower() or "protobuf" in str(e).lower(), e
        pytest.skip("onnxruntime rejects the stub graph before the session is built")
    assert np.allclose(p.mean, [500.0, 500.0])


def test_the_two_tables_really_do_disagree():
    """Guards the premise. If global and per-condition ever coincided, the tests
    above would pass for the wrong reason."""
    n = Normalizer()
    n.stats = _stats(True)["datasets"]
    ids = np.array([2, 3])
    mg, sg = n.vector_for("cmapss", ids)
    mc, sc = n.vector_for("cmapss", ids, "0")
    assert not np.allclose(mg, mc)
    assert float(np.max(sg / sc)) > 100
