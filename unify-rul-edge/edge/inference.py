"""ONNX Runtime inference on the Pi. No PyTorch — that is the point of ONNX
(~50 MB vs ~800 MB, and prebuilt aarch64 wheels).

Loads exactly three artifacts:
    unify_rul.onnx      the frozen graph
    norm_stats.json     training statistics + per-dataset life scales
    sensor_vocab.json   the sensor-ID map

If norm_stats.json is missing or stale, predictions are silently wrong — no
exception, no NaN. So this module refuses to start rather than guessing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# unify.common is numpy-only and is imported by BOTH the trainer and the Pi, so
# the two can never disagree about how a value is normalised. `make deploy` copies
# unify/common alongside edge/.
sys.path.insert(0, str(ROOT))
from unify.common.normalize import Normalizer  # noqa: E402

def artifacts_dir() -> Path:
    # Resolved on every call, not captured in a module-level default — otherwise
    # the env var is frozen at import time and tests cannot redirect it.
    return Path(os.environ.get("UNIFY_ARTIFACTS", Path(__file__).parent / "artifacts"))


def live_map_path() -> Path:
    return Path(os.environ.get("UNIFY_LIVE_MAP", ROOT / "config" / "live_map.json"))


class ArtifactsMissing(RuntimeError):
    pass


class Predictor:
    """Stateless w.r.t. the window — the caller owns the deque."""

    def __init__(self, artifacts: Path | None = None, live_map: Path | None = None,
                 threads: int = 2):
        import onnxruntime as ort

        artifacts = Path(artifacts) if artifacts else artifacts_dir()
        live_map = Path(live_map) if live_map else live_map_path()

        use_int8 = os.environ.get("UNIFY_INT8", "").lower() in ("1", "true", "yes")
        model_name = "unify_rul_int8.onnx" if (use_int8 and (artifacts / "unify_rul_int8.onnx").exists()) else "unify_rul.onnx"
        model_path = artifacts / model_name
        stats_path = artifacts / "norm_stats.json"
        vocab_path = artifacts / "sensor_vocab.json"
        for p in (model_path, stats_path, vocab_path):
            if not p.exists():
                raise ArtifactsMissing(f"missing {p} — run: python -m unify.deploy.export_onnx")

        cfg = json.loads(Path(live_map).read_text())
        self.ref_dataset = cfg["ref_dataset"]
        self.domain_id = int(cfg["domain_id"])
        self.channels = [c for c in cfg["channels"]]
        self.alarm_cfg = cfg.get("alarm", {})

        vocab = {k: v for k, v in json.loads(vocab_path.read_text()).items()
                 if not k.startswith("_")}
        missing = [c for c in self.channels if c not in vocab]
        if missing:
            raise ArtifactsMissing(f"channels not in sensor_vocab.json: {missing}")
        self.sensor_ids = np.array([vocab[c] for c in self.channels], dtype=np.int64)

        stats = json.loads(stats_path.read_text())
        self.window = int(stats.get("window", 30))
        self.life_scale = float(stats.get("scales", {}).get(self.ref_dataset, 1.0))

        if self.ref_dataset not in stats["datasets"]:
            raise ArtifactsMissing(
                f"norm_stats.json has no statistics for ref_dataset "
                f"{self.ref_dataset!r} (has: {sorted(stats['datasets'])}). "
                f"Either train on that dataset or edit config/live_map.json.")

        # Statistics are keyed by sensor ID, so the live rig sending a SUBSET of
        # the trained channels in its own order is just a lookup.
        norm = Normalizer()
        norm.stats = stats["datasets"]
        # The condition matters. If the reference dataset was normalised PER
        # OPERATING CONDITION during training (C-MAPSS is — docs/05 §5.2), then
        # looking the statistics up without one silently returns the global
        # table, whose mean can be 390 units and whose std can be 1105x off from
        # the per-condition table. The model still produces a confident number:
        # a real FD001 engine replayed at true RUL 0 came back as RUL 65.1 on the
        # Pi while the identical checkpoint offline said 0.0. Refuse instead.
        self.condition = cfg.get("condition")
        if self.condition is not None:
            self.condition = str(self.condition)
        if stats["datasets"][self.ref_dataset].get("per_condition") \
                and self.condition is None:
            raise ArtifactsMissing(
                f"{self.ref_dataset} was normalised per operating condition during "
                f"training, so this live map must set \"condition\" to the "
                f"condition ID the rig runs at (available: "
                f"{sorted(stats['datasets'][self.ref_dataset]['cond'])}). Without "
                f"it the global statistics are used and the RUL is wrong by tens "
                f"of cycles while still looking plausible.")
        try:
            self.mean, self.std = norm.vector_for(self.ref_dataset, self.sensor_ids,
                                                  self.condition)
        except KeyError as e:
            raise ArtifactsMissing(
                f"{self.ref_dataset} was not trained with some live channels: {e}")

        opts = ort.SessionOptions()
        # Counter-intuitive but real: for a ~44k-parameter model, thread
        # synchronisation usually costs more than the parallelism gains. Benchmark
        # {1,2,3,4} with edge/bench.py and put the chart in the report.
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts,
                                         providers=["CPUExecutionProvider"])

        self.model_sha = (artifacts / "model.sha256").read_text().strip()[:16] \
            if (artifacts / "model.sha256").exists() else "unknown"
        self.n_outputs = self.sess.get_outputs()[0].shape[-1]

    @property
    def name(self) -> str:
        return f"onnx:{self.model_sha}"

    def predict(self, window: np.ndarray) -> dict:
        """window: (L, S) most recent last. Returns rul/health/latency/attention."""
        x = ((window - self.mean) / self.std).T[None].astype(np.float32)   # (1, S, L)
        S = x.shape[1]
        feeds = {
            "x": x,
            "mask": np.ones((1, S), dtype=bool),
            "sensor_ids": self.sensor_ids[None],
            "domain_id": np.array([self.domain_id], dtype=np.int64),
        }
        t0 = time.perf_counter()
        rul, attn = self.sess.run(["rul", "attn"], feeds)
        latency_ms = (time.perf_counter() - t0) * 1e3

        vals = np.clip(rul[0], 0.0, 1.0)
        if len(vals) >= 3:                       # quantile head: [q_lo, q_med, q_hi]
            # Quantile regression does not guarantee q10 <= q50 <= q90 — the heads
            # are trained independently and can cross, especially early in training.
            # Sorting is the standard cheap repair and keeps the interval valid.
            lo, med, hi = (float(v) for v in np.sort(vals)[[0, len(vals) // 2, -1]])
        else:
            lo = hi = None
            med = float(vals[0])

        return {
            "health": med,
            "rul": med * self.life_scale,
            "rul_lo": None if lo is None else lo * self.life_scale,
            "rul_hi": None if hi is None else hi * self.life_scale,
            "latency_ms": latency_ms,
            # attn: (1, heads, queries, S) -> mean over heads and queries.
            # Free per-prediction interpretability. Present it as an indication of
            # feature usage, NOT a causal explanation — see docs/06 §6.5.
            "attention": attn[0].mean(axis=(0, 1)).tolist(),
        }


class Smoother:
    """EWMA + N-consecutive-breach hysteresis.

    A raw per-second prediction jitters a few percent, so a bare threshold makes
    the alarm flap many times a minute — that is problem P4 from the Review-1
    slides. Six lines removes essentially all of it. The principled version is to
    alarm on the 10th-percentile output of a quantile head (loss.kind: pinball).
    """

    def __init__(self, alpha=0.3, threshold=0.30, consecutive=5):
        self.alpha, self.threshold, self.consecutive = alpha, threshold, consecutive
        self.value = None
        self.breaches = 0

    def update(self, health: float, health_lo: float | None = None):
        self.value = health if self.value is None else \
            self.alpha * health + (1 - self.alpha) * self.value
        # Alarm on the lower bound when we have one: conservative by construction.
        watched = self.value if health_lo is None else health_lo
        self.breaches = self.breaches + 1 if watched < self.threshold else 0
        return self.value, self.breaches >= self.consecutive
