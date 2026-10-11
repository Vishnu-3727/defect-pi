"""Z-score normalisation, fit on TRAIN UNITS ONLY, serialised for the Pi.

Statistics are keyed by (dataset, sensor_id) — NOT by channel position. That
matters more than it looks:

  * C-MAPSS FD001 keeps 14 sensors after the variance filter while FD002 keeps
    more. Positional statistics cannot represent both under one dataset name.
  * The live rig sends a SUBSET of the trained bearing channels, in its own
    order. With ID-keyed statistics that is just a lookup.

Two rules, both of which fail silently when broken:

1. Fit on train only. Computing statistics over the full dataset before splitting
   leaks test information and inflates your results.
2. Ship the fitted statistics to the Pi and reuse them verbatim. Recomputing them
   from live data normalises away the degradation trend you are trying to detect —
   no exception, no NaN, just permanently wrong predictions.

docs/03-system-architecture.md §3.4, docs/05-preprocessing.md §5.2
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-8


def _finish(acc: dict) -> dict:
    """{sensor_id: [n, sum, sumsq]} -> {"<id>": {"mean": m, "std": s}}"""
    out = {}
    for sid, (n, s, ss) in acc.items():
        mean = s / n
        var = max(ss / n - mean * mean, 0.0)          # clamp float error at zero
        out[str(int(sid))] = {"mean": float(mean), "std": float(np.sqrt(var) + EPS)}
    return out


class Normalizer:
    """Per-(dataset, sensor-ID) z-score, with optional per-operating-condition stats.

    C-MAPSS FD002/FD004 run under 6 operating conditions whose effect on the
    sensors is LARGER than the degradation signal. Global z-scoring buries the
    trend in condition variance; per-condition z-scoring recovers it and is the
    single highest-leverage preprocessing step on those subsets.
    """

    def __init__(self):
        self.stats: dict[str, dict] = {}

    def fit(self, units: list, per_condition: set[str] = frozenset()):
        # dataset -> "global"/cond -> sensor_id -> [n, sum, sumsq]
        acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: [0.0, 0.0, 0.0])))

        for u in units:
            sig = u.signals.astype(np.float64)
            buckets = [("global", slice(None))]
            if u.dataset in per_condition:
                buckets += [(str(int(c)), u.condition == c) for c in np.unique(u.condition)]

            for key, sel in buckets:
                block = sig[sel]
                if len(block) == 0:
                    continue
                n = float(len(block))
                col_sum = block.sum(0)
                col_sq = (block ** 2).sum(0)
                target = acc[u.dataset][key]
                for j, sid in enumerate(u.sensor_ids):
                    e = target[int(sid)]
                    e[0] += n
                    e[1] += col_sum[j]
                    e[2] += col_sq[j]

        self.stats = {}
        for ds, keys in acc.items():
            entry = {"per_condition": ds in per_condition,
                     "global": _finish(keys["global"])}
            if entry["per_condition"]:
                entry["cond"] = {k: _finish(v) for k, v in keys.items() if k != "global"}
            self.stats[ds] = entry
        return self

    # ── lookup ────────────────────────────────────────────────────────────────
    def _table(self, dataset: str, cond: str | None) -> dict:
        e = self.stats[dataset]
        if cond is not None and e.get("per_condition"):
            # An operating condition never seen in training falls back to global
            # rather than crashing — the model still gets a sane input.
            return e["cond"].get(cond, e["global"])
        return e["global"]

    def vector_for(self, dataset: str, sensor_ids, cond: str | None = None):
        """(mean, std) arrays aligned to an arbitrary subset of channels, in order."""
        table = self._table(dataset, cond)
        missing = [int(s) for s in sensor_ids if str(int(s)) not in table]
        if missing:
            raise KeyError(
                f"{dataset}: no training statistics for sensor IDs {missing} "
                f"(have {sorted(int(k) for k in table)})")
        mean = np.array([table[str(int(s))]["mean"] for s in sensor_ids], np.float32)
        std = np.array([table[str(int(s))]["std"] for s in sensor_ids], np.float32)
        return mean, std

    def apply(self, unit) -> np.ndarray:
        e = self.stats[unit.dataset]
        if not e.get("per_condition"):
            mean, std = self.vector_for(unit.dataset, unit.sensor_ids)
            return ((unit.signals - mean) / std).astype(np.float32)

        out = np.empty_like(unit.signals)
        for c in np.unique(unit.condition):
            m = unit.condition == c
            mean, std = self.vector_for(unit.dataset, unit.sensor_ids, str(int(c)))
            out[m] = (unit.signals[m] - mean) / std
        return out.astype(np.float32)

    # ── serialisation — this file travels to the Pi next to the .onnx ─────────
    def save(self, path: Path, extra: dict | None = None):
        extra = extra or {}
        # "datasets" holds the per-channel statistics. Letting a caller's extra key
        # shadow it produces a file that looks fine and breaks the Pi at load time.
        assert "datasets" not in extra, "'datasets' is reserved for the channel statistics"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"datasets": self.stats, **extra}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        n = cls()
        n.stats = json.loads(Path(path).read_text())["datasets"]
        return n
