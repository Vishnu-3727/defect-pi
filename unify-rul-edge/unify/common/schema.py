"""The one schema every dataset parser must produce.

Everything downstream — normalisation, windowing, the DataLoader, the model —
consumes only `Unit`. Adding a fourth dataset is therefore one parser file, not
a refactor. That is the whole "Any Dataset" claim, made concrete.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class Unit:
    """One machine, from healthy to failure."""

    unit_id: str            # "cmapss_FD001_u17", "xjtu_35Hz12kN_b3"
    dataset: str            # "cmapss" | "femto" | "xjtu" | "synthetic"
    domain_id: int          # index into the model's domain embedding
    signals: np.ndarray     # (T, S) float32 — T timesteps, S channels
    sensor_ids: np.ndarray  # (S,) int64 — global vocabulary IDs
    rul: np.ndarray         # (T,) float32 — remaining life, NATIVE units, uncapped
    life_total: float       # total life in native units
    unit_of_time: str       # "cycle" | "second" | "minute" | "step"
    condition: np.ndarray = field(default=None)  # (T,) int64 — operating condition, 0 if single

    def __post_init__(self):
        self.signals = np.asarray(self.signals, dtype=np.float32)
        self.rul = np.asarray(self.rul, dtype=np.float32)
        self.sensor_ids = np.asarray(self.sensor_ids, dtype=np.int64)
        if self.condition is None:
            self.condition = np.zeros(len(self.rul), dtype=np.int64)
        self.validate()

    def validate(self):
        T, S = self.signals.shape
        assert len(self.rul) == T, f"{self.unit_id}: rul len {len(self.rul)} != T {T}"
        assert len(self.sensor_ids) == S, f"{self.unit_id}: sensor_ids {len(self.sensor_ids)} != S {S}"
        assert len(self.condition) == T, f"{self.unit_id}: condition len mismatch"
        assert np.isfinite(self.signals).all(), f"{self.unit_id}: non-finite signal values"
        # RUL must count down. A parser that sorts snapshot files lexically
        # ("acc_10.csv" before "acc_2.csv") scrambles the series and trips this.
        assert np.all(np.diff(self.rul) <= 0), f"{self.unit_id}: RUL is not monotonically decreasing"

    @property
    def T(self) -> int:
        return self.signals.shape[0]

    @property
    def S(self) -> int:
        return self.signals.shape[1]

    def capped_rul(self, cap_mode: str, cap: float) -> np.ndarray:
        """Piecewise-linear RUL, normalised to [0,1]. 1.0 = healthy, 0.0 = dead.

        A healthy engine at cycle 1 and cycle 40 look identical to the sensors, so
        asking for different outputs is unlearnable noise. Capping encodes
        "degradation is undetectable while healthy". docs/05-preprocessing.md §5.5
        """
        if cap_mode == "absolute":                    # cycles, e.g. C-MAPSS cap=125
            return np.minimum(self.rul, cap) / cap
        if cap_mode == "fraction":                    # bearings, cap = fraction of own life
            frac = self.rul / max(self.life_total, 1e-9)
            return np.minimum(frac, cap) / cap
        raise ValueError(f"unknown cap_mode {cap_mode!r}")

    def life_scale(self, cap_mode: str, cap: float) -> float:
        """Multiplier that turns normalised RUL back into native time units."""
        return cap if cap_mode == "absolute" else cap * self.life_total

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, signals=self.signals, sensor_ids=self.sensor_ids, rul=self.rul,
            condition=self.condition,
            meta=np.array(json.dumps({
                "unit_id": self.unit_id, "dataset": self.dataset,
                "domain_id": self.domain_id, "life_total": self.life_total,
                "unit_of_time": self.unit_of_time})))

    @classmethod
    def load(cls, path: Path) -> "Unit":
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return cls(signals=z["signals"], sensor_ids=z["sensor_ids"], rul=z["rul"],
                   condition=z["condition"], **m)


def load_sensor_vocab() -> dict[str, int]:
    vocab = json.loads((CONFIG_DIR / "sensor_vocab.json").read_text())
    return {k: v for k, v in vocab.items() if not k.startswith("_")}


def sensor_ids_for(names: list[str]) -> np.ndarray:
    vocab = load_sensor_vocab()
    missing = [n for n in names if n not in vocab]
    if missing:
        raise KeyError(f"not in config/sensor_vocab.json: {missing}")
    return np.array([vocab[n] for n in names], dtype=np.int64)
