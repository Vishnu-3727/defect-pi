"""XJTU-SY bearing parser.

15 LDK UER204 bearings run to COMPLETE failure (not a 20 g cut-off like FEMTO),
across 3 speed/load conditions. Cleaner and better documented than FEMTO — if
your model cannot learn XJTU, it cannot learn anything.

Sampling: 32768 samples at 25.6 kHz (a 1.28 s snapshot) EVERY MINUTE. The longer
snapshot gives far better frequency resolution than FEMTO's 0.1 s if you later
add envelope-spectrum features.

Layout:
  35Hz12kN/Bearing1_1/1.csv, 2.csv, ...     (one file per minute)
  37.5Hz11kN/Bearing2_1/...
  40Hz10kN/Bearing3_1/...
CSV header: Horizontal_vibration_signals, Vertical_vibration_signals
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.features import snapshot_features
from ..common.schema import Unit, sensor_ids_for


def _snapshot_paths(bearing_dir: Path) -> list[Path]:
    # Files are named 1.csv .. 2538.csv — lexical sort would give 1, 10, 100, 2.
    return sorted(bearing_dir.glob("*.csv"),
                  key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))


def load(cfg: dict, split: str = "train") -> list[Unit]:
    """Returns every bearing. XJTU-SY ships no official split — use leave-one-
    bearing-out cross-validation (docs/04-datasets.md), which is what the unit-level
    splitter in joint.py does. With 15 bearings a single random split is
    statistically meaningless."""
    raw = Path(cfg["raw"])
    if not raw.exists():
        return []

    dt = cfg["sample_period_s"]
    units: list[Unit] = []

    for ci, cond_name in enumerate(cfg["conditions"]):
        cond_dir = raw / cond_name
        if not cond_dir.exists():
            continue

        for bd in sorted(d for d in cond_dir.glob("Bearing*") if d.is_dir()):
            snaps = _snapshot_paths(bd)
            if len(snaps) < 2:
                continue

            rows, names = [], None
            for p in snaps:
                df = pd.read_csv(p, engine="c")
                f, names = snapshot_features(df.to_numpy(np.float32)[:, :2],
                                             ["bearing/h", "bearing/v"])
                rows.append(f)
            sig = np.stack(rows)

            T = len(sig)
            rul = (np.arange(T - 1, -1, -1) * dt / 60.0).astype(np.float32)  # minutes

            units.append(Unit(
                unit_id=f"xjtu_{cond_name}_{bd.name}",
                dataset="xjtu",
                domain_id=cfg["domain_id"],
                signals=sig,
                sensor_ids=sensor_ids_for(names),
                rul=rul,
                life_total=float((T - 1) * dt / 60.0),
                unit_of_time=cfg["unit_of_time"],
                condition=np.full(T, ci, np.int64),
            ))
    return units
