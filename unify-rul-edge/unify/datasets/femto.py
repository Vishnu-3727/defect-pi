"""FEMTO / PRONOSTIA bearing parser (IEEE PHM 2012 Data Challenge).

Real bearings run under overload on the PRONOSTIA rig until they self-destruct,
typically in 1-7 hours. 17 bearings across 3 operating conditions.

Sampling: 2560 samples at 25.6 kHz (a 0.1 s snapshot) EVERY 10 SECONDS. So the
data is a sequence of snapshots, not a continuous waveform. Each snapshot becomes
one timestep after feature extraction — you cannot feed 2560 raw points per
sensor into a length-30 window.

Layout on disk:
  Learning_set/Bearing1_1/acc_00001.csv, acc_00002.csv, ...
  Full_Test_Set/Bearing1_3/...
CSV columns: hour, minute, second, microsecond, horiz accel (g), vert accel (g)

Failure: the rig stops when vibration exceeds 20 g. Lives vary 5-20x between
bearings under identical conditions — this is the dataset that teaches humility.
Published RUL errors of 20-50% are normal here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.features import snapshot_features
from ..common.schema import Unit, sensor_ids_for

COND_RE = re.compile(r"Bearing(\d)_(\d)")


def _snapshot_paths(bearing_dir: Path) -> list[Path]:
    """Snapshot files in TIME order.

    Sorting lexically puts acc_00010.csv before acc_00002.csv the moment the
    zero-padding is inconsistent, which silently scrambles the time series. The
    RUL-monotonicity assert in Unit.validate() catches it, but sort numerically
    and it never happens.
    """
    paths = list(bearing_dir.glob("acc_*.csv"))
    return sorted(paths, key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))


def _read_snapshot(path: Path) -> np.ndarray:
    """(N, 2) horizontal + vertical acceleration in g.

    Some FEMTO mirrors use ';' as the separator with ',' as the DECIMAL mark
    (European locale). Getting the separator right but not the decimal mark leaves
    pandas with strings like "-0,412", which silently become object dtype and blow
    up at .to_numpy(np.float32). Sniff both.
    """
    head = path.open("r", errors="ignore").readline()
    if head.count(";") > head.count(","):
        df = pd.read_csv(path, sep=";", header=None, decimal=",", engine="c")
    else:
        df = pd.read_csv(path, sep=",", header=None, engine="c")
    return df.iloc[:, -2:].to_numpy(np.float32)


def _truncation_rul(raw: Path) -> dict[str, float]:
    """True RUL (seconds) at truncation for the 11 test bearings, if provided.

    The challenge published these separately from the archive. Drop a two-column
    CSV at data/raw/femto/test_rul.csv (`Bearing1_3,5730`) and the test bearings
    become usable; without it they are skipped, because labelling a truncated
    series as if it ended at RUL 0 is simply wrong and would poison training.
    """
    p = raw / "test_rul.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, header=None, names=["bearing", "rul_s"])
    return {str(b).strip(): float(r) for b, r in zip(df["bearing"], df["rul_s"])}


def load(cfg: dict, split: str = "train") -> list[Unit]:
    raw = Path(cfg["raw"])
    if not raw.exists():
        return []

    tail_by_bearing = _truncation_rul(raw)

    if split == "train":
        bearing_dirs = [d for d in sorted((raw / "Learning_set").glob("Bearing*"))
                        if d.is_dir()]
        # Some mirrors flatten the archive.
        if not bearing_dirs:
            bearing_dirs = sorted(d for d in raw.glob("Bearing*") if d.is_dir())
    else:
        if not tail_by_bearing:
            return []      # see _truncation_rul — better no data than wrong labels
        test_folder = "Test_set" if (raw / "Test_set").exists() else "Full_Test_Set"
        bearing_dirs = [d for d in sorted((raw / test_folder).glob("Bearing*")) if d.is_dir()]

    dt = cfg["sample_period_s"]
    units: list[Unit] = []

    for bd in bearing_dirs:
        snaps = _snapshot_paths(bd)
        if len(snaps) < 2:
            continue

        rows, names = [], None
        for p in snaps:
            f, names = snapshot_features(_read_snapshot(p), ["bearing/h", "bearing/v"])
            rows.append(f)
        sig = np.stack(rows)                       # (T, 10)

        T = len(sig)
        # Learning-set bearings run to failure (tail 0). Test bearings are cut off
        # early, so their remaining life at the last snapshot is the published
        # value, not zero.
        tail = tail_by_bearing.get(bd.name, 0.0)
        rul = (np.arange(T - 1, -1, -1) * dt + tail).astype(np.float32)
        m = COND_RE.search(bd.name)
        cond = int(m.group(1)) - 1 if m else 0

        units.append(Unit(
            unit_id=f"femto_{split}_{bd.name}",
            dataset="femto",
            domain_id=cfg["domain_id"],
            signals=sig,
            sensor_ids=sensor_ids_for(names),
            rul=rul,
            life_total=float((T - 1) * dt + tail),
            unit_of_time=cfg["unit_of_time"],
            condition=np.full(T, cond, np.int64),
        ))
    return units
