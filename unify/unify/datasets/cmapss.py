"""NASA C-MAPSS turbofan parser.

Simulated (not measured) turbofan degradation from NASA's C-MAPSS Simulink model.
The field benchmark — every RUL paper since 2008 reports on it.

Files per subset: train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt
Columns (26, space separated, no header):
  1     unit number
  2     time in cycles
  3-5   operational settings 1-3 (altitude, Mach, throttle resolver angle)
  6-26  sensor measurements 1-21

Labels: train units run to failure, so RUL = max_cycle - cycle. Test units are
truncated before failure and RUL_FD00X.txt gives the true RUL at each one's last
recorded cycle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.schema import Unit, sensor_ids_for

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
N_CONDITIONS = 6  # FD002/FD004 — the ops columns take 6 discrete values


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df


def _condition_keys(df: pd.DataFrame) -> pd.Series:
    """Quantise the three operating-setting columns into a discrete regime label.

    Rounding to integers, not to 1 decimal, and adding 0.0. Both matter, and both
    were found by running this on the real files rather than by reading it:

    * FD001/FD003 jitter around zero by +-0.0087. `round(1)` maps that to -0.0 for
      half the rows and 0.0 for the other half, and `str(-0.0) != str(0.0)`, so a
      single-regime subset came out with FOUR phantom conditions. Adding 0.0
      collapses signed zero (IEEE 754: -0.0 + 0.0 == +0.0).
    * FD002/FD004 regime 2 has op2 = 0.25, exactly on a 1-decimal rounding
      boundary, so it split into 0.2 and 0.3 and the six documented regimes came
      out as seven.

    Integers give exactly 1/6/1/6 regimes across FD001-FD004, and the train and
    test files produce identical key sets, so no row falls through to the -1
    global-statistics fallback. Pinned by tests/test_datasets.py.
    """
    ops = df[["op1", "op2", "op3"]].round(0) + 0.0
    return ops.astype(int).astype(str).agg("|".join, axis=1)


def _condition_map(train_df: pd.DataFrame) -> dict[str, int]:
    """Fix the condition-ID mapping from the TRAINING file, once per subset."""
    return {k: i for i, k in enumerate(sorted(_condition_keys(train_df).unique()))}


def _cluster_conditions(df: pd.DataFrame, order: dict[str, int]) -> np.ndarray:
    """Label each row with its operating condition.

    FD002/FD004 sensor readings move MORE with operating condition than with
    degradation, so per-condition normalisation is essential there. The three ops
    columns take exactly 6 discrete values, so rounding is enough — no KMeans
    needed (docs/05-preprocessing.md §5.2).

    The mapping comes from the TRAINING file and is reused for test, rather than
    from pd.factorize per file. factorize numbers by order of first appearance, so
    condition 3 in training would have silently meant something else at test time
    and the per-condition statistics would have been applied to the wrong rows.
    A condition absent from training maps to -1 and the Normalizer falls back to
    its global statistics.
    """
    return _condition_keys(df).map(order).fillna(-1).to_numpy(np.int64)


def _usable_sensors(df: pd.DataFrame, threshold: float) -> list[str]:
    """Drop zero-variance channels.

    In FD001, sensors 1, 5, 10, 16, 18 and 19 never change — they carry no
    information. Detected rather than hard-coded so FD002-FD004 (where some of
    them DO vary) are handled automatically.
    """
    cols = [f"s{i}" for i in range(1, 22)]
    return [c for c in cols if df[c].var() > threshold]


def load(cfg: dict, split: str = "train") -> list[Unit]:
    raw = Path(cfg["raw"])
    units: list[Unit] = []

    for sub in cfg["subsets"]:
        train_path = raw / f"train_{sub}.txt"
        if not train_path.exists():
            continue

        # Sensor selection is decided on the TRAINING file only, then applied to
        # both splits — otherwise the test set influences preprocessing.
        train_df = _read(train_path)
        sensors = (_usable_sensors(train_df, cfg["var_threshold"])
                   if cfg["drop_constant_sensors"]
                   else [f"s{i}" for i in range(1, 22)])
        sids = sensor_ids_for([f"cmapss/{c}" for c in sensors])

        order = _condition_map(train_df)

        if split == "train":
            df, rul_final = train_df, None
        else:
            p = raw / f"test_{sub}.txt"
            if not p.exists():
                continue
            df = _read(p)
            rul_path = raw / f"RUL_{sub}.txt"
            # atleast_1d: loadtxt returns a 0-d scalar for a single-unit file,
            # which would make rul_final[i] raise IndexError.
            rul_final = np.atleast_1d(np.loadtxt(rul_path)) if rul_path.exists() else None

        df = df.assign(_cond=_cluster_conditions(df, order))

        for i, (uid, g) in enumerate(df.groupby("unit", sort=True)):
            g = g.sort_values("cycle")
            n = len(g)
            # Test series are truncated, so their remaining life at the final
            # recorded cycle is whatever RUL_FD00X.txt says, not zero.
            tail = 0.0 if rul_final is None else float(rul_final[i])
            rul = np.arange(n - 1, -1, -1, dtype=np.float32) + tail

            units.append(Unit(
                unit_id=f"cmapss_{sub}_{split}_u{int(uid)}",
                dataset="cmapss",
                domain_id=cfg["domain_id"],
                signals=g[sensors].to_numpy(np.float32),
                sensor_ids=sids,
                rul=rul,
                life_total=float(n - 1 + tail),
                unit_of_time=cfg["unit_of_time"],
                condition=g["_cond"].to_numpy(np.int64),
            ))
    return units
