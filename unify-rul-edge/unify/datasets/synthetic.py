"""Synthetic run-to-failure generator.

Exists so the whole stack — preprocessing, training, export, Pi inference — is
runnable and testable BEFORE the ~15 GB of real datasets are downloaded, and so
CI has something deterministic to run. It is not a scientific contribution and
must never appear in your results table.

It emits TWO pseudo-datasets with different channel counts (8 and 3), because a
model whose universality claim is never exercised by mixed-S batches is a model
whose universality claim is untested.

Degradation model (the classic prognostics "hockey stick"):
    health(t) = 1 - (t/T)^p ,  p in [2, 4]
Flat for most of life, then collapsing — which is exactly why the RUL cap exists.

DO NOT TUNE AGAINST THESE NUMBERS. The task has a large irreducible error by
construction: the sensors encode `damage`, but the label is normalised AGE, and
recovering age from damage requires p, which is drawn per unit and is not
observable. At damage 0.5 the true age is anywhere in [0.71, 0.84] depending on
p, i.e. ~0.19 of normalised life, ~34 in the physical scale. So a val RMSE
around 35-42 IS the floor here, and the RandomForest baseline lands in the same
place. That tells you the harness is sound; it tells you nothing about C-MAPSS,
where degradation is physically driven and temporal structure genuinely pays.
"""
from __future__ import annotations

import numpy as np

from ..common.schema import Unit, sensor_ids_for


def _one_unit(rng, uid: str, dataset: str, domain_id: int,
              channel_names: list[str], n_informative: int) -> Unit:
    S = len(channel_names)
    T = int(rng.integers(120, 400))
    t = np.arange(T)

    p = rng.uniform(2.0, 4.0)
    health = 1.0 - (t / (T - 1)) ** p          # 1.0 healthy -> 0.0 dead
    damage = 1.0 - health

    sig = np.empty((T, S), np.float32)
    for c in range(S):
        base = rng.uniform(-2.0, 2.0)
        noise = rng.normal(0, rng.uniform(0.02, 0.12), T)
        if c < n_informative:
            gain = rng.uniform(0.5, 3.0) * rng.choice([-1.0, 1.0])
            if c % 3 == 0:                      # energy-like: grows exponentially
                trend = gain * (np.exp(2.0 * damage) - 1.0)
            elif c % 3 == 1:                    # kurtosis-like: peaks mid-life, falls back
                trend = gain * np.exp(-((damage - 0.55) ** 2) / 0.02)
            else:                               # slow monotone drift
                trend = gain * damage
            sig[:, c] = base + trend + noise
        else:
            sig[:, c] = base + noise            # pure noise — attention should learn to ignore it

    # A couple of discrete operating conditions, with an offset large enough to
    # matter, so per-condition normalisation has something to do.
    n_cond = 2
    cond = rng.integers(0, n_cond, T)
    for c in range(n_cond):
        sig[cond == c] += rng.uniform(-1.5, 1.5, S).astype(np.float32)

    return Unit(
        unit_id=uid, dataset=dataset, domain_id=domain_id,
        signals=sig, sensor_ids=sensor_ids_for(channel_names),
        rul=np.arange(T - 1, -1, -1, dtype=np.float32),
        life_total=float(T - 1), unit_of_time="step",
        condition=cond.astype(np.int64),
    )


def load(cfg: dict, split: str = "train") -> list[Unit]:
    rng = np.random.default_rng(cfg.get("seed", 0) + (0 if split == "train" else 1))
    n = int(cfg["n_units"])
    names = [f"synthetic/c{i}" for i in range(int(cfg["n_sensors"]))]
    return [_one_unit(rng, f"synthetic_{split}_u{i}", "synthetic",
                      cfg["domain_id"], names, n_informative=max(1, len(names) // 2))
            for i in range(n)]


def load_small(cfg: dict, split: str = "train") -> list[Unit]:
    """Same physics, 3 channels instead of 8 — forces mixed-S batches."""
    rng = np.random.default_rng(cfg.get("seed", 0) + 100 + (0 if split == "train" else 1))
    n = int(cfg["n_units"])
    names = [f"synthetic/c{i}" for i in range(3)]
    return [_one_unit(rng, f"synthetic_small_{split}_u{i}", "synthetic_small",
                      cfg["domain_id"], names, n_informative=2)
            for i in range(n)]
