"""Parser correctness, using tiny synthesised copies of each dataset's on-disk
format. These pin bugs that only show up once the real 15 GB is downloaded —
by which point they are expensive to find.
"""
import numpy as np
import pandas as pd
import pytest

from unify.datasets import cmapss, femto, xjtu
from unify.datasets.joint import (WindowDataset, hold_out_official_test,
                                  split_units)
from unify.common.normalize import Normalizer

CMAPSS_CFG = {"raw": "", "subsets": ["FD002"], "domain_id": 0, "unit_of_time": "cycle",
              "cap_mode": "absolute", "cap": 125.0, "drop_constant_sensors": True,
              "var_threshold": 1e-6}


def write_cmapss(dirpath, conditions_train, conditions_test, n_units=3, n_cycles=40):
    """Emit train/test/RUL files in the real 26-column space-separated format."""
    rng = np.random.default_rng(0)

    def block(conds):
        rows = []
        for u in range(1, n_units + 1):
            for c in range(1, n_cycles + 1):
                op = conds[(u + c) % len(conds)]
                sensors = list(rng.normal(500, 10, 21))
                sensors[0] = 518.67           # constant -> must be dropped
                rows.append([u, c, *op, *sensors])
        return pd.DataFrame(rows)

    block(conditions_train).to_csv(dirpath / "train_FD002.txt", sep=" ",
                                   header=False, index=False)
    block(conditions_test).to_csv(dirpath / "test_FD002.txt", sep=" ",
                                  header=False, index=False)
    np.savetxt(dirpath / "RUL_FD002.txt", np.array([20, 35, 50]), fmt="%d")


def test_cmapss_drops_constant_sensors_and_labels_rul(tmp_path):
    write_cmapss(tmp_path, [(0.0, 0.0, 100.0)], [(0.0, 0.0, 100.0)])
    cfg = {**CMAPSS_CFG, "raw": str(tmp_path)}
    units = cmapss.load(cfg, "train")

    assert len(units) == 3
    assert units[0].S == 20, "the constant sensor should have been dropped"
    assert units[0].rul[-1] == 0.0, "train units run to failure"
    assert units[0].rul[0] == units[0].T - 1


def test_cmapss_test_units_carry_the_published_truncation_rul(tmp_path):
    write_cmapss(tmp_path, [(0.0, 0.0, 100.0)], [(0.0, 0.0, 100.0)])
    units = cmapss.load({**CMAPSS_CFG, "raw": str(tmp_path)}, "test")
    # Test series are cut off before failure, so the last cycle is NOT RUL 0.
    assert [u.rul[-1] for u in units] == [20.0, 35.0, 50.0]


def test_cmapss_condition_ids_are_stable_between_train_and_test(tmp_path):
    """The bug this pins: pd.factorize numbers conditions by order of first
    appearance, which differs between the two files. Condition 3 in training would
    silently mean something else at test time, and per-condition normalisation
    would be applied to the wrong rows.
    """
    conds = [(0.0, 0.0, 100.0), (20.0, 0.7, 100.0), (42.0, 0.84, 100.0)]
    # Same three conditions, deliberately encountered in a DIFFERENT order.
    write_cmapss(tmp_path, conds, list(reversed(conds)))
    cfg = {**CMAPSS_CFG, "raw": str(tmp_path)}

    tr = cmapss.load(cfg, "train")
    te = cmapss.load(cfg, "test")
    # Reconstruct the mapping from the raw files and check both splits agree.
    raw_tr = cmapss._read(tmp_path / "train_FD002.txt")
    raw_te = cmapss._read(tmp_path / "test_FD002.txt")
    key_tr = cmapss._condition_keys(raw_tr)
    key_te = cmapss._condition_keys(raw_te)

    cond_tr = np.concatenate([u.condition for u in tr])
    cond_te = np.concatenate([u.condition for u in te])
    map_tr = dict(zip(key_tr, cond_tr))
    map_te = dict(zip(key_te, cond_te))
    assert map_tr == map_te, f"condition IDs disagree: {map_tr} vs {map_te}"


def test_cmapss_condition_keys_survive_signed_zero_and_rounding_boundaries(tmp_path):
    """Two float traps that only appear on the real files.

    FD001/FD003 have a single operating regime, but the op columns jitter by
    +-0.0087 around zero. Rounding to one decimal produced -0.0 for some rows and
    0.0 for others, and str(-0.0) != str(0.0), so one regime became four.

    FD002/FD004 regime 2 has op2 = 0.25, sitting exactly on a one-decimal
    rounding boundary, so banker's rounding split it into 0.2 and 0.3 and the six
    documented regimes became seven.
    """
    jitter = [(-0.0007, 0.0002, 100.0), (0.0087, -0.0006, 100.0),
              (-0.0087, 0.0006, 100.0)]
    write_cmapss(tmp_path, jitter, jitter)
    keys = cmapss._condition_keys(cmapss._read(tmp_path / "train_FD002.txt"))
    assert keys.nunique() == 1, f"single regime split into {sorted(keys.unique())}"

    boundary = [(0.0, 0.0, 100.0), (10.0, 0.2503, 100.0), (10.0, 0.2497, 100.0)]
    write_cmapss(tmp_path, boundary, boundary)
    keys = cmapss._condition_keys(cmapss._read(tmp_path / "train_FD002.txt"))
    assert keys.nunique() == 2, f"op2=0.25 split: {sorted(keys.unique())}"


def test_official_test_units_are_held_out_of_training(tmp_path):
    """Every published C-MAPSS RMSE is measured on test_FDxxx.txt. The parser
    caches train and test side by side, so without this the random 80/20 split
    would train on the very engines the reported number holds out.
    """
    write_cmapss(tmp_path, [(0.0, 0.0, 100.0)], [(0.0, 0.0, 100.0)])
    cfg = {**CMAPSS_CFG, "raw": str(tmp_path)}
    units = cmapss.load(cfg, "train") + cmapss.load(cfg, "test")

    pool, held = hold_out_official_test(units)
    assert held, "no official test units were recognised"
    assert len(pool) + len(held) == len(units)
    assert all("_test_" in u.unit_id for u in held)
    assert not any("_test_" in u.unit_id for u in pool)

    # …and they must not leak back in through the random split.
    train_units, val_units = split_units(pool, 0.2, seed=0)
    ids = {u.unit_id for u in train_units} | {u.unit_id for u in val_units}
    assert ids.isdisjoint({u.unit_id for u in held})


def test_last_window_only_gives_exactly_one_prediction_per_unit(tmp_path):
    """The published C-MAPSS protocol is one prediction per test engine, at its
    final cycle. Scoring all ~120 windows per engine is not comparable to any
    paper and inflates the NASA score by the window count."""
    write_cmapss(tmp_path, [(0.0, 0.0, 100.0)], [(0.0, 0.0, 100.0)], n_units=3)
    cfg = {**CMAPSS_CFG, "raw": str(tmp_path)}
    units = cmapss.load(cfg, "test")
    norm = Normalizer().fit(units)

    full = WindowDataset(units, norm, {"cmapss": cfg}, 30, 1)
    last = WindowDataset(units, norm, {"cmapss": cfg}, 30, 1, last_window_only=True)

    assert len(last) == len(units) < len(full)
    assert len({i["unit"] for i in last}) == len(units)
    # The kept window must be the final one, so its label is the smallest.
    for u in units:
        mine = [i["y"] for i in full.items if i["unit"] == u.unit_id]
        kept = [i["y"] for i in last.items if i["unit"] == u.unit_id]
        assert kept == [min(mine)]


def test_hold_out_is_a_no_op_for_datasets_without_a_published_split(tmp_path):
    """XJTU has no official test set, so every unit must fall through to the
    ordinary random split rather than being silently held out."""
    cond = "35Hz12kN"
    d = tmp_path / cond / "Bearing1_1"
    d.mkdir(parents=True)
    rng = np.random.default_rng(4)
    for k in range(1, 8):
        pd.DataFrame({"Horizontal_vibration_signals": rng.normal(0, 0.1, 128),
                      "Vertical_vibration_signals": rng.normal(0, 0.1, 128)}
                     ).to_csv(d / f"{k}.csv", index=False)

    units = xjtu.load({"raw": str(tmp_path), "domain_id": 2, "unit_of_time": "minute",
                       "sample_period_s": 60.0, "conditions": [cond],
                       "cap_mode": "fraction", "cap": 0.7}, "train")
    assert units, "fixture produced no units"

    pool, held = hold_out_official_test(units)
    assert held == []
    assert len(pool) == len(units)


def test_synthetic_test_units_are_held_out_and_are_genuinely_unseen():
    """The generators emit "<name>_test_uN" ids, so the hold-out catches them.

    That is correct rather than an accident of naming: load(split="test") offsets
    the RNG seed, so those units are independent draws, not the training ones
    relabelled. If they ever became the same draw, holding them out would be
    measuring memorisation.
    """
    from unify.datasets import synthetic
    cfg = {"domain_id": 4, "unit_of_time": "step", "n_units": 4, "n_sensors": 3,
           "seed": 0}
    tr = synthetic.load(cfg, "train")
    te = synthetic.load(cfg, "test")

    pool, held = hold_out_official_test(tr + te)
    assert len(held) == len(te)
    assert len(pool) == len(tr)
    # Independent draws, not a relabelling. Compare over the overlapping prefix:
    # the two units generally differ in length as well, which is itself evidence,
    # but np.allclose cannot broadcast mismatched shapes.
    n = min(tr[0].T, te[0].T)
    assert not np.allclose(tr[0].signals[:n], te[0].signals[:n])


def test_cmapss_handles_a_single_test_unit(tmp_path):
    """np.loadtxt returns a 0-d scalar for a one-line RUL file, which used to raise
    IndexError on rul_final[i]."""
    write_cmapss(tmp_path, [(0.0, 0.0, 100.0)], [(0.0, 0.0, 100.0)], n_units=1)
    np.savetxt(tmp_path / "RUL_FD002.txt", np.array([42]), fmt="%d")
    units = cmapss.load({**CMAPSS_CFG, "raw": str(tmp_path)}, "test")
    assert len(units) == 1 and units[0].rul[-1] == 42.0


# ── FEMTO ─────────────────────────────────────────────────────────────────────
FEMTO_CFG = {"raw": "", "domain_id": 1, "unit_of_time": "second",
             "sample_period_s": 10.0, "snapshot_len": 64, "fs_hz": 25600,
             "cap_mode": "fraction", "cap": 0.7}


def write_femto(root, name, n_snaps, sep=",", decimal="."):
    d = root / "Learning_set" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for k in range(1, n_snaps + 1):
        acc = rng.normal(0, 0.05 + 0.02 * k, (64, 2))
        df = pd.DataFrame({0: 0, 1: 0, 2: k, 3: 0, 4: acc[:, 0], 5: acc[:, 1]})
        df.to_csv(d / f"acc_{k:05d}.csv", sep=sep, decimal=decimal,
                  header=False, index=False)
    return d


def test_femto_extracts_features_and_orders_snapshots(tmp_path):
    write_femto(tmp_path, "Bearing1_1", 12)
    units = femto.load({**FEMTO_CFG, "raw": str(tmp_path)}, "train")
    assert len(units) == 1
    u = units[0]
    assert u.T == 12 and u.S == 10          # 2 channels x 5 features
    assert u.rul[-1] == 0.0 and u.rul[0] == 110.0


def test_femto_reads_european_decimal_commas(tmp_path):
    """Some mirrors ship ';'-separated files with ',' as the decimal mark. Getting
    the separator right but not the decimal leaves pandas holding strings."""
    write_femto(tmp_path, "Bearing1_1", 5, sep=";", decimal=",")
    units = femto.load({**FEMTO_CFG, "raw": str(tmp_path)}, "train")
    assert len(units) == 1
    assert np.isfinite(units[0].signals).all()
    assert units[0].signals.std() > 0, "values parsed as constants — decimal mark ignored"


def test_femto_skips_truncated_test_bearings_without_published_rul(tmp_path):
    """Labelling a truncated series as if it ended at RUL 0 is simply wrong and
    would poison training. Better to return nothing."""
    d = tmp_path / "Full_Test_Set" / "Bearing1_3"
    d.mkdir(parents=True)
    rng = np.random.default_rng(2)
    for k in range(1, 6):
        pd.DataFrame(rng.normal(0, 0.1, (64, 6))).to_csv(
            d / f"acc_{k:05d}.csv", header=False, index=False)

    cfg = {**FEMTO_CFG, "raw": str(tmp_path)}
    assert femto.load(cfg, "test") == []

    (tmp_path / "test_rul.csv").write_text("Bearing1_3,5730\n")
    units = femto.load(cfg, "test")
    assert len(units) == 1 and units[0].rul[-1] == 5730.0


def test_femto_snapshot_order_is_numeric_not_lexical(tmp_path):
    """acc_10.csv must not sort before acc_2.csv. The RUL-monotonicity assert in
    Unit.validate() is the backstop, but ordering is what makes it correct."""
    d = tmp_path / "Learning_set" / "Bearing1_1"
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for k in (1, 2, 10, 11):                       # inconsistent padding on purpose
        pd.DataFrame(rng.normal(0, 0.1, (64, 6))).to_csv(
            d / f"acc_{k}.csv", header=False, index=False)
    paths = femto._snapshot_paths(d)
    assert [p.stem for p in paths] == ["acc_1", "acc_2", "acc_10", "acc_11"]


# ── XJTU ──────────────────────────────────────────────────────────────────────
def test_xjtu_parses_and_converts_to_minutes(tmp_path):
    cond = "35Hz12kN"
    d = tmp_path / cond / "Bearing1_1"
    d.mkdir(parents=True)
    rng = np.random.default_rng(4)
    for k in range(1, 8):
        pd.DataFrame({"Horizontal_vibration_signals": rng.normal(0, 0.1, 128),
                      "Vertical_vibration_signals": rng.normal(0, 0.1, 128)}
                     ).to_csv(d / f"{k}.csv", index=False)

    units = xjtu.load({"raw": str(tmp_path), "domain_id": 2, "unit_of_time": "minute",
                       "sample_period_s": 60.0, "conditions": [cond],
                       "cap_mode": "fraction", "cap": 0.7}, "train")
    assert len(units) == 1
    u = units[0]
    assert u.T == 7 and u.S == 10
    assert u.rul[0] == 6.0 and u.rul[-1] == 0.0     # minutes, not seconds


# ── life scale ────────────────────────────────────────────────────────────────
def test_life_scale_ignores_units_too_short_to_window():
    """Units shorter than L contribute no windows, so they must not be counted in
    the average — doing so deflates the scale and every physical-unit metric."""
    from unify.common.schema import Unit

    def u(uid, T):
        return Unit(unit_id=uid, dataset="synthetic", domain_id=4,
                    signals=np.zeros((T, 2), np.float32),
                    sensor_ids=np.array([50, 51]),
                    rul=np.arange(T - 1, -1, -1, dtype=np.float32),
                    life_total=float(T - 1), unit_of_time="step")

    units = [u("long1", 101), u("long2", 101), u("tiny", 5)]
    norm = Normalizer().fit(units)
    ds = WindowDataset(units, norm, {"synthetic": {"cap_mode": "fraction", "cap": 0.7}},
                       L=30, stride=1)
    assert ds.scales["synthetic"] == pytest.approx(0.7 * 100)
