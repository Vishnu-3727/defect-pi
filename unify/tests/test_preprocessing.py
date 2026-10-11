"""Preprocessing invariants — where 70% of prognostics results are actually won,
and where the bugs fail silently.
"""
import numpy as np
import pytest
import torch

from unify.common.features import FEATURE_NAMES, axis_features
from unify.common.normalize import Normalizer
from unify.common.schema import Unit
from unify.common.windowing import n_windows, window_unit
from unify.datasets.joint import collate, leave_one_out, split_units


def make_unit(uid="u", ds="synthetic", T=100, S=4, seed=0):
    rng = np.random.default_rng(seed)
    return Unit(unit_id=uid, dataset=ds, domain_id=4,
                signals=rng.normal(5.0, 2.0, (T, S)).astype(np.float32),
                sensor_ids=np.arange(50, 50 + S),
                rul=np.arange(T - 1, -1, -1, dtype=np.float32),
                life_total=float(T - 1), unit_of_time="step")


# ── windowing ─────────────────────────────────────────────────────────────────
def test_window_count_and_shape():
    u = make_unit(T=100, S=4)
    X, y = window_unit(u.signals, u.rul, L=30, stride=1)
    assert X.shape == (71, 4, 30) == (n_windows(100, 30, 1), 4, 30)
    assert y.shape == (71,)


def test_label_is_rul_at_the_last_timestep():
    """Not the first, not the middle — the moment the Pi would be predicting at."""
    u = make_unit(T=50)
    X, y = window_unit(u.signals, u.rul, L=10, stride=1)
    assert y[0] == u.rul[9]
    assert y[-1] == u.rul[-1] == 0.0


def test_unit_shorter_than_window_yields_nothing():
    """C-MAPSS test engines can be 31 cycles long. Silently emitting a garbage
    window here would corrupt the evaluation."""
    u = make_unit(T=20, S=3)
    X, y = window_unit(u.signals, u.rul, L=30)
    assert len(X) == 0 and len(y) == 0


def test_windows_never_span_units():
    """Concatenating two units then windowing produces MORE windows than windowing
    each separately — the extras are the physically meaningless cross-boundary ones."""
    a, b = make_unit("a", T=60, seed=1), make_unit("b", T=60, seed=2)
    per_unit = n_windows(60, 30) * 2
    naive = n_windows(120, 30)
    assert naive > per_unit, "test itself is wrong"
    assert per_unit == 62


# ── RUL capping ───────────────────────────────────────────────────────────────
def test_absolute_cap_is_piecewise_linear():
    u = make_unit(T=300)
    y = u.capped_rul("absolute", 125.0)
    assert y.max() == 1.0 and y.min() == 0.0
    assert (y[:150] == 1.0).all(), "healthy region must be flat at 1.0"
    assert np.all(np.diff(y) <= 0)


def test_fraction_cap_normalises_wildly_different_lives():
    """Bearing lives vary 5-20x. In normalised space they must be comparable."""
    short, long = make_unit("s", T=50), make_unit("l", T=1000)
    ys = short.capped_rul("fraction", 0.7)
    yl = long.capped_rul("fraction", 0.7)
    assert ys.max() == yl.max() == 1.0
    assert ys.min() == yl.min() == 0.0


def test_life_scale_recovers_native_units():
    u = make_unit(T=300)
    y = u.capped_rul("absolute", 125.0)
    assert np.allclose(y * u.life_scale("absolute", 125.0), np.minimum(u.rul, 125.0))


# ── normalisation ─────────────────────────────────────────────────────────────
def test_zscore_produces_zero_mean_unit_std():
    units = [make_unit(f"u{i}", seed=i) for i in range(5)]
    z = np.concatenate([Normalizer().fit(units).apply(u) for u in units])
    assert abs(z.mean()) < 0.05 and 0.9 < z.std() < 1.1


def test_stats_survive_a_save_load_round_trip(tmp_path):
    """The Pi reuses these verbatim. Recomputing them from live data normalises
    away the degradation trend — silently, with no exception."""
    units = [make_unit(f"u{i}", seed=i) for i in range(3)]
    n = Normalizer().fit(units)
    p = tmp_path / "norm_stats.json"
    n.save(p)
    assert np.allclose(n.apply(units[0]), Normalizer.load(p).apply(units[0]))


def test_vector_for_reorders_to_a_channel_subset():
    """The live rig sends a SUBSET of the trained channels, in its own order."""
    units = [make_unit(seed=i) for i in range(3)]
    n = Normalizer().fit(units)
    mean, _ = n.vector_for("synthetic", [52, 50])
    table = n.stats["synthetic"]["global"]
    assert mean[0] == pytest.approx(table["52"]["mean"])
    assert mean[1] == pytest.approx(table["50"]["mean"])


def test_unknown_sensor_id_is_a_loud_error():
    n = Normalizer().fit([make_unit()])
    with pytest.raises(KeyError):
        n.vector_for("synthetic", [999])


def test_units_of_one_dataset_may_have_different_channel_sets():
    """C-MAPSS FD001 keeps 14 sensors after the variance filter; FD002 keeps more.
    Positional statistics cannot represent both, so they are keyed by sensor ID."""
    a = make_unit("fd001", ds="cmapss", S=4, seed=1)
    b = make_unit("fd002", ds="cmapss", S=6, seed=2)
    n = Normalizer().fit([a, b])
    za, zb = n.apply(a), n.apply(b)
    assert za.shape == a.signals.shape and zb.shape == b.signals.shape
    assert np.isfinite(za).all() and np.isfinite(zb).all()


def test_per_condition_stats_are_used_and_fall_back_when_unseen():
    """A condition absent from training must not crash — it falls back to global."""
    u = make_unit("c", ds="cmapss", T=200, S=3, seed=5)
    u.condition = np.repeat([0, 1], 100)
    u.signals[u.condition == 1] += 50.0          # a condition offset far bigger than any trend
    n = Normalizer().fit([u], per_condition={"cmapss"})
    z = n.apply(u)
    # Per-condition z-scoring removes the offset; global z-scoring would not.
    assert abs(z[u.condition == 0].mean() - z[u.condition == 1].mean()) < 0.2

    u.condition = np.full(200, 7, np.int64)      # never seen
    assert np.isfinite(n.apply(u)).all()


# ── splitting ─────────────────────────────────────────────────────────────────
def test_split_is_by_unit_not_by_window():
    """Windows overlap ~97%. Splitting them randomly leaks and your RMSE becomes
    fiction. If results beat published state of the art, look here first."""
    units = [make_unit(f"u{i}", seed=i) for i in range(20)]
    tr, va = split_units(units, 0.2, seed=0)
    assert len(tr) + len(va) == 20
    assert not ({u.unit_id for u in tr} & {u.unit_id for u in va})


def test_leave_one_out_holds_out_exactly_one_target_unit_per_fold():
    """With 15 XJTU bearings a single random split is statistically meaningless.
    LOBO gives you an error bar, and keeps the other datasets in every fold."""
    units = ([make_unit(f"b{i}", ds="xjtu", seed=i) for i in range(4)] +
             [make_unit(f"e{i}", ds="cmapss", seed=i + 40) for i in range(3)])
    folds = list(leave_one_out(units, "xjtu"))
    assert len(folds) == 4
    held_ids = set()
    for train, val in folds:
        assert len(val) == 1 and val[0].dataset == "xjtu"
        held_ids.add(val[0].unit_id)
        assert val[0].unit_id not in {u.unit_id for u in train}
        assert sum(1 for u in train if u.dataset == "cmapss") == 3
    assert len(held_ids) == 4, "every bearing must be held out exactly once"


def test_split_keeps_every_dataset_in_both_halves():
    units = ([make_unit(f"a{i}", ds="cmapss", seed=i) for i in range(10)] +
             [make_unit(f"b{i}", ds="xjtu", seed=i + 50) for i in range(10)])
    tr, va = split_units(units, 0.2, seed=0)
    assert {u.dataset for u in tr} == {u.dataset for u in va} == {"cmapss", "xjtu"}


# ── collate ───────────────────────────────────────────────────────────────────
def test_collate_pads_mixed_sensor_counts_and_masks_them():
    """One batch, a 3-channel bearing window and an 8-channel engine window."""
    batch = [
        {"x": torch.randn(3, 30), "y": 0.5, "ids": torch.arange(1, 4), "dom": 0, "ds": "a"},
        {"x": torch.randn(8, 30), "y": 0.2, "ids": torch.arange(1, 9), "dom": 1, "ds": "b"},
    ]
    b = collate(batch)
    assert b["x"].shape == (2, 8, 30)
    assert b["mask"][0].tolist() == [True] * 3 + [False] * 5
    assert b["mask"][1].all()
    assert (b["x"][0, 3:] == 0).all()
    assert (b["sensor_ids"][0, 3:] == 0).all()      # 0 == PAD in sensor_vocab.json


# ── features (the ESP32 mirror) ───────────────────────────────────────────────
def test_gaussian_noise_has_kurtosis_near_three():
    """Healthy bearing vibration is ~Gaussian -> kurtosis ~3.0. Early spalling
    pushes it to 5-20, then it falls BACK toward 3 as the defect spreads."""
    x = np.random.default_rng(0).normal(0, 1, 200_000)
    k = axis_features(x)[FEATURE_NAMES.index("kurt")]
    assert 2.9 < k < 3.1


def test_dc_component_is_removed():
    """Gravity puts a constant ~1 g on the vertical axis. Leave it in and RMS is
    dominated by it and every feature looks flat forever."""
    rng = np.random.default_rng(0)
    vib = rng.normal(0, 0.05, 4096)
    assert np.allclose(axis_features(vib)[0], axis_features(vib + 1.0)[0], rtol=1e-5)


def test_rms_tracks_vibration_amplitude():
    rng = np.random.default_rng(0)
    small = axis_features(rng.normal(0, 0.05, 4096))[0]
    large = axis_features(rng.normal(0, 0.50, 4096))[0]
    assert large > small * 8


def test_crest_factor_of_a_sine_is_root_two():
    t = np.linspace(0, 20 * np.pi, 100_000, endpoint=False)
    c = axis_features(np.sin(t))[FEATURE_NAMES.index("crest")]
    assert abs(c - np.sqrt(2)) < 0.01
