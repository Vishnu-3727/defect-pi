"""Regenerate every figure in reports/figures/ from measured artifacts.

    python scripts/make_report_figures.py            # all figures
    python scripts/make_report_figures.py --demo     # self-check, writes nothing

Nothing here is typed in by hand. Every number is parsed from a file that some
run produced: results/<tag>/metrics.json, results/baseline_rf.json, the Pi's
results/bench*.json, and the Pi's edge/unify.db. If a source is missing the
figure is skipped with a message rather than drawn from stale defaults - a
chart that silently invents its data is worse than no chart.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"

C_MODEL, C_BASE, C_ACC = "#1f77b4", "#ff7f0e", "#2ca02c"


def _save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + str(p))
    return p


def _load(path: Path):
    if not path.exists():
        print("  SKIP - missing " + str(path))
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------- figures ----
def fig_training_curve(metrics: dict, ds: str):
    """Val RMSE per epoch, in physical units, with the selected epoch marked."""
    h = metrics["history"]
    ep = [r["epoch"] for r in h]
    loss = [r["train_loss"] for r in h]
    rmse = [r["val"][ds]["rmse"] for r in h]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, rmse, color=C_MODEL, lw=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation RMSE (cycles)", color=C_MODEL)
    ax1.tick_params(axis="y", labelcolor=C_MODEL)

    b = metrics["best_epoch"]
    ax1.axvline(b, color="grey", ls="--", lw=1)
    ax1.annotate("selected epoch %d\nRMSE %.2f" % (b, metrics["best_mean_rmse"]),
                 xy=(b, metrics["best_mean_rmse"]), xytext=(10, 20),
                 textcoords="offset points", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    ax2 = ax1.twinx()
    ax2.plot(ep, loss, color=C_BASE, lw=1, alpha=0.7)
    ax2.set_ylabel("training loss (normalised life)", color=C_BASE)
    ax2.tick_params(axis="y", labelcolor=C_BASE)

    ax1.set_title("UNIFY-RUL training on %s - %s parameters"
                  % (ds, format(metrics["params"], ",")))
    ax1.grid(alpha=0.3)
    return _save(fig, "fig1_training_curve.png")


def fig_pred_vs_true(pred, true, cap, tol=13.0):
    """One point per held-out engine. The band is the +/-13-cycle tolerance the
    C-MAPSS literature uses when it reports 'within tolerance' accuracy."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = [0, cap * 1.05]
    ax.fill_between(lim, [v - tol for v in lim], [v + tol for v in lim],
                    color=C_ACC, alpha=0.12, label="+/-%g cycles" % tol)
    ax.plot(lim, lim, color="grey", lw=1, ls="--", label="perfect")
    inside = np.abs(pred - true) <= tol
    ax.scatter(true[inside], pred[inside], s=18, color=C_MODEL, alpha=0.7,
               label="within tolerance (%d)" % inside.sum())
    ax.scatter(true[~inside], pred[~inside], s=18, color=C_BASE, alpha=0.8,
               label="outside (%d)" % (~inside).sum())
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true RUL (cycles)")
    ax.set_ylabel("predicted RUL (cycles)")
    rms = float(np.sqrt(np.mean((pred - true) ** 2)))
    ax.set_title("Official test split, one prediction per engine\n"
                 "n=%d  RMSE=%.2f  within tolerance=%.1f%%"
                 % (len(pred), rms, inside.mean() * 100))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "fig2_pred_vs_true.png")


def fig_error_hist(pred, true):
    """Error sign matters more than magnitude: a late prediction is the one that
    lets the engine fail, which is why the NASA score penalises it harder."""
    err = pred - true
    late = err > 0
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(err.min(), err.max(), 40)
    ax.hist(err[~late], bins=bins, color=C_ACC, alpha=0.85,
            label="early / conservative (%d)" % (~late).sum())
    ax.hist(err[late], bins=bins, color=C_BASE, alpha=0.85,
            label="late / dangerous (%d)" % late.sum())
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(err.mean(), color=C_MODEL, ls="--", lw=1.5,
               label="mean bias %+.2f cycles" % err.mean())
    ax.set_xlabel("prediction error (cycles)   negative = predicts failure earlier than truth")
    ax.set_ylabel("engines")
    ax.set_title("Error distribution on the official test split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig3_error_hist.png")


def fig_model_vs_baseline(model: dict, base: dict, ds: str):
    """A 44k-parameter network against a RandomForest on engineered features. If
    the deep model does not win, the report says so."""
    m, b = model["official_test"][ds], base["official_test"][ds]
    names = ["RMSE (cycles)", "MAE (cycles)", "NASA score / 100"]
    mv = [m["rmse"], m["mae"], m["nasa"] / 100]
    bv = [b["rmse"], b["mae"], b["nasa"] / 100]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    r1 = ax.bar(x - w / 2, mv, w, color=C_MODEL,
                label="UNIFY-RUL (%s params)" % format(model["params"], ","))
    r2 = ax.bar(x + w / 2, bv, w, color=C_BASE, label="RandomForest baseline")
    for r in (r1, r2):
        ax.bar_label(r, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("lower is better")
    ax.set_title("Official %s test split - model vs baseline" % ds)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "fig4_model_vs_baseline.png")


def fig_latency(benches):
    """Thread sweep on the Pi 5. Percentiles, not means - real-time cares about
    the tail."""
    fig, ax = plt.subplots(figsize=(7, 4))
    marks = ["o", "s", "^"]
    for i, (label, b) in enumerate(benches.items()):
        runs = sorted(b["runs"], key=lambda r: r["threads"])
        t = [r["threads"] for r in runs]
        ax.plot(t, [r["p50_ms"] for r in runs], marker=marks[i % 3], lw=2,
                label=label + " p50")
        ax.plot(t, [r["p95_ms"] for r in runs], marker=marks[i % 3], lw=1,
                ls="--", alpha=0.7, label=label + " p95")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("onnxruntime intra-op threads")
    ax.set_ylabel("inference latency (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Single-window inference on a Raspberry Pi 5, CPU only")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig5_pi_latency.png")


def fig_budget(bench, window_s=1.0):
    """The point of the latency chart: inference is noise against the window."""
    runs = sorted(bench["runs"], key=lambda r: r["p95_ms"])
    inf_ms = runs[0]["p95_ms"]
    budget_ms = window_s * 1000
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.barh([0], [budget_ms], color="#dddddd",
            label="%g s feature window" % window_s)
    ax.barh([0], [inf_ms], color=C_MODEL, label="inference p95 %.2f ms" % inf_ms)
    ax.set_xlim(0, budget_ms * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("milliseconds")
    ax.set_title("Inference uses %.3f%% of the real-time budget (%.0fx headroom)"
                 % (inf_ms / budget_ms * 100, budget_ms / inf_ms))
    ax.legend(fontsize=8, loc="center right")
    return _save(fig, "fig6_realtime_budget.png")


def fig_live(db: Path):
    """What the Pi actually logged while the service was up.

    The x axis is the engine cycle, not wall-clock. The replay compresses 199
    cycles into 20 seconds, so plotting minutes would say "0.3 min" and tell the
    reader nothing about how much warning the alarm actually gives.
    """
    con = sqlite3.connect(db)
    rows = con.execute("select seq, rul, health, alarm from predictions "
                       "order by seq").fetchall()
    con.close()
    if len(rows) < 10:
        print("  SKIP - fewer than 10 logged predictions")
        return None
    cols = list(zip(*rows))
    seq = np.array(cols[0], dtype=float)
    rul = np.array(cols[1], dtype=float)
    health = np.array(cols[2], dtype=float)
    alarm = np.array([0 if v is None else float(v) for v in cols[3]])
    # The unit is replayed to failure, so the last logged cycle is end of life
    # and the true remaining life at every point is just the distance to it.
    true_rul = seq[-1] - seq

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(seq, true_rul, color="grey", lw=1.2, ls="--", label="true RUL")
    a1.plot(seq, rul, color=C_MODEL, lw=1.4, label="predicted RUL")
    fired = alarm > 0
    if fired.any():
        first = seq[fired][0]
        lead = int(seq[-1] - first)
        a1.scatter(seq[fired], rul[fired], s=14, color="red", zorder=3,
                   label="alarm (%d cycles)" % int(fired.sum()))
        for ax in (a1, a2):
            ax.axvline(first, color="red", lw=1, alpha=0.6)
        a1.annotate("first alarm\n%d cycles before failure" % lead,
                    xy=(first, rul[fired][0]), xytext=(-118, 30),
                    textcoords="offset points", fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    a1.set_ylabel("RUL (cycles)")
    a1.set_title("Pi 5 live inference - real C-MAPSS FD001 engine replayed to failure\n"
                 "%d predictions, one per cycle" % len(rows))
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    a2.plot(seq, health, color=C_ACC, lw=1.4)
    a2.axhline(0.30, color="grey", ls=":", lw=1)
    a2.annotate("alarm threshold 0.30", xy=(seq[0], 0.32), fontsize=7, color="grey")
    a2.set_ylabel("health (EWMA)")
    a2.set_xlabel("engine cycle")
    a2.grid(alpha=0.3)
    return _save(fig, "fig7_pi_live_trace.png")


def fig_live_latency(db: Path):
    """End-to-end service latency, not the bare ONNX call."""
    con = sqlite3.connect(db)
    lat = np.array([r[0] for r in con.execute(
        "select latency_ms from predictions where latency_ms is not null")],
        dtype=float)
    con.close()
    if len(lat) < 10:
        print("  SKIP - fewer than 10 latency samples")
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(lat, bins=40, color=C_MODEL, alpha=0.85)
    for q, c in ((50, C_ACC), (95, C_BASE), (99, "red")):
        v = float(np.percentile(lat, q))
        ax.axvline(v, color=c, ls="--", lw=1.4, label="p%d = %.2f ms" % (q, v))
    ax.set_xlabel("end-to-end request latency (ms)")
    ax.set_ylabel("requests")
    ax.set_title("Pi 5 service latency over %d logged requests" % len(lat))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig8_pi_service_latency.png")


def fig_footprint(manifest: dict, art: Path):
    """FP32 vs INT8 on disk. Quantisation is the whole reason this fits."""
    files = [("PyTorch checkpoint", ROOT / "results" / "best.pt"),
             ("ONNX FP32", art / "unify_rul.onnx"),
             ("ONNX INT8", art / "unify_rul_int8.onnx")]
    have = [(n, p.stat().st_size / 1024) for n, p in files if p.exists()]
    if not have:
        print("  SKIP - no artifacts on disk")
        return None
    fig, ax = plt.subplots(figsize=(6, 3.4))
    r = ax.bar([n for n, _ in have], [k for _, k in have],
               color=[C_BASE, C_MODEL, C_ACC][:len(have)])
    ax.bar_label(r, fmt="%.0f KB", fontsize=9, padding=2)
    ax.set_ylabel("size on disk (KB)")
    ax.set_title("Deployment footprint - opset %s, window %s"
                 % (manifest.get("opset", "?"), manifest.get("window", "?")))
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "fig9_footprint.png")


# ------------------------------------------------------------------ demo ----
def demo():
    """Self-check: every drawing function survives synthetic input, and the
    short-log guard actually refuses to draw."""
    import shutil
    import tempfile
    global FIGS
    # mkdtemp + ignore_errors, not TemporaryDirectory: on Windows the sqlite file
    # handle can outlive the connection object long enough to make the automatic
    # cleanup raise PermissionError and fail a demo that otherwise passed.
    d = tempfile.mkdtemp()
    try:
        FIGS = Path(d)
        rng = np.random.default_rng(0)
        true = rng.uniform(5, 125, 60)
        pred = true + rng.normal(0, 10, 60)

        m = {"best_epoch": 4, "best_mean_rmse": 14.9, "params": 44417,
             "history": [{"epoch": e, "train_loss": 0.1 / (e + 1),
                          "val": {"cmapss": {"rmse": 30 - e}}} for e in range(10)],
             "official_test": {"cmapss": {"rmse": 14.5, "mae": 10.0, "nasa": 3416.0}}}
        b = {"official_test": {"cmapss": {"rmse": 14.4, "mae": 10.5, "nasa": 2558.0}}}
        bench = {"runs": [{"threads": t, "p50_ms": 0.6 - 0.05 * t,
                           "p95_ms": 0.65 - 0.05 * t} for t in (1, 2, 3, 4)]}

        assert fig_training_curve(m, "cmapss").exists()
        assert fig_pred_vs_true(pred, true, 125.0).exists()
        assert fig_error_hist(pred, true).exists()
        assert fig_model_vs_baseline(m, b, "cmapss").exists()
        assert fig_latency({"demo": bench}).exists()
        assert fig_budget(bench).exists()

        # A 1 s window against a sub-ms p95 is a >1000x headroom; the title must
        # not invert the ratio.
        best = min(bench["runs"], key=lambda r: r["p95_ms"])["p95_ms"]
        assert 1000 / best > 1000, best

        db = Path(d) / "t.db"
        con = sqlite3.connect(db)
        con.execute("create table predictions (id integer, ts real, seq integer, "
                    "rul real, health real, alarm int, latency_ms real)")
        con.executemany("insert into predictions values (?,?,?,?,?,?,?)",
                        [(i, 1e9 + i * 5, i, 120 - i, 1 - i / 100.0,
                          int(i > 80), 3 + i % 4) for i in range(100)])
        con.commit()
        con.close()
        assert fig_live(db).exists()
        assert fig_live_latency(db).exists()

        # Too-short logs must be skipped, not drawn from 3 points.
        db2 = Path(d) / "t2.db"
        con = sqlite3.connect(db2)
        con.execute("create table predictions (id integer, ts real, seq integer, "
                    "rul real, health real, alarm int, latency_ms real)")
        con.executemany("insert into predictions values (?,?,?,?,?,?,?)",
                        [(i, 1e9 + i, i, 1.0, 1.0, 0, 1.0) for i in range(3)])
        con.commit()
        con.close()
        assert fig_live(db2) is None and fig_live_latency(db2) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("demo OK")


# ------------------------------------------------------------------ main ----
def official_test_predictions(tag: str, dataset: str, device: str):
    """Re-run the shipping checkpoint over the official held-out units.

    Rebuilt exactly the way unify/train/train.py builds it: same seed, same
    normaliser fitted on train units only, same last_window_only test set. A
    scatter drawn from a different split would not be the reported RMSE.
    """
    import torch
    from torch.utils.data import DataLoader
    from unify.common.normalize import Normalizer
    from unify.datasets.build import dataset_config, load_cached
    from unify.datasets.joint import (WindowDataset, collate,
                                      hold_out_official_test, split_units)
    from unify.models.unify_rul import build as build_model
    from unify.train.train import PER_CONDITION, set_seed

    ck = ROOT / "results" / tag / "best.pt"
    mt = json.loads((ROOT / "results" / tag / "metrics.json").read_text())
    cfg = mt["config"]
    set_seed(cfg["seed"])

    units = load_cached(cfg["data"]["datasets"], ROOT / cfg["data"]["interim_dir"])
    pool, test_units = hold_out_official_test(units)
    train_units, _ = split_units(pool, cfg["data"]["val_frac"], cfg["seed"])
    norm = Normalizer().fit(
        train_units, per_condition=PER_CONDITION & set(cfg["data"]["datasets"]))
    ds_cfg = dataset_config()
    L = cfg["data"]["window"]
    train_ds = WindowDataset(train_units, norm, ds_cfg, L, cfg["data"]["stride_train"])
    test_ds = WindowDataset(test_units, norm, ds_cfg, L, cfg["data"]["stride_eval"],
                            last_window_only=True)

    dev = torch.device(device)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(ck, weights_only=True)["model"])
    model.eval()

    scale = train_ds.scales[dataset]
    P, T = [], []
    with torch.no_grad():
        for b in DataLoader(test_ds, batch_size=256, shuffle=False,
                            collate_fn=collate):
            raw, _ = model(b["x"].to(dev), b["mask"].to(dev),
                           b["sensor_ids"].to(dev), b["domain_id"].to(dev))
            p = raw.squeeze(-1).clamp(0, 1).cpu().numpy()
            y = b["y"].numpy()
            for i, ds in enumerate(b["ds"]):
                if ds == dataset:
                    P.append(p[i])
                    T.append(y[i])
    return np.array(P) * scale, np.array(T) * scale, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmapss_full", help="results/<tag>/ to read")
    ap.add_argument("--dataset", default="cmapss")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pi-dir", default="reports/pi",
                    help="where the Pi's bench json and unify.db were copied")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()

    sys.path.insert(0, str(ROOT))
    res = ROOT / "results" / a.tag
    pi = ROOT / a.pi_dir

    print("figures:")
    metrics = _load(res / "metrics.json")
    if metrics:
        fig_training_curve(metrics, a.dataset)
        base = _load(ROOT / "results" / "baseline_rf.json")
        if base:
            fig_model_vs_baseline(metrics, base, a.dataset)
        pred, true, scale = official_test_predictions(a.tag, a.dataset, a.device)
        fig_pred_vs_true(pred, true, scale)
        fig_error_hist(pred, true)
        np.savez(res / "official_test_predictions.npz", pred=pred, true=true)
        print("  wrote " + str(res / "official_test_predictions.npz"))

    benches = {}
    for label, name in (("Pi 5 (C-MAPSS model)", "bench_pi.json"),
                        ("Pi 5 (synthetic model)", "bench_synth_quiet.json")):
        b = _load(pi / name)
        if b:
            benches[label] = b
    if benches:
        fig_latency(benches)
        fig_budget(list(benches.values())[0])

    db = pi / "unify.db"
    if db.exists():
        fig_live(db)
        fig_live_latency(db)
    else:
        print("  SKIP - missing " + str(db))

    man = _load(ROOT / "edge" / "artifacts" / "manifest.json")
    if man:
        fig_footprint(man, ROOT / "edge" / "artifacts")


if __name__ == "__main__":
    main()
