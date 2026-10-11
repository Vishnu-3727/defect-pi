"""Re-measure accuracy AFTER export and AFTER quantisation.

    python scripts/eval_onnx.py --tag cmapss_full

export_onnx.py ends by printing "NOW RE-MEASURE ACCURACY. A quantisation claim
without an RMSE number beside it is not a result." This is that measurement.

Three numbers on the identical official-test windows:

    torch fp32   - what the training log reported
    onnx  fp32   - proves the export did not change the model
    onnx  int8   - the number that actually ships, and the only honest place to
                   read the quantisation cost from

Writes results/<tag>/onnx_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nasa_score(pred, true):
    """C-MAPSS scoring function. Late predictions (pred > true) cost far more."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0), np.exp(d / 10.0)) - 1.0))


def metrics(pred, true):
    return {
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mae": float(np.mean(np.abs(pred - true))),
        "nasa": nasa_score(pred, true),
        "n": int(len(pred)),
    }


def build_test_batch(tag: str, dataset: str):
    """Same split, same normaliser, same window as training. See
    scripts/make_report_figures.official_test_predictions for why this is
    reconstructed rather than cached."""
    import torch
    from torch.utils.data import DataLoader
    from unify.common.normalize import Normalizer
    from unify.datasets.build import dataset_config, load_cached
    from unify.datasets.joint import (WindowDataset, collate,
                                      hold_out_official_test, split_units)
    from unify.train.train import PER_CONDITION, set_seed

    cfg = json.loads((ROOT / "results" / tag / "metrics.json").read_text())["config"]
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
    batches = list(DataLoader(test_ds, batch_size=256, shuffle=False,
                              collate_fn=collate))
    return cfg, batches, train_ds.scales[dataset], torch


def run_torch(cfg, batches, torch, ck: Path, dataset: str, device="cpu"):
    from unify.models.unify_rul import build as build_model
    dev = torch.device(device)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(ck, weights_only=True)["model"])
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for b in batches:
            raw, _ = model(b["x"].to(dev), b["mask"].to(dev),
                           b["sensor_ids"].to(dev), b["domain_id"].to(dev))
            p = raw.squeeze(-1).clamp(0, 1).cpu().numpy()
            y = b["y"].numpy()
            for i, ds in enumerate(b["ds"]):
                if ds == dataset:
                    P.append(p[i])
                    T.append(y[i])
    return np.array(P), np.array(T)


def run_onnx(onnx_path: Path, batches, dataset: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    P, T = [], []
    for b in batches:
        feed = {}
        for n in names:
            v = b[n].numpy()
            # ONNX keeps the integer inputs as int64 and the mask as float32;
            # feeding a torch bool mask straight through silently fails the shape
            # check on some runtimes, so cast against the declared input type.
            decl = [i for i in sess.get_inputs() if i.name == n][0].type
            if "int64" in decl:
                v = v.astype(np.int64)
            elif "float" in decl:
                v = v.astype(np.float32)
            feed[n] = v
        out = sess.run(None, feed)[0]
        p = np.clip(np.squeeze(out, -1) if out.ndim == 2 and out.shape[1] == 1
                    else out.squeeze(), 0, 1)
        y = b["y"].numpy()
        for i, ds in enumerate(b["ds"]):
            if ds == dataset:
                P.append(np.atleast_1d(p)[i])
                T.append(y[i])
    return np.array(P), np.array(T)


def demo():
    """Self-check for the two things here that can silently be wrong: the NASA
    asymmetry and the metric arithmetic."""
    t = np.array([50.0, 50.0])
    early = np.array([40.0, 40.0])     # predicts failure sooner - safe
    late = np.array([60.0, 60.0])      # predicts failure later - dangerous
    assert nasa_score(late, t) > nasa_score(early, t), "late must cost more"
    assert abs(nasa_score(t, t)) < 1e-9, "a perfect prediction scores 0"

    m = metrics(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    assert abs(m["rmse"] - np.sqrt(2.0)) < 1e-9, m
    assert abs(m["mae"] - 1.0) < 1e-9, m
    assert m["n"] == 2
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmapss_full")
    ap.add_argument("--dataset", default="cmapss")
    ap.add_argument("--artifacts", default="edge/artifacts")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()

    art = ROOT / a.artifacts
    cfg, batches, scale, torch = build_test_batch(a.tag, a.dataset)

    rows = {}
    p, t = run_torch(cfg, batches, torch, ROOT / "results" / a.tag / "best.pt",
                     a.dataset)
    rows["torch_fp32"] = metrics(p * scale, t * scale)
    ref = p

    for label, f in (("onnx_fp32", "unify_rul.onnx"),
                     ("onnx_int8", "unify_rul_int8.onnx")):
        path = art / f
        if not path.exists():
            print("SKIP " + label + " - missing " + str(path))
            continue
        p, t = run_onnx(path, batches, a.dataset)
        rows[label] = metrics(p * scale, t * scale)
        rows[label]["max_abs_delta_vs_torch"] = float(np.max(np.abs(p - ref)) * scale)

    print("%-12s %9s %9s %12s %14s" % ("", "RMSE", "MAE", "NASA", "max|d| vs torch"))
    for k, v in rows.items():
        print("%-12s %9.3f %9.3f %12.1f %14s"
              % (k, v["rmse"], v["mae"], v["nasa"],
                 ("%.3f" % v["max_abs_delta_vs_torch"])
                 if "max_abs_delta_vs_torch" in v else "-"))

    if "onnx_int8" in rows and "onnx_fp32" in rows:
        d = rows["onnx_int8"]["rmse"] - rows["onnx_fp32"]["rmse"]
        print("\nquantisation cost: %+.3f cycles RMSE (%+.2f%%)"
              % (d, 100 * d / rows["onnx_fp32"]["rmse"]))

    out = ROOT / "results" / a.tag / "onnx_eval.json"
    out.write_text(json.dumps({"dataset": a.dataset, "scale": scale,
                               "results": rows}, indent=2))
    print("wrote " + str(out))


if __name__ == "__main__":
    main()
