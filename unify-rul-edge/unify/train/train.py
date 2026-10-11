"""Joint multi-dataset training.

    python -m unify.train.train --config config/default.yaml
    python -m unify.train.train --datasets synthetic synthetic_small --epochs 20
    python -m unify.train.train --overfit          # sanity check: 8 samples -> loss ~0

Run --overfit BEFORE any real training run. If the model cannot memorise 8
samples, the bug is in the model or the data path, and no amount of epochs will
fix it.

Batches MIX datasets on every step. Training dataset-by-dataset causes
catastrophic forgetting; mixed batches are what force one shared representation,
which is the entire "Universal" claim.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..common.normalize import Normalizer
from ..datasets.build import build as build_cache, dataset_config, load_cached
from ..datasets.joint import (WindowDataset, collate, hold_out_official_test,
                              split_units)
from ..models.losses import build_loss, nasa_soft
from ..models.unify_rul import build as build_model
from .evaluate import evaluate, format_metrics

ROOT = Path(__file__).resolve().parents[2]

# C-MAPSS FD002/FD004 sensor readings move more with operating condition than with
# degradation. Per-condition statistics recover the trend. docs/05 §5.2
PER_CONDITION = {"cmapss", "synthetic", "synthetic_small"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(cfg):
    warm, total = cfg["optim"]["warmup_epochs"], cfg["optim"]["epochs"]
    floor = cfg["optim"]["min_lr"] / cfg["optim"]["lr"]

    def f(epoch):
        if epoch < warm:                       # linear warm-up — attention diverges without it
            return (epoch + 1) / max(warm, 1)
        p = (epoch - warm) / max(total - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--patience", type=int, help="early-stop patience, in epochs")
    ap.add_argument("--tag", help="write results under results/<tag>/ instead of "
                                  "overwriting the main checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit", action="store_true",
                    help="memorise 8 samples; loss must go to ~0")
    ap.add_argument("--sensor-self-attention", action="store_true", default=None,
                    help="enable inter-sensor self-attention layer")
    ap.add_argument("--residual", action="store_true", default=None,
                    help="enable residual skip connections in sensor encoder")
    ap.add_argument("--encoder-type", choices=["plain", "residual", "multiscale", "auto"],
                    help="sensor encoder architecture variant")
    ap.add_argument("--loss-kind", choices=["smooth_l1", "asym_smooth_l1", "asym_mse", "mse", "pinball"],
                    help="override loss function kind")
    ap.add_argument("--late-weight", type=float, help="penalty weight for late predictions")
    ap.add_argument("--nasa-weight", type=float, help="weight for differentiable NASA soft loss term")
    ap.add_argument("--pretrained", "--resume", dest="pretrained",
                    help="path to pretrained checkpoint to warm-start weights")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    # `is not None`, not truthiness — --seed 0 is a legitimate seed and would be
    # silently ignored by `if a.seed:`.
    if a.datasets is not None:
        cfg["data"]["datasets"] = a.datasets
    if a.epochs is not None:
        cfg["optim"]["epochs"] = a.epochs
    if a.batch_size is not None:
        cfg["optim"]["batch_size"] = a.batch_size
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.lr is not None:
        cfg["optim"]["lr"] = a.lr
    if a.patience is not None:
        cfg["optim"]["early_stop_patience"] = a.patience
    if a.sensor_self_attention is not None:
        cfg["model"]["sensor_self_attention"] = a.sensor_self_attention
    if a.residual is not None:
        cfg["model"]["residual"] = a.residual
    if a.encoder_type is not None:
        cfg["model"]["encoder_type"] = a.encoder_type
    if a.loss_kind is not None:
        cfg["loss"]["kind"] = a.loss_kind
    if a.late_weight is not None:
        cfg["loss"]["late_weight"] = a.late_weight
    if a.nasa_weight is not None:
        cfg["loss"]["nasa_weight"] = a.nasa_weight
    if a.tag:
        # Sweeps must not clobber the shipping checkpoint or, worse, the artifacts
        # the Pi is served from. Everything this run writes goes under the tag.
        cfg["out"]["ckpt"] = f"results/{a.tag}/best.pt"
        cfg["out"]["metrics"] = f"results/{a.tag}/metrics.json"
        cfg["out"]["artifacts"] = f"results/{a.tag}/artifacts"
        (ROOT / "results" / a.tag / "artifacts").mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device(a.device)
    names = cfg["data"]["datasets"]
    interim = ROOT / cfg["data"]["interim_dir"]

    # ── data ──────────────────────────────────────────────────────────────────
    build_cache(names, interim)
    units = load_cached(names, interim)
    if not units:
        raise SystemExit(
            f"no units found for {names}. Run:\n"
            f"  python -m unify.datasets.build --datasets {' '.join(names)}\n"
            f"Datasets not downloaded yet? Use --datasets synthetic synthetic_small.")

    ds_cfg = dataset_config()
    # Three splits, not two. val drives early stopping and checkpoint selection,
    # so reporting it as the headline number would be selection bias; the official
    # test units are never trained on and never looked at until the run is over.
    pool_units, test_units = hold_out_official_test(units)
    train_units, val_units = split_units(pool_units, cfg["data"]["val_frac"], cfg["seed"])
    print(f"units: {len(train_units)} train / {len(val_units)} val "
          f"/ {len(test_units)} official test "
          f"({len(set(u.dataset for u in units))} datasets)")

    # Fit normalisation on TRAIN UNITS ONLY. Fitting on everything leaks.
    norm = Normalizer().fit(train_units, per_condition=PER_CONDITION & set(names))

    L = cfg["data"]["window"]
    train_ds = WindowDataset(train_units, norm, ds_cfg, L, cfg["data"]["stride_train"])
    val_ds = WindowDataset(val_units, norm, ds_cfg, L, cfg["data"]["stride_eval"])
    print(f"windows: {len(train_ds)} train / {len(val_ds)} val  {train_ds.per_dataset_counts}")
    print(f"life scales (normalised -> native units): "
          f"{ {k: round(v, 1) for k, v in train_ds.scales.items()} }")

    if a.overfit:
        train_ds.items = train_ds.items[:8]
        val_ds = train_ds
        cfg["optim"].update(batch_size=8, epochs=300, warmup_epochs=0,
                            early_stop_patience=10 ** 6)
        # Capacity check, not a generalisation check. Dropout injects noise that
        # keeps the TRAIN-mode loss around 0.05 even when the model has memorised
        # the batch perfectly, which would make this test useless.
        cfg["model"].update(dropout=0.0, attn_dropout=0.0)

    bs = cfg["optim"]["batch_size"]
    nw = cfg["optim"]["num_workers"]
    sampler = (train_ds.balanced_sampler()
               if cfg["data"]["balance_datasets"] and not a.overfit else None)
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                          shuffle=sampler is None, collate_fn=collate,
                          num_workers=nw, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate,
                        num_workers=nw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    print(f"model: {model.n_params()} params "
          f"({model.n_params() * 4 / 1024:.0f} KB fp32, {model.n_params() / 1024:.0f} KB int8)")

    if a.pretrained:
        pre_path = Path(a.pretrained)
        if not pre_path.is_absolute():
            pre_path = ROOT / pre_path
        if pre_path.exists():
            print(f"Loading pretrained weights from {pre_path} (strict=False)...")
            ckpt = torch.load(pre_path, map_location=device, weights_only=True)
            sd = ckpt["model"] if "model" in ckpt else ckpt
            model_sd = model.state_dict()
            filtered_sd = {}
            for k, v in sd.items():
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered_sd[k] = v
                    else:
                        print(f"Skipping layer {k} due to shape mismatch: {model_sd[k].shape} vs {v.shape}")
            model.load_state_dict(filtered_sd, strict=False)
            print(f"Pretrained weights loaded successfully! ({len(filtered_sd)}/{len(model_sd)} layers matched)")
        else:
            print(f"Pretrained checkpoint {pre_path} not found; training from scratch.")

    criterion = build_loss(cfg["loss"])
    nasa_w = cfg["loss"].get("nasa_weight", 0.0)
    cmapss_scale = train_ds.scales.get("cmapss", 125.0)
    median_idx = (cfg["loss"]["quantiles"].index(0.5)
                  if cfg["loss"]["kind"] == "pinball" else None)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))

    # ── loop ──────────────────────────────────────────────────────────────────
    best, best_epoch, history = float("inf"), -1, []
    ckpt_path = ROOT / cfg["out"]["ckpt"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        total, n_batches, t0 = 0.0, 0, time.time()

        for b in train_dl:
            opt.zero_grad(set_to_none=True)
            out, _ = model(b["x"].to(device), b["mask"].to(device),
                           b["sensor_ids"].to(device), b["domain_id"].to(device))
            y = b["y"].to(device)
            loss = criterion(out, y)
            if nasa_w > 0:
                # The NASA score's constants (13 and 10) are in CYCLES — it is
                # defined for C-MAPSS only. Applying it to bearing samples, whose
                # life scale is completely different, is meaningless, so restrict
                # the term to the C-MAPSS rows of the batch.
                sel = torch.tensor([d == "cmapss" for d in b["ds"]], device=device)
                if sel.any():
                    p = out[:, median_idx] if median_idx is not None else out.squeeze(-1)
                    loss = loss + nasa_w * nasa_soft(
                        p[sel].unsqueeze(-1), y[sel], cmapss_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            total += loss.item()
            n_batches += 1

        sched.step()
        train_loss = total / max(n_batches, 1)
        m = evaluate(model, val_dl, val_ds.scales, device, median_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": m})

        flag = ""
        val_metric = m["mean"].get("norm_rmse", m["mean"]["rmse"])
        if val_metric < best:
            best, best_epoch = val_metric, epoch
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "scales": train_ds.scales, "norm": norm.stats,
                        "epoch": epoch, "val": m}, ckpt_path)
            flag = " *"
        print(f"ep {epoch:3d} loss {train_loss:.5f} | {format_metrics(m)} "
              f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s{flag}")

        if epoch - best_epoch >= cfg["optim"]["early_stop_patience"]:
            print(f"early stop: no improvement for {cfg['optim']['early_stop_patience']} epochs")
            break

    if a.overfit:
        # Measure in EVAL mode: BatchNorm running statistics, no dropout — the same
        # mode the deployed model runs in.
        model.eval()
        with torch.no_grad():
            b = collate(train_ds.items)
            out, _ = model(b["x"].to(device), b["mask"].to(device),
                           b["sensor_ids"].to(device), b["domain_id"].to(device))
            p = out[:, median_idx] if median_idx is not None else out.squeeze(-1)
            err = float((p.cpu() - b["y"]).abs().max())
        ok = err < 0.02          # < 2% of full life on every one of the 8 samples
        print(f"\noverfit check: max abs error {err:.5f} (normalised life) "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("The model cannot memorise 8 samples. The bug is in the model or "
                  "the data path — do not start a real training run.")
        raise SystemExit(0 if ok else 1)

    # Normalisation stats are written next to the checkpoint AND shipped to the Pi
    # by export_onnx.py. Losing them means silently wrong predictions on device.
    # NB: the extra keys must not collide with Normalizer.save's own "datasets"
    # block, which holds the per-channel statistics. Naming this one "datasets"
    # silently overwrote them and the Pi failed with KeyError: 'sensor_ids'.
    norm.save(ROOT / cfg["out"]["artifacts"] / "norm_stats.json",
              extra={"scales": train_ds.scales, "window": L,
                     "dataset_config": {k: ds_cfg[k] for k in names}})

    # ── official test split ───────────────────────────────────────────────────
    # Restore the BEST checkpoint first. Evaluating whatever weights training
    # happened to stop on reports a different model than the one that ships.
    # weights_only=True: the checkpoint holds only tensors, dicts and primitives,
    # so there is no reason to let the unpickler run arbitrary code.
    ckpt = torch.load(ckpt_path, weights_only=True)
    test_metrics = None
    if test_units:
        model.load_state_dict(ckpt["model"])
        test_ds = WindowDataset(test_units, norm, ds_cfg, L, cfg["data"]["stride_eval"],
                                last_window_only=True)
        if len(test_ds):
            test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                 collate_fn=collate, num_workers=nw)
            # train_ds.scales, not test_ds.scales: the normalised prediction must be
            # converted back to physical units with the same constant the labels were
            # divided by, and that constant is a property of the training set.
            test_metrics, test_preds, test_trues = evaluate(
                model, test_dl, train_ds.scales, device, median_idx, return_predictions=True)
            print(f"\nofficial test split, one prediction per unit "
                  f"({len(test_ds)} units): {format_metrics(test_metrics)}")

            # Save test predictions for outlier diagnostics and report figures
            pred_out_path = Path(ROOT / cfg["out"]["ckpt"]).parent / "official_test_predictions.npz"
            npz_dict = {}
            for ds_name in test_preds:
                npz_dict[f"pred_{ds_name}"] = test_preds[ds_name]
                npz_dict[f"true_{ds_name}"] = test_trues[ds_name]
                ds_units = [u.unit_id for u in test_units if ds_name in u.unit_id][:len(test_preds[ds_name])]
                npz_dict[f"unit_ids_{ds_name}"] = np.array(ds_units)
            if "cmapss" in test_preds:
                npz_dict["pred"] = test_preds["cmapss"]
                npz_dict["true"] = test_trues["cmapss"]
                npz_dict["unit_ids"] = npz_dict["unit_ids_cmapss"]
            if npz_dict:
                np.savez_compressed(pred_out_path, **npz_dict)

    metrics_path = ROOT / cfg["out"]["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "best_epoch": best_epoch, "best_mean_rmse": best,
        "final": history[-1]["val"], "best_val": ckpt["val"],
        "official_test": test_metrics,
        "params": model.n_params(), "config": cfg, "history": history,
    }, indent=2, default=str))
    print(f"\nbest mean RMSE {best:.3f} at epoch {best_epoch} (validation)\n"
          f"checkpoint {ckpt_path}\nmetrics    {metrics_path}")


if __name__ == "__main__":
    main()
