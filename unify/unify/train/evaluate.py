"""Evaluation — always PER DATASET and always in PHYSICAL units.

Reporting "normalised RMSE 0.09" is meaningless to a reviewer. Denormalise by the
per-dataset life scale first, then compute RMSE / MAE / NASA score, so the numbers
are directly comparable to published literature.

Also: report per dataset, never only the mean. A mean across three datasets can
hide a total collapse on one of them.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.losses import mae, nasa_score, rmse


@torch.no_grad()
def evaluate(model, loader, scales: dict[str, float], device, median_index: int | None = None,
             return_predictions: bool = False):
    """Returns {dataset: {rmse, mae, nasa, n}} plus a "mean" entry over datasets."""
    model.eval()
    preds: dict[str, list] = {}
    trues: dict[str, list] = {}
    los: dict[str, list] = {}
    his: dict[str, list] = {}

    for b in loader:
        raw, _ = model(b["x"].to(device), b["mask"].to(device),
                       b["sensor_ids"].to(device), b["domain_id"].to(device))
        # For quantile models the point estimate is the median head.
        p = raw[:, median_index] if median_index is not None else raw.squeeze(-1)
        p = p.clamp(0.0, 1.0).cpu().numpy()
        y = b["y"].numpy()

        has_quantiles = (raw.shape[-1] >= 3)
        if has_quantiles:
            sorted_q = torch.sort(raw, dim=-1).values
            q_lo = sorted_q[:, 0].clamp(0.0, 1.0).cpu().numpy()
            q_hi = sorted_q[:, -1].clamp(0.0, 1.0).cpu().numpy()

        for i, ds in enumerate(b["ds"]):
            preds.setdefault(ds, []).append(p[i])
            trues.setdefault(ds, []).append(y[i])
            if has_quantiles:
                los.setdefault(ds, []).append(q_lo[i])
                his.setdefault(ds, []).append(q_hi[i])

    out: dict[str, dict] = {}
    scaled_preds: dict[str, np.ndarray] = {}
    scaled_trues: dict[str, np.ndarray] = {}
    for ds in preds:
        s = scales.get(ds, 1.0)
        p = np.array(preds[ds]) * s          # -> cycles / seconds / minutes
        t = np.array(trues[ds]) * s
        scaled_preds[ds] = p
        scaled_trues[ds] = t
        res = {
            "rmse": rmse(p, t),
            "mae": mae(p, t),
            "nasa": nasa_score(p, t) if ds == "cmapss" else None,
            "n": int(len(p)),
            "scale": float(s),
        }
        if ds in los:
            lo_arr = np.array(los[ds]) * s
            hi_arr = np.array(his[ds]) * s
            res["coverage_80"] = float(np.mean((t >= lo_arr) & (t <= hi_arr)))
            res["interval_width"] = float(np.mean(hi_arr - lo_arr))
        out[ds] = res

    # Mean of the per-dataset RMSEs, plus dimensionless normalized RMSE (rmse / scale)
    # so datasets measured in seconds and cycles are evaluated on an equal scale.
    norm_rmses = [v["rmse"] / v["scale"] for v in out.values() if "scale" in v and v["scale"] > 0]
    out["mean"] = {
        "rmse": float(np.mean([v["rmse"] for v in out.values()])),
        "norm_rmse": float(np.mean(norm_rmses)) if norm_rmses else float(np.mean([v["rmse"] for v in out.values()])),
    }
    if return_predictions:
        return out, scaled_preds, scaled_trues
    return out


def format_metrics(m: dict) -> str:
    parts = []
    for ds, v in sorted(m.items()):
        if ds == "mean":
            continue
        s = f"{ds} rmse={v['rmse']:.2f}"
        if v.get("coverage_80") is not None:
            s += f" cov={v['coverage_80']*100:.1f}%"
        if v.get("nasa") is not None:
            s += f" nasa={v['nasa']:.0f}"
        parts.append(s)
    mean_info = f"MEAN rmse={m['mean']['rmse']:.2f}"
    if "norm_rmse" in m["mean"]:
        mean_info += f" norm={m['mean']['norm_rmse']:.4f}"
    parts.append(mean_info)
    return " | ".join(parts)
