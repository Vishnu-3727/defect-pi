"""Losses and metrics.

Key asymmetry: predicting failure TOO LATE is dangerous (the engine fails in
service); predicting too early is merely wasteful. MSE does not know that. The
NASA scoring function does, and so does `asym_mse`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── training losses ───────────────────────────────────────────────────────────
def smooth_l1(pred, y, beta: float = 0.1):
    """Quadratic near zero, linear far away. More stable than MSE on the noisy
    bearing datasets, where outliers are often pathological units rather than
    signal. Good default."""
    return F.smooth_l1_loss(pred.squeeze(-1), y, beta=beta)


def mse(pred, y, **_):
    return F.mse_loss(pred.squeeze(-1), y)


def asym_mse(pred, y, late_weight: float = 2.0, **_):
    """Weighted MSE — asymmetry without the NASA score's exponential drama.

    Two lines, numerically stable, and a strong answer to "how did you address the
    asymmetric cost of errors?". Note the sign: predicting a LARGER normalised RUL
    than the truth means claiming more life remains than there is = late = dangerous.
    """
    p = pred.squeeze(-1)
    w = torch.where(p > y, late_weight, 1.0)
    return (w * (p - y) ** 2).mean()


def asym_smooth_l1(pred, y, beta: float = 0.1, late_weight: float = 2.0, **_):
    """Asymmetric Smooth L1 (Huber) loss.

    Combines Huber robustness to extreme sensor outliers with an asymmetric
    penalty multiplier for late predictions (pred > y). Reduces the devastating
    NASA score blowups while preventing gradient explosion on noisy bearing data.
    """
    p = pred.squeeze(-1)
    diff = p - y
    abs_diff = torch.abs(diff)
    base = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    w = torch.where(p > y, late_weight, 1.0)
    return (w * base).mean()


def pinball(pred, y, quantiles=(0.1, 0.5, 0.9), **_):
    """Quantile regression — the fix for problem P4 (false alarms).

    The head emits one value per quantile, so the dashboard can show
    "RUL 47 min (10-90%: 31-68)" and the alarm can fire on the LOWER quantile
    rather than on a noisy point estimate. That is principled false-alarm control
    instead of a hand-tuned threshold.
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = y - pred[:, i]
        loss = loss + torch.max(q * e, (q - 1.0) * e).mean()
    return loss / len(quantiles)


def nasa_soft(pred, y, scale: float, clamp: float = 60.0, **_):
    """Differentiable NASA score, clamped so one bad sample cannot blow up training.

    Use as a small auxiliary term (weight 0.01-0.1), never as the sole loss — the
    exponential produces enormous gradients.
    """
    d = (pred.squeeze(-1) - y) * scale                 # physical units
    d = d.clamp(-clamp, clamp)
    return torch.where(d < 0, torch.exp(-d / 13.0) - 1.0,
                       torch.exp(d / 10.0) - 1.0).mean()


def build_loss(cfg: dict):
    kind = cfg["kind"]
    if kind == "smooth_l1":
        return lambda p, y: smooth_l1(p, y, cfg["beta"])
    if kind == "mse":
        return mse
    if kind == "asym_mse":
        return lambda p, y: asym_mse(p, y, cfg["late_weight"])
    if kind == "asym_smooth_l1":
        return lambda p, y: asym_smooth_l1(p, y, beta=cfg.get("beta", 0.1), late_weight=cfg.get("late_weight", 2.0))
    if kind == "pinball":
        qs = tuple(cfg["quantiles"])
        return lambda p, y: pinball(p, y, qs)
    raise ValueError(f"unknown loss {kind!r}")


# ── evaluation metrics, computed in PHYSICAL units ────────────────────────────
def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def nasa_score(pred: np.ndarray, y: np.ndarray) -> float:
    """PHM08 scoring function. Lower is better; 0 is perfect.

        d = pred - true       (positive = LATE = dangerous)
        s = exp(-d/13) - 1    if d < 0
            exp( d/10) - 1    if d >= 0

    The constants 13 and 10 are in CYCLES, so this must be computed on capped
    physical values. Running it on normalised [0,1] targets gives a meaningless
    number that cannot be compared to any published result.

    Defined for C-MAPSS. Reporting it for bearings is non-standard — say so if
    you do.
    """
    d = pred - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))
