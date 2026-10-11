"""Time-domain vibration features.

MIRRORS firmware/esp32/src/features (axisFeatures). If these two implementations
disagree, the model sees a different feature distribution at inference than it
was trained on and the predictions are quietly wrong — the same class of bug as
losing the normalisation statistics. tests/test_features_parity.py pins them.

Why these features (docs/05-preprocessing.md §5.3):
  rms   — overall vibration energy, the workhorse trend
  kurt  — "spikiness". ~3.0 for Gaussian (healthy). Jumps to 5-20 at early
          spalling, then FALLS BACK toward 3 as the defect spreads, so it is
          non-monotonic and only useful alongside rms.
  skew  — asymmetry
  crest — peak/rms, early fault indicator
  p2p   — amplitude swing

Note there is no separate "std": after DC removal std(x) == rms(x) exactly, so it
would be a duplicate channel. ID 31 stays reserved in the vocab, unused.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["rms", "kurt", "skew", "crest", "p2p"]

# What the ESP32 actually computes and sends — a SUBSET. The mask handles the
# rest. See config/sensor_vocab.json.
LIVE_FEATURE_NAMES = ["rms", "kurt", "crest"]


def axis_features(x: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """Features for one channel's snapshot. x: (N,) or (B, N). Returns (5,) or (B, 5)."""
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None]

    mean = x.mean(axis=1, keepdims=True)
    d = x - mean if remove_dc else x
    # Gravity puts a constant ~1 g on the vertical axis. Leave it in and RMS is
    # dominated by it, every feature looks flat forever, and the model learns
    # nothing. This subtraction is not optional.

    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    peak = np.abs(d).max(axis=1)

    rms = np.sqrt(m2)
    safe2 = np.maximum(m2, 1e-24)
    safe_rms = np.maximum(rms, 1e-12)

    out = np.stack([
        rms,                                   # rms
        m4 / (safe2 ** 2),                     # kurtosis, ~3.0 for Gaussian
        m3 / (safe2 ** 1.5),                   # skewness
        peak / safe_rms,                       # crest factor
        d.max(axis=1) - d.min(axis=1),         # peak-to-peak
    ], axis=1).astype(np.float32)

    return out[0] if single else out


def snapshot_features(snap: np.ndarray, channel_prefixes: list[str],
                      subset: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Reduce one raw snapshot to a flat feature vector.

    snap: (N, C) raw waveform — e.g. FEMTO (2560, 2), XJTU (32768, 2)
    channel_prefixes: e.g. ["bearing/h", "bearing/v"]
    returns: (C*len(subset),) float32, and the matching sensor-name list

    This is the step that turns a 25.6 kHz waveform into something a length-30
    window can hold. Without it L would have to be 2560 and the model would not
    fit on a Pi.
    """
    names_wanted = subset or FEATURE_NAMES
    idx = [FEATURE_NAMES.index(n) for n in names_wanted]

    vals, names = [], []
    for c, prefix in enumerate(channel_prefixes):
        f = axis_features(snap[:, c])
        vals.append(f[idx])
        names += [f"{prefix}_{n}" for n in names_wanted]
    return np.concatenate(vals).astype(np.float32), names
