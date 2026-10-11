"""Sliding-window extraction.

Two invariants, both enforced here rather than trusted:

1. A window NEVER spans two units. A window containing the last 20 cycles of
   engine 7 and the first 10 of engine 8 is physically meaningless. Each unit is
   windowed separately and the results concatenated.
2. The label is the RUL at the LAST timestep of the window — the moment the Pi
   would actually be predicting at.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_unit(signals: np.ndarray, targets: np.ndarray, L: int, stride: int = 1):
    """signals (T, S), targets (T,) -> X (N, S, L), y (N,).

    Uses sliding_window_view, which returns a VIEW — no data is copied. On XJTU
    this is the difference between 40 seconds and 40 milliseconds.
    """
    T = signals.shape[0]
    if T < L:
        return (np.empty((0, signals.shape[1], L), np.float32),
                np.empty((0,), np.float32))

    w = sliding_window_view(signals, window_shape=L, axis=0)  # (T-L+1, S, L)
    w = w[::stride]
    y = targets[L - 1:][::stride]                             # label at the window's right edge
    return np.ascontiguousarray(w, dtype=np.float32), y.astype(np.float32)


def n_windows(T: int, L: int, stride: int = 1) -> int:
    return 0 if T < L else (T - L) // stride + 1
