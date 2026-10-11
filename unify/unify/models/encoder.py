"""Shared per-sensor 1-D CNN encoder — the mechanism behind "Any Dataset".

The SAME weights encode sensor 1, sensor 2, ... sensor S. Sensors are folded into
the batch dimension, so S only ever changes the size of an internal reshape and
never the shape of a weight:

    (B, S, L) -> (B*S, 1, L) -> CNN -> (B*S, D) -> (B, S, D)

A conventional RUL model writes nn.Linear(21*30, ...) and is permanently welded to
21 sensors. That is problem P1 from the Review-1 slides.

Why 1-D and not 2-D convolution: a 2-D kernel assumes both axes are
translation-invariant. Along the sensor axis that is false — CSV column order is
arbitrary and neighbouring columns are not physically related — and a kernel
spanning k sensors would hard-code the sensor count. Convolve along time only,
treat sensors as a set, aggregate with attention.

Global average pooling (not flatten) at the end removes the dependence on L too,
so in PyTorch the same weights accept any window length. Note the ONNX export
deliberately PINS L (dynamic_axes marks only batch and sensors) so that a window
mismatch on the Pi fails loudly instead of silently producing garbage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """One channel's window (1, L) -> a D-dimensional descriptor.

    Receptive field: conv k=5, then k=3, pool 2, then k=3 -> ~19 timesteps of the
    30-step window. Layer 1 learns edges/spikes/levels, layer 2 combinations
    ("rise then plateau"), layer 3 window-scale trends.
    """

    def __init__(self, d_model: int = 64, norm: str = "group", residual: bool = False):
        super().__init__()
        # GroupNorm(1, C) normalises within each sample, so it is independent of
        # batch composition. BatchNorm is NOT safe here: sensors are folded into
        # the batch dimension, so a mixed-sensor-count batch feeds zero-padded rows
        # into the batch statistics and the normalisation shifts with how much
        # padding a batch happens to contain. It also misbehaves at batch size 1,
        # which is exactly how the Pi runs. Set norm="batch" to compare.
        Norm = nn.BatchNorm1d if norm == "batch" else (lambda c: nn.GroupNorm(1, c))
        self.residual = residual

        if not residual:
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), Norm(32), nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 3, padding=1), Norm(64), nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(64, d_model, 3, padding=1), Norm(d_model), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.c1 = nn.Conv1d(1, 32, 5, padding=2)
            self.n1 = Norm(32)
            self.act1 = nn.ReLU(inplace=True)

            self.c2 = nn.Conv1d(32, 64, 3, padding=1)
            self.n2 = Norm(64)
            self.skip1 = nn.Conv1d(1, 64, 1, bias=False)
            self.act2 = nn.ReLU(inplace=True)

            self.pool = nn.MaxPool1d(2)

            self.c3 = nn.Conv1d(64, d_model, 3, padding=1)
            self.n3 = Norm(d_model)
            self.skip2 = nn.Conv1d(64, d_model, 1, bias=False) if 64 != d_model else nn.Identity()
            self.act3 = nn.ReLU(inplace=True)

            self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, L) -> (N, d_model)
        if not self.residual:
            return self.net(x).squeeze(-1)

        h1 = self.act1(self.n1(self.c1(x)))
        h2 = self.act2(self.n2(self.c2(h1)) + self.skip1(x))
        h_pooled = self.pool(h2)
        h3 = self.act3(self.n3(self.c3(h_pooled)) + self.skip2(h_pooled))
        return self.gap(h3).squeeze(-1)


class MultiScaleSensorEncoder(nn.Module):
    """Multi-scale temporal encoder — captures both high-frequency shock bursts
    (bearing degradation) and low-frequency degradation trends (turbofan drift)
    simultaneously using parallel convolution branches.

    Architecture:
        Stage 1 — Multi-scale parallel branches:
            Branch_3  : Conv1d(1, c1, kernel_size=3)  → local patterns / spikes
            Branch_5  : Conv1d(1, c2, kernel_size=5)  → medium-scale features
            Branch_9  : Conv1d(1, c3, kernel_size=9)  → long-range trends
            → Concatenate along channel dim → Conv1d(c_total, 32, 1) fusion

        Stage 2 — Standard residual conv:
            Conv1d(32, 64, 3)  + skip(32 → 64)
            MaxPool1d(2)
            Conv1d(64, d_model, 3) + skip(64 → d_model)
            AdaptiveAvgPool1d(1)

    The multi-scale front-end adds only ~1k parameters over the single-kernel
    baseline but dramatically improves cross-domain generalisation by letting the
    model match temporal scales to each domain's degradation dynamics.

    Motivation: turbofan degradation is a slow drift over hundreds of cycles, while
    bearing faults manifest as sudden impulse bursts spanning 2–3 samples. A single
    kernel size forces a compromise; parallel branches eliminate it.
    """

    def __init__(self, d_model: int = 64, norm: str = "group",
                 branch_channels: tuple[int, ...] = (12, 12, 8),
                 kernel_sizes: tuple[int, ...] = (3, 5, 9)):
        super().__init__()
        assert len(branch_channels) == len(kernel_sizes)
        Norm = nn.BatchNorm1d if norm == "batch" else (lambda c: nn.GroupNorm(1, c))

        c_total = sum(branch_channels)

        # --- Stage 1: Multi-scale parallel conv branches ---
        self.branches = nn.ModuleList()
        for c, k in zip(branch_channels, kernel_sizes):
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, c, k, padding=k // 2),
                Norm(c),
                nn.ReLU(inplace=True),
            ))
        # 1x1 pointwise fusion: mixes the concatenated multi-scale features
        self.fuse = nn.Sequential(
            nn.Conv1d(c_total, 32, 1),
            Norm(32),
            nn.ReLU(inplace=True),
        )

        # --- Stage 2: standard residual conv (identical to SensorEncoder) ---
        self.c2 = nn.Conv1d(32, 64, 3, padding=1)
        self.n2 = Norm(64)
        self.skip1 = nn.Conv1d(32, 64, 1, bias=False)
        self.act2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(2)

        self.c3 = nn.Conv1d(64, d_model, 3, padding=1)
        self.n3 = Norm(d_model)
        self.skip2 = nn.Conv1d(64, d_model, 1, bias=False) if 64 != d_model else nn.Identity()
        self.act3 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, L) -> (N, d_model)

        # Multi-scale feature extraction
        branch_outs = [branch(x) for branch in self.branches]
        h1 = torch.cat(branch_outs, dim=1)  # (N, c_total, L)
        h1 = self.fuse(h1)                  # (N, 32, L)

        # Standard residual stages
        h2 = self.act2(self.n2(self.c2(h1)) + self.skip1(h1))
        h_pooled = self.pool(h2)
        h3 = self.act3(self.n3(self.c3(h_pooled)) + self.skip2(h_pooled))
        return self.gap(h3).squeeze(-1)

