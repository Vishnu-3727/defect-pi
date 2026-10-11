"""Masked cross-attention pooling — turns a variable-size sensor set into a
fixed-size context vector.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

M *learned* query vectors attend over the S sensor tokens, so the output is
(B, M*D) no matter what S is. The alternatives all fail one requirement:

  concatenate (B, S*D)  — output shape welded to S. Fatal.
  mean pool             — handles variable S, but treats a dead constant channel
                          exactly like the one screaming about failure.
  cross-attention       — variable S, learned INPUT-DEPENDENT weighting, missing
                          sensors handled by the mask, and the weights are a free
                          per-prediction explanation for the dashboard.

The 1/sqrt(d_k) matters: without it the dot products have variance ~d_k, pushing
softmax into saturation where gradients vanish.

Written with plain matmuls rather than nn.MultiheadAttention so the ONNX export
traces cleanly with a dynamic sensor axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Finite, not -inf: a fully-masked row would produce NaN, and -inf exports to
# ONNX less predictably. -1e4 also survives fp16.
MASK_FILL = -1e4


class CrossAttentionPool(nn.Module):
    def __init__(self, d_model: int = 64, n_queries: int = 1, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.m = n_queries

        self.query = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """tokens (B, S, D), mask (B, S) bool -> context (B, M*D), attn (B, h, M, S)."""
        B, S, D = tokens.shape

        q = self.Wq(self.query.expand(B, -1, -1)).view(B, self.m, self.h, self.dk).transpose(1, 2)
        k = self.Wk(tokens).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.Wv(tokens).view(B, S, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)          # (B, h, M, S)
        # Mask BEFORE softmax. Masking after would leave the weights not summing to 1.
        scores = scores.masked_fill(~mask[:, None, None, :], MASK_FILL)
        attn = scores.softmax(dim=-1)

        out = self.drop(attn) @ v                                       # (B, h, M, dk)
        out = out.transpose(1, 2).reshape(B, self.m, D)
        return self.Wo(out).reshape(B, self.m * D), attn


class SensorSelfAttention(nn.Module):
    """Optional: let sensors inform each other before pooling.

    Lets the model represent conjunctions like "temperature up AND pressure down".
    One extra layer, typically worth 1-2 RMSE. Off by default — get the baseline
    working first (docs/06-model.md §6.12).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        B, S, D = tokens.shape
        q = self.Wq(tokens).view(B, S, self.h, self.dk).transpose(1, 2)
        k = self.Wk(tokens).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.Wv(tokens).view(B, S, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        scores = scores.masked_fill(~mask[:, None, None, :], MASK_FILL)
        out = (self.drop(scores.softmax(-1)) @ v).transpose(1, 2).reshape(B, S, D)
        # Zero the padded rows so they cannot leak into the residual stream.
        return self.norm(tokens + self.Wo(out)) * mask[..., None]
