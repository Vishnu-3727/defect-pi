"""UNIFY-RUL — the full model.

    x (B, S, L)                    windows, any S
      -> reshape (B*S, 1, L)
      -> shared 1-D CNN            same weights for every sensor and dataset
      -> (B, S, D)
      -> + sensor embedding        identity the shared encoder deliberately erased
      -> + domain embedding        lets the model apply rig-specific corrections
      -> masked cross-attention    variable-size set -> fixed context vector
      -> MLP head
      -> normalised RUL in [0,1]   dimensionless, so all datasets share one loss scale

~44k parameters, ~176 KB fp32. Sized to stay inside the Pi's 1 MB shared L2 cache.

Every operation is a plain tensor op — no .item(), no data-dependent branching, no
hard-coded shapes — so torch.onnx.export traces it cleanly with dynamic batch AND
sensor axes. That is a design constraint, not luck.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttentionPool, SensorSelfAttention
from .encoder import MultiScaleSensorEncoder, SensorEncoder


class UnifyRUL(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_queries: int = 1,
                 sensor_vocab: int = 64, n_domains: int = 8, dropout: float = 0.2,
                 attn_dropout: float = 0.1, sigmoid_output: bool = False,
                 n_outputs: int = 1, sensor_self_attention: bool = False,
                 norm: str = "group", residual: bool = False,
                 encoder_type: str = "auto"):
        super().__init__()
        # encoder_type: "plain", "residual", "multiscale", or "auto" (default).
        # "auto" resolves to "residual" if residual=True else "plain" for backward compat.
        if encoder_type == "auto":
            encoder_type = "residual" if residual else "plain"
        if encoder_type == "multiscale":
            self.encoder = MultiScaleSensorEncoder(d_model, norm=norm)
        else:
            self.encoder = SensorEncoder(d_model, norm=norm,
                                         residual=(encoder_type == "residual"))
        # padding_idx=0 keeps the PAD row at exactly zero and out of the gradient.
        self.sensor_emb = nn.Embedding(sensor_vocab, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(n_domains, d_model)
        self.self_attn = (SensorSelfAttention(d_model, n_heads, attn_dropout)
                          if sensor_self_attention else None)
        self.pool = CrossAttentionPool(d_model, n_queries, n_heads, attn_dropout)
        self.head = nn.Sequential(
            nn.Linear(n_queries * d_model, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )
        self.sigmoid_output = sigmoid_output
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                sensor_ids: torch.Tensor, domain_id: torch.Tensor):
        """x (B,S,L) float | mask (B,S) bool | sensor_ids (B,S) long | domain_id (B,) long

        Returns rul (B, n_outputs) in [0,1]-ish, and attn (B, heads, M, S).
        """
        B, S, L = x.shape

        # The reshape that carries the whole universality claim: S moves into the
        # batch dimension, so no weight shape depends on it.
        h = self.encoder(x.reshape(B * S, 1, L)).reshape(B, S, -1)

        h = h + self.sensor_emb(sensor_ids) + self.domain_emb(domain_id)[:, None, :]
        h = h * mask[..., None]                       # padded tokens contribute nothing

        if self.self_attn is not None:
            h = self.self_attn(h, mask)

        ctx, attn = self.pool(h, mask)
        out = self.head(ctx)
        if self.sigmoid_output:
            # Bounds the output, but saturates where the RUL cap piles most labels
            # at exactly 1.0. Off by default; see config/default.yaml.
            out = torch.sigmoid(out)
        return out, attn

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(cfg: dict) -> UnifyRUL:
    m = cfg["model"]
    n_out = (len(cfg["loss"]["quantiles"]) if cfg["loss"]["kind"] == "pinball" else 1)
    return UnifyRUL(
        d_model=m["d_model"], n_heads=m["n_heads"], n_queries=m["n_queries"],
        sensor_vocab=m["sensor_vocab"], n_domains=m["n_domains"],
        dropout=m["dropout"], attn_dropout=m["attn_dropout"],
        sigmoid_output=m["sigmoid_output"], n_outputs=n_out,
        sensor_self_attention=m.get("sensor_self_attention", False),
        norm=m.get("norm", "group"),
        residual=m.get("residual", False),
        encoder_type=m.get("encoder_type", "auto"),
    )
