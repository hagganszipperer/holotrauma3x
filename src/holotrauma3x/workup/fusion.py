from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.protocol import FusionSpec


class CrossModalFusion(nn.Module):
    def __init__(self, spec: FusionSpec, dim_volume: int, dim_dental: int, dim_ceph: int) -> None:
        super().__init__()
        self.spec = spec
        enc = spec.enc_dim
        self.volume_proj = nn.Linear(dim_volume, enc)
        self.dental_proj = nn.Linear(dim_dental, enc)
        self.ceph_proj = nn.Linear(dim_ceph, enc)
        self.attention = nn.MultiheadAttention(enc, spec.heads, batch_first=True)
        self.merge = nn.Sequential(
            nn.Linear(enc * 4, enc),
            nn.LayerNorm(enc),
            nn.GELU(),
            nn.Linear(enc, enc),
        )

    def forward(
        self, f_volume: torch.Tensor, f_dental: torch.Tensor, f_ceph: torch.Tensor, mode: str
    ) -> torch.Tensor:
        v = self.volume_proj(f_volume)
        d = self.dental_proj(f_dental)
        c = self.ceph_proj(f_ceph)
        wv, wd, wc = self.spec.weight_volume, self.spec.weight_dental, self.spec.weight_cephalogram
        if mode == "late":
            late: torch.Tensor = (wv * v + wd * d + wc * c) / (wv + wd + wc)
            return late
        if mode == "early":
            merged = wv * v + wd * d + wc * c
            early: torch.Tensor = self.merge(torch.cat([v, d, c, merged], dim=-1))
            return early
        if mode == "concat":
            joined: torch.Tensor = self.merge(
                torch.cat([wv * v, wd * d, wc * c, torch.zeros_like(v)], dim=-1)
            )
            return joined
        query = v.unsqueeze(1)
        context = torch.stack([d, c], dim=1)
        attended, _ = self.attention(query, context, context)
        f_vd = attended.squeeze(1)
        fused: torch.Tensor = self.merge(torch.cat([wv * v, wd * d, wc * c, f_vd], dim=-1))
        return fused
