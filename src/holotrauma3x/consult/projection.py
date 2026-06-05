from __future__ import annotations

import torch
from torch import nn


class ClipProjection(nn.Module):
    def __init__(self, clip_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(clip_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.emphasis = nn.Parameter(torch.ones(proj_dim))

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        projected = self.dropout(self.norm(self.linear(visual)))
        result: torch.Tensor = projected * self.emphasis
        return result
