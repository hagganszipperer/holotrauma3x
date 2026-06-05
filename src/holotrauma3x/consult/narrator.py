from __future__ import annotations

import torch
from torch import nn


class VisionToLanguageDecoder(nn.Module):
    def __init__(self, proj_dim: int, hidden: int, heads: int, layers: int, vocab: int) -> None:
        super().__init__()
        self.memory = nn.Linear(proj_dim, hidden)
        self.queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.vocabulary = nn.Linear(hidden, vocab)

    def forward(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = projected.shape[0]
        memory = self.memory(projected).unsqueeze(1)
        queries = self.queries.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        pooled = decoded.mean(dim=1)
        logits = self.vocabulary(pooled)
        return pooled, logits
