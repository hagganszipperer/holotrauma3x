from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.protocol import VolumetricSpec


class InjuryAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, tokens)
        return pooled, weights


class VolumetricTransformer(nn.Module):
    def __init__(self, spec: VolumetricSpec, volume_size: int) -> None:
        super().__init__()
        if volume_size % spec.patch != 0:
            raise ValueError("volume_size must be divisible by patch")
        self.spec = spec
        side = volume_size // spec.patch
        tokens = side * side * side
        self.embed = nn.Conv3d(1, spec.dim, kernel_size=spec.patch, stride=spec.patch)
        self.position = nn.Parameter(torch.randn(1, tokens, spec.dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=spec.dim,
                nhead=spec.heads,
                dim_feedforward=spec.feedforward,
                dropout=spec.dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(spec.layers)
        )
        self.pool_every = max(1, spec.layers // 3)
        self.attention_pool = InjuryAttentionPool(spec.dim)

    @staticmethod
    def _halve(tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        if length % 2 != 0:
            return tokens
        return tokens.reshape(batch, length // 2, 2, dim).mean(dim=2)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(volume)
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + self.position
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens)
            last = index == len(self.layers) - 1
            if (index + 1) % self.pool_every == 0 and not last:
                tokens = self._halve(tokens)
        pooled, _ = self.attention_pool(tokens)
        result: torch.Tensor = pooled
        return result
