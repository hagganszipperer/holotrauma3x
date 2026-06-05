from __future__ import annotations

from typing import cast

import torch
from torch import nn


class TraumaKnowledgeBase(nn.Module):
    def __init__(self, size: int, dim: int, top_m: int, seed: int = 17) -> None:
        super().__init__()
        self.top_m = top_m
        generator = torch.Generator().manual_seed(seed)
        passages = torch.randn(size, dim, generator=generator)
        passages = passages / (passages.norm(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("passages", passages)

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = query / (query.norm(dim=-1, keepdim=True) + 1e-6)
        passages = cast(torch.Tensor, self.passages)
        relevance = normalized @ passages.t()
        scores, indices = relevance.topk(min(self.top_m, passages.shape[0]), dim=-1)
        retrieved = passages[indices]
        context = (torch.softmax(scores, dim=-1).unsqueeze(-1) * retrieved).sum(dim=1)
        return context, scores
