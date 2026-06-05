from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.consult.deliberation import (
    ApiReasoner,
    DirectRegressor,
    GrammarReasoner,
    ReasoningEngine,
    ReasoningOutput,
)
from holotrauma3x.consult.knowledge import TraumaKnowledgeBase
from holotrauma3x.consult.narrator import VisionToLanguageDecoder
from holotrauma3x.consult.projection import ClipProjection
from holotrauma3x.protocol import CoReasoningSpec


def _build_engine(spec: CoReasoningSpec, pooled_dim: int, context_dim: int) -> ReasoningEngine:
    if not spec.enabled or spec.engine == "regression":
        return DirectRegressor(pooled_dim)
    if spec.engine == "api":
        return ApiReasoner(
            pooled_dim, context_dim, 50.0, spec.confidence_threshold, "gpt-4-turbo-2024-04-09"
        )
    return GrammarReasoner(pooled_dim, context_dim, 50.0, spec.confidence_threshold)


class CoReasoner(nn.Module):
    def __init__(self, spec: CoReasoningSpec, visual_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.projection = ClipProjection(visual_dim, spec.proj_dim, 0.15)
        self.narrator = VisionToLanguageDecoder(
            spec.proj_dim, spec.decoder_hidden, spec.decoder_heads, spec.decoder_layers, spec.vocab
        )
        self.knowledge = TraumaKnowledgeBase(spec.knowledge_size, spec.proj_dim, spec.top_m)
        self.engine = _build_engine(spec, spec.decoder_hidden, spec.proj_dim)

    def forward(self, visual: torch.Tensor) -> ReasoningOutput:
        projected = self.projection(visual)
        pooled, _ = self.narrator(projected)
        context, scores = self.knowledge.retrieve(projected)
        return self.engine.deliberate(pooled, context, scores)
