from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from holotrauma3x.consult.severity import airway_band, fiss_band
from holotrauma3x.protocol import InjuryState

_SEGMENTS = 4


@dataclass
class ReasoningOutput:
    injury: InjuryState
    confidence: torch.Tensor
    narrative: list[str]


class ReasoningEngine(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.area = nn.Linear(in_dim, 1)
        self.volume = nn.Linear(in_dim, 1)
        self.fracture = nn.Linear(in_dim, _SEGMENTS * 3)
        self.occlusion = nn.Linear(in_dim, 4)
        self.fiss = nn.Linear(in_dim, 1)
        self.confidence = nn.Linear(in_dim, 1)

    def _heads(self, latent: torch.Tensor) -> InjuryState:
        area = 30.0 + 40.0 * torch.sigmoid(self.area(latent))
        volume = 8.0 + 18.0 * torch.sigmoid(self.volume(latent))
        fracture = self.fracture(latent).reshape(-1, _SEGMENTS, 3)
        occlusion = torch.tanh(self.occlusion(latent))
        fiss = 1.0 + 8.0 * torch.sigmoid(self.fiss(latent))
        return InjuryState(
            airway_area=area,
            airway_volume=volume,
            fracture_offsets=fracture,
            occlusion=occlusion,
            fiss=fiss,
        )

    def deliberate(
        self, pooled: torch.Tensor, context: torch.Tensor, scores: torch.Tensor
    ) -> ReasoningOutput:
        raise NotImplementedError


class GrammarReasoner(ReasoningEngine):
    def __init__(
        self, pooled_dim: int, context_dim: int, critical: float, threshold: float
    ) -> None:
        super().__init__(pooled_dim + context_dim)
        self.critical = critical
        self.threshold = threshold

    def deliberate(
        self, pooled: torch.Tensor, context: torch.Tensor, scores: torch.Tensor
    ) -> ReasoningOutput:
        latent = torch.cat([pooled, context], dim=-1)
        injury = self._heads(latent)
        grounding = torch.sigmoid(scores.mean(dim=-1, keepdim=True))
        confidence = 0.5 * torch.sigmoid(self.confidence(latent)) + 0.5 * grounding
        narrative = self._render(injury, confidence)
        return ReasoningOutput(injury=injury, confidence=confidence, narrative=narrative)

    def _render(self, injury: InjuryState, confidence: torch.Tensor) -> list[str]:
        lines: list[str] = []
        for i in range(injury.airway_area.shape[0]):
            area = float(injury.airway_area[i].item())
            band = airway_band(area, self.critical)
            offset = float(injury.fracture_offsets[i].norm(dim=-1).mean().item())
            overjet = float(injury.occlusion[i, 2].item())
            severity = fiss_band(float(injury.fiss[i].item()))
            conf = float(confidence[i].item())
            review = " senior-review" if conf < self.threshold else ""
            verdict = "below" if area < self.critical else "meets"
            lines.append(
                f"[life-saving] airway {area:.0f} mm2 {verdict} {self.critical:.0f} ({band}); "
                f"[stabilization] reduce {_SEGMENTS} segments offset {offset:.1f} mm; "
                f"[functional] class-I overjet {overjet:.1f}; "
                f"[risk] {severity} confidence {conf:.2f}{review}"
            )
        return lines


class DirectRegressor(ReasoningEngine):
    def __init__(self, pooled_dim: int) -> None:
        super().__init__(pooled_dim)

    def deliberate(
        self, pooled: torch.Tensor, context: torch.Tensor, scores: torch.Tensor
    ) -> ReasoningOutput:
        injury = self._heads(pooled)
        confidence = torch.sigmoid(self.confidence(pooled))
        empty = ["" for _ in range(pooled.shape[0])]
        return ReasoningOutput(injury=injury, confidence=confidence, narrative=empty)


class ApiReasoner(GrammarReasoner):
    def __init__(
        self, pooled_dim: int, context_dim: int, critical: float, threshold: float, model: str
    ) -> None:
        super().__init__(pooled_dim, context_dim, critical, threshold)
        self.model = model
        self.client: Any = None

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        try:
            import openai

            self.client = openai.OpenAI()
        except Exception:
            self.client = None
