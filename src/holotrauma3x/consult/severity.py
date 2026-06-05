from __future__ import annotations

import torch


def fiss_band(value: float) -> str:
    if value <= 2.0:
        return "mild"
    if value <= 4.0:
        return "moderate"
    if value <= 6.0:
        return "severe"
    return "critical"


def airway_band(area: float, critical: float) -> str:
    if area < critical * 0.64:
        return "severe"
    if area < critical:
        return "moderate"
    if area < critical * 1.4:
        return "partial"
    return "clear"


def needs_review(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    return confidence.squeeze(-1) < threshold
