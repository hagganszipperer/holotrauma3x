from __future__ import annotations

import torch

_SUBSTEPS = 20
_MARGIN_SCALE = 0.01


def clearance(config: torch.Tensor, obstacles: torch.Tensor, radius: float) -> float:
    if obstacles.numel() == 0:
        return float("inf")
    distances = (obstacles - config).norm(dim=-1)
    return float(distances.min()) - radius


def collision_free(
    source: torch.Tensor,
    target: torch.Tensor,
    obstacles: torch.Tensor,
    radius: float,
    safety_margin: float,
) -> bool:
    if obstacles.numel() == 0:
        return True
    buffer = radius + _MARGIN_SCALE * safety_margin
    steps = torch.linspace(0.0, 1.0, _SUBSTEPS)
    for t in steps:
        point = source + (target - source) * t
        distances = (obstacles - point).norm(dim=-1)
        if bool((distances <= buffer).any()):
            return False
    return True
