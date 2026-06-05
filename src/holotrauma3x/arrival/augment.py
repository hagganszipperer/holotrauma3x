from __future__ import annotations

import torch

from holotrauma3x.protocol import ArrivalSpec


class TraumaAugment:
    def __init__(self, spec: ArrivalSpec, generator: torch.Generator) -> None:
        self.spec = spec
        self.generator = generator

    def _draw(self) -> bool:
        return bool(torch.rand(1, generator=self.generator) < self.spec.augment_prob)

    def volume(self, x: torch.Tensor) -> torch.Tensor:
        if not self._draw():
            return x
        noise = 0.015 * torch.randn(x.shape, generator=self.generator)
        return x + noise

    def points(self, x: torch.Tensor) -> torch.Tensor:
        if not self._draw():
            return x
        keep = torch.rand(x.shape[0], generator=self.generator) > 0.15
        if not bool(keep.any()):
            return x
        scale = 0.95 + 0.1 * torch.rand(1, generator=self.generator)
        return x[keep] * scale

    def radiograph(self, x: torch.Tensor) -> torch.Tensor:
        if not self._draw():
            return x
        brightness = 0.9 + 0.2 * torch.rand(1, generator=self.generator)
        return torch.clamp(x * brightness, 0.0, 1.0)
