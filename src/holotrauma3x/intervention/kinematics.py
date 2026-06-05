from __future__ import annotations

from dataclasses import dataclass

import torch

from holotrauma3x.protocol import RobotSpec


@dataclass(frozen=True)
class JointLimits:
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]

    @classmethod
    def from_spec(cls, spec: RobotSpec) -> JointLimits:
        return cls(velocity=spec.velocity_limits, acceleration=spec.acceleration_limits)

    def velocity_tensor(self) -> torch.Tensor:
        return torch.tensor(self.velocity)


def steer(source: torch.Tensor, target: torch.Tensor, step: float) -> torch.Tensor:
    direction = target - source
    distance = float(direction.norm())
    if distance <= step or distance < 1e-9:
        return target.clone()
    return source + direction * (step / distance)


def cost_time(
    source: torch.Tensor,
    target: torch.Tensor,
    limits: JointLimits,
    alpha_distance: float,
    alpha_time: float,
) -> float:
    delta = (target - source).abs()
    distance = float((target - source).norm())
    velocity = limits.velocity_tensor()[: delta.shape[0]]
    travel = float((delta / velocity.clamp_min(1e-6)).max())
    return alpha_distance * distance + alpha_time * travel
