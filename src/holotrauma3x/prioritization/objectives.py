from __future__ import annotations

import torch
import torch.nn.functional as F

from holotrauma3x.prioritization.weights import PriorityWeights


def airway_loss(
    area: torch.Tensor,
    volume: torch.Tensor,
    target_volume: torch.Tensor,
    beta: tuple[float, float],
    critical: float,
) -> torch.Tensor:
    shortfall = F.relu(critical - area)
    cubic = beta[0] * shortfall.pow(3)
    target = beta[1] * (volume - target_volume).pow(2)
    return (cubic + target).mean()


def stabilization_loss(
    offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    stability: torch.Tensor,
    gamma: tuple[float, float],
) -> torch.Tensor:
    reduction = gamma[0] * (offsets - target_offsets).pow(2).sum(dim=(-1, -2))
    fixation = gamma[1] * F.relu(1.0 - stability).sum(dim=-1)
    return (reduction + fixation).mean()


def occlusion_loss(
    occlusion: torch.Tensor,
    target: torch.Tensor,
    alpha: tuple[float, float, float, float],
) -> torch.Tensor:
    weights = torch.tensor(alpha, device=occlusion.device)
    return (weights * (occlusion - target).pow(2)).sum(dim=-1).mean()


def symmetry_loss(symmetry: torch.Tensor) -> torch.Tensor:
    return symmetry.pow(2).sum(dim=-1).mean()


def aesthetic_loss(aesthetic: torch.Tensor) -> torch.Tensor:
    return aesthetic.pow(2).sum(dim=-1).mean()


def joint_loss(condyle: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (condyle - target).pow(2).sum(dim=-1).mean()


def total_objective(
    weights: PriorityWeights,
    airway: torch.Tensor,
    stabilization: torch.Tensor,
    occlusion: torch.Tensor,
    symmetry: torch.Tensor,
    aesthetic: torch.Tensor,
    joint: torch.Tensor,
) -> torch.Tensor:
    return (
        weights.airway * airway
        + weights.stabilization * stabilization
        + weights.function * occlusion
        + weights.longterm * (symmetry + aesthetic)
        + joint
    )
