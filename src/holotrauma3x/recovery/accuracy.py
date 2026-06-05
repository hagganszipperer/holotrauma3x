from __future__ import annotations

import math

import torch


def positional_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction[..., :3] - target[..., :3]).abs().mean())


def rotational_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    radians = (prediction[..., 3:] - target[..., 3:]).abs().mean()
    return float(radians) * 180.0 / math.pi


def airway_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).abs().mean())


def condyle_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).norm(dim=-1).mean())


def occlusal_class_i(occlusion: torch.Tensor, tolerance: float = 0.5) -> float:
    within = (occlusion.abs() <= tolerance).all(dim=-1)
    return 100.0 * float(within.float().mean())
