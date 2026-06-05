from __future__ import annotations

import torch
import torch.nn.functional as F


def _percentile_clip(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    flat = x.flatten()
    lo = torch.quantile(flat, low)
    hi = torch.quantile(flat, high)
    return torch.clamp(x, float(lo), float(hi))


def prepare_cbct(volume: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, 3) / 27.0
    smoothed = F.conv3d(volume.unsqueeze(0), kernel, padding=1).squeeze(0)
    clipped = _percentile_clip(smoothed, 0.005, 0.995)
    mean = clipped.mean()
    std = clipped.std().clamp_min(1e-6)
    return (clipped - mean) / std


def farthest_point_sample(points: torch.Tensor, count: int) -> torch.Tensor:
    n = points.shape[0]
    if count >= n:
        return points
    selected = torch.zeros(count, dtype=torch.long)
    distances = torch.full((n,), float("inf"))
    current = 0
    for i in range(count):
        selected[i] = current
        delta = points - points[current]
        dist = (delta * delta).sum(dim=1)
        distances = torch.minimum(distances, dist)
        current = int(torch.argmax(distances))
    return points[selected]


def remove_outliers(points: torch.Tensor, k: int, std_ratio: float) -> torch.Tensor:
    if points.shape[0] <= k:
        return points
    centroid = points.mean(dim=0)
    radial = (points - centroid).norm(dim=1)
    threshold = radial.mean() + std_ratio * radial.std()
    keep = radial <= threshold
    if not bool(keep.any()):
        return points
    return points[keep]


def normalize_unit_sphere(points: torch.Tensor) -> torch.Tensor:
    centroid = points.mean(dim=0)
    centered = points - centroid
    scale = centered.norm(dim=1).max().clamp_min(1e-6)
    normalized: torch.Tensor = centered / scale
    return normalized


def prepare_cast(points: torch.Tensor, target: int, k: int, std_ratio: float) -> torch.Tensor:
    cleaned = remove_outliers(points, k, std_ratio)
    sampled = farthest_point_sample(cleaned, min(target, cleaned.shape[0]))
    return normalize_unit_sphere(sampled)


def prepare_cephalogram(image: torch.Tensor) -> torch.Tensor:
    clipped = _percentile_clip(image, 0.02, 0.98)
    lo = clipped.amin()
    hi = clipped.amax().clamp_min(float(lo) + 1e-6)
    return (clipped - lo) / (hi - lo)
