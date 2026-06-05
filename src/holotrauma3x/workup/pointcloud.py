from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.protocol import PointSpec

_LADDER: tuple[tuple[int, ...], ...] = (
    (32, 32, 64),
    (64, 64, 128),
    (128, 128, 256),
    (256, 256, 512),
)
_NEIGHBORS = 32


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    inner = -2.0 * torch.matmul(src, dst.transpose(1, 2))
    src_sq = (src * src).sum(dim=-1, keepdim=True)
    dst_sq = (dst * dst).sum(dim=-1).unsqueeze(1)
    return inner + src_sq + dst_sq


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch = points.shape[0]
    view = [batch] + [1] * (idx.dim() - 1)
    expand = list(idx.shape)
    arange = torch.arange(batch, device=points.device).view(view).expand(expand)
    return points[arange, idx]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    batch, total, _ = xyz.shape
    npoint = min(npoint, total)
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, total), float("inf"), device=xyz.device)
    current = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    batch_index = torch.arange(batch, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = current
        centroid = xyz[batch_index, current].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        current = torch.argmax(distance, dim=-1)
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    batch, total, _ = xyz.shape
    _, npoint, _ = new_xyz.shape
    dists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(total, device=xyz.device).view(1, 1, total).repeat(batch, npoint, 1)
    group_idx = group_idx.clone()
    group_idx[dists > radius * radius] = total
    width = min(nsample, total)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :width]
    first = group_idx[:, :, 0:1].repeat(1, 1, width)
    mask = group_idx == total
    group_idx[mask] = first[mask]
    return group_idx


class SetAbstraction(nn.Module):
    def __init__(self, in_channels: int, mlp: tuple[int, ...], radius: float, npoint: int) -> None:
        super().__init__()
        self.radius = radius
        self.npoint = npoint
        blocks: list[nn.Module] = []
        channels = in_channels
        for out_channels in mlp:
            blocks.append(nn.Conv2d(channels, out_channels, 1))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
            channels = out_channels
        self.network = nn.Sequential(*blocks)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centroid_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, centroid_idx)
        group_idx = query_ball_point(self.radius, _NEIGHBORS, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
        if features is None:
            grouped = grouped_xyz
        else:
            grouped_feat = index_points(features, group_idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        grouped = grouped.permute(0, 3, 1, 2)
        encoded = self.network(grouped)
        pooled = encoded.max(dim=-1)[0].transpose(1, 2)
        return new_xyz, pooled


class PointNetTrauma(nn.Module):
    def __init__(self, spec: PointSpec) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        for ladder, radius, npoint in zip(_LADDER, spec.radii, spec.samples, strict=False):
            self.layers.append(SetAbstraction(in_channels, ladder, radius, npoint))
            in_channels = ladder[-1] + 3
        self.head = nn.Linear(_LADDER[-1][-1], spec.out_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points
        features: torch.Tensor | None = None
        for layer in self.layers:
            xyz, features = layer(xyz, features)
        assert features is not None
        pooled = features.max(dim=1)[0]
        projected: torch.Tensor = self.head(pooled)
        return projected
