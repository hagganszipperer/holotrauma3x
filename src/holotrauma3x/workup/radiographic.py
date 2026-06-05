from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.protocol import RadiographSpec

_STAGES: tuple[tuple[int, int], ...] = ((64, 3), (128, 4), (256, 6), (512, 3))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        activated: torch.Tensor = self.relu(out + identity)
        return activated


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1, hidden, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.relu(self.reduce(feature) + self.bias)
        logits = self.score(hidden)
        batch, _, height, width = logits.shape
        weights = torch.softmax(logits.reshape(batch, 1, height * width), dim=-1)
        flat = feature.reshape(batch, feature.shape[1], height * width)
        pooled = torch.einsum("bck,bok->bc", flat, weights)
        return pooled, weights.reshape(batch, height, width)


class CephalometricEncoder(nn.Module):
    def __init__(self, spec: RadiographSpec) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        stages: list[nn.Module] = []
        in_channels = 64
        for index, (channels, blocks) in enumerate(_STAGES):
            stride = 1 if index == 0 else 2
            stages.append(BasicBlock(in_channels, channels, stride))
            for _ in range(blocks - 1):
                stages.append(BasicBlock(channels, channels, 1))
            in_channels = channels
        self.backbone = nn.Sequential(*stages)
        self.project = nn.Conv2d(512, spec.out_dim, 1)
        self.attention = SpatialAttention(spec.out_dim, spec.reduction)

    def forward(self, radiograph: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(self.stem(radiograph))
        projected = self.project(feature)
        pooled, _ = self.attention(projected)
        result: torch.Tensor = pooled
        return result
