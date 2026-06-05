from __future__ import annotations

from typing import TypedDict

import torch
from torch.utils.data import Dataset

from holotrauma3x.protocol import ArrivalSpec

_LATENT = 16
_SEGMENTS = 4


class Sample(TypedDict):
    volume: torch.Tensor
    points: torch.Tensor
    radiograph: torch.Tensor
    airway_area: torch.Tensor
    airway_volume: torch.Tensor
    fracture_offsets: torch.Tensor
    occlusion: torch.Tensor
    fiss: torch.Tensor
    maxilla: torch.Tensor
    mandible: torch.Tensor
    condyle: torch.Tensor
    center: torch.Tensor


def _basis(rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(rows, cols, generator=generator)
    normalized: torch.Tensor = matrix / (matrix.norm(dim=0, keepdim=True) + 1e-6)
    return normalized


class TraumaCohort(Dataset[Sample]):
    def __init__(self, spec: ArrivalSpec, size: int, seed: int, centers: int = 12) -> None:
        self.spec = spec
        self.size = size
        self.seed = seed
        self.centers = centers
        grid = spec.volume_size
        self.volume_basis = _basis(grid * grid * grid, _LATENT, seed + 1)
        self.point_basis = _basis(spec.point_count * 3, _LATENT, seed + 2)
        self.image_basis = _basis(spec.radiograph_size * spec.radiograph_size, _LATENT, seed + 3)
        self.area_head = _basis(_LATENT, 1, seed + 4)
        self.fracture_head = _basis(_LATENT, _SEGMENTS * 3, seed + 5)
        self.occlusion_head = _basis(_LATENT, 4, seed + 6)
        self.maxilla_head = _basis(_LATENT, 6, seed + 7)
        self.mandible_head = _basis(_LATENT, 6, seed + 8)
        self.condyle_head = _basis(_LATENT, 3, seed + 9)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Sample:
        generator = torch.Generator().manual_seed(self.seed * 100003 + index)
        latent = torch.randn(_LATENT, generator=generator)
        grid = self.spec.volume_size

        volume = (self.volume_basis @ latent).reshape(1, grid, grid, grid)
        volume = volume + 0.05 * torch.randn(volume.shape, generator=generator)
        points = (self.point_basis @ latent).reshape(self.spec.point_count, 3)
        points = points + 0.02 * torch.randn(points.shape, generator=generator)
        radiograph = (self.image_basis @ latent).reshape(
            1, self.spec.radiograph_size, self.spec.radiograph_size
        )
        radiograph = radiograph + 0.05 * torch.randn(radiograph.shape, generator=generator)

        severity = torch.sigmoid(self.area_head.squeeze(-1) @ latent)
        airway_area = 30.0 + 40.0 * severity
        airway_volume = 8.0 + 18.0 * severity
        fracture_offsets = (self.fracture_head.t() @ latent).reshape(_SEGMENTS, 3)
        occlusion = torch.tanh(self.occlusion_head.t() @ latent)
        fiss = 1.0 + 8.0 * (1.0 - severity)
        maxilla = 0.5 * (self.maxilla_head.t() @ latent)
        mandible = 0.5 * (self.mandible_head.t() @ latent)
        condyle = 0.5 * (self.condyle_head.t() @ latent)
        center = torch.tensor(index % self.centers, dtype=torch.long)

        return Sample(
            volume=volume,
            points=points,
            radiograph=radiograph,
            airway_area=airway_area.reshape(1),
            airway_volume=airway_volume.reshape(1),
            fracture_offsets=fracture_offsets,
            occlusion=occlusion,
            fiss=fiss.reshape(1),
            maxilla=maxilla,
            mandible=mandible,
            condyle=condyle,
            center=center,
        )
