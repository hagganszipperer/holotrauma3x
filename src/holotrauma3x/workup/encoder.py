from __future__ import annotations

import torch
from torch import nn

from holotrauma3x.protocol import EncoderSpec, TriadInputs
from holotrauma3x.workup.fusion import CrossModalFusion
from holotrauma3x.workup.pointcloud import PointNetTrauma
from holotrauma3x.workup.radiographic import CephalometricEncoder
from holotrauma3x.workup.volumetric import VolumetricTransformer


class TriadicEncoder(nn.Module):
    def __init__(self, spec: EncoderSpec, volume_size: int, modalities: tuple[str, ...]) -> None:
        super().__init__()
        self.spec = spec
        self.modalities = modalities
        self.volume = VolumetricTransformer(spec.volumetric, volume_size)
        self.point = PointNetTrauma(spec.point)
        self.ceph = CephalometricEncoder(spec.radiograph)
        self.fusion = CrossModalFusion(
            spec.fusion, spec.volumetric.dim, spec.point.out_dim, spec.radiograph.out_dim
        )

    def forward(self, inputs: TriadInputs) -> torch.Tensor:
        f_volume = self.volume(inputs.volume)
        if "dental" in self.modalities:
            f_dental = self.point(inputs.points)
        else:
            f_dental = torch.zeros(
                f_volume.shape[0], self.spec.point.out_dim, device=f_volume.device
            )
        if "cephalogram" in self.modalities:
            f_ceph = self.ceph(inputs.radiograph)
        else:
            f_ceph = torch.zeros(
                f_volume.shape[0], self.spec.radiograph.out_dim, device=f_volume.device
            )
        fused: torch.Tensor = self.fusion(f_volume, f_dental, f_ceph, self.spec.fusion_mode)
        return fused
