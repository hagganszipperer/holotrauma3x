from __future__ import annotations

from dataclasses import dataclass

import torch

from holotrauma3x.prioritization.objectives import (
    aesthetic_loss,
    airway_loss,
    joint_loss,
    occlusion_loss,
    stabilization_loss,
    symmetry_loss,
    total_objective,
)
from holotrauma3x.prioritization.weights import PriorityWeights
from holotrauma3x.protocol import InjuryState, OptimizationSpec

_FIXATION = 4
_TARGET_VOLUME = 18.0


@dataclass
class PlanResult:
    airway_area: torch.Tensor
    airway_volume: torch.Tensor
    offsets: torch.Tensor
    occlusion: torch.Tensor
    stability: torch.Tensor
    losses: list[float]


class PriorityPlanner:
    def __init__(self, spec: OptimizationSpec) -> None:
        self.spec = spec
        self.weights = PriorityWeights.from_spec(spec)

    def solve(self, prediction: InjuryState) -> PlanResult:
        spec = self.spec
        area0 = prediction.airway_area.detach()
        volume0 = prediction.airway_volume.detach()
        offsets0 = prediction.fracture_offsets.detach()
        occlusion0 = prediction.occlusion.detach()
        batch = area0.shape[0]
        device = area0.device

        airway_gain = torch.zeros(batch, 1, device=device, requires_grad=True)
        volume_gain = torch.zeros(batch, 1, device=device, requires_grad=True)
        correction = torch.zeros_like(offsets0, requires_grad=True)
        occlusion_fix = torch.zeros_like(occlusion0, requires_grad=True)
        condyle = torch.zeros(batch, 3, device=device, requires_grad=True)
        symmetry = torch.zeros(batch, 4, device=device, requires_grad=True)
        aesthetic = torch.zeros(batch, 4, device=device, requires_grad=True)
        stability_logit = torch.zeros(batch, _FIXATION, device=device, requires_grad=True)

        variables = [
            airway_gain,
            volume_gain,
            correction,
            occlusion_fix,
            condyle,
            symmetry,
            aesthetic,
            stability_logit,
        ]
        optimizer = torch.optim.Adam(
            variables, lr=spec.learning_rate, betas=(spec.beta1, spec.beta2)
        )
        target_volume = torch.full_like(volume0, _TARGET_VOLUME)
        target_offsets = torch.zeros_like(offsets0)
        target_occlusion = torch.zeros_like(occlusion0)
        target_condyle = torch.zeros_like(condyle)
        losses: list[float] = []

        for _ in range(spec.max_iterations):
            optimizer.zero_grad()
            area = area0 + airway_gain
            volume = volume0 + volume_gain
            offsets = offsets0 + correction
            occlusion = occlusion0 + occlusion_fix
            stability = torch.sigmoid(stability_logit)
            loss = total_objective(
                self.weights,
                airway_loss(area, volume, target_volume, spec.airway_beta, spec.airway_critical),
                stabilization_loss(offsets, target_offsets, stability, spec.stabilization_gamma),
                occlusion_loss(occlusion, target_occlusion, spec.occlusion_alpha),
                symmetry_loss(symmetry),
                aesthetic_loss(aesthetic),
                joint_loss(condyle, target_condyle),
            )
            torch.autograd.backward(loss)
            optimizer.step()
            losses.append(float(loss.item()))

        return PlanResult(
            airway_area=(area0 + airway_gain).detach(),
            airway_volume=(volume0 + volume_gain).detach(),
            offsets=(offsets0 + correction).detach(),
            occlusion=(occlusion0 + occlusion_fix).detach(),
            stability=torch.sigmoid(stability_logit).detach(),
            losses=losses,
        )
