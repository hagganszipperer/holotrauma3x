from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from holotrauma3x.consult.coreasoner import CoReasoner
from holotrauma3x.consult.deliberation import ReasoningOutput
from holotrauma3x.intervention.policy import EmergencyPolicy
from holotrauma3x.intervention.rrtstar import EmergencyNeuralRRTStar
from holotrauma3x.prioritization.planner import PriorityPlanner
from holotrauma3x.protocol import Protocol, SurgicalPlan, TriadInputs
from holotrauma3x.workup.encoder import TriadicEncoder


@dataclass
class PathwayOutput:
    fused: torch.Tensor
    maxilla: torch.Tensor
    mandible: torch.Tensor
    condyle: torch.Tensor
    reasoning: ReasoningOutput


class RepositioningHead(nn.Module):
    def __init__(self, enc_dim: int) -> None:
        super().__init__()
        self.maxilla = nn.Linear(enc_dim, 6)
        self.mandible = nn.Linear(enc_dim, 6)
        self.condyle = nn.Linear(enc_dim, 3)

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.maxilla(fused), self.mandible(fused), self.condyle(fused)


class HoloTrauma3x(nn.Module):
    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.protocol = protocol
        enc_dim = protocol.encoder.fusion.enc_dim
        self.encoder = TriadicEncoder(
            protocol.encoder, protocol.arrival.volume_size, protocol.arrival.modalities
        )
        self.reposition = RepositioningHead(enc_dim)
        self.coreasoner = CoReasoner(protocol.coreasoning, enc_dim)
        self.policy = EmergencyPolicy(protocol.robot)
        self.goal_proj = nn.Linear(enc_dim, protocol.robot.dof)
        self.planner = PriorityPlanner(protocol.optimization)
        self.navigator = EmergencyNeuralRRTStar(protocol.robot, self.policy)

    def forward(self, inputs: TriadInputs) -> PathwayOutput:
        fused = self.encoder(inputs)
        maxilla, mandible, condyle = self.reposition(fused)
        reasoning = self.coreasoner(fused)
        return PathwayOutput(
            fused=fused,
            maxilla=maxilla,
            mandible=mandible,
            condyle=condyle,
            reasoning=reasoning,
        )

    def plan(self, inputs: TriadInputs, seed: int = 42) -> SurgicalPlan:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs)
        if was_training:
            self.train()
        refined = self.planner.solve(output.reasoning.injury)
        dof = self.protocol.robot.dof
        grid = self.protocol.robot.obstacle_grid
        uncertainty = self.protocol.robot.uncertainty_grid
        goal = torch.tanh(self.goal_proj(output.fused[0].detach())).detach()
        start = torch.zeros(dof)
        obstacles = torch.empty(0, dof)
        trajectory = self.navigator.plan(
            start,
            goal,
            obstacles,
            torch.zeros(grid, grid, grid),
            torch.zeros(uncertainty, uncertainty, uncertainty),
            radius=0.5,
            urgency=1.0,
            seed=seed,
        )
        return SurgicalPlan(
            maxilla=output.maxilla,
            mandible=output.mandible,
            condyle=output.condyle,
            airway_volume=refined.airway_volume,
            airway_area=refined.airway_area,
            trajectory=trajectory.configs,
            confidence=output.reasoning.confidence,
            narrative=output.reasoning.narrative,
        )


def build_pathway(protocol: Protocol) -> HoloTrauma3x:
    return HoloTrauma3x(protocol)
