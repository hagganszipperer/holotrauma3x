from __future__ import annotations

import torch

from holotrauma3x.prioritization.planner import PriorityPlanner
from holotrauma3x.protocol import InjuryState, OptimizationSpec


def _compromised(batch: int = 2) -> InjuryState:
    return InjuryState(
        airway_area=torch.full((batch, 1), 32.0),
        airway_volume=torch.full((batch, 1), 12.0),
        fracture_offsets=torch.ones(batch, 4, 3),
        occlusion=0.5 * torch.ones(batch, 4),
        fiss=torch.full((batch, 1), 6.0),
    )


def test_planner_secures_airway() -> None:
    planner = PriorityPlanner(OptimizationSpec(max_iterations=2000, learning_rate=0.12))
    result = planner.solve(_compromised())
    assert float(result.airway_area.min()) >= 49.0


def test_planner_loss_decreases() -> None:
    planner = PriorityPlanner(OptimizationSpec(max_iterations=200, learning_rate=2e-2))
    result = planner.solve(_compromised())
    assert result.losses[-1] < result.losses[0]
