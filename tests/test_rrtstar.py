from __future__ import annotations

import torch

from holotrauma3x.intervention.collision import collision_free
from holotrauma3x.intervention.policy import EmergencyPolicy
from holotrauma3x.intervention.rrtstar import EmergencyNeuralRRTStar
from holotrauma3x.protocol import RobotSpec


def _planner(max_iterations: int) -> EmergencyNeuralRRTStar:
    spec = RobotSpec(dof=2, max_iterations=max_iterations, obstacle_grid=4, uncertainty_grid=4)
    return EmergencyNeuralRRTStar(spec, EmergencyPolicy(spec))


def test_reaches_goal() -> None:
    planner = _planner(1500)
    start = torch.zeros(2)
    goal = torch.tensor([0.8, -0.6])
    trajectory = planner.plan(
        start,
        goal,
        torch.empty(0, 2),
        torch.zeros(4, 4, 4),
        torch.zeros(4, 4, 4),
        radius=0.5,
        urgency=1.0,
        seed=3,
    )
    assert trajectory.success
    for i in range(trajectory.configs.shape[0] - 1):
        assert collision_free(
            trajectory.configs[i], trajectory.configs[i + 1], torch.empty(0, 2), 0.5, 5.0
        )


def test_timeout_escalates() -> None:
    planner = _planner(3)
    start = torch.zeros(2)
    goal = torch.tensor([3.0, 3.0])
    obstacles = torch.zeros(1, 2)
    trajectory = planner.plan(
        start, goal, obstacles, torch.zeros(4, 4, 4), torch.zeros(4, 4, 4), 0.6, 1.0, seed=1
    )
    assert not trajectory.success


def test_collision_blocks_path() -> None:
    obstacle = torch.zeros(1, 2)
    assert not collision_free(
        torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]), obstacle, 0.5, 5.0
    )
