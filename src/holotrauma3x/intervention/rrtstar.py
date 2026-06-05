from __future__ import annotations

from dataclasses import dataclass

import torch

from holotrauma3x.intervention.collision import collision_free
from holotrauma3x.intervention.kinematics import JointLimits, cost_time, steer
from holotrauma3x.intervention.policy import EmergencyPolicy
from holotrauma3x.protocol import RobotSpec

_BOUND = 3.2
_GOAL_BIAS = 0.1
_MAX_REWIRE = 5


@dataclass
class Trajectory:
    configs: torch.Tensor
    success: bool
    iterations: int


class EmergencyNeuralRRTStar:
    def __init__(self, spec: RobotSpec, policy: EmergencyPolicy) -> None:
        self.spec = spec
        self.policy = policy
        self.limits = JointLimits.from_spec(spec)

    def _sample(
        self,
        nodes: list[torch.Tensor],
        goal: torch.Tensor,
        obstacle_grid: torch.Tensor,
        uncertainty_grid: torch.Tensor,
        urgency: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        roll = float(torch.rand(1, generator=generator))
        if roll < _GOAL_BIAS:
            return goal.clone()
        if roll < _GOAL_BIAS + self.spec.exploration or not self.spec.neural_guidance:
            return (2.0 * torch.rand(self.spec.dof, generator=generator) - 1.0) * _BOUND
        nearest = min(nodes, key=lambda node: float((node - goal).norm()))
        return self.policy.act(nearest, goal, obstacle_grid, uncertainty_grid, urgency, generator)

    def plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        obstacle_grid: torch.Tensor,
        uncertainty_grid: torch.Tensor,
        radius: float,
        urgency: float,
        seed: int,
    ) -> Trajectory:
        generator = torch.Generator().manual_seed(seed)
        nodes = [start.clone()]
        parents = [-1]
        costs = [0.0]
        margin = self.spec.safety_margin

        for iteration in range(self.spec.max_iterations):
            sample = self._sample(nodes, goal, obstacle_grid, uncertainty_grid, urgency, generator)
            nearest = min(range(len(nodes)), key=lambda j: float((nodes[j] - sample).norm()))
            candidate = steer(nodes[nearest], sample, self.spec.step_size)
            if not collision_free(nodes[nearest], candidate, obstacles, radius, margin):
                continue

            near = [
                j
                for j in range(len(nodes))
                if float((nodes[j] - candidate).norm()) < self.spec.connection_radius
            ]
            best = nearest
            best_cost = costs[nearest] + cost_time(
                nodes[nearest],
                candidate,
                self.limits,
                self.spec.alpha_distance,
                self.spec.alpha_time,
            )
            for j in near:
                tentative = costs[j] + cost_time(
                    nodes[j], candidate, self.limits, self.spec.alpha_distance, self.spec.alpha_time
                )
                if tentative < best_cost and collision_free(
                    nodes[j], candidate, obstacles, radius, margin
                ):
                    best = j
                    best_cost = tentative

            nodes.append(candidate)
            parents.append(best)
            costs.append(best_cost)
            new_index = len(nodes) - 1

            for j in near[:_MAX_REWIRE]:
                rewired = best_cost + cost_time(
                    candidate, nodes[j], self.limits, self.spec.alpha_distance, self.spec.alpha_time
                )
                if rewired < costs[j] and collision_free(
                    candidate, nodes[j], obstacles, radius, margin
                ):
                    parents[j] = new_index
                    costs[j] = rewired

            if float((candidate - goal).norm()) < self.spec.goal_threshold:
                return self._extract(nodes, parents, new_index, iteration + 1)

        return Trajectory(
            configs=start.unsqueeze(0), success=False, iterations=self.spec.max_iterations
        )

    @staticmethod
    def _extract(
        nodes: list[torch.Tensor], parents: list[int], index: int, iterations: int
    ) -> Trajectory:
        path: list[torch.Tensor] = []
        cursor = index
        while cursor != -1:
            path.append(nodes[cursor])
            cursor = parents[cursor]
        path.reverse()
        configs = torch.stack(path)
        if configs.shape[0] >= 3:
            smoothed = configs.clone()
            smoothed[1:-1] = (configs[:-2] + configs[1:-1] + configs[2:]) / 3.0
            configs = smoothed
        return Trajectory(configs=configs, success=True, iterations=iterations)
