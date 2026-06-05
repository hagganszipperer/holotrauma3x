from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from holotrauma3x.protocol import RobotSpec

_POOL = 2


class EmergencyPolicy(nn.Module):
    def __init__(self, spec: RobotSpec) -> None:
        super().__init__()
        self.dof = spec.dof
        self.guidance = spec.connection_radius
        descriptor = _POOL * _POOL * _POOL
        in_dim = spec.dof + descriptor + descriptor + spec.dof + 1
        blocks: list[nn.Module] = []
        width = in_dim
        for _ in range(spec.policy_layers):
            blocks.append(nn.Linear(width, spec.policy_units))
            blocks.append(nn.BatchNorm1d(spec.policy_units))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(spec.policy_dropout))
            width = spec.policy_units
        self.trunk = nn.Sequential(*blocks)
        self.mean = nn.Linear(width, spec.dof)
        self.log_std = nn.Linear(width, spec.dof)

    @staticmethod
    def _descriptor(grid: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(grid.reshape(1, 1, *grid.shape), _POOL)
        return pooled.flatten()

    def _features(
        self,
        config: torch.Tensor,
        goal: torch.Tensor,
        obstacle_grid: torch.Tensor,
        uncertainty_grid: torch.Tensor,
        urgency: float,
    ) -> torch.Tensor:
        return torch.cat(
            [
                config,
                self._descriptor(obstacle_grid),
                self._descriptor(uncertainty_grid),
                goal,
                torch.tensor([urgency]),
            ]
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(features)
        return self.mean(latent), self.log_std(latent)

    def act(
        self,
        config: torch.Tensor,
        goal: torch.Tensor,
        obstacle_grid: torch.Tensor,
        uncertainty_grid: torch.Tensor,
        urgency: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        features = self._features(config, goal, obstacle_grid, uncertainty_grid, urgency)
        with torch.no_grad():
            delta, log_std = self.forward(features.unsqueeze(0))
        if was_training:
            self.train()
        direction = goal - config
        distance = float(direction.norm())
        if distance > 1e-9:
            stride = min(self.guidance, distance)
            toward = direction * (stride / distance)
        else:
            toward = torch.zeros_like(direction)
        mean = config + toward + 0.1 * delta.squeeze(0)
        std = log_std.squeeze(0).clamp(-4.0, 1.0).exp()
        noise = torch.randn(mean.shape, generator=generator)
        return mean + std * noise
