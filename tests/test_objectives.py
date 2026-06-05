from __future__ import annotations

import torch

from holotrauma3x.prioritization.objectives import airway_loss, occlusion_loss, stabilization_loss


def test_airway_loss_closed_form() -> None:
    area = torch.tensor([[44.0]])
    volume = torch.tensor([[20.0]])
    target = torch.tensor([[18.0]])
    expected = 5.0 * (6.0**3) + 1.8 * (2.0**2)
    value = airway_loss(area, volume, target, (5.0, 1.8), 50.0)
    assert abs(float(value) - expected) < 1e-3


def test_stabilization_loss_closed_form() -> None:
    offsets = torch.zeros(1, 2, 3)
    offsets[0, 0, 0] = 2.0
    target = torch.zeros(1, 2, 3)
    stability = torch.zeros(1, 2)
    expected = 3.0 * 4.0 + 2.0 * 2.0
    value = stabilization_loss(offsets, target, stability, (3.0, 2.0))
    assert abs(float(value) - expected) < 1e-3


def test_occlusion_weighting() -> None:
    occlusion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = torch.zeros(1, 4)
    value = occlusion_loss(occlusion, target, (2.0, 1.5, 1.0, 1.0))
    assert abs(float(value) - 2.0) < 1e-5
