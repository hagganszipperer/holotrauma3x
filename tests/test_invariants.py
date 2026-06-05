from __future__ import annotations

import torch

from holotrauma3x.prioritization.objectives import airway_loss
from holotrauma3x.prioritization.weights import PriorityWeights
from holotrauma3x.protocol import OptimizationSpec
from holotrauma3x.workup.radiographic import SpatialAttention
from holotrauma3x.workup.volumetric import InjuryAttentionPool


def test_attention_pool_normalized() -> None:
    pool = InjuryAttentionPool(16)
    tokens = torch.randn(3, 7, 16)
    _, weights = pool(tokens)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)


def test_spatial_attention_normalized() -> None:
    attention = SpatialAttention(8, 4)
    feature = torch.randn(2, 8, 5, 5)
    _, weights = attention(feature)
    assert torch.allclose(weights.reshape(2, -1).sum(dim=1), torch.ones(2), atol=1e-5)


def test_airway_penalty_monotonic() -> None:
    volume = torch.full((1, 1), 18.0)
    target = torch.full((1, 1), 18.0)
    losses = [
        airway_loss(torch.full((1, 1), area), volume, target, (5.0, 1.8), 50.0)
        for area in (10.0, 30.0, 45.0, 50.0, 60.0)
    ]
    for earlier, later in zip(losses, losses[1:], strict=False):
        assert later <= earlier + 1e-6
    assert float(losses[-1]) == 0.0


def test_priority_order() -> None:
    weights = PriorityWeights.from_spec(OptimizationSpec())
    assert weights.is_ordered()
    assert weights.airway == 10.0


def test_priority_collapse_when_disabled() -> None:
    weights = PriorityWeights.from_spec(OptimizationSpec(priority_aware=False))
    assert weights.airway == weights.longterm == 1.0
