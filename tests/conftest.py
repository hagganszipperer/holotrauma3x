from __future__ import annotations

import pytest

from holotrauma3x.protocol import (
    ArrivalSpec,
    CoReasoningSpec,
    EncoderSpec,
    FusionSpec,
    OptimizationSpec,
    PointSpec,
    Protocol,
    RadiographSpec,
    RobotSpec,
    TrainingSpec,
    VolumetricSpec,
)


def tiny_protocol() -> Protocol:
    return Protocol(
        experiment="test",
        arrival=ArrivalSpec(
            cohort_size=16, volume_size=8, point_count=256, radiograph_size=32, fps_points=256
        ),
        encoder=EncoderSpec(
            volumetric=VolumetricSpec(
                patch=4, layers=3, dim=32, heads=4, feedforward=64, dropout=0.0
            ),
            point=PointSpec(radii=(0.2, 0.4, 0.6, 0.8), samples=(64, 32, 16, 8), out_dim=32),
            radiograph=RadiographSpec(out_dim=32, reduction=4),
            fusion=FusionSpec(enc_dim=64, heads=4),
        ),
        coreasoning=CoReasoningSpec(
            clip_dim=64,
            proj_dim=48,
            decoder_layers=2,
            decoder_heads=4,
            decoder_hidden=32,
            vocab=64,
            knowledge_size=128,
            top_m=5,
        ),
        optimization=OptimizationSpec(max_iterations=20),
        robot=RobotSpec(dof=2, max_iterations=600, obstacle_grid=4, uncertainty_grid=4),
        training=TrainingSpec(epochs=4, batch_size=4, max_steps=8, folds=2, centers=3),
    )


@pytest.fixture()
def protocol() -> Protocol:
    return tiny_protocol()
