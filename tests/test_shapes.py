from __future__ import annotations

import torch

from holotrauma3x.arrival.cohort import TraumaCohort
from holotrauma3x.pathway import build_pathway
from holotrauma3x.protocol import Protocol, TriadInputs


def _batch(protocol: Protocol, size: int = 4) -> TriadInputs:
    dataset = TraumaCohort(
        protocol.arrival, protocol.arrival.cohort_size, 42, protocol.training.centers
    )
    volume = torch.stack([dataset[i]["volume"] for i in range(size)])
    points = torch.stack([dataset[i]["points"] for i in range(size)])
    radiograph = torch.stack([dataset[i]["radiograph"] for i in range(size)])
    return TriadInputs(volume=volume, points=points, radiograph=radiograph)


def test_fused_dimension(protocol: Protocol) -> None:
    model = build_pathway(protocol)
    output = model(_batch(protocol))
    assert output.fused.shape == (4, protocol.encoder.fusion.enc_dim)


def test_reposition_dimensions(protocol: Protocol) -> None:
    model = build_pathway(protocol)
    output = model(_batch(protocol))
    assert output.maxilla.shape == (4, 6)
    assert output.mandible.shape == (4, 6)
    assert output.condyle.shape == (4, 3)


def test_injury_dimensions(protocol: Protocol) -> None:
    model = build_pathway(protocol)
    injury = model(_batch(protocol)).reasoning.injury
    assert injury.airway_area.shape == (4, 1)
    assert injury.occlusion.shape == (4, 4)
    assert injury.fracture_offsets.shape == (4, 4, 3)
