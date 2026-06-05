from __future__ import annotations

import torch

from holotrauma3x.arrival.cohort import TraumaCohort
from holotrauma3x.pathway import build_pathway
from holotrauma3x.protocol import Protocol, TriadInputs
from holotrauma3x.rounds.seeding import set_seed


def test_cohort_is_reproducible(protocol: Protocol) -> None:
    first = TraumaCohort(protocol.arrival, 8, 42, 3)[0]
    second = TraumaCohort(protocol.arrival, 8, 42, 3)[0]
    assert torch.equal(first["volume"], second["volume"])
    assert torch.equal(first["airway_area"], second["airway_area"])


def test_forward_is_reproducible(protocol: Protocol) -> None:
    dataset = TraumaCohort(protocol.arrival, 8, 42, 3)
    inputs = TriadInputs(
        dataset[0]["volume"].unsqueeze(0),
        dataset[0]["points"].unsqueeze(0),
        dataset[0]["radiograph"].unsqueeze(0),
    )
    set_seed(0)
    model_a = build_pathway(protocol).eval()
    set_seed(0)
    model_b = build_pathway(protocol).eval()
    with torch.no_grad():
        out_a = model_a(inputs).fused
        out_b = model_b(inputs).fused
    assert torch.allclose(out_a, out_b, atol=1e-6)
