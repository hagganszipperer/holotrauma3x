from __future__ import annotations

from torch.utils.data import DataLoader

from holotrauma3x.arrival.cohort import TraumaCohort
from holotrauma3x.pathway import build_pathway
from holotrauma3x.protocol import Protocol
from holotrauma3x.rounds.loop import Trainer


def test_gradients_reach_encoder(protocol: Protocol) -> None:
    model = build_pathway(protocol)
    dataset = TraumaCohort(
        protocol.arrival, protocol.arrival.cohort_size, 42, protocol.training.centers
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    loss = Trainer(protocol.training).loss(model, batch)
    loss.backward()
    weight = model.encoder.volume.embed.weight
    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0.0
