from __future__ import annotations

from holotrauma3x.arrival.cohort import TraumaCohort
from holotrauma3x.pathway import build_pathway
from holotrauma3x.protocol import Protocol
from holotrauma3x.rounds.loop import Trainer
from holotrauma3x.rounds.seeding import set_seed


def test_two_step_smoke_decreases(protocol: Protocol) -> None:
    set_seed(protocol.training.seed)
    model = build_pathway(protocol)
    dataset = TraumaCohort(
        protocol.arrival, protocol.arrival.cohort_size, 42, protocol.training.centers
    )
    history = Trainer(protocol.training).fit(model, dataset)
    assert len(history.step_losses) >= 2
    assert history.step_losses[-1] < history.step_losses[0]
