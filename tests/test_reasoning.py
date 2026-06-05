from __future__ import annotations

import torch

from holotrauma3x.consult.deliberation import DirectRegressor, GrammarReasoner
from holotrauma3x.consult.severity import airway_band, needs_review


def test_grammar_emits_ranked_trace() -> None:
    reasoner = GrammarReasoner(8, 6, critical=50.0, threshold=0.75)
    output = reasoner.deliberate(torch.randn(3, 8), torch.randn(3, 6), torch.rand(3, 5))
    assert len(output.narrative) == 3
    assert all("[life-saving]" in line for line in output.narrative)
    assert bool(((output.confidence >= 0.0) & (output.confidence <= 1.0)).all())


def test_regression_has_no_narrative() -> None:
    reasoner = DirectRegressor(8)
    output = reasoner.deliberate(torch.randn(2, 8), torch.randn(2, 6), torch.rand(2, 5))
    assert output.narrative == ["", ""]


def test_review_flag_threshold() -> None:
    confidence = torch.tensor([[0.9], [0.6], [0.74]])
    flagged = needs_review(confidence, 0.75)
    assert flagged.tolist() == [False, True, True]


def test_airway_band_boundaries() -> None:
    assert airway_band(20.0, 50.0) == "severe"
    assert airway_band(45.0, 50.0) == "moderate"
    assert airway_band(60.0, 50.0) == "partial"
    assert airway_band(80.0, 50.0) == "clear"
