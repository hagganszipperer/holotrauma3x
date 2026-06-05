from __future__ import annotations

import torch

from holotrauma3x.consult.knowledge import TraumaKnowledgeBase


def test_topm_count() -> None:
    base = TraumaKnowledgeBase(64, 16, top_m=7)
    query = torch.randn(3, 16)
    _, scores = base.retrieve(query)
    assert scores.shape == (3, 7)


def test_scores_descending() -> None:
    base = TraumaKnowledgeBase(64, 16, top_m=5)
    _, scores = base.retrieve(torch.randn(2, 16))
    diffs = scores[:, :-1] - scores[:, 1:]
    assert bool((diffs >= -1e-6).all())


def test_matches_numpy_reference() -> None:
    base = TraumaKnowledgeBase(40, 8, top_m=3, seed=11)
    query = torch.randn(1, 8)
    _, scores = base.retrieve(query)
    passages = base.passages
    normalized = query / query.norm(dim=-1, keepdim=True)
    reference = (normalized @ passages.t()).topk(3, dim=-1)[0]
    assert torch.allclose(scores, reference, atol=1e-5)
