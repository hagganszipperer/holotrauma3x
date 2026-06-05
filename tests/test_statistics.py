from __future__ import annotations

import numpy as np
from scipy import stats

from holotrauma3x.recovery.statistics import bca_bootstrap_ci, holm_bonferroni, permutation_test


def test_bca_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0.42, 0.05, size=64)
    low, high = bca_bootstrap_ci(data, resamples=500, seed=7)
    assert low < data.mean() < high


def test_permutation_agrees_with_scipy() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.8, 1.0, size=40)
    ours = permutation_test(a, b, resamples=4000, seed=5)
    reference = stats.permutation_test(
        (a, b),
        lambda x, y: abs(x.mean() - y.mean()),
        permutation_type="independent",
        n_resamples=4000,
        random_state=5,
    ).pvalue
    assert abs(ours - reference) < 0.05


def test_holm_monotone_and_bounded() -> None:
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
