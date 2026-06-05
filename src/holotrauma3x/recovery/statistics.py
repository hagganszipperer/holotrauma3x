from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bca_bootstrap_ci(
    data: np.ndarray, resamples: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(data, dtype=float)
    observed = float(values.mean())
    n = values.shape[0]
    boot = np.array([values[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    proportion = float((boot < observed).mean())
    proportion = min(max(proportion, 1.0 / resamples), 1.0 - 1.0 / resamples)
    z0 = float(norm.ppf(proportion))
    jackknife = np.array([np.delete(values, i).mean() for i in range(n)])
    centered = jackknife.mean() - jackknife
    denom = 6.0 * (centered**2).sum() ** 1.5
    acceleration = float((centered**3).sum() / denom) if denom != 0 else 0.0
    lower_z = float(norm.ppf(alpha / 2.0))
    upper_z = float(norm.ppf(1.0 - alpha / 2.0))
    lower_a = norm.cdf(z0 + (z0 + lower_z) / (1.0 - acceleration * (z0 + lower_z)))
    upper_a = norm.cdf(z0 + (z0 + upper_z) / (1.0 - acceleration * (z0 + upper_z)))
    low = float(np.quantile(boot, lower_a))
    high = float(np.quantile(boot, upper_a))
    return low, high


def permutation_test(
    first: np.ndarray, second: np.ndarray, resamples: int = 10000, seed: int = 42
) -> float:
    rng = np.random.default_rng(seed)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n = a.shape[0]
    count = 0
    for _ in range(resamples):
        rng.shuffle(pooled)
        if abs(pooled[:n].mean() - pooled[n:].mean()) >= observed:
            count += 1
    return (count + 1) / (resamples + 1)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        scaled = (total - rank) * pvalues[index]
        running = max(running, min(scaled, 1.0))
        adjusted[index] = running
    return adjusted
