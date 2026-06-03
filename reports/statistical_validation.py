from __future__ import annotations

import numpy as np
from scipy import stats


def lo_sharpe_test(returns: np.ndarray, freq: int = 252) -> dict[str, float | bool]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        raise ValueError("At least 3 returns are required.")
    std = r.std(ddof=1)
    if std == 0:
        raise ValueError("Returns must be non-constant.")

    sharpe = float(r.mean() / std * np.sqrt(freq))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))
    variance = max((1 + 0.5 * sharpe**2 * (kurt / 4 - skew**2 / 3 + 0.25)) / len(r), 1e-12)
    z_stat = float(sharpe / np.sqrt(variance))
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    return {
        "sharpe": sharpe,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def block_bootstrap_ci(
    returns: np.ndarray,
    block: int = 21,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        raise ValueError("At least 3 returns are required.")
    block = max(1, min(block, len(r)))
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_bootstrap):
        indices: list[int] = []
        while len(indices) < len(r):
            start = int(rng.integers(0, len(r) - block + 1))
            indices.extend(range(start, start + block))
        sampled = r[indices[: len(r)]]
        std = sampled.std(ddof=1)
        samples.append(np.nan if std == 0 else sampled.mean() / std * np.sqrt(252))

    arr = np.asarray(samples, dtype=float)
    return {
        "ci_low": float(np.nanpercentile(arr, 2.5)),
        "ci_high": float(np.nanpercentile(arr, 97.5)),
        "ci_width": float(np.nanpercentile(arr, 97.5) - np.nanpercentile(arr, 2.5)),
    }
