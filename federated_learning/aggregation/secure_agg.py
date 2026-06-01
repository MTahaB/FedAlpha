from __future__ import annotations

import numpy as np


def simulate_secure_sum(
    updates: list[np.ndarray],
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate Bonawitz-style pairwise masks whose sum cancels at the server."""
    if not updates:
        raise ValueError("updates cannot be empty.")

    arrays = [np.asarray(update, dtype=float) for update in updates]
    rng = np.random.default_rng(seed)
    masked = [array.copy() for array in arrays]

    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            mask = rng.normal(0, 1, size=arrays[i].shape)
            masked[i] += mask
            masked[j] -= mask

    return np.sum(masked, axis=0), masked


def topk_sparsify(gradient: np.ndarray, k: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1].")
    gradient = np.asarray(gradient, dtype=float)
    threshold = np.quantile(np.abs(gradient).ravel(), 1 - k)
    mask = np.abs(gradient) >= threshold
    return gradient * mask, mask
