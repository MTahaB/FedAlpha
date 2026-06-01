from __future__ import annotations

import numpy as np


def trimmed_mean_aggregate(updates: list[np.ndarray], trim_ratio: float = 0.1) -> np.ndarray:
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    if not updates:
        raise ValueError("updates cannot be empty.")

    stacked = np.stack([np.asarray(update, dtype=float) for update in updates])
    k = int(len(updates) * trim_ratio)
    sorted_updates = np.sort(stacked, axis=0)
    if k == 0:
        return sorted_updates.mean(axis=0)
    return sorted_updates[k:-k].mean(axis=0)


def trimmed_mean_layers(
    updates: list[list[np.ndarray]],
    trim_ratio: float = 0.1,
) -> list[np.ndarray]:
    n_layers = len(updates[0])
    return [
        trimmed_mean_aggregate([client_update[layer_idx] for client_update in updates], trim_ratio)
        for layer_idx in range(n_layers)
    ]
