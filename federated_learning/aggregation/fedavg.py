from __future__ import annotations

import numpy as np


ArrayLikeTree = list[np.ndarray] | tuple[np.ndarray, ...]


def _as_layers(update: np.ndarray | ArrayLikeTree) -> list[np.ndarray]:
    if isinstance(update, np.ndarray):
        return [update]
    return [np.asarray(layer, dtype=float) for layer in update]


def fedavg(
    updates: list[np.ndarray | ArrayLikeTree],
    weights: list[float] | None = None,
) -> list[np.ndarray]:
    if not updates:
        raise ValueError("updates cannot be empty.")

    layers = [_as_layers(update) for update in updates]
    n_layers = len(layers[0])
    if any(len(layer_set) != n_layers for layer_set in layers):
        raise ValueError("All updates must have the same number of layers.")

    if weights is None:
        weights_array = np.ones(len(updates), dtype=float) / len(updates)
    else:
        weights_array = np.asarray(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()

    aggregated = []
    for layer_idx in range(n_layers):
        stacked = np.stack([client_layers[layer_idx] for client_layers in layers])
        aggregated.append(np.tensordot(weights_array, stacked, axes=(0, 0)))
    return aggregated
