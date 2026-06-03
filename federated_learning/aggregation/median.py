from __future__ import annotations

import numpy as np

from federated_learning.aggregation.common import ArrayTree, validate_layer_sets


def median_aggregate(updates: list[np.ndarray]) -> np.ndarray:
    if not updates:
        raise ValueError("updates cannot be empty.")
    stacked = np.stack([np.asarray(update, dtype=float) for update in updates])
    return np.median(stacked, axis=0)


def median_layers(updates: list[ArrayTree]) -> list[np.ndarray]:
    layers = validate_layer_sets(updates)
    n_layers = len(layers[0])
    return [median_aggregate([client[layer_idx] for client in layers]) for layer_idx in range(n_layers)]
