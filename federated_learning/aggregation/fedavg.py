from __future__ import annotations

import numpy as np

from federated_learning.aggregation.common import ArrayTree, normalize_weights, validate_layer_sets


def fedavg(
    updates: list[np.ndarray | ArrayTree],
    weights: list[float] | None = None,
) -> list[np.ndarray]:
    layers = validate_layer_sets(updates)
    n_layers = len(layers[0])
    weights_array = normalize_weights(weights, len(updates))

    aggregated = []
    for layer_idx in range(n_layers):
        stacked = np.stack([client_layers[layer_idx] for client_layers in layers])
        aggregated.append(np.tensordot(weights_array, stacked, axes=(0, 0)))
    return aggregated
