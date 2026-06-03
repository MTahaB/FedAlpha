from __future__ import annotations

import numpy as np

from federated_learning.aggregation.common import ArrayTree, flatten_layers, unflatten_layers, validate_layer_sets


def krum_index(updates: list[np.ndarray | ArrayTree], byzantine_clients: int = 1) -> int:
    if len(updates) < 2 * byzantine_clients + 3:
        raise ValueError("Krum requires n >= 2f + 3 clients.")

    layers = validate_layer_sets(updates)
    flat = [flatten_layers(update) for update in layers]
    scores = []
    neighbor_count = len(flat) - byzantine_clients - 2

    for i, candidate in enumerate(flat):
        distances = []
        for j, other in enumerate(flat):
            if i != j:
                distances.append(float(np.sum((candidate - other) ** 2)))
        scores.append(sum(sorted(distances)[:neighbor_count]))

    return int(np.argmin(scores))


def krum(updates: list[np.ndarray], byzantine_clients: int = 1) -> np.ndarray:
    selected = krum_index(updates, byzantine_clients=byzantine_clients)
    return np.asarray(updates[selected], dtype=float)


def krum_layers(updates: list[ArrayTree], byzantine_clients: int = 1) -> list[np.ndarray]:
    layers = validate_layer_sets(updates)
    selected = krum_index(layers, byzantine_clients=byzantine_clients)
    return unflatten_layers(flatten_layers(layers[selected]), layers[0])
