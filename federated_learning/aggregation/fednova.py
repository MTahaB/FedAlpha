from __future__ import annotations

import numpy as np

from federated_learning.aggregation.common import ArrayTree, normalize_weights, validate_layer_sets


def parameter_deltas(
    global_params: ArrayTree,
    client_params: list[ArrayTree],
) -> list[list[np.ndarray]]:
    global_layers = validate_layer_sets([global_params])[0]
    client_layers = validate_layer_sets(client_params)
    deltas: list[list[np.ndarray]] = []
    for layers in client_layers:
        deltas.append([client - global_ for client, global_ in zip(layers, global_layers)])
    return deltas


def normalize_deltas_by_steps(
    deltas: list[ArrayTree],
    local_steps: list[int | float],
) -> list[list[np.ndarray]]:
    if len(deltas) != len(local_steps):
        raise ValueError("local_steps must match number of client updates.")
    layer_sets = validate_layer_sets(deltas)
    normalized: list[list[np.ndarray]] = []
    for layers, steps in zip(layer_sets, local_steps):
        tau = max(float(steps), 1.0)
        normalized.append([layer / tau for layer in layers])
    return normalized


def fednova(
    global_params: ArrayTree,
    client_params: list[ArrayTree],
    num_examples: list[int | float],
    local_steps: list[int | float],
    robust_aggregator: str = "none",
    trim_ratio: float = 0.1,
    byzantine_clients: int = 1,
) -> list[np.ndarray]:
    if not client_params:
        raise ValueError("client_params cannot be empty.")

    global_layers = validate_layer_sets([global_params])[0]
    weights = normalize_weights(num_examples, len(client_params))
    taus = np.asarray([max(float(step), 1.0) for step in local_steps], dtype=float)
    tau_eff = float(np.dot(weights, taus))

    deltas = parameter_deltas(global_layers, client_params)
    normalized_deltas = normalize_deltas_by_steps(deltas, local_steps)
    robust_name = robust_aggregator.lower()

    if robust_name == "none":
        aggregate_delta_by_layer = []
        for layer_idx in range(len(global_layers)):
            stacked = np.stack([client_layers[layer_idx] for client_layers in normalized_deltas])
            aggregate_delta_by_layer.append(np.tensordot(weights, stacked, axes=(0, 0)))
    elif robust_name == "median":
        from federated_learning.aggregation.median import median_layers

        aggregate_delta_by_layer = median_layers(normalized_deltas)
    elif robust_name == "trimmed_mean":
        from federated_learning.aggregation.trimmed_mean import trimmed_mean_layers

        aggregate_delta_by_layer = trimmed_mean_layers(normalized_deltas, trim_ratio=trim_ratio)
    elif robust_name == "krum":
        from federated_learning.aggregation.krum import krum_layers

        aggregate_delta_by_layer = krum_layers(normalized_deltas, byzantine_clients=byzantine_clients)
    else:
        raise ValueError(f"Unknown robust aggregator: {robust_aggregator}")

    aggregated: list[np.ndarray] = []
    for layer_idx, global_layer in enumerate(global_layers):
        aggregated.append(global_layer + tau_eff * aggregate_delta_by_layer[layer_idx])
    return aggregated
