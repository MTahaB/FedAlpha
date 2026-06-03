from __future__ import annotations

import numpy as np

from federated_learning.aggregation.common import ArrayTree, validate_layer_sets


def simulate_byzantine_updates(
    updates: list[ArrayTree],
    malicious_clients: int,
    attack: str = "sign_flip",
    scale: float = 10.0,
    seed: int | None = None,
) -> list[list[np.ndarray]]:
    if malicious_clients < 0:
        raise ValueError("malicious_clients must be non-negative.")
    if malicious_clients > len(updates):
        raise ValueError("malicious_clients cannot exceed number of updates.")

    layers = validate_layer_sets(updates)
    rng = np.random.default_rng(seed)
    poisoned = [[layer.copy() for layer in client_layers] for client_layers in layers]
    malicious_indices = list(range(malicious_clients))

    for idx in malicious_indices:
        if attack == "sign_flip":
            poisoned[idx] = [-scale * layer for layer in poisoned[idx]]
        elif attack == "gaussian":
            poisoned[idx] = [
                rng.normal(loc=0.0, scale=scale, size=layer.shape).astype(float)
                for layer in poisoned[idx]
            ]
        elif attack == "constant":
            poisoned[idx] = [np.full_like(layer, fill_value=scale, dtype=float) for layer in poisoned[idx]]
        else:
            raise ValueError(f"Unknown byzantine attack: {attack}")
    return poisoned
