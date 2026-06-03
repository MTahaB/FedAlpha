from __future__ import annotations

from collections.abc import Callable

import numpy as np


def monte_carlo_shapley(
    client_weights: dict[str, np.ndarray],
    eval_fn: Callable[[np.ndarray | None], float],
    n_samples: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Approximate FL client contributions with Monte Carlo Shapley values."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not client_weights:
        return {}

    clients = list(client_weights)
    shapley = {client: 0.0 for client in clients}
    rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        permutation = list(rng.permutation(clients))
        previous_value = float(eval_fn(None))

        for idx, client in enumerate(permutation):
            subset = [np.asarray(client_weights[name], dtype=float) for name in permutation[: idx + 1]]
            averaged = np.mean(subset, axis=0)
            current_value = float(eval_fn(averaged))
            shapley[client] += (current_value - previous_value) / n_samples
            previous_value = current_value

    return shapley
