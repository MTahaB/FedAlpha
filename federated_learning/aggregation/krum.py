from __future__ import annotations

import numpy as np


def krum(updates: list[np.ndarray], byzantine_clients: int = 1) -> np.ndarray:
    if len(updates) < 2 * byzantine_clients + 3:
        raise ValueError("Krum requires n >= 2f + 3 clients.")

    flat = [np.asarray(update, dtype=float).ravel() for update in updates]
    scores = []
    neighbor_count = len(flat) - byzantine_clients - 2

    for i, candidate in enumerate(flat):
        distances = []
        for j, other in enumerate(flat):
            if i != j:
                distances.append(float(np.sum((candidate - other) ** 2)))
        scores.append(sum(sorted(distances)[:neighbor_count]))

    return np.asarray(updates[int(np.argmin(scores))], dtype=float)
