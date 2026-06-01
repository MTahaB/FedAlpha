from __future__ import annotations

import numpy as np


def fedprox_penalty(local_params: list[np.ndarray], global_params: list[np.ndarray], mu: float = 0.01) -> float:
    if len(local_params) != len(global_params):
        raise ValueError("local_params and global_params must have the same length.")
    total = 0.0
    for local, global_ in zip(local_params, global_params):
        total += float(np.sum((np.asarray(local) - np.asarray(global_)) ** 2))
    return 0.5 * mu * total
