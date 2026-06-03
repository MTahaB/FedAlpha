from __future__ import annotations

import numpy as np


def topk_sparsify(gradient: np.ndarray, k_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Keep the k% largest absolute values and return compact values plus flat indices."""
    if not 0 < k_ratio <= 1:
        raise ValueError("k_ratio must be in (0, 1].")
    flat = np.asarray(gradient, dtype=float).ravel()
    k = max(1, int(len(flat) * k_ratio))
    indices = np.argpartition(np.abs(flat), -k)[-k:]
    indices = indices[np.argsort(indices)]
    return flat[indices], indices


def reconstruct(values: np.ndarray, indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    full = np.zeros(int(np.prod(shape)), dtype=float)
    full[np.asarray(indices, dtype=int)] = np.asarray(values, dtype=float)
    return full.reshape(shape)


def bytes_saved(original: np.ndarray, values: np.ndarray, indices: np.ndarray) -> int:
    original_bytes = np.asarray(original).nbytes
    sparse_bytes = np.asarray(values).nbytes + np.asarray(indices).nbytes
    return int(max(0, original_bytes - sparse_bytes))
