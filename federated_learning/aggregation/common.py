from __future__ import annotations

import numpy as np


ArrayTree = list[np.ndarray] | tuple[np.ndarray, ...]


def as_layers(update: np.ndarray | ArrayTree) -> list[np.ndarray]:
    if isinstance(update, np.ndarray):
        return [np.asarray(update, dtype=float)]
    return [np.asarray(layer, dtype=float) for layer in update]


def validate_layer_sets(updates: list[np.ndarray | ArrayTree]) -> list[list[np.ndarray]]:
    if not updates:
        raise ValueError("updates cannot be empty.")

    layers = [as_layers(update) for update in updates]
    n_layers = len(layers[0])
    shapes = [layer.shape for layer in layers[0]]
    for layer_set in layers:
        if len(layer_set) != n_layers:
            raise ValueError("All updates must have the same number of layers.")
        if [layer.shape for layer in layer_set] != shapes:
            raise ValueError("All update layers must have matching shapes.")
    return layers


def flatten_layers(layers: ArrayTree) -> np.ndarray:
    return np.concatenate([np.asarray(layer, dtype=float).ravel() for layer in layers])


def unflatten_layers(flat: np.ndarray, template: ArrayTree) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    cursor = 0
    for layer in template:
        layer_array = np.asarray(layer, dtype=float)
        size = layer_array.size
        arrays.append(np.asarray(flat[cursor : cursor + size], dtype=float).reshape(layer_array.shape))
        cursor += size
    if cursor != len(flat):
        raise ValueError("Flat update size does not match template.")
    return arrays


def normalize_weights(weights: list[float] | np.ndarray | None, n_items: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_items, dtype=float) / n_items
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (n_items,):
        raise ValueError("weights must match number of updates.")
    total = weights_array.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return weights_array / total
