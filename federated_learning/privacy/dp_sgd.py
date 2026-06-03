from __future__ import annotations

import numpy as np


def clip_and_add_noise(
    gradient: np.ndarray,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    norm = np.linalg.norm(gradient.ravel())
    scale = min(1.0, max_grad_norm / (norm + 1e-12))
    clipped = gradient * scale
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_multiplier * max_grad_norm, size=gradient.shape)
    return clipped + noise


def clip_model_update(layers: list[np.ndarray], max_norm: float = 1.0) -> list[np.ndarray]:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")
    total_norm = float(np.sqrt(sum(np.sum(np.asarray(layer, dtype=float) ** 2) for layer in layers)))
    scale = min(1.0, max_norm / (total_norm + 1e-12))
    return [np.asarray(layer, dtype=float) * scale for layer in layers]


def add_local_dp_noise(
    layers: list[np.ndarray],
    max_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    seed: int | None = None,
) -> list[np.ndarray]:
    clipped = clip_model_update(layers, max_norm=max_norm)
    if noise_multiplier <= 0:
        return clipped
    rng = np.random.default_rng(seed)
    return [
        layer + rng.normal(0.0, noise_multiplier * max_norm, size=layer.shape)
        for layer in clipped
    ]


def estimate_local_dp_epsilon(
    noise_multiplier: float,
    local_steps: int,
    delta: float = 1e-5,
) -> float:
    if local_steps <= 0:
        raise ValueError("local_steps must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if noise_multiplier <= 0:
        return float("inf")
    return float(np.sqrt(2.0 * local_steps * np.log(1.0 / delta)) / noise_multiplier)


def make_private_with_opacus(model, optimizer, data_loader, *, epochs: int, epsilon: float, delta: float = 1e-5):
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise RuntimeError("Install opacus with `pip install -r requirements.txt`.") from exc

    privacy_engine = PrivacyEngine()
    return privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=1.0,
    )
