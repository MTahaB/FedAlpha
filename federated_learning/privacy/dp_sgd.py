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
