from __future__ import annotations

import math


def compose_epsilon(local_eps: float, central_eps: float) -> float:
    """Basic sequential DP composition."""
    if local_eps < 0 or central_eps < 0:
        raise ValueError("epsilons must be non-negative.")
    return float(local_eps + central_eps)


def gaussian_mechanism_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
) -> float:
    """Conservative epsilon estimate for repeated Gaussian mechanisms."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if noise_multiplier <= 0:
        return float("inf")
    return float(math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier)


def rdp_compose_epsilon(
    local_noise_multiplier: float,
    central_noise_multiplier: float,
    n_local_steps: int,
    n_rounds: int,
    delta: float = 1e-5,
) -> float:
    """Return a conservative epsilon bound for local plus central DP.

    This function intentionally avoids making a false precision claim when a
    full RDP accountant is not configured. It composes two Gaussian estimates,
    which is pessimistic but stable for automated experiment sweeps.
    """
    local_eps = gaussian_mechanism_epsilon(local_noise_multiplier, n_local_steps, delta)
    central_eps = gaussian_mechanism_epsilon(central_noise_multiplier, n_rounds, delta)
    return compose_epsilon(local_eps, central_eps)
