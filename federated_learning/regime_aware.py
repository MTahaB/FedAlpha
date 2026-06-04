from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientContributionStats:
    client_id: str
    n_examples: int
    validation_sharpe: float
    validation_volatility: float
    client_regime: str


def compute_regime_aware_weights(
    stats: list[ClientContributionStats],
    *,
    current_regime: str,
    min_performance_factor: float = 0.05,
) -> dict[str, float]:
    """Compute FedAlpha client weights from size, validation, regime, stability."""
    if not stats:
        return {}

    raw: dict[str, float] = {}
    for item in stats:
        size_factor = max(float(item.n_examples), 1.0)
        performance_factor = max(1.0 + float(item.validation_sharpe), min_performance_factor)
        regime_factor = 1.5 if item.client_regime == current_regime else 0.5
        stability_factor = 1.0 / (1.0 + max(float(item.validation_volatility), 0.0))
        raw[item.client_id] = size_factor * performance_factor * regime_factor * stability_factor

    total = sum(raw.values())
    if total <= 0:
        equal = 1.0 / len(raw)
        return {client_id: equal for client_id in raw}
    return {client_id: weight / total for client_id, weight in raw.items()}


def aggregate_regime_aware_ridge_states(
    states: list[dict],
    stats: list[ClientContributionStats],
    *,
    current_regime: str,
) -> dict:
    if not states:
        raise ValueError("states cannot be empty.")
    if len(states) != len(stats):
        raise ValueError("states and stats must have the same length.")

    feature_columns = states[0]["feature_columns"]
    if any(state["feature_columns"] != feature_columns for state in states):
        raise ValueError("All ridge states must share feature columns.")

    weights_by_client = compute_regime_aware_weights(stats, current_regime=current_regime)
    weights = np.asarray([weights_by_client[item.client_id] for item in stats], dtype=float)
    coefs = np.stack([np.asarray(state["coef"], dtype=float) for state in states])
    aggregate_coef = np.tensordot(weights, coefs, axes=(0, 0))
    return {
        "model_type": "federated_ridge_fedalpha_regime_aware",
        "alpha": float(states[0].get("alpha", 1.0)),
        "fit_intercept": bool(states[0].get("fit_intercept", True)),
        "feature_columns": feature_columns,
        "coef": aggregate_coef.tolist(),
        "client_weights": weights.tolist(),
        "client_weight_map": weights_by_client,
        "current_regime": current_regime,
        "client_stats": [item.__dict__ for item in stats],
    }


def personalize_ridge_state(
    global_state: dict,
    local_state: dict,
    *,
    local_blend: float = 0.25,
) -> dict:
    """Personalize a Ridge state by shrinking the global model toward a local model."""
    if not 0.0 <= local_blend <= 1.0:
        raise ValueError("local_blend must be between 0 and 1.")
    if global_state["feature_columns"] != local_state["feature_columns"]:
        raise ValueError("Global and local states must share feature columns.")

    global_coef = np.asarray(global_state["coef"], dtype=float)
    local_coef = np.asarray(local_state["coef"], dtype=float)
    coef = (1.0 - local_blend) * global_coef + local_blend * local_coef
    return {
        **global_state,
        "model_type": "federated_ridge_fedalpha_personalized",
        "coef": coef.tolist(),
        "personalization": {
            "local_blend": float(local_blend),
            "local_model_type": local_state.get("model_type", "ridge"),
        },
    }
