import pytest

from federated_learning.regime_aware import (
    ClientContributionStats,
    aggregate_regime_aware_ridge_states,
    compute_regime_aware_weights,
    personalize_ridge_state,
)


def _state(coef):
    return {
        "model_type": "ridge",
        "alpha": 1.0,
        "fit_intercept": True,
        "feature_columns": ["intercept", "x1"],
        "coef": coef,
    }


def test_regime_aware_weights_reward_validation_and_regime_match():
    stats = [
        ClientContributionStats("a", 100, 0.8, 0.10, "bull"),
        ClientContributionStats("b", 100, -0.5, 0.40, "crisis"),
    ]

    weights = compute_regime_aware_weights(stats, current_regime="bull")

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > weights["b"]


def test_regime_aware_aggregation_uses_computed_weights():
    stats = [
        ClientContributionStats("a", 100, 0.8, 0.10, "bull"),
        ClientContributionStats("b", 100, -0.5, 0.40, "crisis"),
    ]
    aggregate = aggregate_regime_aware_ridge_states(
        [_state([1.0, 2.0]), _state([5.0, 6.0])],
        stats,
        current_regime="bull",
    )

    assert aggregate["model_type"] == "federated_ridge_fedalpha_regime_aware"
    assert aggregate["client_weight_map"]["a"] > aggregate["client_weight_map"]["b"]
    assert aggregate["coef"][0] < 3.0


def test_personalize_ridge_state_blends_global_and_local_coefficients():
    personalized = personalize_ridge_state(
        _state([1.0, 3.0]),
        _state([5.0, 7.0]),
        local_blend=0.25,
    )

    assert personalized["model_type"] == "federated_ridge_fedalpha_personalized"
    assert personalized["coef"] == pytest.approx([2.0, 4.0])
