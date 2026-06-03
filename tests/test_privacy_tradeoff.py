import numpy as np

from federated_learning.experiments.run_privacy_tradeoff import simulate_privacy_tradeoff
from federated_learning.privacy import add_local_dp_noise, estimate_local_dp_epsilon


def test_estimate_local_dp_epsilon_decreases_with_noise():
    eps_low_noise = estimate_local_dp_epsilon(0.5, local_steps=2)
    eps_high_noise = estimate_local_dp_epsilon(1.0, local_steps=2)

    assert eps_high_noise < eps_low_noise


def test_add_local_dp_noise_preserves_layer_shapes():
    layers = [np.ones(3), np.ones((2, 2))]
    private_layers = add_local_dp_noise(layers, max_norm=1.0, noise_multiplier=0.1, seed=1)

    assert [layer.shape for layer in private_layers] == [(3,), (2, 2)]


def test_simulate_privacy_tradeoff_returns_local_and_central_rows():
    frame = simulate_privacy_tradeoff([0.5, 1.0], seed=7, n_days=80)

    assert set(frame["mode"]) == {"central_dp", "local_dp"}
    assert set(frame["noise_multiplier"]) == {0.5, 1.0}
    assert np.isfinite(frame["sharpe"]).all()
