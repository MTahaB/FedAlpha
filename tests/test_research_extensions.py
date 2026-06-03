import numpy as np
import pytest

from federated_learning.compression import bytes_saved, reconstruct, topk_sparsify
from federated_learning.privacy_accounting import compose_epsilon, rdp_compose_epsilon
from federated_learning.shapley import monte_carlo_shapley
from oracle.ipfs_storage import model_hash_bytes32, model_hash_hex
from quant.regimes import MarketRegimeDetector
from reports.statistical_validation import block_bootstrap_ci, lo_sharpe_test


def test_privacy_accounting_composes_local_and_central_epsilons():
    assert compose_epsilon(1.0, 0.5) == 1.5
    assert rdp_compose_epsilon(1.0, 2.0, n_local_steps=2, n_rounds=3) > 0


def test_topk_compression_roundtrip_and_saves_bytes():
    gradient = np.array([[1.0, -10.0], [0.1, 3.0]])
    values, indices = topk_sparsify(gradient, k_ratio=0.5)
    restored = reconstruct(values, indices, gradient.shape)

    assert np.count_nonzero(restored) == 2
    assert restored[0, 1] == -10.0
    assert bytes_saved(gradient, values, indices) >= 0


def test_monte_carlo_shapley_contributions_sum_to_full_value():
    weights = {"a": np.array([1.0]), "b": np.array([3.0])}

    def eval_fn(weight):
        return 0.0 if weight is None else float(weight.sum())

    shapley = monte_carlo_shapley(weights, eval_fn, n_samples=50, seed=7)

    assert set(shapley) == {"a", "b"}
    assert sum(shapley.values()) == pytest.approx(2.0)


def test_market_regime_detector_predicts_named_regimes_and_client_weights():
    returns = np.r_[np.repeat(0.001, 50), np.repeat(-0.002, 50), np.linspace(-0.08, 0.08, 50)]
    detector = MarketRegimeDetector().fit(returns)
    regimes = detector.predict(returns[-10:])

    assert set(regimes).issubset({"bull", "bear", "crisis"})
    weights = detector.client_regime_weights({"a": "bull", "b": "bull", "c": "crisis"})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > weights["c"]


def test_ipfs_hash_helpers_return_bytes32_and_hex():
    payload = b"model-bytes"

    assert len(model_hash_bytes32(payload)) == 32
    assert model_hash_hex(payload).startswith("0x")
    assert len(model_hash_hex(payload)) == 66


def test_statistical_validation_returns_significance_and_bootstrap_ci():
    returns = np.linspace(-0.01, 0.02, 80)

    lo_result = lo_sharpe_test(returns)
    ci = block_bootstrap_ci(returns, n_bootstrap=100, seed=1)

    assert "p_value" in lo_result
    assert ci["ci_high"] >= ci["ci_low"]


def test_ditto_personalization_smoke():
    torch = pytest.importorskip("torch")
    from federated_learning.personalization import DITTO

    model = torch.nn.Linear(1, 1)
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    personalized = DITTO(lambda_reg=0.01, lr=0.05, n_steps=5).personalize(model, x, y)

    assert isinstance(personalized, torch.nn.Module)
