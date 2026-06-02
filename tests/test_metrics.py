import numpy as np

from quant.metrics import compute_max_drawdown, compute_sharpe, sharpe_significance_test


def test_compute_sharpe_positive_for_positive_mean_returns():
    returns = np.array([0.001, 0.002, -0.0005, 0.0015, 0.001])
    assert compute_sharpe(returns) > 0


def test_compute_max_drawdown():
    returns = np.array([0.1, -0.2, 0.05])
    assert round(compute_max_drawdown(returns), 3) == -0.2


def test_sharpe_significance_shape():
    result = sharpe_significance_test(np.repeat(0.001, 30) + np.linspace(-0.0001, 0.0001, 30))
    assert {"sharpe", "ci_lower", "ci_upper", "p_value", "significant"}.issubset(result)


def test_sharpe_significance_detects_strong_positive_series():
    returns = np.array([0.001 + ((i % 5) - 2) * 0.0001 for i in range(260)])
    result = sharpe_significance_test(returns, confidence=0.90)
    assert result["significant"] is True
