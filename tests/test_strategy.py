import pandas as pd

from quant.strategy import compute_target_leverage, construct_long_short_portfolio


def test_construct_long_short_portfolio_is_dollar_neutral():
    predictions = pd.Series({"A": 0.3, "B": 0.2, "C": -0.1, "D": -0.2})
    weights = construct_long_short_portfolio(predictions, top_k=1, bottom_k=1)
    assert weights["A"] == 1.0
    assert weights["D"] == -1.0
    assert abs(weights.sum()) < 1e-12


def test_compute_target_leverage_caps():
    assert compute_target_leverage(0.01, target_vol=0.10, max_leverage=1.5) == 1.5
