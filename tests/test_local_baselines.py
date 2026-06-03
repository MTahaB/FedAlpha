import numpy as np
import pandas as pd

from federated_learning.experiments.run_local_baselines import (
    chronological_split,
    prepare_supervised_frame,
    run_baselines,
)


def _research_panel(n_days=320, tickers=("A", "B", "C", "D")):
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    date_axis = np.arange(n_days)
    trend = np.repeat(100 + np.linspace(0, 25, n_days) + np.sin(date_axis / 5) * 2.5, len(tickers))
    ticker_offsets = np.tile(np.arange(len(tickers)) * 3.0, n_days)
    seasonal = np.sin(np.arange(len(index)) / 11) * 0.5
    close = trend + ticker_offsets + seasonal
    volume_cycle = np.repeat(1_000_000 + np.cos(date_axis / 7) * 50_000, len(tickers))
    ticker_volume = np.tile(np.arange(len(tickers)) * 10_000, n_days)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume_cycle + ticker_volume,
        },
        index=index,
    )


def test_prepare_supervised_frame_uses_real_ohlcv_features():
    x, y = prepare_supervised_frame(_research_panel(), horizon=5)

    assert not x.empty
    assert len(x) == len(y)
    assert "market_regime_code" in x


def test_chronological_split_keeps_future_out_of_training():
    x, y = prepare_supervised_frame(_research_panel(), horizon=5)
    x_train, x_test, y_train, y_test = chronological_split(x, y, test_start="2020-01-01")

    assert x_train.index.get_level_values("date").max() < pd.Timestamp("2020-01-01")
    assert x_test.index.get_level_values("date").min() >= pd.Timestamp("2020-01-01")
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_run_baselines_runs_ridge_on_real_panel():
    results = run_baselines(_research_panel(), model_names=["ridge"], horizon=5)

    assert results["ridge"]["status"] == "ok"
    assert results["ridge"]["n_train"] > 0
    assert "sharpe_ratio" in results["ridge"]["metrics"]
