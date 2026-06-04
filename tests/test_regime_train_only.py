import numpy as np
import pandas as pd

from quant.pipeline import embargoed_train_end, prepare_supervised_dataset, resolve_supervised_split_date


def _ohlcv_panel(n_days=330, tickers=("A", "B", "C")):
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    base = np.repeat(100 + np.linspace(0, 40, n_days), len(tickers))
    ticker_offsets = np.tile(np.arange(len(tickers)) * 2.0, n_days)
    close = base + ticker_offsets + np.sin(np.arange(len(index)) / 13)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + np.tile(np.arange(len(tickers)) * 10_000, n_days),
        },
        index=index,
    )


def test_prepare_supervised_dataset_uses_train_fitted_regimes(monkeypatch):
    ohlcv = _ohlcv_panel()
    split_date = resolve_supervised_split_date(ohlcv, horizon=5)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("regime fitting must happen through train_fitted_regime_labels only")

    monkeypatch.setattr("quant.features.add_regime_labels", fail_if_called)
    dataset = prepare_supervised_dataset(
        ohlcv,
        horizon=5,
        regime_train_end=embargoed_train_end(split_date, embargo_days=5),
    )

    assert "market_regime_code" in dataset.features
    assert {"market_regime_bull", "market_regime_bear", "market_regime_crisis"}.issubset(dataset.features)
