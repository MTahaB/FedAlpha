import numpy as np
import pandas as pd

from quant.features import REGIME_LABELS, add_regime_features, add_regime_labels, build_features


def _ohlcv_panel(n_days=320, tickers=("A", "B", "C")):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    base = np.repeat(np.linspace(100, 140, n_days), len(tickers))
    offsets = np.tile(np.arange(len(tickers)) * 5, n_days)
    close = base + offsets
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


def test_add_regime_labels_returns_named_volatility_regimes():
    returns = pd.Series(
        np.r_[np.repeat(0.0005, 40), np.repeat(0.001, 40), np.repeat(-0.002, 40)],
        index=pd.bdate_range("2020-01-01", periods=120),
    )
    volatility = pd.Series(
        np.r_[np.repeat(0.01, 40), np.repeat(0.03, 40), np.repeat(0.08, 40)],
        index=returns.index,
    )

    regimes = add_regime_labels(returns, market_volatility=volatility)

    assert regimes.notna().all()
    assert set(regimes.unique()).issubset(set(REGIME_LABELS))


def test_build_features_without_regime_labels_is_causal_only():
    ohlcv = _ohlcv_panel()
    market_returns = ohlcv["close"].unstack("ticker").pct_change().mean(axis=1)

    features = build_features(ohlcv, market_returns=market_returns)

    assert "return_1d" in features
    assert "market_return_1d" in features
    assert "market_regime_code" not in features


def test_add_regime_features_adds_numeric_market_regime_columns():
    ohlcv = _ohlcv_panel()
    market_returns = ohlcv["close"].unstack("ticker").pct_change().mean(axis=1)
    regime_labels = pd.Series("bull", index=market_returns.index, name="regime")
    features = add_regime_features(build_features(ohlcv, market_returns=market_returns), regime_labels)

    assert "market_regime_code" in features
    assert {"market_regime_bull", "market_regime_bear", "market_regime_crisis"}.issubset(features)
    assert features[["market_regime_bull", "market_regime_bear", "market_regime_crisis"]].dtypes.eq(float).all()


def test_build_features_uses_supplied_regime_labels_without_refitting(monkeypatch):
    ohlcv = _ohlcv_panel()
    market_returns = ohlcv["close"].unstack("ticker").pct_change().mean(axis=1)
    regime_labels = pd.Series("bull", index=market_returns.index, name="regime")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("regimes should be supplied by the walk-forward train fit")

    monkeypatch.setattr("quant.features.add_regime_labels", fail_if_called)
    features = build_features(ohlcv, market_returns=market_returns, regime_labels=regime_labels)

    assert features["market_regime_bull"].dropna().eq(1.0).all()
