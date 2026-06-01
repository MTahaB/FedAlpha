from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(ohlcv: pd.DataFrame, market_returns: pd.Series | None = None) -> pd.DataFrame:
    """Create causal daily features for a MultiIndex OHLCV frame."""
    if not isinstance(ohlcv.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex [date, ticker].")

    ohlcv = ohlcv.sort_index().copy()
    grouped = ohlcv.groupby(level="ticker", group_keys=False)

    out = pd.DataFrame(index=ohlcv.index)
    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).replace(0, np.nan)

    returns = grouped["close"].pct_change()
    out["return_1d"] = returns
    out["log_return_1d"] = np.log1p(returns)
    out["momentum_5d"] = grouped["close"].pct_change(5)
    out["momentum_20d"] = grouped["close"].pct_change(20)
    out["volatility_20d"] = returns.groupby(level="ticker", group_keys=False).rolling(20).std().droplevel(0)
    out["skewness_20d"] = returns.groupby(level="ticker", group_keys=False).rolling(20).skew().droplevel(0)
    out["dollar_volume"] = close * volume
    out["amihud_20d"] = (returns.abs() / volume).groupby(level="ticker", group_keys=False).rolling(20).mean().droplevel(0)
    out["high_52w_ratio"] = close / grouped["close"].rolling(252).max().droplevel(0)
    out["ma_20_ratio"] = close / grouped["close"].rolling(20).mean().droplevel(0) - 1
    out["ma_60_ratio"] = close / grouped["close"].rolling(60).mean().droplevel(0) - 1
    out["rsi_14"] = grouped["close"].transform(_rsi)
    out["volume_z_20d"] = grouped["volume"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
    )
    out["earnings_surprise_proxy"] = out["volume_z_20d"] * np.sign(out["return_1d"])

    if market_returns is not None:
        aligned = market_returns.reindex(out.index.get_level_values("date")).to_numpy()
        out["market_return_1d"] = aligned
        out["excess_return_1d"] = out["return_1d"] - aligned

    return out.replace([np.inf, -np.inf], np.nan)


def add_regime_labels(
    market_returns: pd.Series,
    market_volatility: pd.Series | None = None,
    n_components: int = 3,
) -> pd.Series:
    """Fit a Gaussian HMM when hmmlearn is available; otherwise return volatility terciles."""
    features = pd.DataFrame({"return": market_returns})
    features["volatility"] = (
        market_volatility if market_volatility is not None else market_returns.rolling(20).std()
    )
    features = features.dropna()

    try:
        from hmmlearn import hmm

        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
        regimes = model.fit(features.to_numpy()).predict(features.to_numpy())
        return pd.Series(regimes, index=features.index, name="regime")
    except ImportError:
        buckets = pd.qcut(features["volatility"], q=n_components, labels=False, duplicates="drop")
        return buckets.rename("regime")
