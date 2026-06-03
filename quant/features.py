from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS = ("bull", "bear", "crisis")


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
        dates = out.index.get_level_values("date")
        aligned = market_returns.reindex(dates).to_numpy()
        out["market_return_1d"] = aligned
        out["excess_return_1d"] = out["return_1d"] - aligned
        regimes = add_regime_labels(market_returns).reindex(dates).ffill().infer_objects(copy=False)
        out["market_regime_code"] = regimes.map({label: idx for idx, label in enumerate(REGIME_LABELS)}).to_numpy()
        for label in REGIME_LABELS:
            out[f"market_regime_{label}"] = (regimes == label).astype(float).to_numpy()

    return out.replace([np.inf, -np.inf], np.nan)


def add_regime_labels(
    market_returns: pd.Series,
    market_volatility: pd.Series | None = None,
    n_components: int = 3,
) -> pd.Series:
    """Fit regimes on market volatility and return bull/bear/crisis labels.

    The fitted HMM uses volatility only, then labels hidden states by realized
    volatility from low to high. If hmmlearn is unavailable, volatility quantiles
    provide a deterministic fallback with the same output contract.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    volatility = market_volatility if market_volatility is not None else market_returns.rolling(20).std()
    features = pd.DataFrame({"volatility": volatility.astype(float)}).dropna()
    if features.empty:
        return pd.Series(index=market_returns.index, dtype=object, name="regime")

    n_components = min(n_components, len(features))

    try:
        from hmmlearn import hmm

        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        raw_regimes = model.fit(features[["volatility"]].to_numpy()).predict(
            features[["volatility"]].to_numpy()
        )
    except ImportError:
        raw_regimes = pd.qcut(
            features["volatility"],
            q=n_components,
            labels=False,
            duplicates="drop",
        ).to_numpy()

    names = _regime_names(len(pd.unique(raw_regimes)))
    ordered_states = (
        pd.DataFrame({"raw_regime": raw_regimes, "volatility": features["volatility"].to_numpy()})
        .groupby("raw_regime")["volatility"]
        .mean()
        .sort_values()
        .index
    )
    mapping = {state: names[position] for position, state in enumerate(ordered_states)}
    return pd.Series(raw_regimes, index=features.index).map(mapping).rename("regime")


def _regime_names(n_components: int) -> list[str]:
    if n_components == 1:
        return ["crisis"]
    if n_components == 2:
        return ["bull", "crisis"]
    if n_components == 3:
        return list(REGIME_LABELS)
    middle = [f"transition_{idx}" for idx in range(1, n_components - 1)]
    return ["bull", *middle, "crisis"]
