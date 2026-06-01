from __future__ import annotations

import pandas as pd


def make_forward_return_labels(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
    benchmark_forward_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Create forward-return labels without using future information in features."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    grouped = ohlcv.sort_index().groupby(level="ticker", group_keys=False)
    close = ohlcv["close"].astype(float)
    forward_return = grouped["close"].shift(-horizon) / close - 1

    labels = pd.DataFrame(index=ohlcv.index)
    labels[f"forward_return_{horizon}d"] = forward_return

    if benchmark_forward_returns is not None:
        bench = benchmark_forward_returns.reindex(labels.index.get_level_values("date")).to_numpy()
        labels[f"forward_alpha_{horizon}d"] = forward_return.to_numpy() - bench

    return labels


def align_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    joined = features.join(labels[[target]], how="inner").dropna()
    return joined.drop(columns=[target]), joined[target]
