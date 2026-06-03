from __future__ import annotations

import pandas as pd

from quant.strategy import portfolio_returns


def buy_and_hold_returns(benchmark_close: pd.Series) -> pd.Series:
    return benchmark_close.sort_index().pct_change(fill_method=None).dropna().rename("buy_and_hold")


def equal_weight_returns(asset_returns: pd.DataFrame) -> pd.Series:
    if asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must contain at least one asset.")
    weights = pd.DataFrame(1.0 / asset_returns.shape[1], index=asset_returns.index, columns=asset_returns.columns)
    return portfolio_returns(weights, asset_returns).rename("equal_weight")


def momentum_20d_long_short(asset_returns: pd.DataFrame, top_k: int = 10, bottom_k: int = 10) -> pd.Series:
    momentum = (1 + asset_returns).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    weights = pd.DataFrame(0.0, index=asset_returns.index, columns=asset_returns.columns)

    for date, row in momentum.iterrows():
        clean = row.dropna().sort_values(ascending=False)
        if clean.empty:
            continue
        longs = clean.index[: min(top_k, len(clean))]
        short_count = min(bottom_k, max(len(clean) - len(longs), 0))
        shorts = clean.index[-short_count:] if short_count else []
        if len(longs):
            weights.loc[date, longs] = 1 / len(longs)
        if len(shorts):
            weights.loc[date, shorts] = -1 / len(shorts)

    return portfolio_returns(weights, asset_returns).rename("momentum_20d")
