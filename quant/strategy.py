from __future__ import annotations

import numpy as np
import pandas as pd


def compute_target_leverage(
    realized_vol_20d: float,
    target_vol: float = 0.10,
    max_leverage: float = 1.5,
) -> float:
    if realized_vol_20d <= 0 or np.isnan(realized_vol_20d):
        return 0.0
    return float(min(target_vol / realized_vol_20d, max_leverage))


def construct_long_short_portfolio(
    predictions: pd.Series,
    top_k: int = 5,
    bottom_k: int = 5,
    vol_target: float = 0.10,
    current_vol: float | None = None,
    max_leverage: float = 2.0,
) -> pd.Series:
    """Rank predictions into dollar-neutral long-short weights."""
    clean = predictions.dropna().sort_values(ascending=False)
    weights = pd.Series(0.0, index=predictions.index, dtype=float)
    if clean.empty:
        return weights

    long_count = min(top_k, len(clean))
    short_count = min(bottom_k, max(len(clean) - long_count, 0))

    if long_count:
        weights.loc[clean.index[:long_count]] = 1.0 / long_count
    if short_count:
        weights.loc[clean.index[-short_count:]] = -1.0 / short_count

    if current_vol is not None and current_vol > 0:
        leverage = min(vol_target / current_vol, max_leverage)
        weights *= leverage

    return weights


def compute_transaction_costs(
    trades: pd.Series | pd.DataFrame,
    fixed_bps: float = 5.0,
    market_impact_bps: float = 3.0,
) -> float:
    gross_turnover = float(trades.abs().sum().sum())
    return gross_turnover * (fixed_bps + market_impact_bps) / 10_000


def portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    fixed_bps: float = 5.0,
    market_impact_bps: float = 3.0,
) -> pd.Series:
    aligned_weights, aligned_returns = weights.align(asset_returns, join="inner", axis=0)
    gross = (aligned_weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
    turnover = aligned_weights.diff().fillna(aligned_weights)
    costs = turnover.apply(
        lambda row: compute_transaction_costs(row, fixed_bps, market_impact_bps),
        axis=1,
    )
    return (gross - costs).rename("portfolio_return")
