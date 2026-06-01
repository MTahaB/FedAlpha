from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.strategy import construct_long_short_portfolio, portfolio_returns


@dataclass(frozen=True)
class WalkForwardWindow:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def mask(self, index: pd.Index, start: str, end: str):
        raw_dates = index.get_level_values("date") if isinstance(index, pd.MultiIndex) else index
        dates = pd.to_datetime(raw_dates)
        return (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))


def make_expanding_windows() -> list[WalkForwardWindow]:
    return [
        WalkForwardWindow("wf_2019", "2014-01-01", "2018-01-01", "2018-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
        WalkForwardWindow("wf_2020", "2014-01-01", "2019-01-01", "2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
        WalkForwardWindow("wf_2021", "2014-01-01", "2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
        WalkForwardWindow("wf_2022", "2014-01-01", "2021-01-01", "2021-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
        WalkForwardWindow("wf_2023_2025", "2014-01-01", "2022-01-01", "2022-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
    ]


def predictions_to_weight_panel(
    predictions: pd.Series,
    top_k: int = 5,
    bottom_k: int = 5,
) -> pd.DataFrame:
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("Predictions must use MultiIndex [date, ticker].")

    weights = []
    for date, day_predictions in predictions.groupby(level="date"):
        ranked = day_predictions.droplevel("date")
        day_weights = construct_long_short_portfolio(ranked, top_k=top_k, bottom_k=bottom_k)
        day_weights.name = date
        weights.append(day_weights)

    return pd.DataFrame(weights).sort_index().fillna(0.0)


def asset_return_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    returns = ohlcv["close"].groupby(level="ticker").pct_change()
    return returns.unstack("ticker").sort_index()


def backtest_predictions(
    predictions: pd.Series,
    ohlcv: pd.DataFrame,
    top_k: int = 5,
    bottom_k: int = 5,
    fixed_bps: float = 5.0,
    market_impact_bps: float = 3.0,
) -> pd.Series:
    weights = predictions_to_weight_panel(predictions, top_k=top_k, bottom_k=bottom_k)
    returns = asset_return_panel(ohlcv)
    return portfolio_returns(weights, returns, fixed_bps, market_impact_bps)
