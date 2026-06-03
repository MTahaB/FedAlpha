from __future__ import annotations

import numpy as np
import pandas as pd


def research_panel(n_days: int = 320, tickers: tuple[str, ...] = ("A", "B", "C", "D")) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        base = 100 + ticker_idx * 7
        trend = np.linspace(0, 12 + ticker_idx, n_days)
        seasonal = np.sin(np.arange(n_days) / (7 + ticker_idx)) * (1.0 + ticker_idx * 0.15)
        close = base + trend + seasonal
        for date, price in zip(dates, close):
            rows.append(
                {
                    "date": date.isoformat(),
                    "ticker": ticker,
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000 + ticker_idx * 20_000,
                }
            )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index(["date", "ticker"]).sort_index()


def write_ohlcv_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(path, index=False)
