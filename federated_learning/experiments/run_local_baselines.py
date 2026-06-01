from __future__ import annotations

import numpy as np
import pandas as pd

from quant.metrics import summarize_performance
from quant.models import RidgeSignalModel


def synthetic_panel(n_days: int = 500, n_assets: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    signal = rng.normal(size=len(index))
    noise = rng.normal(scale=0.02, size=len(index))
    forward = 0.001 * signal + noise
    features = pd.DataFrame({"signal": signal}, index=index)
    labels = pd.Series(forward, index=index, name="forward_return_5d")
    return features, labels


def main() -> None:
    x, y = synthetic_panel()
    cutoff = int(len(x) * 0.7)
    model = RidgeSignalModel(alpha=1.0).fit(x.iloc[:cutoff].to_numpy(), y.iloc[:cutoff].to_numpy())
    preds = pd.Series(model.predict(x.iloc[cutoff:].to_numpy()), index=x.iloc[cutoff:].index)
    actual = y.loc[preds.index]

    daily_returns = []
    for date, day_preds in preds.groupby(level="date"):
        day_actual = actual.loc[date]
        ranked = day_preds.droplevel("date").sort_values(ascending=False)
        if len(ranked) < 2:
            continue
        long_return = day_actual.loc[ranked.index[0]]
        short_return = day_actual.loc[ranked.index[-1]]
        daily_returns.append(0.5 * long_return - 0.5 * short_return)

    print(summarize_performance(pd.Series(daily_returns)))


if __name__ == "__main__":
    main()
