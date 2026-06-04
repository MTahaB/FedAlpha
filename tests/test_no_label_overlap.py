import numpy as np
import pandas as pd

from quant.labels import make_forward_return_labels
from quant.pipeline import purged_time_split


def test_purge_keeps_train_forward_labels_before_test_window():
    dates = pd.bdate_range("2021-01-01", periods=35)
    tickers = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    close = np.repeat(100 + np.arange(len(dates)), len(tickers)) + np.tile([0.0, 1.0], len(dates))
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )
    x = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = make_forward_return_labels(ohlcv, horizon=5)["forward_return_5d"].dropna()
    x = x.loc[y.index]

    test_start = pd.Timestamp("2021-02-01")
    x_train, x_test, y_train, y_test = purged_time_split(x, y, test_start=test_start, horizon=5)

    train_decision_dates = pd.Index(x_train.index.get_level_values("date").unique())
    train_label_end_dates = train_decision_dates + pd.offsets.BDay(5)
    assert train_label_end_dates.max() < test_start
    assert x_test.index.get_level_values("date").min() >= test_start
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
