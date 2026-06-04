import numpy as np
import pandas as pd

from quant.pipeline import purged_time_split


def _supervised_panel(n_days=30, tickers=("A", "B")):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    x = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(np.arange(len(index), dtype=float), index=index, name="target")
    return x, y


def test_purged_time_split_removes_forward_label_overlap():
    x, y = _supervised_panel()
    x_train, x_test, y_train, y_test = purged_time_split(
        x,
        y,
        test_start="2020-01-20",
        horizon=5,
    )

    assert x_train.index.get_level_values("date").max() < pd.Timestamp("2020-01-20") - pd.offsets.BDay(5)
    assert x_test.index.get_level_values("date").min() >= pd.Timestamp("2020-01-20")
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_purged_time_split_respects_train_start_and_test_end():
    x, y = _supervised_panel(n_days=45)
    x_train, x_test, _, _ = purged_time_split(
        x,
        y,
        train_start="2020-01-08",
        test_start="2020-02-03",
        test_end="2020-02-10",
        horizon=5,
    )

    assert x_train.index.get_level_values("date").min() >= pd.Timestamp("2020-01-08")
    assert x_train.index.get_level_values("date").max() < pd.Timestamp("2020-02-03") - pd.offsets.BDay(5)
    assert x_test.index.get_level_values("date").min() >= pd.Timestamp("2020-02-03")
    assert x_test.index.get_level_values("date").max() < pd.Timestamp("2020-02-10")
