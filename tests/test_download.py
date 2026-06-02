import pandas as pd

from data.download import format_yfinance_ohlcv
from quant.data_loader import load_ohlcv_csv


def test_format_yfinance_ohlcv_roundtrip(tmp_path):
    columns = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    raw = pd.DataFrame(
        [[10, 11, 9, 10.5, 10.4, 1000], [10.5, 12, 10, 11.5, 11.4, 1200]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=columns,
    )

    formatted = format_yfinance_ohlcv(raw, ["AAPL"])
    path = tmp_path / "ohlcv.csv"
    formatted.to_csv(path, index=False)
    loaded = load_ohlcv_csv(path)

    assert loaded.index.names == ["date", "ticker"]
    assert loaded.loc[(pd.Timestamp("2024-01-02"), "AAPL"), "close"] == 10.5
