import pandas as pd

from data.download import format_yfinance_ohlcv, load_sp100_tickers
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


def test_load_sp100_tickers_reads_symbol_table(monkeypatch):
    tickers = [f"T{i:03d}" for i in range(100)]

    def fake_read_html(url):
        return [pd.DataFrame({"Symbol": tickers})]

    monkeypatch.setattr(pd, "read_html", fake_read_html)

    assert load_sp100_tickers("https://example.test") == tickers


def test_load_sp100_tickers_normalizes_yahoo_class_tickers(monkeypatch):
    tickers = ["BRK.B", "BF.B", *[f"T{i:03d}" for i in range(98)]]

    def fake_read_html(url):
        return [pd.DataFrame({"Symbol": tickers})]

    monkeypatch.setattr(pd, "read_html", fake_read_html)

    assert load_sp100_tickers("https://example.test")[:2] == ["BRK-B", "BF-B"]
