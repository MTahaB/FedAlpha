from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SP100_SAMPLE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "JNJ",
    "UNH",
    "PFE",
    "MRK",
    "CAT",
    "GE",
    "HON",
    "BA",
    "XOM",
    "CVX",
]


FIELD_ALIASES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj_close": "adj_close",
    "volume": "volume",
}


def _normalize(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _ticker_and_field(first: object, second: object) -> tuple[str, str]:
    first_norm = _normalize(first)
    second_norm = _normalize(second)
    if first_norm in FIELD_ALIASES:
        return str(second), FIELD_ALIASES[first_norm]
    return str(first), FIELD_ALIASES.get(second_norm, second_norm)


def format_yfinance_ohlcv(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Downloaded frame is empty.")

    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    per_ticker: dict[str, pd.DataFrame] = {}

    if isinstance(frame.columns, pd.MultiIndex):
        for column in frame.columns:
            ticker, field = _ticker_and_field(column[0], column[1])
            if field not in FIELD_ALIASES.values():
                continue
            if ticker.lower().startswith("unnamed") and len(tickers) == 1:
                ticker = tickers[0]
            per_ticker.setdefault(ticker, pd.DataFrame(index=frame.index))[field] = frame[column]
    else:
        ticker = tickers[0] if len(tickers) == 1 else "UNKNOWN"
        single = frame.rename(columns={column: _normalize(column) for column in frame.columns})
        per_ticker[ticker] = single[[column for column in single.columns if column in FIELD_ALIASES.values()]]

    pieces = []
    for ticker, ticker_frame in per_ticker.items():
        ticker_frame = ticker_frame.reset_index(names="date")
        ticker_frame["ticker"] = ticker
        pieces.append(ticker_frame)

    long = pd.concat(pieces, ignore_index=True)
    columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    available = [column for column in columns if column in long.columns]
    return long[available].sort_values(["date", "ticker"]).reset_index(drop=True)


def download_yfinance(tickers: list[str], start: str, end: str, output_dir: Path) -> Path:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `pip install -r requirements.txt`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ohlcv.csv"
    frame = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker")
    format_yfinance_ohlcv(frame, tickers).to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV data for FedAlpha.")
    parser.add_argument("--preset", choices=["sp100"], default="sp100")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers or SP100_SAMPLE
    path = download_yfinance(tickers, args.start, args.end, args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
