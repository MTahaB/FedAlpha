from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from quant.data_loader import load_ohlcv_csv, save_client_partitions


SP100_SAMPLE = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AMAT",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BKNG",
    "BLK",
    "BMY",
    "BNY",
    "BRK-B",
    "C",
    "CAT",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "EMR",
    "FDX",
    "GD",
    "GE",
    "GEV",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "INTU",
    "ISRG",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "LRCX",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "MU",
    "NEE",
    "NFLX",
    "NKE",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PLTR",
    "PM",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TMO",
    "TMUS",
    "TSLA",
    "TXN",
    "UBER",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
]
SP100_SOURCE_URL = "https://en.wikipedia.org/wiki/S%26P_100"
DEFAULT_CLIENT_GROUPS = {
    "tech": [
        "AAPL",
        "ACN",
        "ADBE",
        "AMAT",
        "AMD",
        "AMZN",
        "AVGO",
        "CRM",
        "CSCO",
        "GOOG",
        "GOOGL",
        "IBM",
        "INTC",
        "INTU",
        "LRCX",
        "META",
        "MSFT",
        "MU",
        "NFLX",
        "NOW",
        "NVDA",
        "ORCL",
        "PLTR",
        "QCOM",
        "TXN",
    ],
    "finance": [
        "AXP",
        "BAC",
        "BLK",
        "BNY",
        "BRK-B",
        "C",
        "COF",
        "GS",
        "JPM",
        "MA",
        "MS",
        "SCHW",
        "USB",
        "V",
        "WFC",
    ],
    "healthcare_industrials": [
        "ABBV",
        "ABT",
        "AMGN",
        "AMT",
        "BA",
        "BKNG",
        "BMY",
        "CAT",
        "CL",
        "CMCSA",
        "COP",
        "COST",
        "CVS",
        "CVX",
        "DE",
        "DHR",
        "DIS",
        "DUK",
        "EMR",
        "FDX",
        "GD",
        "GE",
        "GEV",
        "GILD",
        "GM",
        "HD",
        "HON",
        "ISRG",
        "JNJ",
        "KO",
        "LIN",
        "LLY",
        "LMT",
        "LOW",
        "MCD",
        "MDLZ",
        "MDT",
        "MMM",
        "MO",
        "MRK",
        "NEE",
        "NKE",
        "PEP",
        "PFE",
        "PG",
        "PM",
        "RTX",
        "SBUX",
        "SO",
        "SPG",
        "T",
        "TMO",
        "TMUS",
        "TSLA",
        "UBER",
        "UNH",
        "UNP",
        "UPS",
        "VZ",
        "WMT",
        "XOM",
    ],
}


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


def _to_yfinance_ticker(ticker: object) -> str:
    return str(ticker).strip().replace(".", "-")


def load_sp100_tickers(source_url: str = SP100_SOURCE_URL) -> list[str]:
    """Load the current S&P 100 ticker list, falling back to a bundled liquid sample."""
    try:
        tables = pd.read_html(source_url)
    except Exception:
        return SP100_SAMPLE

    for table in tables:
        normalized_columns = {_normalize(column): column for column in table.columns}
        symbol_column = normalized_columns.get("symbol") or normalized_columns.get("ticker")
        if symbol_column is None:
            continue
        tickers = [_to_yfinance_ticker(value) for value in table[symbol_column].dropna()]
        tickers = [ticker for ticker in tickers if ticker and ticker.lower() != "nan"]
        if len(tickers) >= 80:
            return tickers
    return SP100_SAMPLE


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
    if hasattr(yf, "set_tz_cache_location"):
        cache_dir = output_dir / "yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
    path = output_dir / "ohlcv.csv"
    frame = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker")
    format_yfinance_ohlcv(frame, tickers).to_csv(path, index=False)
    return path


def write_default_client_partitions(
    data_path: Path,
    output_dir: Path = Path("data/client_partitions"),
) -> dict[str, Path]:
    ohlcv = load_ohlcv_csv(data_path)
    available = set(ohlcv.index.get_level_values("ticker"))
    groups = {
        name: [ticker for ticker in tickers if ticker in available]
        for name, tickers in DEFAULT_CLIENT_GROUPS.items()
    }
    groups = {name: tickers for name, tickers in groups.items() if tickers}
    if not groups:
        raise ValueError("No default client partition tickers are available in the downloaded data.")
    return save_client_partitions(ohlcv, groups, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV data for FedAlpha.")
    parser.add_argument("--preset", choices=["sp100"], default="sp100")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--client-partitions-dir", type=Path, default=Path("data/client_partitions"))
    parser.add_argument("--skip-client-partitions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers or load_sp100_tickers()
    path = download_yfinance(tickers, args.start, args.end, args.output_dir)
    print(f"Wrote {path}")
    if not args.skip_client_partitions:
        partitions = write_default_client_partitions(path, args.client_partitions_dir)
        for name, partition_path in partitions.items():
            print(f"Wrote {name} partition: {partition_path}")


if __name__ == "__main__":
    main()
