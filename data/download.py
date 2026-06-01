from __future__ import annotations

import argparse
from pathlib import Path


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


def download_yfinance(tickers: list[str], start: str, end: str, output_dir: Path) -> Path:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `pip install -r requirements.txt`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ohlcv.csv"
    frame = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker")
    frame.to_csv(path)
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
