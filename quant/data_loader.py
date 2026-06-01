from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


def normalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower().replace(" ", "_") for column in frame.columns}
    return frame.rename(columns=renamed)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load long-format OHLCV data with Date/Ticker columns or a saved MultiIndex."""
    frame = pd.read_csv(path)
    frame = normalize_ohlcv_columns(frame)

    if {"date", "ticker"}.issubset(frame.columns):
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
        frame = frame.set_index(["date", "ticker"]).sort_index()
    elif {"unnamed:_0", "unnamed:_1"}.issubset(frame.columns):
        frame = frame.rename(columns={"unnamed:_0": "date", "unnamed:_1": "ticker"})
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
        frame = frame.set_index(["date", "ticker"]).sort_index()

    missing = REQUIRED_OHLCV_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("OHLCV data must be indexed by date and ticker.")

    frame.index = frame.index.set_names(["date", "ticker"])
    return frame.sort_index()


def save_client_partitions(
    ohlcv: pd.DataFrame,
    groups: dict[str, list[str]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, tickers in groups.items():
        partition = ohlcv.loc[ohlcv.index.get_level_values("ticker").isin(tickers)].copy()
        path = output_path / name / "ohlcv.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        partition.to_csv(path)
        written[name] = path

    return written
