from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.download import download_yfinance, load_sp100_tickers
from quant.backtest import backtest_predictions
from quant.data_loader import load_ohlcv_csv
from quant.metrics import summarize_performance
from quant.models import OptionalDependencyMissing, RidgeSignalModel, TreeBoostingSignalModel
from quant.pipeline import chronological_split, market_return_series, prepare_supervised_dataset


ModelFactory = Callable[[], object]
DEFAULT_DATA_PATH = Path("data/raw/ohlcv.csv")


def load_research_ohlcv(
    data_path: Path = DEFAULT_DATA_PATH,
    download_if_missing: bool = True,
    start: str = "2014-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    if data_path.exists():
        return load_ohlcv_csv(data_path)
    if not download_if_missing:
        raise FileNotFoundError(
            f"{data_path} not found. Run `python data/download.py --preset sp100 "
            f"--start {start} --end {end}` first."
        )

    tickers = load_sp100_tickers()
    written = download_yfinance(tickers, start, end, data_path.parent)
    return load_ohlcv_csv(written)


def prepare_supervised_frame(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    try:
        dataset = prepare_supervised_dataset(ohlcv, horizon=horizon)
    except ValueError as exc:
        raise ValueError(
            "No supervised rows remain after feature/label alignment. "
            "Use real OHLCV history with enough observations for rolling features "
            "(roughly 252+ trading days) instead of the tiny smoke fixture."
        ) from exc
    return dataset.features, dataset.labels


def baseline_factories(model_names: list[str]) -> dict[str, ModelFactory]:
    factories: dict[str, ModelFactory] = {
        "ridge": lambda: RidgeSignalModel(alpha=1.0),
        "lightgbm": TreeBoostingSignalModel.lightgbm,
        "xgboost": TreeBoostingSignalModel.xgboost,
    }
    unknown = sorted(set(model_names).difference(factories))
    if unknown:
        raise ValueError(f"Unknown baseline model(s): {unknown}")
    return {name: factories[name] for name in model_names}


def run_baselines(
    ohlcv: pd.DataFrame,
    model_names: list[str],
    horizon: int = 5,
    test_start: str | None = None,
) -> dict[str, dict]:
    x, y = prepare_supervised_frame(ohlcv, horizon=horizon)
    x_train, x_test, y_train, _ = chronological_split(x, y, test_start=test_start)

    results: dict[str, dict] = {}
    for name, factory in baseline_factories(model_names).items():
        model = factory()
        try:
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            predictions = pd.Series(model.predict(x_test.to_numpy()), index=x_test.index, name=name)
        except OptionalDependencyMissing as exc:
            results[name] = {"status": "skipped", "reason": str(exc)}
            continue

        returns = backtest_predictions(predictions, ohlcv)
        results[name] = {
            "status": "ok",
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "metrics": summarize_performance(returns),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local non-FL baselines on real OHLCV data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--models", nargs="+", default=["ridge", "lightgbm", "xgboost"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ohlcv = load_research_ohlcv(
        data_path=args.data_path,
        download_if_missing=not args.no_download,
        start=args.start,
        end=args.end,
    )
    results = run_baselines(
        ohlcv,
        model_names=args.models,
        horizon=args.horizon,
        test_start=args.test_start,
    )
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
