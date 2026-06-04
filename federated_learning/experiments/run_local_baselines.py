from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant.backtest import asset_return_panel, make_expanding_windows
from quant.benchmarks import buy_and_hold_returns, equal_weight_returns, momentum_20d_long_short
from data.download import download_yfinance, load_sp100_tickers
from quant.backtest import backtest_predictions
from quant.data_loader import load_ohlcv_csv
from quant.metrics import summarize_performance
from quant.models import OptionalDependencyMissing, RidgeSignalModel, TreeBoostingSignalModel
from quant.pipeline import (
    chronological_split,
    embargoed_train_end,
    prepare_supervised_dataset,
    purged_time_split,
    resolve_supervised_split_date,
)


ModelFactory = Callable[[], object]
DEFAULT_DATA_PATH = Path("data/raw/ohlcv.csv")
DEFAULT_REPORTS_DIR = Path("reports")


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
        split_date = resolve_supervised_split_date(ohlcv, horizon=horizon)
        dataset = prepare_supervised_dataset(
            ohlcv,
            horizon=horizon,
            regime_train_end=embargoed_train_end(split_date, embargo_days=horizon),
        )
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
    embargo_days: int = 5,
) -> dict[str, dict]:
    split_date = resolve_supervised_split_date(ohlcv, horizon=horizon, test_start=test_start)
    dataset = prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        regime_train_end=embargoed_train_end(split_date, embargo_days=embargo_days),
    )
    x, y = dataset.features, dataset.labels
    x_train, x_test, y_train, _ = chronological_split(
        x,
        y,
        test_start=test_start,
        embargo_days=embargo_days,
    )

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
            "embargo_days": int(embargo_days),
            "regime_fit": "train_only",
            "metrics": summarize_performance(returns),
        }
    return results


def _slice_dates(frame: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    index = pd.to_datetime(frame.index.get_level_values("date") if isinstance(frame.index, pd.MultiIndex) else frame.index)
    return frame.loc[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))]


def _metrics_row(window: str, method: str, returns: pd.Series, **extra: object) -> dict[str, object]:
    metrics = summarize_performance(returns)
    return {
        "window": window,
        "method": method,
        "status": "ok",
        "n_returns": int(returns.dropna().shape[0]),
        **metrics,
        **extra,
    }


def run_walk_forward_baselines(
    ohlcv: pd.DataFrame,
    model_names: list[str],
    horizon: int = 5,
    embargo_days: int = 5,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Run naive and ML baselines on the research walk-forward windows."""
    asset_returns = asset_return_panel(ohlcv)
    market_close = ohlcv["close"].unstack("ticker").sort_index().mean(axis=1)

    rows: list[dict[str, object]] = []
    return_series: dict[str, pd.Series] = {}

    for window in make_expanding_windows():
        train_cutoff = embargoed_train_end(pd.Timestamp(window.test_start), embargo_days=embargo_days)
        dataset = prepare_supervised_dataset(
            ohlcv,
            horizon=horizon,
            regime_train_start=window.train_start,
            regime_train_end=train_cutoff,
        )
        test_asset_returns = _slice_dates(asset_returns, window.test_start, window.test_end)
        buy_hold = _slice_dates(buy_and_hold_returns(market_close), window.test_start, window.test_end)
        equal_weight = equal_weight_returns(test_asset_returns).rename("equal_weight")
        momentum = _slice_dates(
            momentum_20d_long_short(_slice_dates(asset_returns, window.train_start, window.test_end)),
            window.test_start,
            window.test_end,
        ).rename("momentum_20d")

        for method, returns in {
            "buy_hold": buy_hold,
            "equal_weight": equal_weight,
            "momentum_20d": momentum,
        }.items():
            rows.append(
                _metrics_row(
                    window.name,
                    method,
                    returns,
                    embargo_days=embargo_days,
                    regime_fit="not_used",
                )
            )
            return_series[f"{window.name}:{method}"] = returns

        x_train, x_test, y_train, _ = purged_time_split(
            dataset.features,
            dataset.labels,
            train_start=window.train_start,
            test_start=window.test_start,
            test_end=window.test_end,
            horizon=embargo_days,
        )

        if x_train.empty or x_test.empty:
            for name in model_names:
                rows.append(
                    {
                        "window": window.name,
                        "method": name,
                        "status": "skipped",
                        "n_returns": 0,
                        "embargo_days": embargo_days,
                        "regime_fit": "train_only",
                        "reason": "empty train or test split",
                    }
                )
            continue

        for name, factory in baseline_factories(model_names).items():
            model = factory()
            try:
                model.fit(x_train.to_numpy(), y_train.to_numpy())
                predictions = pd.Series(model.predict(x_test.to_numpy()), index=x_test.index, name=name)
                returns = backtest_predictions(predictions, ohlcv)
            except OptionalDependencyMissing as exc:
                rows.append(
                    {
                        "window": window.name,
                        "method": name,
                        "status": "skipped",
                        "n_returns": 0,
                        "embargo_days": embargo_days,
                        "regime_fit": "train_only",
                        "reason": str(exc),
                    }
                )
                continue
            rows.append(
                _metrics_row(
                    window.name,
                    name,
                    returns,
                    embargo_days=embargo_days,
                    regime_fit="train_only",
                )
            )
            return_series[f"{window.name}:{name}"] = returns.rename(name)

    return pd.DataFrame(rows), return_series


def write_baseline_reports(
    summary: pd.DataFrame,
    return_series: dict[str, pd.Series],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "summary_csv": reports_dir / "baselines_summary.csv",
        "summary_json": reports_dir / "baselines_summary.json",
    }
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(summary.replace({np.nan: None}).to_dict(orient="records"), indent=2, default=float),
        encoding="utf-8",
    )

    for idx, window in enumerate(make_expanding_windows(), start=1):
        window_summary = summary[summary["window"] == window.name]
        path = reports_dir / f"baselines_window_{idx}.csv"
        window_summary.to_csv(path, index=False)
        paths[f"window_{idx}"] = path

        window_returns = {
            key.split(":", 1)[1]: series
            for key, series in return_series.items()
            if key.startswith(f"{window.name}:")
        }
        if window_returns:
            returns_path = reports_dir / f"baselines_window_{idx}_returns.csv"
            pd.concat(window_returns, axis=1).to_csv(returns_path)
            paths[f"window_{idx}_returns"] = returns_path

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local non-FL baselines on real OHLCV data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--models", nargs="+", default=["ridge", "lightgbm", "xgboost"])
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--embargo-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ohlcv = load_research_ohlcv(
        data_path=args.data_path,
        download_if_missing=not args.no_download,
        start=args.start,
        end=args.end,
    )
    if args.single_split:
        results = run_baselines(
            ohlcv,
            model_names=args.models,
            horizon=args.horizon,
            test_start=args.test_start,
            embargo_days=args.embargo_days,
        )
        print(json.dumps(results, indent=2, default=float))
        return

    summary, return_series = run_walk_forward_baselines(
        ohlcv,
        model_names=args.models,
        horizon=args.horizon,
        embargo_days=args.embargo_days,
    )
    paths = write_baseline_reports(summary, return_series, reports_dir=args.reports_dir)
    print(json.dumps({"summary": summary.to_dict(orient="records"), "paths": paths}, indent=2, default=str))


if __name__ == "__main__":
    main()
