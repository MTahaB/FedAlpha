from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from federated_learning.experiments.run_federated import DEFAULT_PARTITIONS, run_federated_pipeline
from quant.backtest import asset_return_panel, backtest_predictions
from quant.benchmarks import equal_weight_returns
from quant.data_loader import load_ohlcv_csv
from quant.metrics import summarize_performance
from quant.models import RidgeSignalModel
from quant.pipeline import (
    chronological_split,
    embargoed_train_end,
    prepare_supervised_dataset,
    resolve_supervised_split_date,
    run_centralized_pipeline,
)


def _date_slice(frame: pd.DataFrame | pd.Series, start: pd.Timestamp) -> pd.DataFrame | pd.Series:
    raw_dates = frame.index.get_level_values("date") if isinstance(frame.index, pd.MultiIndex) else frame.index
    return frame.loc[pd.to_datetime(raw_dates) >= start]


def _table_row(method: str, metrics: dict[str, float], **metadata: object) -> dict[str, object]:
    return {
        "method": method,
        "annual_return": metrics.get("annual_return"),
        "annual_volatility": metrics.get("annual_volatility"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "sortino_ratio": metrics.get("sortino_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar_ratio": metrics.get("calmar_ratio"),
        **metadata,
    }


def run_local_ridge_ensemble(
    partition_paths: list[Path],
    *,
    split_date: pd.Timestamp,
    reports_dir: Path,
    horizon: int = 5,
    embargo_days: int = 5,
) -> dict[str, object]:
    client_returns: dict[str, pd.Series] = {}

    for idx, path in enumerate(partition_paths, start=1):
        ohlcv = load_ohlcv_csv(path)
        dataset = prepare_supervised_dataset(
            ohlcv,
            horizon=horizon,
            regime_train_end=embargoed_train_end(split_date, embargo_days=embargo_days),
        )
        x_train, x_test, y_train, _ = chronological_split(
            dataset.features,
            dataset.labels,
            test_start=split_date.date().isoformat(),
            embargo_days=embargo_days,
        )
        model = RidgeSignalModel(alpha=1.0).fit(x_train.to_numpy(), y_train.to_numpy())
        predictions = pd.Series(
            model.predict(x_test.to_numpy()),
            index=x_test.index,
            name=f"client_{idx}",
        )
        client_returns[f"client_{idx}"] = backtest_predictions(predictions, ohlcv).rename(f"client_{idx}")

    returns = pd.concat(client_returns, axis=1).mean(axis=1).rename("local_ridge")
    metrics = summarize_performance(returns)
    reports_dir.mkdir(parents=True, exist_ok=True)
    returns.to_frame("portfolio_return").to_csv(reports_dir / "local_ridge_returns.csv")
    (reports_dir / "local_ridge_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float),
        encoding="utf-8",
    )
    return {"returns": returns, "metrics": metrics}


def run_comparison_table(
    *,
    partition_paths: list[Path] = DEFAULT_PARTITIONS,
    evaluation_data_path: Path = Path("data/raw/ohlcv.csv"),
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
    horizon: int = 5,
    train_fraction: float = 0.70,
    test_start: str | None = None,
    embargo_days: int = 5,
    robust_aggregator: str = "median",
) -> pd.DataFrame:
    ohlcv = load_ohlcv_csv(evaluation_data_path)
    split_date = resolve_supervised_split_date(
        ohlcv,
        horizon=horizon,
        train_fraction=train_fraction,
        test_start=test_start,
    )
    split_label = split_date.date().isoformat()

    equal_weight = equal_weight_returns(_date_slice(asset_return_panel(ohlcv), split_date))
    centralized = run_centralized_pipeline(
        evaluation_data_path,
        reports_dir=reports_dir,
        models_dir=models_dir,
        horizon=horizon,
        test_start=split_label,
        embargo_days=embargo_days,
    )
    local = run_local_ridge_ensemble(
        partition_paths,
        split_date=split_date,
        reports_dir=reports_dir,
        horizon=horizon,
        embargo_days=embargo_days,
    )
    fedavg = run_federated_pipeline(
        partition_paths,
        evaluation_data_path,
        reports_dir=reports_dir,
        models_dir=models_dir,
        horizon=horizon,
        train_fraction=train_fraction,
        test_start=split_label,
        embargo_days=embargo_days,
        round_id=30,
        robust_aggregator="fedavg",
        reports_prefix="federated",
    )
    robust = run_federated_pipeline(
        partition_paths,
        evaluation_data_path,
        reports_dir=reports_dir,
        models_dir=models_dir,
        horizon=horizon,
        train_fraction=train_fraction,
        test_start=split_label,
        embargo_days=embargo_days,
        round_id=31,
        robust_aggregator=robust_aggregator,
        reports_prefix="federated_robust",
    )

    metadata = {
        "split_date": split_label,
        "embargo_days": int(embargo_days),
        "regime_fit": "train_only",
    }
    rows = [
        _table_row("Equal Weight", summarize_performance(equal_weight), model_scope="all_assets", **metadata),
        _table_row("Centralized Ridge", centralized["metrics"], model_scope="all_assets", **metadata),
        _table_row("Local Ridge", local["metrics"], model_scope="client_isolated", **metadata),
        _table_row("FedAvg", fedavg["metrics"], model_scope="federated", **metadata),
        _table_row(
            f"Robust Aggregation ({robust_aggregator})",
            robust["metrics"],
            model_scope="federated",
            **metadata,
        ),
    ]
    table = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "comparison_table.csv", index=False)
    (reports_dir / "comparison_table.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=float),
        encoding="utf-8",
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the main FedAlpha method comparison table.")
    parser.add_argument("--partition-paths", nargs="*", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--evaluation-data-path", type=Path, default=Path("data/raw/ohlcv.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--robust-aggregator", default="median")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = run_comparison_table(
        partition_paths=args.partition_paths,
        evaluation_data_path=args.evaluation_data_path,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        test_start=args.test_start,
        embargo_days=args.embargo_days,
        robust_aggregator=args.robust_aggregator,
    )
    print(table.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
