from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant.data_loader import load_ohlcv_csv
from quant.pipeline import (
    aggregate_ridge_states,
    evaluate_model_state,
    prepare_supervised_dataset,
    save_pipeline_artifacts,
    train_local_model,
    validate_with_oracle,
)


DEFAULT_PARTITIONS = [
    Path("data/client_partitions/tech/ohlcv.csv"),
    Path("data/client_partitions/finance/ohlcv.csv"),
    Path("data/client_partitions/healthcare_industrials/ohlcv.csv"),
]


def run_federated_pipeline(
    partition_paths: list[Path],
    evaluation_data_path: Path,
    *,
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
    horizon: int = 5,
    alpha: float = 1.0,
    oracle_url: str | None = None,
    round_id: int = 1,
) -> dict:
    if len(partition_paths) < 2:
        raise ValueError("At least two client partitions are required.")

    evaluation_ohlcv = load_ohlcv_csv(evaluation_data_path)
    evaluation_dataset = prepare_supervised_dataset(evaluation_ohlcv, horizon=horizon)
    feature_columns = list(evaluation_dataset.features.columns)

    local_results = [
        train_local_model(
            partition,
            feature_columns=feature_columns,
            horizon=horizon,
            alpha=alpha,
        )
        for partition in partition_paths
    ]
    global_state = aggregate_ridge_states(
        [result.model_state for result in local_results],
        [result.n_examples for result in local_results],
    )
    predictions, returns, metrics = evaluate_model_state(global_state, evaluation_ohlcv, horizon=horizon)
    oracle_response = validate_with_oracle(
        global_state,
        returns,
        oracle_url=oracle_url,
        round_id=round_id,
        anchor_blockchain=True,
    )

    training_history = pd.DataFrame(
        [
            {
                "round": round_id,
                "client": f"client_{idx + 1}",
                "examples": result.n_examples,
                "loss": result.train_loss,
            }
            for idx, result in enumerate(local_results)
        ]
    )
    save_pipeline_artifacts(
        prefix="federated",
        reports_dir=reports_dir,
        models_dir=models_dir,
        predictions=predictions,
        returns=returns,
        metrics=metrics,
        model_state=global_state,
        oracle_response=oracle_response,
        training_history=training_history,
    )
    return {
        "model_state": global_state,
        "metrics": metrics,
        "oracle": oracle_response,
        "clients": [
            {
                "n_examples": result.n_examples,
                "train_loss": result.train_loss,
            }
            for result in local_results
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a connected FedAlpha federated Ridge pipeline.")
    parser.add_argument("--partition-paths", nargs="*", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--evaluation-data-path", type=Path, default=Path("data/raw/ohlcv.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--oracle-url", default=None)
    parser.add_argument("--round-id", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_federated_pipeline(
        args.partition_paths,
        args.evaluation_data_path,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        horizon=args.horizon,
        alpha=args.alpha,
        oracle_url=args.oracle_url,
        round_id=args.round_id,
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
