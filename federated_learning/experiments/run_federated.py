from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from federated_learning.aggregation import fedavg, krum_layers, median_layers, simulate_byzantine_updates, trimmed_mean_layers
from quant.data_loader import load_ohlcv_csv
from quant.pipeline import (
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
ATTACK_ALIASES = {
    "none": "none",
    "sign_flip": "sign_flip",
    "gaussian": "gaussian",
    "gaussian_noise": "gaussian",
    "random_weights": "constant",
    "constant": "constant",
}


def _aggregate_ridge_states_for_fl(
    states: list[dict],
    weights: list[int | float],
    *,
    robust_aggregator: str = "fedavg",
    malicious_attack: str = "none",
    malicious_client_indices: list[int] | None = None,
    attack_scale: float = 10.0,
) -> dict:
    if not states:
        raise ValueError("states cannot be empty.")
    feature_columns = states[0]["feature_columns"]
    if any(state["feature_columns"] != feature_columns for state in states):
        raise ValueError("All ridge states must share feature columns.")

    attack = ATTACK_ALIASES.get(malicious_attack)
    if attack is None:
        raise ValueError(f"Unknown malicious attack: {malicious_attack}")

    updates = [[np.asarray(state["coef"], dtype=float)] for state in states]
    if attack != "none":
        indices = malicious_client_indices or [len(updates) - 1]
        updates = simulate_byzantine_updates(
            updates,
            malicious_clients=len(indices),
            attack=attack,
            scale=attack_scale,
            seed=42,
            malicious_indices=indices,
        )

    aggregator = robust_aggregator.lower()
    if aggregator == "fedavg":
        aggregate_coef = fedavg(updates, weights=weights)[0]
    elif aggregator == "median":
        aggregate_coef = median_layers(updates)[0]
    elif aggregator in {"trimmed_mean", "trimmed-mean"}:
        aggregate_coef = trimmed_mean_layers(updates, trim_ratio=0.2)[0]
    elif aggregator == "krum":
        byzantine_clients = len(malicious_client_indices or ([] if attack == "none" else [len(updates) - 1]))
        aggregate_coef = krum_layers(updates, byzantine_clients=byzantine_clients)[0]
    else:
        raise ValueError(f"Unknown robust aggregator: {robust_aggregator}")

    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    return {
        "model_type": f"federated_ridge_{aggregator}",
        "alpha": float(states[0].get("alpha", 1.0)),
        "fit_intercept": bool(states[0].get("fit_intercept", True)),
        "feature_columns": feature_columns,
        "coef": aggregate_coef.astype(float).tolist(),
        "client_weights": weights_array.tolist(),
        "robust_aggregator": aggregator,
        "malicious_attack": malicious_attack,
        "malicious_client_indices": malicious_client_indices or ([] if attack == "none" else [len(states) - 1]),
    }


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
    robust_aggregator: str = "fedavg",
    malicious_attack: str = "none",
    malicious_client_indices: list[int] | None = None,
    attack_scale: float = 10.0,
    reports_prefix: str = "federated",
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
    global_state = _aggregate_ridge_states_for_fl(
        [result.model_state for result in local_results],
        [result.n_examples for result in local_results],
        robust_aggregator=robust_aggregator,
        malicious_attack=malicious_attack,
        malicious_client_indices=malicious_client_indices,
        attack_scale=attack_scale,
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
        prefix=reports_prefix,
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
        "robust_aggregator": robust_aggregator,
        "malicious_attack": malicious_attack,
        "malicious_client_indices": malicious_client_indices or [],
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
    parser.add_argument("--robust-aggregator", default=os.getenv("FL_ROBUST_AGGREGATOR", "fedavg"))
    parser.add_argument("--malicious-attack", default=os.getenv("MALICIOUS_ATTACK", "none"))
    parser.add_argument("--malicious-client-indices", nargs="*", type=int, default=None)
    parser.add_argument("--attack-scale", type=float, default=10.0)
    parser.add_argument("--reports-prefix", default="federated")
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
        robust_aggregator=args.robust_aggregator,
        malicious_attack=args.malicious_attack,
        malicious_client_indices=args.malicious_client_indices,
        attack_scale=args.attack_scale,
        reports_prefix=args.reports_prefix,
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
