from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from federated_learning.experiments.run_federated import DEFAULT_PARTITIONS, run_federated_pipeline


def run_byzantine_comparison(
    *,
    partition_paths: list[Path] = DEFAULT_PARTITIONS,
    evaluation_data_path: Path = Path("data/raw/ohlcv.csv"),
    reports_dir: Path = Path("reports"),
    models_dir: Path = Path("models"),
    malicious_attack: str = "sign_flip",
    malicious_client_indices: list[int] | None = None,
    attack_scale: float = 10.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    malicious_client_indices = malicious_client_indices or [2]

    for round_id, aggregator in [(10, "median"), (11, "fedavg")]:
        result = run_federated_pipeline(
            partition_paths,
            evaluation_data_path,
            reports_dir=reports_dir,
            models_dir=models_dir,
            round_id=round_id,
            robust_aggregator=aggregator,
            malicious_attack=malicious_attack,
            malicious_client_indices=malicious_client_indices,
            attack_scale=attack_scale,
            reports_prefix=f"federated_{aggregator}_{malicious_attack}",
        )
        rows.append(
            {
                "round_id": round_id,
                "robust_aggregator": aggregator,
                "malicious_attack": malicious_attack,
                "malicious_client_indices": ",".join(str(idx) for idx in malicious_client_indices),
                "validated": bool(result["oracle"]["validated"]),
                "validation_score": float(result["oracle"]["validation_score"]),
                **result["metrics"],
            }
        )

    comparison = pd.DataFrame(rows)
    fedavg = comparison.loc[comparison["robust_aggregator"] == "fedavg"].iloc[0]
    comparison["sharpe_delta_vs_fedavg"] = comparison["sharpe_ratio"] - float(fedavg["sharpe_ratio"])
    comparison["annual_return_delta_vs_fedavg"] = comparison["annual_return"] - float(fedavg["annual_return"])
    comparison["max_drawdown_delta_vs_fedavg"] = comparison["max_drawdown"] - float(fedavg["max_drawdown"])
    comparison["attack_detection_rate"] = comparison.apply(
        lambda row: float(
            row["robust_aggregator"] != "fedavg"
            and row["annual_return_delta_vs_fedavg"] >= 0
            and row["max_drawdown_delta_vs_fedavg"] >= 0
        ),
        axis=1,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "byzantine_comparison.csv", index=False)
    (reports_dir / "byzantine_comparison.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2, default=float),
        encoding="utf-8",
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare robust vs non-robust FL under Byzantine attack.")
    parser.add_argument("--partition-paths", nargs="*", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--evaluation-data-path", type=Path, default=Path("data/raw/ohlcv.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--malicious-attack", default="sign_flip")
    parser.add_argument("--malicious-client-indices", nargs="*", type=int, default=[2])
    parser.add_argument("--attack-scale", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_byzantine_comparison(
        partition_paths=args.partition_paths,
        evaluation_data_path=args.evaluation_data_path,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        malicious_attack=args.malicious_attack,
        malicious_client_indices=args.malicious_client_indices,
        attack_scale=args.attack_scale,
    )
    print(comparison.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
