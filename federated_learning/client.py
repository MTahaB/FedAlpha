from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from quant.pipeline import train_local_model


@dataclass(frozen=True)
class FedAlphaClientConfig:
    client_id: str
    partition_path: Path
    target_epsilon: float = 1.0
    local_epochs: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one FedAlpha local Ridge client on an OHLCV partition.")
    parser.add_argument("--client-id", default="A")
    parser.add_argument("--partition-path", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_local_model(args.partition_path, horizon=args.horizon, alpha=args.alpha)
    print(
        json.dumps(
            {
                "client_id": args.client_id,
                "n_examples": result.n_examples,
                "train_loss": result.train_loss,
                "model_state": result.model_state,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
