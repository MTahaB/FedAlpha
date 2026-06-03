from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from federated_learning.privacy import estimate_local_dp_epsilon
from quant.metrics import compute_sharpe


def simulate_privacy_tradeoff(
    noise_multipliers: list[float],
    seed: int = 42,
    n_days: int = 252,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0008, 0.01, size=n_days)
    rows: list[dict[str, float | str]] = []

    for mode in ["central_dp", "local_dp"]:
        for noise in noise_multipliers:
            if mode == "central_dp":
                epsilon = 8.0 / max(noise, 0.25)
                noisy_returns = signal + rng.normal(0.0, noise * 0.0006, size=n_days)
            else:
                epsilon = estimate_local_dp_epsilon(noise, local_steps=2) if noise > 0 else float("inf")
                noisy_returns = signal + rng.normal(0.0, noise * 0.0012, size=n_days)

            rows.append(
                {
                    "mode": mode,
                    "noise_multiplier": float(noise),
                    "epsilon": float(epsilon) if np.isfinite(epsilon) else float("inf"),
                    "sharpe": compute_sharpe(noisy_returns),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare central DP and local DP epsilon-Sharpe tradeoffs.")
    parser.add_argument("--noise", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = simulate_privacy_tradeoff(args.noise, seed=args.seed)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.reports_dir / "privacy_tradeoff.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
