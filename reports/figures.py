from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd


style.use("seaborn-v0_8-paper")

PALETTE = {
    "buy_hold": "#AAAAAA",
    "momentum_20d": "#888888",
    "ridge": "#377EB8",
    "lightgbm": "#4DAF4A",
    "xgboost": "#A65628",
    "centralized": "#FF7F00",
    "fl_fednova": "#984EA3",
    "fl_dp_eps1": "#E41A1C",
    "fl_ditto": "#F781BF",
}


def fig_sharpe_comparison(results_df: pd.DataFrame, out: str | Path) -> None:
    windows = list(results_df["window"].dropna().unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(max(4, len(windows) * 3.2), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, window in zip(axes, windows, strict=False):
        data = results_df[(results_df["window"] == window) & (results_df["status"] == "ok")]
        data = data.sort_values("sharpe_ratio")
        ax.barh(
            data["method"],
            data["sharpe_ratio"],
            color=[PALETTE.get(method, "#333333") for method in data["method"]],
            alpha=0.85,
        )
        ax.axvline(0, color="black", lw=0.5, ls="--")
        ax.set_title(str(window), fontsize=10)
        ax.set_xlabel("Sharpe Ratio")

    fig.suptitle("Sharpe Ratio by Walk-Forward Window", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_epsilon_sharpe(dp_results: pd.DataFrame, out: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, data in dp_results.groupby("mode"):
        ax.plot(data["epsilon"], data["sharpe"], "o-", label=mode, linewidth=2)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xscale("log")
    ax.set_title("Privacy vs Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_regime_performance(results_df: pd.DataFrame, out: str | Path) -> None:
    regimes = ["bull", "bear", "crisis"]
    methods = list(results_df["method"].dropna().unique())
    x = np.arange(len(regimes))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, method in enumerate(methods):
        data = results_df[results_df["method"] == method]
        sharpes = [data[data["regime"] == regime]["sharpe"].mean() for regime in regimes]
        ax.bar(x + idx * width, sharpes, width, label=method, alpha=0.85)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(["Bull", "Bear", "Crisis"])
    ax.set_ylabel("Mean Sharpe Ratio")
    ax.set_title("Performance by Market Regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FedAlpha report figures from CSV artifacts.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    baselines_path = args.reports_dir / "baselines_summary.csv"
    if baselines_path.exists():
        fig_sharpe_comparison(
            pd.read_csv(baselines_path),
            args.reports_dir / "figure_1_sharpe_comparison.png",
        )

    privacy_path = args.reports_dir / "privacy_tradeoff.csv"
    if privacy_path.exists():
        fig_epsilon_sharpe(
            pd.read_csv(privacy_path),
            args.reports_dir / "figure_2_epsilon_sharpe.png",
        )


if __name__ == "__main__":
    main()
