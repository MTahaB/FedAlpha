from __future__ import annotations

import pandas as pd

from quant.metrics import bootstrap_metrics, sharpe_significance_test, summarize_performance


def validate_return_series(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    min_sharpe: float = 0.75,
    max_drawdown: float = -0.20,
) -> dict:
    performance = summarize_performance(returns, benchmark_returns)
    sharpe_test = sharpe_significance_test(returns, confidence=0.90)
    bootstrap = bootstrap_metrics(returns, n_bootstrap=500, confidence=0.90)

    passed = (
        performance["sharpe_ratio"] >= min_sharpe
        and performance["max_drawdown"] >= max_drawdown
        and bool(sharpe_test["significant"])
    )

    sharpe_ratio = performance["sharpe_ratio"]
    validation_score = 0 if pd.isna(sharpe_ratio) else int(max(0, min(100, sharpe_ratio / min_sharpe * 100)))

    return {
        "performance": performance,
        "statistical_tests": {
            "sharpe": sharpe_test,
            "bootstrap": bootstrap,
        },
        "validation_criteria": {
            "min_sharpe_threshold": min_sharpe,
            "max_drawdown_threshold": max_drawdown,
            "sharpe_significant": bool(sharpe_test["significant"]),
        },
        "validated": bool(passed),
        "validation_score": validation_score,
    }
