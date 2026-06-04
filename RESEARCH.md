# FedAlpha v2 Research Plan

## Questions

1. Does Federated Learning improve performance compared with isolated local models?
2. What is the performance cost of Differential Privacy in trading signals?
3. Does a federated model generalize better across market regimes?
4. Does post-FL personalization improve each institution's local Sharpe ratio?

## Baseline Hierarchy

| Baseline | Purpose |
|---|---|
| Buy and Hold S&P 500 | Passive reference |
| Equal Weight | Naive rebalanced equity allocation |
| Momentum 20D | Simple quant rule |
| Local Ridge per institution | Isolated learning limit |
| Centralized Ridge | Upper bound without privacy constraint |
| FL + FedAvg | Basic collaboration |
| FL + FedProx | Non-IID robustness |
| FL + DP-SGD | Privacy-performance tradeoff |
| FL + Byzantine defense | Malicious-client robustness |
| FL + fine-tuning | Personalized federation |
| FedAlpha Regime-Aware | Client weights use data size, validation Sharpe, regime compatibility, and stability |
| FedAlpha Personalized | Regime-aware global model plus local shrinkage |

## FedAlpha Contribution

FedAlpha Regime-Aware Personalized FL tests whether a market-regime aware aggregation rule improves out-of-sample institutional performance under non-IID financial data. Client weights combine sample size, validation Sharpe, current-regime compatibility, and validation volatility. The personalized variant then shrinks the global Ridge coefficients toward each client's local Ridge coefficients.

## Walk-Forward Windows

| Window | Train | Validation | Test |
|---|---|---|---|
| 1 | 2014-2018 | 2018-2019 | 2019 |
| 2 | 2014-2019 | 2019-2020 | 2020 |
| 3 | 2014-2020 | 2020-2021 | 2021 |
| 4 | 2014-2021 | 2021-2022 | 2022 |
| 5 | 2014-2022 | 2022-2023 | 2023-2025 |

Hyperparameters are selected on validation only. The final test estimate retrains the selected configuration on train plus validation, then measures the held-out test window.

## Experiment Matrix

The main report should include:

- Sharpe, Sortino, Calmar, max drawdown, alpha, beta, tracking error.
- Lo-style Sharpe significance test.
- Block bootstrap confidence intervals.
- Performance by market regime.
- Epsilon-vs-Sharpe curve for DP-SGD.
- Communication reduction from top-k sparsification.
- Byzantine attack simulation with FedAvg vs trimmed mean/Krum.

## Anti-Leakage Rules

- Features at date `t` may use only information available at or before `t`.
- Labels are forward returns from `t+1` to `t+horizon`.
- Scalers, imputers, model selection, and HMM regimes must be fitted inside each training window.
- Regime labels are produced by fitting on train returns only, then applying the fitted detector or fitted fallback quantile thresholds to validation/test returns.
- A five-session purge is applied before every test window for the five-day forward-return label.
- Test windows are never used for hyperparameter selection.
- Transaction costs are applied to weight changes, not final weights.

## Current Data Limitation

The current experiment uses today's S&P 100 constituents for a historical backtest and is therefore subject to survivorship bias. Results are interpreted as pipeline validation, not deployable trading performance.
