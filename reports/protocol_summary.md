# FedAlpha Research Protocol

- Universe: current S&P 100 constituents loaded from the local OHLCV dataset.
- Label: five-session forward return.
- Validation: alpha, top-k, and bottom-k are selected on the validation window only.
- Final test: models are retrained on train plus validation before measuring the test window.
- Regimes: the main study uses deterministic volatility-quantile regime features fitted on train returns only inside each walk-forward window; the HMM detector follows the same train-only contract and can be swapped in for slower runs.
- FedAlpha contribution: regime-aware aggregation weights clients by sample size, validation Sharpe, current-regime compatibility, and validation volatility; the personalized variant shrinks the global Ridge model toward each local client model.
- Purge: a five-session purge prevents forward-label overlap with validation and test windows.
- Transaction costs: fixed plus market-impact costs are charged on turnover.

Survivorship bias notice: The current experiment uses today's S&P 100 constituents for a historical backtest and is therefore subject to survivorship bias. Results are interpreted as pipeline validation, not deployable trading performance.
