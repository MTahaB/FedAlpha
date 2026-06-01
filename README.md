# FedAlpha v2

Federated Learning, privacy-preserving collaboration, and decentralized validation for quantitative trading.

FedAlpha v2 studies whether several financial institutions can collaboratively train stronger trading signals than their isolated local models while preserving data confidentiality and validating the learning process through an oracle/blockchain layer.

## Project Thesis

The central research question is:

> Can Federated Learning help financial institutions produce more robust trading signals than individual local models, while preserving client data privacy and adding cryptographic/process robustness to model validation?

The first implementation priority is the quantitative research core: clean data, anti-leakage features, strict walk-forward validation, long-short portfolio construction, statistical tests, and baselines. Federated Learning, Differential Privacy, blockchain governance, oracle validation, MLflow, and dashboarding are layered on top of that core.

## Repository Map

```text
data/                   Data download entrypoint and local data folders
quant/                  Quant research pipeline, features, labels, models, backtests
federated_learning/     Aggregation, DP helpers, Flower server/client skeletons
oracle/                 FastAPI validation service, model hash registry, stats tests
blockchain/             Solidity contracts and Hardhat project skeleton
dashboard/              Streamlit dashboard
clients/                Dockerized institutional clients
experiments/            Notebook placeholders and experiment notes
tests/                  Unit tests for quant and FL primitives
reports/                Generated charts and final report artifacts
```

## Quick Start

Create an environment with Python 3.11 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the core tests:

```bash
python -m pytest
```

Download market data:

```bash
python data/download.py --preset sp100 --start 2014-01-01 --end 2025-01-01
```

Run the first offline experiment on synthetic data:

```bash
python -m federated_learning.experiments.run_local_baselines
```

Launch the oracle API:

```bash
uvicorn oracle.validation_api:app --reload --port 8000
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Implementation Order

1. Quant pipeline and anti-leakage checks.
2. Walk-forward baselines.
3. Federated Learning with FedAvg, FedProx, and Byzantine defenses.
4. Differential Privacy with Opacus and epsilon-vs-Sharpe experiments.
5. Docker Compose for distributed simulation.
6. Blockchain staking, slashing, aggregator selection, and DAO parameters.
7. Oracle statistical validation and model hashing.
8. MLflow experiment tracking.
9. Streamlit dashboard.
10. Research report and notebooks.

## Success Criteria

The project is successful if it rigorously demonstrates at least one of:

- `Sharpe(FL) > mean(Sharpe(local models))` with statistical support.
- `Sharpe(FL + DP, epsilon=1) > 80% * Sharpe(centralized model)`.
- `Sharpe(FL)` is more robust than the best local model in stressed regimes such as 2020 or 2022.

Negative results are still valuable if the experiment design is clean and well documented.
