from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from urllib import request as urllib_request

import numpy as np
import pandas as pd

from quant.backtest import backtest_predictions
from quant.data_loader import load_ohlcv_csv
from quant.features import build_features
from quant.labels import make_forward_return_labels
from quant.metrics import summarize_performance
from quant.models import RidgeSignalModel


TARGET_COLUMN = "target"


@dataclass(frozen=True)
class SupervisedDataset:
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series


@dataclass(frozen=True)
class LocalTrainingResult:
    model: RidgeSignalModel
    model_state: dict
    n_examples: int
    train_loss: float


def market_return_series(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv["close"].unstack("ticker").sort_index()
    return close.pct_change().mean(axis=1).rename("market_return")


def prepare_supervised_dataset(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
    feature_columns: list[str] | None = None,
) -> SupervisedDataset:
    market_returns = market_return_series(ohlcv)
    features = build_features(ohlcv, market_returns=market_returns)
    labels = make_forward_return_labels(ohlcv, horizon=horizon)
    target = f"forward_return_{horizon}d"
    x = features.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if feature_columns is not None:
        x = x.reindex(columns=feature_columns, fill_value=0.0)
    joined = x.join(labels[[target]], how="inner").rename(columns={target: TARGET_COLUMN}).dropna()
    if joined.empty:
        raise ValueError("No supervised rows remain after feature/label alignment.")
    return SupervisedDataset(
        ohlcv=ohlcv,
        features=joined.drop(columns=[TARGET_COLUMN]),
        labels=joined[TARGET_COLUMN],
    )


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.70,
    test_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.Index(sorted(pd.unique(x.index.get_level_values("date"))))
    if len(dates) < 3:
        raise ValueError("Need at least three unique dates for a chronological split.")

    if test_start is not None:
        split_date = pd.Timestamp(test_start)
    else:
        cutoff = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
        split_date = pd.Timestamp(dates[cutoff])

    date_index = pd.to_datetime(x.index.get_level_values("date"))
    train_mask = date_index < split_date
    test_mask = date_index >= split_date
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"Split date {split_date.date()} produced an empty train or test set.")
    return x.loc[train_mask], x.loc[test_mask], y.loc[train_mask], y.loc[test_mask]


def model_state_from_ridge(model: RidgeSignalModel, feature_columns: list[str]) -> dict:
    if model.coef_ is None:
        raise RuntimeError("Model must be fitted before serializing.")
    return {
        "model_type": "ridge",
        "alpha": float(model.alpha),
        "fit_intercept": bool(model.fit_intercept),
        "feature_columns": list(feature_columns),
        "coef": model.coef_.astype(float).tolist(),
    }


def ridge_from_state(state: dict) -> RidgeSignalModel:
    model = RidgeSignalModel(alpha=float(state.get("alpha", 1.0)), fit_intercept=bool(state.get("fit_intercept", True)))
    model.coef_ = np.asarray(state["coef"], dtype=float)
    return model


def train_local_model(
    partition_path: str | Path,
    *,
    feature_columns: list[str] | None = None,
    horizon: int = 5,
    alpha: float = 1.0,
    train_fraction: float = 0.70,
    test_start: str | None = None,
) -> LocalTrainingResult:
    dataset = prepare_supervised_dataset(load_ohlcv_csv(partition_path), horizon=horizon, feature_columns=feature_columns)
    x_train, _, y_train, _ = chronological_split(
        dataset.features,
        dataset.labels,
        train_fraction=train_fraction,
        test_start=test_start,
    )
    model = RidgeSignalModel(alpha=alpha).fit(x_train.to_numpy(), y_train.to_numpy())
    preds = model.predict(x_train.to_numpy())
    train_loss = float(np.mean((preds - y_train.to_numpy()) ** 2))
    return LocalTrainingResult(
        model=model,
        model_state=model_state_from_ridge(model, list(x_train.columns)),
        n_examples=int(len(x_train)),
        train_loss=train_loss,
    )


def aggregate_ridge_states(states: list[dict], weights: list[int | float]) -> dict:
    if not states:
        raise ValueError("states cannot be empty.")
    feature_columns = states[0]["feature_columns"]
    if any(state["feature_columns"] != feature_columns for state in states):
        raise ValueError("All ridge states must share feature columns.")

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.sum() <= 0:
        raise ValueError("weights must sum to a positive value.")
    weights_array = weights_array / weights_array.sum()
    coefs = np.stack([np.asarray(state["coef"], dtype=float) for state in states])
    aggregate_coef = np.tensordot(weights_array, coefs, axes=(0, 0))
    return {
        "model_type": "federated_ridge_fedavg",
        "alpha": float(states[0].get("alpha", 1.0)),
        "fit_intercept": bool(states[0].get("fit_intercept", True)),
        "feature_columns": feature_columns,
        "coef": aggregate_coef.tolist(),
        "client_weights": weights_array.tolist(),
    }


def evaluate_model_state(
    state: dict,
    ohlcv: pd.DataFrame,
    *,
    horizon: int = 5,
    train_fraction: float = 0.70,
    test_start: str | None = None,
    top_k: int = 5,
    bottom_k: int = 5,
) -> tuple[pd.Series, pd.Series, dict]:
    dataset = prepare_supervised_dataset(ohlcv, horizon=horizon, feature_columns=state["feature_columns"])
    _, x_test, _, _ = chronological_split(
        dataset.features,
        dataset.labels,
        train_fraction=train_fraction,
        test_start=test_start,
    )
    model = ridge_from_state(state)
    predictions = pd.Series(model.predict(x_test.to_numpy()), index=x_test.index, name="prediction")
    returns = backtest_predictions(predictions, ohlcv, top_k=top_k, bottom_k=bottom_k)
    metrics = summarize_performance(returns)
    return predictions, returns, metrics


def validate_with_oracle(
    model_state: dict,
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    oracle_url: str | None = None,
    round_id: int = 1,
    anchor_blockchain: bool = True,
) -> dict:
    payload = {
        "model_state": model_state,
        "returns": [float(value) for value in returns.dropna().tolist()],
        "benchmark_returns": None
        if benchmark_returns is None
        else [float(value) for value in benchmark_returns.dropna().tolist()],
        "min_sharpe": 0.1,
        "max_drawdown": -0.5,
        "round_id": round_id,
        "anchor_blockchain": anchor_blockchain,
    }
    if oracle_url:
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            oracle_url.rstrip("/") + "/validate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    from oracle.validation_api import ValidationRequest, validate

    return validate(ValidationRequest(**payload))


def save_pipeline_artifacts(
    *,
    prefix: str,
    reports_dir: str | Path,
    models_dir: str | Path,
    predictions: pd.Series,
    returns: pd.Series,
    metrics: dict,
    model_state: dict,
    oracle_response: dict | None = None,
    training_history: pd.DataFrame | None = None,
) -> dict[str, Path]:
    report_path = Path(reports_dir)
    model_path = Path(models_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "predictions": report_path / f"{prefix}_predictions.csv",
        "returns": report_path / f"{prefix}_returns.csv",
        "metrics": report_path / f"{prefix}_metrics.json",
        "model": model_path / f"{prefix}_model.pkl",
    }
    predictions.rename("prediction").to_frame().to_csv(paths["predictions"])
    returns.rename("portfolio_return").to_frame().to_csv(paths["returns"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    with paths["model"].open("wb") as handle:
        pickle.dump(model_state, handle)

    if oracle_response is not None:
        paths["oracle"] = report_path / f"{prefix}_oracle_response.json"
        paths["oracle"].write_text(json.dumps(oracle_response, indent=2, default=float), encoding="utf-8")
        (report_path / "oracle_response.json").write_text(json.dumps(oracle_response, indent=2, default=float), encoding="utf-8")
    if training_history is not None:
        paths["training_history"] = report_path / f"{prefix}_training_history.csv"
        training_history.to_csv(paths["training_history"], index=False)
        if prefix == "federated":
            training_history.to_csv(report_path / "federated_training_history.csv", index=False)
    return paths


def run_centralized_pipeline(
    data_path: str | Path,
    *,
    reports_dir: str | Path = "reports",
    models_dir: str | Path = "models",
    horizon: int = 5,
    alpha: float = 1.0,
    test_start: str | None = None,
    oracle_url: str | None = None,
) -> dict:
    ohlcv = load_ohlcv_csv(data_path)
    dataset = prepare_supervised_dataset(ohlcv, horizon=horizon)
    x_train, x_test, y_train, _ = chronological_split(dataset.features, dataset.labels, test_start=test_start)
    model = RidgeSignalModel(alpha=alpha).fit(x_train.to_numpy(), y_train.to_numpy())
    state = model_state_from_ridge(model, list(x_train.columns))
    predictions = pd.Series(model.predict(x_test.to_numpy()), index=x_test.index, name="prediction")
    returns = backtest_predictions(predictions, ohlcv)
    metrics = summarize_performance(returns)
    oracle_response = validate_with_oracle(state, returns, oracle_url=oracle_url, round_id=1)
    save_pipeline_artifacts(
        prefix="centralized",
        reports_dir=reports_dir,
        models_dir=models_dir,
        predictions=predictions,
        returns=returns,
        metrics=metrics,
        model_state=state,
        oracle_response=oracle_response,
    )
    return {"model_state": state, "metrics": metrics, "oracle": oracle_response}
