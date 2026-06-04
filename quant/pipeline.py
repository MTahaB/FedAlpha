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
from quant.features import REGIME_LABELS, build_features
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
    return close.pct_change(fill_method=None).mean(axis=1).rename("market_return")


def resolve_split_date(
    dates: pd.Index,
    train_fraction: float = 0.70,
    test_start: str | None = None,
) -> pd.Timestamp:
    unique_dates = pd.Index(sorted(pd.to_datetime(pd.unique(dates))))
    if len(unique_dates) < 3:
        raise ValueError("Need at least three unique dates for a chronological split.")
    if test_start is not None:
        return pd.Timestamp(test_start)
    cutoff = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * train_fraction)))
    return pd.Timestamp(unique_dates[cutoff])


def embargoed_train_end(split_date: pd.Timestamp | str, embargo_days: int = 5) -> pd.Timestamp:
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative.")
    split = pd.Timestamp(split_date)
    if embargo_days == 0:
        return split
    return split - pd.offsets.BDay(embargo_days)


def train_fitted_regime_labels(
    market_returns: pd.Series,
    *,
    train_start: str | pd.Timestamp | None = None,
    train_end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Fit regime detection on train returns only, then label all available dates."""
    returns = market_returns.astype(float).sort_index()
    train_mask = pd.Series(True, index=returns.index)
    if train_start is not None:
        train_mask &= returns.index >= pd.Timestamp(train_start)
    if train_end is not None:
        train_mask &= returns.index < pd.Timestamp(train_end)

    train_returns = returns.loc[train_mask].dropna()
    if len(train_returns) < 30:
        return _train_quantile_regime_labels(returns, train_returns)

    valid_returns = returns.dropna()
    try:
        from quant.regimes import MarketRegimeDetector

        detector = MarketRegimeDetector().fit(train_returns.to_numpy())
        labels = pd.Series(
            detector.predict(valid_returns.to_numpy()),
            index=valid_returns.index,
            name="regime",
        )
        return labels.reindex(returns.index).ffill().bfill().rename("regime")
    except Exception:
        return _train_quantile_regime_labels(returns, train_returns)


def _train_quantile_regime_labels(returns: pd.Series, train_returns: pd.Series) -> pd.Series:
    volatility = returns.rolling(20).std()
    train_volatility = train_returns.rolling(20).std().dropna()
    if train_volatility.empty:
        return pd.Series("crisis", index=returns.index, name="regime")

    low, high = train_volatility.quantile([1 / 3, 2 / 3]).tolist()
    labels = pd.Series(index=returns.index, dtype=object, name="regime")
    labels.loc[volatility <= low] = REGIME_LABELS[0]
    labels.loc[(volatility > low) & (volatility <= high)] = REGIME_LABELS[1]
    labels.loc[volatility > high] = REGIME_LABELS[2]
    return labels.ffill().bfill().fillna("crisis").rename("regime")


def prepare_supervised_dataset(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
    feature_columns: list[str] | None = None,
    regime_labels: pd.Series | None = None,
    regime_train_start: str | pd.Timestamp | None = None,
    regime_train_end: str | pd.Timestamp | None = None,
) -> SupervisedDataset:
    market_returns = market_return_series(ohlcv)
    if regime_labels is None and (regime_train_start is not None or regime_train_end is not None):
        regime_labels = train_fitted_regime_labels(
            market_returns,
            train_start=regime_train_start,
            train_end=regime_train_end,
        )
    features = build_features(ohlcv, market_returns=market_returns, regime_labels=regime_labels)
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


def resolve_supervised_split_date(
    ohlcv: pd.DataFrame,
    *,
    horizon: int = 5,
    train_fraction: float = 0.70,
    test_start: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.Timestamp:
    if test_start is not None:
        return pd.Timestamp(test_start)
    preliminary = prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        feature_columns=feature_columns,
    )
    return resolve_split_date(
        preliminary.features.index.get_level_values("date"),
        train_fraction=train_fraction,
    )


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.70,
    test_start: str | None = None,
    embargo_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.Index(sorted(pd.unique(x.index.get_level_values("date"))))
    split_date = resolve_split_date(dates, train_fraction=train_fraction, test_start=test_start)
    return purged_time_split(
        x,
        y,
        test_start=split_date,
        horizon=embargo_days,
    )


def purged_time_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    train_start: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split with a purge before the test window.

    The forward label at decision date `t` uses future returns through
    `t + horizon`, so the last `horizon` train sessions are excluded before
    the test starts.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    split_date = pd.Timestamp(test_start)
    train_cutoff = embargoed_train_end(split_date, embargo_days=horizon)

    date_index = pd.to_datetime(x.index.get_level_values("date"))
    train_mask = date_index < train_cutoff
    if train_start is not None:
        train_mask &= date_index >= pd.Timestamp(train_start)
    test_mask = date_index >= split_date
    if test_end is not None:
        test_mask &= date_index < pd.Timestamp(test_end)
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
    embargo_days: int = 5,
) -> LocalTrainingResult:
    ohlcv = load_ohlcv_csv(partition_path)
    split_date = resolve_supervised_split_date(
        ohlcv,
        horizon=horizon,
        feature_columns=feature_columns,
        train_fraction=train_fraction,
        test_start=test_start,
    )
    dataset = prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        feature_columns=feature_columns,
        regime_train_end=embargoed_train_end(split_date, embargo_days=embargo_days),
    )
    x_train, _, y_train, _ = chronological_split(
        dataset.features,
        dataset.labels,
        train_fraction=train_fraction,
        test_start=split_date.date().isoformat(),
        embargo_days=embargo_days,
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
    embargo_days: int = 5,
    top_k: int = 5,
    bottom_k: int = 5,
) -> tuple[pd.Series, pd.Series, dict]:
    split_date = resolve_supervised_split_date(
        ohlcv,
        horizon=horizon,
        feature_columns=state["feature_columns"],
        train_fraction=train_fraction,
        test_start=test_start,
    )
    dataset = prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        feature_columns=state["feature_columns"],
        regime_train_end=embargoed_train_end(split_date, embargo_days=embargo_days),
    )
    _, x_test, _, _ = chronological_split(
        dataset.features,
        dataset.labels,
        train_fraction=train_fraction,
        test_start=split_date.date().isoformat(),
        embargo_days=embargo_days,
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
    embargo_days: int = 5,
    oracle_url: str | None = None,
) -> dict:
    ohlcv = load_ohlcv_csv(data_path)
    split_date = resolve_supervised_split_date(ohlcv, horizon=horizon, test_start=test_start)
    dataset = prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        regime_train_end=embargoed_train_end(split_date, embargo_days=embargo_days),
    )
    x_train, x_test, y_train, _ = chronological_split(
        dataset.features,
        dataset.labels,
        test_start=split_date.date().isoformat(),
        embargo_days=embargo_days,
    )
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
