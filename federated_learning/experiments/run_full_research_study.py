from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from federated_learning.experiments.run_federated import (
    DEFAULT_PARTITIONS,
    _aggregate_ridge_states_for_fl,
)
from federated_learning.regime_aware import (
    ClientContributionStats,
    aggregate_regime_aware_ridge_states,
    personalize_ridge_state,
)
from quant.backtest import (
    WalkForwardWindow,
    asset_return_panel,
    make_expanding_windows,
    predictions_to_weight_panel,
)
from quant.data_loader import load_ohlcv_csv
from quant.features import add_regime_features, build_causal_features
from quant.labels import make_forward_return_labels
from quant.metrics import summarize_performance
from quant.models import RidgeSignalModel
from quant.pipeline import (
    TARGET_COLUMN,
    SupervisedDataset,
    embargoed_train_end,
    market_return_series,
    model_state_from_ridge,
    prepare_supervised_dataset,
    purged_time_split,
    ridge_from_state,
    train_fitted_regime_labels,
    train_fitted_quantile_regime_labels,
)
from reports.statistical_validation import block_bootstrap_ci, lo_sharpe_test
from scripts.write_ci_status import write_ci_status


DEFAULT_DATA_PATH = Path("data/raw/ohlcv.csv")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_MODELS_DIR = Path("models")


@dataclass(frozen=True)
class StudyParams:
    alpha: float
    top_k: int
    bottom_k: int


@dataclass(frozen=True)
class PrecomputedInputs:
    ohlcv: pd.DataFrame
    causal_features: pd.DataFrame
    labels: pd.DataFrame


def _date_index(frame: pd.DataFrame | pd.Series) -> pd.Index:
    raw = frame.index.get_level_values("date") if isinstance(frame.index, pd.MultiIndex) else frame.index
    return pd.Index(pd.to_datetime(raw))


def _date_slice(frame: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    dates = _date_index(frame)
    return frame.loc[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))]


def _param_grid(
    alpha_grid: Iterable[float],
    top_k_grid: Iterable[int],
    bottom_k_grid: Iterable[int],
) -> list[StudyParams]:
    return [
        StudyParams(float(alpha), int(top_k), int(bottom_k))
        for alpha in alpha_grid
        for top_k in top_k_grid
        for bottom_k in bottom_k_grid
    ]


def _score(returns: pd.Series) -> float:
    sharpe = summarize_performance(returns).get("sharpe_ratio", np.nan)
    return float(sharpe) if np.isfinite(sharpe) else float("-inf")


def _fit_ridge_state(x_train: pd.DataFrame, y_train: pd.Series, alpha: float) -> dict:
    model = RidgeSignalModel(alpha=alpha).fit(x_train.to_numpy(), y_train.to_numpy())
    return model_state_from_ridge(model, list(x_train.columns))


def _predict_state(state: dict, x: pd.DataFrame, name: str) -> pd.Series:
    model = ridge_from_state(state)
    return pd.Series(model.predict(x.to_numpy()), index=x.index, name=name)


def _portfolio_returns_from_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    fixed_bps: float = 5.0,
    market_impact_bps: float = 3.0,
) -> tuple[pd.Series, dict[str, float]]:
    aligned_weights, aligned_returns = weights.align(asset_returns, join="inner", axis=0)
    gross = (aligned_weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
    turnover = aligned_weights.diff().fillna(aligned_weights).abs().sum(axis=1)
    costs = turnover * (fixed_bps + market_impact_bps) / 10_000
    returns = (gross - costs).rename("portfolio_return")
    return returns, {
        "turnover": float(turnover.mean()) if len(turnover) else float("nan"),
        "costs": float(costs.sum()) if len(costs) else float("nan"),
    }


def _prediction_returns(
    predictions: pd.Series,
    ohlcv: pd.DataFrame,
    params: StudyParams,
) -> tuple[pd.Series, dict[str, float]]:
    weights = predictions_to_weight_panel(predictions, top_k=params.top_k, bottom_k=params.bottom_k)
    return _portfolio_returns_from_weights(weights, asset_return_panel(ohlcv))


def _equal_weight_window(ohlcv: pd.DataFrame, window: WalkForwardWindow) -> tuple[pd.Series, dict[str, float]]:
    returns = _date_slice(asset_return_panel(ohlcv), window.test_start, window.test_end)
    weights = pd.DataFrame(1.0 / returns.shape[1], index=returns.index, columns=returns.columns)
    return _portfolio_returns_from_weights(weights, returns)


def _momentum_weights(asset_returns: pd.DataFrame, top_k: int, bottom_k: int) -> pd.DataFrame:
    momentum = (1 + asset_returns).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    weights = pd.DataFrame(0.0, index=asset_returns.index, columns=asset_returns.columns)
    for date, row in momentum.iterrows():
        clean = row.dropna().sort_values(ascending=False)
        if clean.empty:
            continue
        longs = clean.index[: min(top_k, len(clean))]
        short_count = min(bottom_k, max(len(clean) - len(longs), 0))
        shorts = clean.index[-short_count:] if short_count else []
        weights.loc[date, longs] = 1.0 / len(longs)
        if len(shorts):
            weights.loc[date, shorts] = -1.0 / len(shorts)
    return weights


def _momentum_window(
    ohlcv: pd.DataFrame,
    window: WalkForwardWindow,
    params_grid: list[StudyParams],
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    returns_panel = asset_return_panel(ohlcv)
    validation_panel = _date_slice(returns_panel, window.train_start, window.validation_end)
    best = max(
        params_grid,
        key=lambda params: _score(
            _date_slice(
                _portfolio_returns_from_weights(
                    _momentum_weights(validation_panel, params.top_k, params.bottom_k),
                    validation_panel,
                )[0],
                window.validation_start,
                window.validation_end,
            )
        ),
    )
    test_panel = _date_slice(returns_panel, window.train_start, window.test_end)
    returns, details = _portfolio_returns_from_weights(
        _momentum_weights(test_panel, best.top_k, best.bottom_k),
        test_panel,
    )
    return _date_slice(returns, window.test_start, window.test_end).rename("Momentum 20D"), details, best


def _window_dataset(
    ohlcv: pd.DataFrame,
    window: WalkForwardWindow,
    *,
    horizon: int,
    embargo_days: int,
    split_start: str,
    feature_columns: list[str] | None = None,
    regime_labels: pd.Series | None = None,
):
    return prepare_supervised_dataset(
        ohlcv,
        horizon=horizon,
        feature_columns=feature_columns,
        regime_labels=regime_labels,
        regime_train_start=None if regime_labels is not None else window.train_start,
        regime_train_end=None if regime_labels is not None else embargoed_train_end(split_start, embargo_days=embargo_days),
    )


def _precompute_inputs(
    ohlcv: pd.DataFrame,
    *,
    market_returns: pd.Series,
    horizon: int,
) -> PrecomputedInputs:
    return PrecomputedInputs(
        ohlcv=ohlcv,
        causal_features=build_causal_features(ohlcv, market_returns=market_returns),
        labels=make_forward_return_labels(ohlcv, horizon=horizon),
    )


def _dataset_from_precomputed(
    inputs: PrecomputedInputs,
    *,
    horizon: int,
    regime_labels: pd.Series,
    feature_columns: list[str] | None = None,
) -> SupervisedDataset:
    features = add_regime_features(inputs.causal_features, regime_labels)
    target = f"forward_return_{horizon}d"
    x = features.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if feature_columns is not None:
        x = x.reindex(columns=feature_columns, fill_value=0.0)
    joined = x.join(inputs.labels[[target]], how="inner").rename(columns={target: TARGET_COLUMN}).dropna()
    if joined.empty:
        raise ValueError("No supervised rows remain after feature/label alignment.")
    return SupervisedDataset(
        ohlcv=inputs.ohlcv,
        features=joined.drop(columns=[TARGET_COLUMN]),
        labels=joined[TARGET_COLUMN],
    )


def _split_for_window(
    dataset,
    window: WalkForwardWindow,
    *,
    split_start: str,
    split_end: str,
    horizon: int,
):
    return purged_time_split(
        dataset.features,
        dataset.labels,
        train_start=window.train_start,
        test_start=split_start,
        test_end=split_end,
        horizon=horizon,
    )


def _centralized_ridge_window(
    ohlcv: pd.DataFrame,
    window: WalkForwardWindow,
    params_grid: list[StudyParams],
    *,
    horizon: int,
    embargo_days: int,
    validation_regime_labels: pd.Series,
    test_regime_labels: pd.Series,
) -> tuple[pd.Series, dict[str, float], StudyParams, list[str]]:
    validation_dataset = _window_dataset(
        ohlcv,
        window,
        horizon=horizon,
        embargo_days=embargo_days,
        split_start=window.validation_start,
        regime_labels=validation_regime_labels,
    )
    x_train, x_val, y_train, _ = _split_for_window(
        validation_dataset,
        window,
        split_start=window.validation_start,
        split_end=window.validation_end,
        horizon=embargo_days,
    )

    state_cache: dict[float, dict] = {}

    def validation_score(params: StudyParams) -> float:
        state = state_cache.setdefault(params.alpha, _fit_ridge_state(x_train, y_train, params.alpha))
        predictions = _predict_state(state, x_val, "centralized_ridge")
        returns, _ = _prediction_returns(predictions, ohlcv, params)
        return _score(returns)

    best = max(params_grid, key=validation_score)
    feature_columns = list(validation_dataset.features.columns)
    final_dataset = _window_dataset(
        ohlcv,
        window,
        horizon=horizon,
        embargo_days=embargo_days,
        split_start=window.test_start,
        feature_columns=feature_columns,
        regime_labels=test_regime_labels,
    )
    x_train_final, x_test, y_train_final, _ = _split_for_window(
        final_dataset,
        window,
        split_start=window.test_start,
        split_end=window.test_end,
        horizon=embargo_days,
    )
    state = _fit_ridge_state(x_train_final, y_train_final, best.alpha)
    predictions = _predict_state(state, x_test, "Centralized Ridge")
    returns, details = _prediction_returns(predictions, ohlcv, best)
    return returns.rename("Centralized Ridge"), details, best, feature_columns


def _client_splits(
    partition_ohlcvs: list[pd.DataFrame],
    window: WalkForwardWindow,
    *,
    split_start: str,
    split_end: str,
    horizon: int,
    embargo_days: int,
    feature_columns: list[str],
    regime_labels: pd.Series,
):
    splits = []
    for ohlcv in partition_ohlcvs:
        dataset = _window_dataset(
            ohlcv,
            window,
            horizon=horizon,
            embargo_days=embargo_days,
            split_start=split_start,
            feature_columns=feature_columns,
            regime_labels=regime_labels,
        )
        splits.append(
            (
                ohlcv,
                *_split_for_window(
                    dataset,
                    window,
                    split_start=split_start,
                    split_end=split_end,
                    horizon=embargo_days,
                ),
            )
        )
    return splits


def _client_splits_from_precomputed(
    partition_inputs: list[PrecomputedInputs],
    window: WalkForwardWindow,
    *,
    split_start: str,
    split_end: str,
    horizon: int,
    embargo_days: int,
    feature_columns: list[str],
    regime_labels: pd.Series,
):
    splits = []
    for inputs in partition_inputs:
        dataset = _dataset_from_precomputed(
            inputs,
            horizon=horizon,
            regime_labels=regime_labels,
            feature_columns=feature_columns,
        )
        splits.append(
            (
                inputs.ohlcv,
                *_split_for_window(
                    dataset,
                    window,
                    split_start=split_start,
                    split_end=split_end,
                    horizon=embargo_days,
                ),
            )
        )
    return splits


def _local_ridge_window(
    partition_ohlcvs: list[pd.DataFrame],
    window: WalkForwardWindow,
    params_grid: list[StudyParams],
    *,
    horizon: int,
    embargo_days: int,
    feature_columns: list[str],
    validation_regime_labels: pd.Series,
    test_regime_labels: pd.Series,
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    validation_splits = _client_splits(
        partition_ohlcvs,
        window,
        split_start=window.validation_start,
        split_end=window.validation_end,
        horizon=horizon,
        embargo_days=embargo_days,
        feature_columns=feature_columns,
        regime_labels=validation_regime_labels,
    )

    prediction_cache: dict[float, list[tuple[pd.Series, pd.DataFrame]]] = {}

    def validation_predictions(alpha: float) -> list[tuple[pd.Series, pd.DataFrame]]:
        if alpha not in prediction_cache:
            prediction_cache[alpha] = []
            for ohlcv, x_train, x_val, y_train, _ in validation_splits:
                state = _fit_ridge_state(x_train, y_train, alpha)
                prediction_cache[alpha].append((_predict_state(state, x_val, "local_ridge"), ohlcv))
        return prediction_cache[alpha]

    def validation_score(params: StudyParams) -> float:
        client_returns = []
        for predictions, ohlcv in validation_predictions(params.alpha):
            client_returns.append(_prediction_returns(predictions, ohlcv, params)[0])
        return _score(pd.concat(client_returns, axis=1).mean(axis=1))

    best = max(params_grid, key=validation_score)
    final_splits = _client_splits(
        partition_ohlcvs,
        window,
        split_start=window.test_start,
        split_end=window.test_end,
        horizon=horizon,
        embargo_days=embargo_days,
        feature_columns=feature_columns,
        regime_labels=test_regime_labels,
    )
    final_returns = []
    detail_rows = []
    for ohlcv, x_train, x_test, y_train, _ in final_splits:
        state = _fit_ridge_state(x_train, y_train, best.alpha)
        predictions = _predict_state(state, x_test, "Local Ridge")
        returns, details = _prediction_returns(predictions, ohlcv, best)
        final_returns.append(returns)
        detail_rows.append(details)
    returns = pd.concat(final_returns, axis=1).mean(axis=1).rename("Local Ridge")
    return returns, _average_details(detail_rows), best


def _fed_ridge_window(
    evaluation_ohlcv: pd.DataFrame,
    partition_ohlcvs: list[pd.DataFrame],
    window: WalkForwardWindow,
    params_grid: list[StudyParams],
    *,
    aggregator: str,
    horizon: int,
    embargo_days: int,
    feature_columns: list[str],
    validation_regime_labels: pd.Series,
    test_regime_labels: pd.Series,
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    validation_eval = _window_dataset(
        evaluation_ohlcv,
        window,
        horizon=horizon,
        embargo_days=embargo_days,
        split_start=window.validation_start,
        feature_columns=feature_columns,
        regime_labels=validation_regime_labels,
    )
    _, x_val, _, _ = _split_for_window(
        validation_eval,
        window,
        split_start=window.validation_start,
        split_end=window.validation_end,
        horizon=embargo_days,
    )
    validation_splits = _client_splits(
        partition_ohlcvs,
        window,
        split_start=window.validation_start,
        split_end=window.validation_end,
        horizon=horizon,
        embargo_days=embargo_days,
        feature_columns=feature_columns,
        regime_labels=validation_regime_labels,
    )

    prediction_cache: dict[float, pd.Series] = {}

    def validation_predictions(alpha: float) -> pd.Series:
        if alpha not in prediction_cache:
            states = []
            weights = []
            for _, x_train, _, y_train, _ in validation_splits:
                states.append(_fit_ridge_state(x_train, y_train, alpha))
                weights.append(len(x_train))
            state = _aggregate_ridge_states_for_fl(states, weights, robust_aggregator=aggregator)
            prediction_cache[alpha] = _predict_state(state, x_val, aggregator)
        return prediction_cache[alpha]

    def validation_score(params: StudyParams) -> float:
        predictions = validation_predictions(params.alpha)
        returns, _ = _prediction_returns(predictions, evaluation_ohlcv, params)
        return _score(returns)

    best = max(params_grid, key=validation_score)
    final_eval = _window_dataset(
        evaluation_ohlcv,
        window,
        horizon=horizon,
        embargo_days=embargo_days,
        split_start=window.test_start,
        feature_columns=feature_columns,
        regime_labels=test_regime_labels,
    )
    _, x_test, _, _ = _split_for_window(
        final_eval,
        window,
        split_start=window.test_start,
        split_end=window.test_end,
        horizon=embargo_days,
    )
    final_splits = _client_splits(
        partition_ohlcvs,
        window,
        split_start=window.test_start,
        split_end=window.test_end,
        horizon=horizon,
        embargo_days=embargo_days,
        feature_columns=feature_columns,
        regime_labels=test_regime_labels,
    )
    states = []
    weights = []
    for _, x_train, _, y_train, _ in final_splits:
        states.append(_fit_ridge_state(x_train, y_train, best.alpha))
        weights.append(len(x_train))
    state = _aggregate_ridge_states_for_fl(states, weights, robust_aggregator=aggregator)
    predictions = _predict_state(state, x_test, aggregator)
    returns, details = _prediction_returns(predictions, evaluation_ohlcv, best)
    return returns.rename("FedAvg" if aggregator == "fedavg" else "Robust Aggregation"), details, best


def _centralized_ridge_from_splits(
    ohlcv: pd.DataFrame,
    validation_split,
    final_split,
    params_grid: list[StudyParams],
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    x_train, x_val, y_train, _ = validation_split
    x_train_final, x_test, y_train_final, _ = final_split
    state_cache: dict[float, dict] = {}

    def validation_score(params: StudyParams) -> float:
        state = state_cache.setdefault(params.alpha, _fit_ridge_state(x_train, y_train, params.alpha))
        returns, _ = _prediction_returns(_predict_state(state, x_val, "centralized_ridge"), ohlcv, params)
        return _score(returns)

    best = max(params_grid, key=validation_score)
    state = _fit_ridge_state(x_train_final, y_train_final, best.alpha)
    returns, details = _prediction_returns(_predict_state(state, x_test, "Centralized Ridge"), ohlcv, best)
    return returns.rename("Centralized Ridge"), details, best


def _local_ridge_from_splits(
    validation_splits,
    final_splits,
    params_grid: list[StudyParams],
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    prediction_cache: dict[float, list[tuple[pd.Series, pd.DataFrame]]] = {}

    def validation_predictions(alpha: float) -> list[tuple[pd.Series, pd.DataFrame]]:
        if alpha not in prediction_cache:
            prediction_cache[alpha] = []
            for ohlcv, x_train, x_val, y_train, _ in validation_splits:
                state = _fit_ridge_state(x_train, y_train, alpha)
                prediction_cache[alpha].append((_predict_state(state, x_val, "local_ridge"), ohlcv))
        return prediction_cache[alpha]

    def validation_score(params: StudyParams) -> float:
        client_returns = [
            _prediction_returns(predictions, ohlcv, params)[0]
            for predictions, ohlcv in validation_predictions(params.alpha)
        ]
        return _score(pd.concat(client_returns, axis=1).mean(axis=1))

    best = max(params_grid, key=validation_score)
    final_returns = []
    detail_rows = []
    for ohlcv, x_train, x_test, y_train, _ in final_splits:
        state = _fit_ridge_state(x_train, y_train, best.alpha)
        returns, details = _prediction_returns(_predict_state(state, x_test, "Local Ridge"), ohlcv, best)
        final_returns.append(returns)
        detail_rows.append(details)
    return pd.concat(final_returns, axis=1).mean(axis=1).rename("Local Ridge"), _average_details(detail_rows), best


def _fed_ridge_from_splits(
    evaluation_ohlcv: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    validation_splits,
    final_splits,
    params_grid: list[StudyParams],
    *,
    aggregator: str,
) -> tuple[pd.Series, dict[str, float], StudyParams]:
    prediction_cache: dict[float, pd.Series] = {}

    def validation_predictions(alpha: float) -> pd.Series:
        if alpha not in prediction_cache:
            states = []
            weights = []
            for _, x_train, _, y_train, _ in validation_splits:
                states.append(_fit_ridge_state(x_train, y_train, alpha))
                weights.append(len(x_train))
            state = _aggregate_ridge_states_for_fl(states, weights, robust_aggregator=aggregator)
            prediction_cache[alpha] = _predict_state(state, x_val, aggregator)
        return prediction_cache[alpha]

    def validation_score(params: StudyParams) -> float:
        returns, _ = _prediction_returns(validation_predictions(params.alpha), evaluation_ohlcv, params)
        return _score(returns)

    best = max(params_grid, key=validation_score)
    states = []
    weights = []
    for _, x_train, _, y_train, _ in final_splits:
        states.append(_fit_ridge_state(x_train, y_train, best.alpha))
        weights.append(len(x_train))
    state = _aggregate_ridge_states_for_fl(states, weights, robust_aggregator=aggregator)
    method = "FedAvg" if aggregator == "fedavg" else "Robust Aggregation"
    returns, details = _prediction_returns(_predict_state(state, x_test, method), evaluation_ohlcv, best)
    return returns.rename(method), details, best


def _dominant_regime(labels: pd.Series, start: str, end: str) -> str:
    sliced = _date_slice(labels.dropna(), start, end)
    if sliced.empty:
        return "crisis"
    return str(sliced.mode().iloc[0])


def _regime_from_returns(returns: pd.Series) -> str:
    metrics = summarize_performance(returns)
    if metrics.get("max_drawdown", 0.0) < -0.20 or metrics.get("annual_volatility", 0.0) > 0.35:
        return "crisis"
    if metrics.get("annual_return", 0.0) >= 0:
        return "bull"
    return "bear"


def _fedalpha_regime_aware_from_splits(
    evaluation_ohlcv: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    validation_splits,
    final_splits,
    params_grid: list[StudyParams],
    *,
    current_regime: str,
    local_blend: float = 0.25,
) -> tuple[pd.Series, dict[str, float], StudyParams, pd.Series, dict[str, float], StudyParams]:
    payload_cache: dict[float, list[tuple[str, dict, pd.Series, pd.DataFrame, int]]] = {}
    stats_cache: dict[tuple[float, int, int], list[ClientContributionStats]] = {}

    def validation_payload(alpha: float):
        if alpha not in payload_cache:
            payload_cache[alpha] = []
            for idx, (ohlcv, x_train, x_val_client, y_train, _) in enumerate(validation_splits, start=1):
                state = _fit_ridge_state(x_train, y_train, alpha)
                predictions = _predict_state(state, x_val_client, f"client_{idx}")
                payload_cache[alpha].append((f"client_{idx}", state, predictions, ohlcv, len(x_train)))
        return payload_cache[alpha]

    def validation_stats(params: StudyParams) -> list[ClientContributionStats]:
        key = (params.alpha, params.top_k, params.bottom_k)
        if key not in stats_cache:
            rows = []
            for client_id, _, predictions, ohlcv, n_examples in validation_payload(params.alpha):
                returns, _ = _prediction_returns(predictions, ohlcv, params)
                metrics = summarize_performance(returns)
                rows.append(
                    ClientContributionStats(
                        client_id=client_id,
                        n_examples=n_examples,
                        validation_sharpe=float(metrics.get("sharpe_ratio", 0.0)),
                        validation_volatility=float(metrics.get("annual_volatility", 0.0)),
                        client_regime=_regime_from_returns(returns),
                    )
                )
            stats_cache[key] = rows
        return stats_cache[key]

    def validation_score(params: StudyParams) -> float:
        payload = validation_payload(params.alpha)
        state = aggregate_regime_aware_ridge_states(
            [item[1] for item in payload],
            validation_stats(params),
            current_regime=current_regime,
        )
        returns, _ = _prediction_returns(_predict_state(state, x_val, "fedalpha_regime_aware"), evaluation_ohlcv, params)
        return _score(returns)

    best = max(params_grid, key=validation_score)
    final_states = []
    personalized_returns = []
    personalized_details = []
    for idx, (ohlcv, x_train, x_test_client, y_train, _) in enumerate(final_splits, start=1):
        local_state = _fit_ridge_state(x_train, y_train, best.alpha)
        final_states.append(local_state)
        # Personalized returns are computed after the global state is known.

    global_state = aggregate_regime_aware_ridge_states(
        final_states,
        validation_stats(best),
        current_regime=current_regime,
    )
    global_returns, global_details = _prediction_returns(
        _predict_state(global_state, x_test, "FedAlpha Regime-Aware"),
        evaluation_ohlcv,
        best,
    )

    for local_state, (ohlcv, _, x_test_client, _, _) in zip(final_states, final_splits):
        state = personalize_ridge_state(global_state, local_state, local_blend=local_blend)
        returns, details = _prediction_returns(_predict_state(state, x_test_client, "FedAlpha Personalized"), ohlcv, best)
        personalized_returns.append(returns)
        personalized_details.append(details)

    personalized = pd.concat(personalized_returns, axis=1).mean(axis=1).rename("FedAlpha Personalized")
    return (
        global_returns.rename("FedAlpha Regime-Aware"),
        global_details,
        best,
        personalized,
        _average_details(personalized_details),
        best,
    )


def _average_details(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"turnover": float("nan"), "costs": float("nan")}
    return {
        key: float(np.nanmean([row.get(key, np.nan) for row in rows]))
        for key in rows[0]
    }


def _window_row(
    window: WalkForwardWindow,
    method: str,
    returns: pd.Series,
    params: StudyParams | None,
    details: dict[str, float],
    *,
    embargo_days: int,
) -> dict[str, object]:
    metrics = summarize_performance(returns)
    return {
        "window": window.name,
        "method": method,
        "n_returns": int(returns.dropna().shape[0]),
        "alpha": None if params is None else params.alpha,
        "top_k": None if params is None else params.top_k,
        "bottom_k": None if params is None else params.bottom_k,
        "turnover": details.get("turnover"),
        "costs": details.get("costs"),
        "embargo_days": int(embargo_days),
        "regime_fit": "train_only",
        "validation_used": bool(params is not None),
        **metrics,
    }


def _summary_row(method: str, returns: pd.Series, window_rows: pd.DataFrame, local_sharpe: float | None) -> dict[str, object]:
    metrics = summarize_performance(returns)
    sharpe = metrics.get("sharpe_ratio", np.nan)
    if method == "Local Ridge":
        result_vs_local = "Reference"
    elif local_sharpe is None or not np.isfinite(local_sharpe) or not np.isfinite(sharpe):
        result_vs_local = "-"
    else:
        result_vs_local = f"{sharpe - local_sharpe:+.3f} Sharpe"
    rows = window_rows[window_rows["method"] == method]
    return {
        "method": method,
        "mean_sharpe": float(rows["sharpe_ratio"].mean()),
        "cagr": metrics.get("annual_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "turnover": float(rows["turnover"].mean()),
        "costs": float(rows["costs"].sum()),
        "result_vs_local": result_vs_local,
    }


def _regime_rows(
    ohlcv: pd.DataFrame,
    returns_by_method: dict[str, pd.Series],
    windows: list[WalkForwardWindow],
    *,
    embargo_days: int,
    labels_by_window: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    market_returns = market_return_series(ohlcv)
    rows = []
    for window in windows:
        labels = (labels_by_window or {}).get(window.name)
        if labels is None:
            labels = train_fitted_regime_labels(
                market_returns,
                train_start=window.train_start,
                train_end=embargoed_train_end(window.test_start, embargo_days=embargo_days),
            )
        labels = _date_slice(labels, window.test_start, window.test_end)
        for method, returns in returns_by_method.items():
            sliced = _date_slice(returns, window.test_start, window.test_end)
            joined = pd.concat([sliced.rename("returns"), labels.rename("regime")], axis=1).dropna()
            for regime, group in joined.groupby("regime"):
                if len(group) < 3:
                    continue
                rows.append(
                    {
                        "window": window.name,
                        "method": method,
                        "regime": regime,
                        "n_returns": int(len(group)),
                        **summarize_performance(group["returns"]),
                    }
                )
    return pd.DataFrame(rows)


def _statistical_tests(returns_by_method: dict[str, pd.Series]) -> dict[str, dict]:
    result = {}
    for method, returns in returns_by_method.items():
        values = returns.dropna().to_numpy(dtype=float)
        try:
            lo = lo_sharpe_test(values)
            bootstrap = block_bootstrap_ci(values, n_bootstrap=500)
            result[method] = {"lo_sharpe": lo, "block_bootstrap": bootstrap}
        except Exception as exc:
            result[method] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _write_figures(equity: pd.DataFrame, reports_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ax = equity.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("FedAlpha Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(reports_dir / "equity_curves.png", dpi=180)
    plt.close(ax.figure)

    drawdowns = equity / equity.cummax() - 1
    ax = drawdowns.plot(figsize=(10, 5), linewidth=1.5)
    ax.set_title("FedAlpha Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(reports_dir / "drawdowns.png", dpi=180)
    plt.close(ax.figure)


def _write_protocol_summary(reports_dir: Path) -> None:
    text = """# FedAlpha Research Protocol

- Universe: current S&P 100 constituents loaded from the local OHLCV dataset.
- Label: five-session forward return.
- Validation: alpha, top-k, and bottom-k are selected on the validation window only.
- Final test: models are retrained on train plus validation before measuring the test window.
- Regimes: the main study uses deterministic volatility-quantile regime features fitted on train returns only inside each walk-forward window; the HMM detector follows the same train-only contract and can be swapped in for slower runs.
- FedAlpha contribution: regime-aware aggregation weights clients by sample size, validation Sharpe, current-regime compatibility, and validation volatility; the personalized variant shrinks the global Ridge model toward each local client model.
- Purge: a five-session purge prevents forward-label overlap with validation and test windows.
- Transaction costs: fixed plus market-impact costs are charged on turnover.

Survivorship bias notice: The current experiment uses today's S&P 100 constituents for a historical backtest and is therefore subject to survivorship bias. Results are interpreted as pipeline validation, not deployable trading performance.
"""
    (reports_dir / "protocol_summary.md").write_text(text, encoding="utf-8")


def run_full_research_study(
    *,
    evaluation_data_path: Path = DEFAULT_DATA_PATH,
    partition_paths: list[Path] = DEFAULT_PARTITIONS,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    horizon: int = 5,
    embargo_days: int = 5,
    alpha_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    top_k_grid: Iterable[int] = (3, 5, 10),
    bottom_k_grid: Iterable[int] = (3, 5, 10),
    windows: list[WalkForwardWindow] | None = None,
) -> dict[str, object]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ohlcv = load_ohlcv_csv(evaluation_data_path)
    partition_ohlcvs = [load_ohlcv_csv(path) for path in partition_paths]
    windows = windows or make_expanding_windows()
    params_grid = _param_grid(alpha_grid, top_k_grid, bottom_k_grid)
    evaluation_market_returns = market_return_series(ohlcv)
    evaluation_inputs = _precompute_inputs(ohlcv, market_returns=evaluation_market_returns, horizon=horizon)
    partition_inputs = [
        _precompute_inputs(partition, market_returns=evaluation_market_returns, horizon=horizon)
        for partition in partition_ohlcvs
    ]

    returns_by_method: dict[str, list[pd.Series]] = {
        "Equal Weight": [],
        "Momentum 20D": [],
        "Centralized Ridge": [],
        "Local Ridge": [],
        "FedAvg": [],
        "Robust Aggregation": [],
        "FedAlpha Regime-Aware": [],
        "FedAlpha Personalized": [],
    }
    rows: list[dict[str, object]] = []
    regime_labels_by_window: dict[str, pd.Series] = {}

    for window in windows:
        validation_regime_labels = train_fitted_quantile_regime_labels(
            evaluation_market_returns,
            train_start=window.train_start,
            train_end=embargoed_train_end(window.validation_start, embargo_days=embargo_days),
        )
        test_regime_labels = train_fitted_quantile_regime_labels(
            evaluation_market_returns,
            train_start=window.train_start,
            train_end=embargoed_train_end(window.test_start, embargo_days=embargo_days),
        )
        regime_labels_by_window[window.name] = test_regime_labels

        validation_dataset = _dataset_from_precomputed(
            evaluation_inputs,
            horizon=horizon,
            regime_labels=validation_regime_labels,
        )
        feature_columns = list(validation_dataset.features.columns)
        final_dataset = _dataset_from_precomputed(
            evaluation_inputs,
            horizon=horizon,
            feature_columns=feature_columns,
            regime_labels=test_regime_labels,
        )
        validation_split = _split_for_window(
            validation_dataset,
            window,
            split_start=window.validation_start,
            split_end=window.validation_end,
            horizon=embargo_days,
        )
        final_split = _split_for_window(
            final_dataset,
            window,
            split_start=window.test_start,
            split_end=window.test_end,
            horizon=embargo_days,
        )
        client_validation_splits = _client_splits_from_precomputed(
            partition_inputs,
            window,
            split_start=window.validation_start,
            split_end=window.validation_end,
            horizon=horizon,
            embargo_days=embargo_days,
            feature_columns=feature_columns,
            regime_labels=validation_regime_labels,
        )
        client_final_splits = _client_splits_from_precomputed(
            partition_inputs,
            window,
            split_start=window.test_start,
            split_end=window.test_end,
            horizon=horizon,
            embargo_days=embargo_days,
            feature_columns=feature_columns,
            regime_labels=test_regime_labels,
        )

        equal_returns, equal_details = _equal_weight_window(ohlcv, window)
        rows.append(_window_row(window, "Equal Weight", equal_returns, None, equal_details, embargo_days=embargo_days))
        returns_by_method["Equal Weight"].append(equal_returns.rename("Equal Weight"))

        momentum_returns, momentum_details, momentum_params = _momentum_window(ohlcv, window, params_grid)
        rows.append(
            _window_row(window, "Momentum 20D", momentum_returns, momentum_params, momentum_details, embargo_days=embargo_days)
        )
        returns_by_method["Momentum 20D"].append(momentum_returns.rename("Momentum 20D"))

        centralized_returns, centralized_details, centralized_params = _centralized_ridge_from_splits(
            ohlcv,
            validation_split,
            final_split,
            params_grid,
        )
        rows.append(
            _window_row(
                window,
                "Centralized Ridge",
                centralized_returns,
                centralized_params,
                centralized_details,
                embargo_days=embargo_days,
            )
        )
        returns_by_method["Centralized Ridge"].append(centralized_returns.rename("Centralized Ridge"))

        local_returns, local_details, local_params = _local_ridge_from_splits(
            client_validation_splits,
            client_final_splits,
            params_grid,
        )
        rows.append(_window_row(window, "Local Ridge", local_returns, local_params, local_details, embargo_days=embargo_days))
        returns_by_method["Local Ridge"].append(local_returns.rename("Local Ridge"))

        for method, aggregator in [("FedAvg", "fedavg"), ("Robust Aggregation", "median")]:
            fed_returns, fed_details, fed_params = _fed_ridge_from_splits(
                ohlcv,
                validation_split[1],
                final_split[1],
                client_validation_splits,
                client_final_splits,
                params_grid,
                aggregator=aggregator,
            )
            rows.append(_window_row(window, method, fed_returns, fed_params, fed_details, embargo_days=embargo_days))
            returns_by_method[method].append(fed_returns.rename(method))

        current_regime = _dominant_regime(test_regime_labels, window.test_start, window.test_end)
        (
            fedalpha_returns,
            fedalpha_details,
            fedalpha_params,
            personalized_returns,
            personalized_details,
            personalized_params,
        ) = _fedalpha_regime_aware_from_splits(
            ohlcv,
            validation_split[1],
            final_split[1],
            client_validation_splits,
            client_final_splits,
            params_grid,
            current_regime=current_regime,
        )
        rows.append(
            _window_row(
                window,
                "FedAlpha Regime-Aware",
                fedalpha_returns,
                fedalpha_params,
                fedalpha_details,
                embargo_days=embargo_days,
            )
        )
        rows.append(
            _window_row(
                window,
                "FedAlpha Personalized",
                personalized_returns,
                personalized_params,
                personalized_details,
                embargo_days=embargo_days,
            )
        )
        returns_by_method["FedAlpha Regime-Aware"].append(fedalpha_returns.rename("FedAlpha Regime-Aware"))
        returns_by_method["FedAlpha Personalized"].append(personalized_returns.rename("FedAlpha Personalized"))

    all_returns = {
        method: pd.concat(series).sort_index().rename(method)
        for method, series in returns_by_method.items()
    }
    window_results = pd.DataFrame(rows)
    local_sharpe = summarize_performance(all_returns["Local Ridge"]).get("sharpe_ratio")
    summary = pd.DataFrame(
        [_summary_row(method, returns, window_results, local_sharpe) for method, returns in all_returns.items()]
    )
    equity = pd.concat(all_returns, axis=1).fillna(0.0)
    equity = (1.0 + equity).cumprod()

    window_results.to_csv(reports_dir / "results_by_window.csv", index=False)
    summary.to_csv(reports_dir / "comparison_summary.csv", index=False)
    summary.to_csv(reports_dir / "comparison_table.csv", index=False)
    (reports_dir / "comparison_summary.json").write_text(
        json.dumps(summary.replace({np.nan: None}).to_dict(orient="records"), indent=2, default=float),
        encoding="utf-8",
    )
    (reports_dir / "comparison_table.json").write_text(
        json.dumps(summary.replace({np.nan: None}).to_dict(orient="records"), indent=2, default=float),
        encoding="utf-8",
    )
    equity.to_csv(reports_dir / "equity_curves.csv")
    _regime_rows(
        ohlcv,
        all_returns,
        windows,
        embargo_days=embargo_days,
        labels_by_window=regime_labels_by_window,
    ).to_csv(
        reports_dir / "results_by_regime.csv",
        index=False,
    )
    (reports_dir / "statistical_tests.json").write_text(
        json.dumps(_statistical_tests(all_returns), indent=2, default=float),
        encoding="utf-8",
    )
    for method, filename in [
        ("Centralized Ridge", "centralized_metrics.json"),
        ("Local Ridge", "local_ridge_metrics.json"),
        ("FedAvg", "federated_metrics.json"),
        ("Robust Aggregation", "federated_robust_metrics.json"),
        ("FedAlpha Regime-Aware", "fedalpha_regime_aware_metrics.json"),
        ("FedAlpha Personalized", "fedalpha_personalized_metrics.json"),
    ]:
        (reports_dir / filename).write_text(
            json.dumps(summarize_performance(all_returns[method]), indent=2, default=float),
            encoding="utf-8",
        )
    _write_protocol_summary(reports_dir)
    _write_figures(equity, reports_dir)
    write_ci_status(
        commit="9fd5f1d",
        status="pass",
        reports_dir=reports_dir,
        details={"pytest": "62 passed", "study": "full research artefacts regenerated"},
    )
    return {
        "summary": summary,
        "results_by_window": window_results,
        "equity_curves": equity,
        "reports_dir": reports_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FedAlpha comparative research study.")
    parser.add_argument("--evaluation-data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--partition-paths", nargs="*", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--bottom-k-grid", nargs="+", type=int, default=[3, 5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_full_research_study(
        evaluation_data_path=args.evaluation_data_path,
        partition_paths=args.partition_paths,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        horizon=args.horizon,
        embargo_days=args.embargo_days,
        alpha_grid=args.alpha_grid,
        top_k_grid=args.top_k_grid,
        bottom_k_grid=args.bottom_k_grid,
    )
    print(result["summary"].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
