from __future__ import annotations

from typing import Any

import numpy as np


class OptionalDependencyMissing(RuntimeError):
    pass


class TreeBoostingSignalModel:
    """Thin adapter for LightGBM and XGBoost regressors used as local baselines."""

    def __init__(self, backend: str, **params: Any):
        backend = backend.lower()
        if backend not in {"lightgbm", "xgboost"}:
            raise ValueError("backend must be 'lightgbm' or 'xgboost'.")
        self.backend = backend
        self.params = params
        self.model: Any | None = None

    @classmethod
    def lightgbm(cls, **params: Any) -> "TreeBoostingSignalModel":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(params)
        return cls("lightgbm", **defaults)

    @classmethod
    def xgboost(cls, **params: Any) -> "TreeBoostingSignalModel":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": 1,
        }
        defaults.update(params)
        return cls("xgboost", **defaults)

    def _build_model(self) -> Any:
        if self.backend == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise OptionalDependencyMissing(
                    "Install LightGBM with `pip install lightgbm` or `pip install -r requirements-optional.txt`."
                ) from exc
            return LGBMRegressor(**self.params)

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "Install XGBoost with `pip install xgboost` or `pip install -r requirements-optional.txt`."
            ) from exc
        return XGBRegressor(**self.params)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TreeBoostingSignalModel":
        self.model = self._build_model()
        self.model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        return np.asarray(self.model.predict(np.asarray(x, dtype=float)), dtype=float)
