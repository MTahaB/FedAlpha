from __future__ import annotations

import numpy as np


class RidgeSignalModel:
    """Small Ridge regression implementation with no sklearn dependency."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(x)), x])
        return x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeSignalModel":
        design = self._design(x)
        target = np.asarray(y, dtype=float)
        penalty = np.eye(design.shape[1]) * self.alpha
        if self.fit_intercept:
            penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        return self._design(x) @ self.coef_
