from __future__ import annotations

from collections import Counter

import numpy as np


class MarketRegimeDetector:
    """HMM-style detector for bull, bear, and crisis volatility regimes."""

    REGIME_NAMES = ("bull", "bear", "crisis")

    def __init__(self, n_states: int = 3, random_state: int = 42):
        if n_states <= 0:
            raise ValueError("n_states must be positive.")
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self._state_to_label: dict[int, str] = {}

    def fit(self, returns_train: np.ndarray) -> "MarketRegimeDetector":
        values = np.asarray(returns_train, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if len(values) < self.n_states:
            raise ValueError("Not enough returns to fit market regimes.")
        x = values.reshape(-1, 1)

        try:
            from hmmlearn import hmm

            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=self.random_state,
            )
            states = self.model.fit(x).predict(x)
        except ImportError:
            self.model = None
            states = self._quantile_states(values)

        ordered = (
            sorted(np.unique(states), key=lambda state: float(np.std(values[states == state])))
            if len(np.unique(states)) > 1
            else list(np.unique(states))
        )
        names = self._regime_names(len(ordered))
        self._state_to_label = {int(state): names[idx] for idx, state in enumerate(ordered)}
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        values = np.asarray(returns, dtype=float).reshape(-1)
        if self.model is not None:
            states = self.model.predict(values.reshape(-1, 1))
        else:
            states = self._quantile_states(values)
        return np.asarray([self._state_to_label.get(int(state), "crisis") for state in states], dtype=object)

    def client_regime_weights(self, client_regimes: dict[str, str]) -> dict[str, float]:
        if not client_regimes:
            return {}
        dominant = Counter(client_regimes.values()).most_common(1)[0][0]
        raw = {
            client: 1.5 if regime == dominant else 0.5
            for client, regime in client_regimes.items()
        }
        total = sum(raw.values())
        return {client: weight / total for client, weight in raw.items()}

    def _quantile_states(self, values: np.ndarray) -> np.ndarray:
        volatility = np.abs(values - np.mean(values))
        quantiles = np.quantile(volatility, np.linspace(0, 1, self.n_states + 1))
        return np.clip(np.digitize(volatility, quantiles[1:-1], right=True), 0, self.n_states - 1)

    def _regime_names(self, n_states: int) -> list[str]:
        if n_states == 1:
            return ["crisis"]
        if n_states == 2:
            return ["bull", "crisis"]
        if n_states == 3:
            return list(self.REGIME_NAMES)
        return ["bull", *[f"transition_{idx}" for idx in range(1, n_states - 1)], "crisis"]
