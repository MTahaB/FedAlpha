import numpy as np
import pytest

from quant.regimes import MarketRegimeDetector


def test_quantile_regime_thresholds_are_fitted_once_and_reused():
    train_returns = np.r_[
        np.repeat(0.001, 40),
        np.repeat(-0.002, 40),
        np.linspace(-0.03, 0.03, 40),
    ]
    test_returns = np.array([0.20, -0.25, 0.15])

    detector = MarketRegimeDetector(use_hmm=False).fit(train_returns)
    thresholds = detector.quantile_thresholds_.copy()
    regimes = detector.predict(test_returns)

    assert detector.quantile_thresholds_ == pytest.approx(thresholds)
    assert set(regimes).issubset({"bull", "bear", "crisis"})


def test_quantile_regime_predict_requires_fit():
    detector = MarketRegimeDetector(use_hmm=False)

    with pytest.raises(RuntimeError):
        detector.predict(np.array([0.01, -0.02]))
