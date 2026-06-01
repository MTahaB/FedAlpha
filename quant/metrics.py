from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _as_series(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def compute_annual_return(returns: pd.Series | np.ndarray | list[float]) -> float:
    r = _as_series(returns)
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def compute_annual_volatility(returns: pd.Series | np.ndarray | list[float]) -> float:
    r = _as_series(returns)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def compute_sharpe(returns: pd.Series | np.ndarray | list[float], risk_free: float = 0.0) -> float:
    r = _as_series(returns) - risk_free / TRADING_DAYS
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def compute_sortino(returns: pd.Series | np.ndarray | list[float], risk_free: float = 0.0) -> float:
    r = _as_series(returns) - risk_free / TRADING_DAYS
    downside = r[r < 0].std(ddof=1)
    if len(r) < 2 or downside == 0 or np.isnan(downside):
        return float("nan")
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS))


def compute_max_drawdown(returns: pd.Series | np.ndarray | list[float]) -> float:
    r = _as_series(returns)
    if r.empty:
        return float("nan")
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def compute_calmar(returns: pd.Series | np.ndarray | list[float]) -> float:
    annual_return = compute_annual_return(returns)
    max_dd = abs(compute_max_drawdown(returns))
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(annual_return / max_dd)


def compute_information_ratio(
    returns: pd.Series | np.ndarray | list[float],
    benchmark_returns: pd.Series | np.ndarray | list[float],
) -> float:
    active = _as_series(returns).reset_index(drop=True) - _as_series(benchmark_returns).reset_index(drop=True)
    if len(active) < 2 or active.std(ddof=1) == 0:
        return float("nan")
    return float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_significance_test(
    returns: pd.Series | np.ndarray | list[float],
    risk_free: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    r = _as_series(returns) - risk_free / TRADING_DAYS
    if len(r) < 3:
        raise ValueError("At least 3 returns are required.")

    sharpe = compute_sharpe(r)
    autocorr = r.autocorr(lag=1)
    if np.isnan(autocorr):
        autocorr = 0.0

    var_sharpe = max((1 / len(r)) * (1 + 0.5 * sharpe**2 - autocorr * sharpe**2), 1e-12)
    se_sharpe = np.sqrt(var_sharpe) * np.sqrt(TRADING_DAYS)
    t_stat = sharpe / se_sharpe
    norm = NormalDist()
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
    z = norm.inv_cdf(1 - (1 - confidence) / 2)

    return {
        "sharpe": round(float(sharpe), 3),
        "ci_lower": round(float(sharpe - z * se_sharpe), 3),
        "ci_upper": round(float(sharpe + z * se_sharpe), 3),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < (1 - confidence)),
    }


def block_bootstrap_sample(values: pd.Series, block_size: int, rng: np.random.Generator) -> pd.Series:
    values = _as_series(values).reset_index(drop=True)
    n = len(values)
    if n == 0:
        return values
    block_size = max(1, min(block_size, n))
    starts = rng.integers(0, n - block_size + 1, size=int(np.ceil(n / block_size)))
    chunks = [values.iloc[start : start + block_size] for start in starts]
    return pd.concat(chunks, ignore_index=True).iloc[:n]


def bootstrap_metrics(
    returns: pd.Series | np.ndarray | list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    r = _as_series(returns)
    rng = np.random.default_rng(seed)
    block_size = max(1, int(np.sqrt(len(r))))
    samples = {"sharpe": [], "max_dd": [], "annual_return": []}

    for _ in range(n_bootstrap):
        sample = block_bootstrap_sample(r, block_size, rng)
        samples["sharpe"].append(compute_sharpe(sample))
        samples["max_dd"].append(compute_max_drawdown(sample))
        samples["annual_return"].append(compute_annual_return(sample))

    alpha = (1 - confidence) / 2
    result: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        result[key] = {
            "mean": float(np.nanmean(arr)),
            "ci_lower": float(np.nanpercentile(arr, alpha * 100)),
            "ci_upper": float(np.nanpercentile(arr, (1 - alpha) * 100)),
        }
    return result


def summarize_performance(
    returns: pd.Series | np.ndarray | list[float],
    benchmark_returns: pd.Series | np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    summary = {
        "annual_return": compute_annual_return(returns),
        "annual_volatility": compute_annual_volatility(returns),
        "sharpe_ratio": compute_sharpe(returns),
        "sortino_ratio": compute_sortino(returns),
        "max_drawdown": compute_max_drawdown(returns),
        "calmar_ratio": compute_calmar(returns),
    }
    if benchmark_returns is not None:
        summary["information_ratio"] = compute_information_ratio(returns, benchmark_returns)
    return summary
