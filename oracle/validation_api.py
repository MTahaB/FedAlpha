from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from oracle.backtest_runner import validate_return_series
from oracle.model_registry import compute_model_hash


class ValidationRequest(BaseModel):
    model_state: dict
    returns: list[float]
    benchmark_returns: list[float] | None = None
    min_sharpe: float = 0.75
    max_drawdown: float = -0.20


app = FastAPI(title="FedAlpha Oracle", version="2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate")
def validate(request: ValidationRequest) -> dict:
    returns = pd.Series(request.returns, dtype=float)
    benchmark = pd.Series(request.benchmark_returns, dtype=float) if request.benchmark_returns else None
    result = validate_return_series(returns, benchmark, request.min_sharpe, request.max_drawdown)
    result["model_hash"] = compute_model_hash(request.model_state)
    result["validation_timestamp"] = datetime.now(timezone.utc).isoformat()
    return result
