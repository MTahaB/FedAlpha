from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    renamed = {
        column: "date"
        for column in frame.columns
        if str(column).lower().startswith("unnamed")
    }
    return frame.rename(columns=renamed)


def load_dashboard_results(report_dir: str | Path = "reports") -> dict:
    root = Path(report_dir)
    metrics = _read_json(root / "federated_metrics.json") or _read_json(root / "centralized_metrics.json")
    oracle = _read_json(root / "oracle_response.json")
    training = _read_csv(root / "federated_training_history.csv")
    returns = _read_csv(root / "federated_returns.csv")
    privacy = _read_csv(root / "privacy_tradeoff.csv")
    blockchain_events = _read_json(root / "blockchain_events.json") if (root / "blockchain_events.json").exists() else []

    latest_round = 0
    if not training.empty and "round" in training:
        latest_round = int(training["round"].max())

    return {
        "round": latest_round,
        "metrics": metrics,
        "oracle": oracle,
        "training": training,
        "returns": returns,
        "privacy": privacy,
        "blockchain_events": blockchain_events,
    }


def participant_rows(results: dict) -> pd.DataFrame:
    training = results.get("training", pd.DataFrame())
    if training.empty:
        return pd.DataFrame(
            [
                {"institution": "A", "examples": 0, "loss": None, "status": "pending"},
                {"institution": "B", "examples": 0, "loss": None, "status": "pending"},
                {"institution": "C", "examples": 0, "loss": None, "status": "pending"},
            ]
        )
    rows = []
    for idx, row in training.iterrows():
        rows.append(
            {
                "institution": chr(ord("A") + idx),
                "examples": int(row.get("examples", 0)),
                "loss": float(row.get("loss", 0.0)),
                "status": "trained",
            }
        )
    return pd.DataFrame(rows)
