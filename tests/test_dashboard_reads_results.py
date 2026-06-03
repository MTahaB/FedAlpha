import json

import pandas as pd

from dashboard.results import load_dashboard_results, participant_rows


def test_dashboard_reads_results_from_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "federated_metrics.json").write_text(
        json.dumps({"sharpe_ratio": 1.23, "max_drawdown": -0.04}),
        encoding="utf-8",
    )
    (reports / "oracle_response.json").write_text(
        json.dumps({"validated": True, "model_hash": "0xabc"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"round": 1, "client": "client_1", "examples": 100, "loss": 0.01},
            {"round": 1, "client": "client_2", "examples": 90, "loss": 0.02},
        ]
    ).to_csv(reports / "federated_training_history.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01"], "portfolio_return": [0.01]}).to_csv(
        reports / "federated_returns.csv",
        index=False,
    )
    (reports / "blockchain_events.json").write_text(
        json.dumps([{"round_id": 1, "model_hash": "0xabc"}]),
        encoding="utf-8",
    )

    results = load_dashboard_results(reports)
    participants = participant_rows(results)

    assert results["round"] == 1
    assert results["metrics"]["sharpe_ratio"] == 1.23
    assert results["oracle"]["validated"] is True
    assert len(results["blockchain_events"]) == 1
    assert list(participants["status"]) == ["trained", "trained"]
