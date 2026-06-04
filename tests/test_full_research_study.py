import json

from federated_learning.experiments.run_full_research_study import run_full_research_study
from quant.backtest import WalkForwardWindow

from helpers import research_panel, write_ohlcv_csv


def test_full_research_study_writes_core_artifacts(tmp_path):
    panel = research_panel(n_days=520, tickers=("A", "B", "C", "D", "E", "F"))
    evaluation_path = tmp_path / "raw" / "ohlcv.csv"
    write_ohlcv_csv(panel, evaluation_path)

    partition_paths = []
    for idx, tickers in enumerate([("A", "B"), ("C", "D"), ("E", "F")], start=1):
        path = tmp_path / f"client_{idx}" / "ohlcv.csv"
        write_ohlcv_csv(panel.loc[panel.index.get_level_values("ticker").isin(tickers)], path)
        partition_paths.append(path)

    window = WalkForwardWindow(
        "wf_smoke",
        "2023-01-02",
        "2024-04-01",
        "2024-04-01",
        "2024-07-01",
        "2024-07-01",
        "2025-01-01",
    )
    result = run_full_research_study(
        evaluation_data_path=evaluation_path,
        partition_paths=partition_paths,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        alpha_grid=[0.1],
        top_k_grid=[2],
        bottom_k_grid=[2],
        windows=[window],
    )

    reports = tmp_path / "reports"
    expected = [
        "protocol_summary.md",
        "comparison_summary.csv",
        "comparison_summary.json",
        "comparison_table.csv",
        "comparison_table.json",
        "equity_curves.csv",
        "results_by_window.csv",
        "results_by_regime.csv",
        "statistical_tests.json",
        "ci_status.json",
    ]
    for name in expected:
        assert (reports / name).exists()

    assert result["summary"]["method"].tolist() == [
        "Equal Weight",
        "Momentum 20D",
        "Centralized Ridge",
        "Local Ridge",
        "FedAvg",
        "Robust Aggregation",
        "FedAlpha Regime-Aware",
        "FedAlpha Personalized",
    ]
    ci_status = json.loads((reports / "ci_status.json").read_text(encoding="utf-8"))
    assert ci_status["visible_result"] == "PASS for commit 9fd5f1d"
    assert "survivorship bias" in (reports / "protocol_summary.md").read_text(encoding="utf-8").lower()
