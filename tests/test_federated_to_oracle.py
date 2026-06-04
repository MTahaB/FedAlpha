from federated_learning.experiments.run_federated import run_federated_pipeline
from federated_learning.experiments.run_byzantine_comparison import run_byzantine_comparison
from federated_learning.experiments.run_comparison_table import run_comparison_table

from helpers import research_panel, write_ohlcv_csv


def test_federated_pipeline_sends_global_model_to_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDALPHA_REPORT_DIR", str(tmp_path / "reports"))
    panel = research_panel(tickers=("A", "B", "C", "D", "E", "F"))
    evaluation_path = tmp_path / "raw" / "ohlcv.csv"
    write_ohlcv_csv(panel, evaluation_path)

    partition_paths = []
    for idx, tickers in enumerate([("A", "B"), ("C", "D"), ("E", "F")], start=1):
        path = tmp_path / f"client_{idx}" / "ohlcv.csv"
        partition = panel.loc[panel.index.get_level_values("ticker").isin(tickers)]
        write_ohlcv_csv(partition, path)
        partition_paths.append(path)

    result = run_federated_pipeline(
        partition_paths,
        evaluation_path,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        round_id=4,
    )

    assert result["model_state"]["model_type"] == "federated_ridge_fedavg"
    assert len(result["clients"]) == 3
    assert result["oracle"]["model_hash"].startswith("0x")
    assert result["oracle"]["blockchain_anchor"]["round_id"] == 4
    assert (tmp_path / "reports" / "federated_returns.csv").exists()
    assert (tmp_path / "reports" / "oracle_response.json").exists()


def test_byzantine_comparison_writes_robustness_report(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDALPHA_REPORT_DIR", str(tmp_path / "reports"))
    panel = research_panel(tickers=("A", "B", "C", "D", "E", "F"))
    evaluation_path = tmp_path / "raw" / "ohlcv.csv"
    write_ohlcv_csv(panel, evaluation_path)

    partition_paths = []
    for idx, tickers in enumerate([("A", "B"), ("C", "D"), ("E", "F")], start=1):
        path = tmp_path / f"client_{idx}" / "ohlcv.csv"
        partition = panel.loc[panel.index.get_level_values("ticker").isin(tickers)]
        write_ohlcv_csv(partition, path)
        partition_paths.append(path)

    comparison = run_byzantine_comparison(
        partition_paths=partition_paths,
        evaluation_data_path=evaluation_path,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )

    assert set(comparison["robust_aggregator"]) == {"median", "fedavg"}
    assert (tmp_path / "reports" / "byzantine_comparison.csv").exists()


def test_comparison_table_materializes_required_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDALPHA_REPORT_DIR", str(tmp_path / "reports"))
    panel = research_panel(tickers=("A", "B", "C", "D", "E", "F"))
    evaluation_path = tmp_path / "raw" / "ohlcv.csv"
    write_ohlcv_csv(panel, evaluation_path)

    partition_paths = []
    for idx, tickers in enumerate([("A", "B"), ("C", "D"), ("E", "F")], start=1):
        path = tmp_path / f"client_{idx}" / "ohlcv.csv"
        partition = panel.loc[panel.index.get_level_values("ticker").isin(tickers)]
        write_ohlcv_csv(partition, path)
        partition_paths.append(path)

    table = run_comparison_table(
        partition_paths=partition_paths,
        evaluation_data_path=evaluation_path,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )

    assert table["method"].tolist() == [
        "Equal Weight",
        "Centralized Ridge",
        "Local Ridge",
        "FedAvg",
        "Robust Aggregation (median)",
    ]
    assert table["embargo_days"].eq(5).all()
    assert table["regime_fit"].eq("train_only").all()
    assert (tmp_path / "reports" / "comparison_table.csv").exists()
