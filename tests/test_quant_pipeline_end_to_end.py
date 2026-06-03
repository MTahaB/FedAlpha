from quant.pipeline import run_centralized_pipeline

from helpers import research_panel, write_ohlcv_csv


def test_quant_pipeline_end_to_end_writes_model_reports_and_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDALPHA_REPORT_DIR", str(tmp_path / "reports"))
    data_path = tmp_path / "data" / "ohlcv.csv"
    write_ohlcv_csv(research_panel(), data_path)

    result = run_centralized_pipeline(
        data_path,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        horizon=5,
    )

    assert result["model_state"]["model_type"] == "ridge"
    assert "sharpe_ratio" in result["metrics"]
    assert result["oracle"]["model_hash"].startswith("0x")
    assert (tmp_path / "reports" / "centralized_predictions.csv").exists()
    assert (tmp_path / "reports" / "centralized_returns.csv").exists()
    assert (tmp_path / "reports" / "centralized_metrics.json").exists()
    assert (tmp_path / "models" / "centralized_model.pkl").exists()
