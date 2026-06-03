from quant.pipeline import train_local_model

from helpers import research_panel, write_ohlcv_csv


def test_fl_clients_train_real_model_from_partitions(tmp_path):
    panel = research_panel(tickers=("A", "B", "C", "D", "E", "F"))
    partitions = {
        "tech": ("A", "B"),
        "finance": ("C", "D"),
        "healthcare_industrials": ("E", "F"),
    }

    results = []
    for name, tickers in partitions.items():
        path = tmp_path / name / "ohlcv.csv"
        partition = panel.loc[panel.index.get_level_values("ticker").isin(tickers)]
        write_ohlcv_csv(partition, path)
        results.append(train_local_model(path))

    assert len(results) == 3
    assert all(result.n_examples > 30 for result in results)
    assert all(result.model_state["coef"] for result in results)
    assert all(result.train_loss >= 0 for result in results)
