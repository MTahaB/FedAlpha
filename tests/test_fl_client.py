import numpy as np
import pandas as pd

from clients.institution_a.train import FedAlphaLinearClient, load_real_dataset, synthetic_dataset


def test_fedalpha_linear_client_fit_updates_parameters():
    x, y = synthetic_dataset("A", feature_dim=4)
    client = FedAlphaLinearClient("A", x, y, learning_rate=0.05, local_epochs=2).numpy_client
    initial = client.get_parameters({})

    fitted, examples, metrics = client.fit(initial, {"server_round": 1})

    assert examples == len(x)
    assert metrics["client_id"] == "A"
    assert metrics["loss"] >= 0
    assert metrics["local_steps"] == 2
    assert not np.allclose(initial[0], fitted[0])


def test_fedalpha_linear_client_evaluate_returns_loss():
    x, y = synthetic_dataset("B", feature_dim=4)
    client = FedAlphaLinearClient("B", x, y, learning_rate=0.05, local_epochs=1).numpy_client

    loss, examples, metrics = client.evaluate(client.get_parameters({}), {"server_round": 0})

    assert examples == len(x)
    assert metrics["client_id"] == "B"
    assert loss >= 0


def test_load_real_dataset_keeps_rows_when_long_window_feature_is_all_nan(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=250)
    rows = []
    for ticker, offset in [("AAPL", 0.0), ("MSFT", 5.0)]:
        close = 100.0 + offset + np.linspace(0, 10, len(dates))
        for date, price in zip(dates, close):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "open": price - 0.25,
                    "high": price + 0.5,
                    "low": price - 0.75,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "ohlcv.csv", index=False)

    dataset = load_real_dataset(tmp_path, feature_dim=8)

    assert dataset is not None
    x, y = dataset
    assert x.shape[0] > 30
    assert x.shape[1] == 8
    assert len(y) == x.shape[0]


def test_local_dp_client_adds_noise_and_reports_epsilon():
    x, y = synthetic_dataset("A", feature_dim=4)
    client = FedAlphaLinearClient(
        "A",
        x,
        y,
        learning_rate=0.05,
        local_epochs=2,
        local_dp_noise_multiplier=0.5,
        local_dp_max_norm=1.0,
    ).numpy_client

    _, _, metrics = client.fit(client.get_parameters({}), {"server_round": 1})

    assert metrics["local_dp_epsilon"] > 0
    assert metrics["malicious_attack"] == "none"


def test_malicious_client_can_flip_update_for_simulation():
    x, y = synthetic_dataset("A", feature_dim=4)
    honest = FedAlphaLinearClient("A", x, y, learning_rate=0.05, local_epochs=1).numpy_client
    malicious = FedAlphaLinearClient(
        "A",
        x,
        y,
        learning_rate=0.05,
        local_epochs=1,
        malicious_attack="sign_flip",
        malicious_scale=5.0,
    ).numpy_client

    honest_params, _, _ = honest.fit(honest.get_parameters({}), {"server_round": 1})
    malicious_params, _, metrics = malicious.fit(malicious.get_parameters({}), {"server_round": 1})

    assert metrics["malicious_attack"] == "sign_flip"
    assert np.linalg.norm(malicious_params[0]) > np.linalg.norm(honest_params[0])
