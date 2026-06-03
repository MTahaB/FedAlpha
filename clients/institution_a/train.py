from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from federated_learning.privacy import add_local_dp_noise, estimate_local_dp_epsilon


DEFAULT_FEATURE_DIM = 8


def _client_seed(client_id: str) -> int:
    return 10_000 + sum((idx + 1) * ord(char) for idx, char in enumerate(client_id))


def synthetic_dataset(client_id: str, feature_dim: int = DEFAULT_FEATURE_DIM) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_client_seed(client_id))
    n_rows = 256
    x = rng.normal(size=(n_rows, feature_dim))
    base_weights = np.linspace(0.08, -0.04, feature_dim)
    client_shift = (ord(client_id[:1] or "A") - ord("A")) * 0.01
    y = x @ (base_weights + client_shift) + rng.normal(scale=0.03, size=n_rows)
    return x.astype(float), y.astype(float)


def load_real_dataset(data_dir: Path, feature_dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = data_dir / "ohlcv.csv"
    if not path.exists():
        return None

    try:
        from quant.data_loader import load_ohlcv_csv
        from quant.pipeline import prepare_supervised_dataset

        dataset = prepare_supervised_dataset(load_ohlcv_csv(path), horizon=5)
        if len(dataset.features) < 30:
            return None

        x = dataset.features.iloc[:, :feature_dim].copy()
        if x.shape[1] < feature_dim:
            for idx in range(x.shape[1], feature_dim):
                x[f"pad_{idx}"] = 0.0
        return x.to_numpy(dtype=float), dataset.labels.to_numpy(dtype=float)
    except Exception as exc:
        print({"event": "client_real_data_fallback", "reason": str(exc)}, flush=True)
        return None


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


class FedAlphaLinearClient:
    def __init__(
        self,
        client_id: str,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.05,
        local_epochs: int = 2,
        local_dp_noise_multiplier: float = 0.0,
        local_dp_max_norm: float = 1.0,
        malicious_attack: str = "none",
        malicious_scale: float = 10.0,
    ):
        try:
            import flwr as fl
        except ImportError as exc:
            raise RuntimeError("Install Flower with `pip install -r requirements-client.txt`.") from exc

        class _Client(fl.client.NumPyClient):
            def __init__(inner_self):
                inner_self.weights = np.zeros(x.shape[1], dtype=float)
                inner_self.bias = np.array([0.0], dtype=float)

            def get_parameters(inner_self, config):
                return [inner_self.weights.copy(), inner_self.bias.copy()]

            def set_parameters(inner_self, parameters):
                if parameters:
                    inner_self.weights = np.asarray(parameters[0], dtype=float).copy()
                    inner_self.bias = np.asarray(parameters[1], dtype=float).copy()

            def fit(inner_self, parameters, config):
                inner_self.set_parameters(parameters)
                global_weights = inner_self.weights.copy()
                global_bias = inner_self.bias.copy()
                for _ in range(local_epochs):
                    preds = x @ inner_self.weights + inner_self.bias[0]
                    err = preds - y
                    inner_self.weights -= learning_rate * ((x.T @ err) / len(x))
                    inner_self.bias[0] -= learning_rate * float(err.mean())

                update_layers = [inner_self.weights - global_weights, inner_self.bias - global_bias]
                server_round = int(config.get("server_round") or 0)
                if local_dp_noise_multiplier > 0:
                    update_layers = add_local_dp_noise(
                        update_layers,
                        max_norm=local_dp_max_norm,
                        noise_multiplier=local_dp_noise_multiplier,
                        seed=_client_seed(client_id) + server_round,
                    )

                if malicious_attack != "none":
                    if malicious_attack == "sign_flip":
                        update_layers = [-malicious_scale * layer for layer in update_layers]
                    elif malicious_attack == "constant":
                        update_layers = [
                            np.full_like(layer, fill_value=malicious_scale, dtype=float)
                            for layer in update_layers
                        ]
                    elif malicious_attack == "gaussian":
                        rng = np.random.default_rng(_client_seed(client_id) + 1000 + server_round)
                        update_layers = [
                            rng.normal(0.0, malicious_scale, size=layer.shape)
                            for layer in update_layers
                        ]
                    else:
                        raise ValueError(f"Unknown malicious attack: {malicious_attack}")

                inner_self.weights = global_weights + update_layers[0]
                inner_self.bias = global_bias + update_layers[1]

                loss = inner_self._loss()
                local_dp_epsilon = estimate_local_dp_epsilon(
                    local_dp_noise_multiplier,
                    local_epochs,
                )
                epsilon_metric = local_dp_epsilon if np.isfinite(local_dp_epsilon) else -1.0
                print(
                    {
                        "event": "client_fit",
                        "client_id": client_id,
                        "server_round": server_round,
                        "loss": loss,
                        "examples": len(x),
                        "local_steps": local_epochs,
                        "local_dp_noise": local_dp_noise_multiplier,
                        "local_dp_epsilon": "inf" if not np.isfinite(local_dp_epsilon) else local_dp_epsilon,
                        "malicious_attack": malicious_attack,
                    },
                    flush=True,
                )
                return (
                    inner_self.get_parameters(config),
                    len(x),
                    {
                        "loss": loss,
                        "client_id": client_id,
                        "local_steps": local_epochs,
                        "local_dp_epsilon": epsilon_metric,
                        "malicious_attack": malicious_attack,
                    },
                )

            def evaluate(inner_self, parameters, config):
                inner_self.set_parameters(parameters)
                loss = inner_self._loss()
                print(
                    {
                        "event": "client_evaluate",
                        "client_id": client_id,
                        "server_round": config.get("server_round"),
                        "loss": loss,
                        "examples": len(x),
                    },
                    flush=True,
                )
                return loss, len(x), {"loss": loss, "client_id": client_id}

            def _loss(inner_self):
                err = (x @ inner_self.weights + inner_self.bias[0]) - y
                return float(np.mean(err**2))

        self.numpy_client = _Client()


def build_client_from_env() -> tuple[FedAlphaLinearClient, str]:
    client_id = os.getenv("CLIENT_ID", "A")
    data_dir = Path(os.getenv("CLIENT_DATA_DIR", "/data"))
    feature_dim = int(os.getenv("MODEL_DIM", str(DEFAULT_FEATURE_DIM)))
    learning_rate = float(os.getenv("LOCAL_LR", "0.05"))
    local_epochs = int(os.getenv("LOCAL_EPOCHS", "2"))
    local_dp_noise_multiplier = float(os.getenv("LOCAL_DP_NOISE_MULTIPLIER", "0.0"))
    local_dp_max_norm = float(os.getenv("LOCAL_DP_MAX_NORM", "1.0"))
    malicious_attack = os.getenv("MALICIOUS_ATTACK", "none").lower()
    malicious_scale = float(os.getenv("MALICIOUS_SCALE", "10.0"))

    dataset = load_real_dataset(data_dir, feature_dim)
    data_source = "real_ohlcv" if dataset is not None else "synthetic_fallback"
    if dataset is None:
        dataset = synthetic_dataset(client_id, feature_dim)
    x, y = dataset
    x = standardize(x)
    return (
        FedAlphaLinearClient(
            client_id,
            x,
            y,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            local_dp_noise_multiplier=local_dp_noise_multiplier,
            local_dp_max_norm=local_dp_max_norm,
            malicious_attack=malicious_attack,
            malicious_scale=malicious_scale,
        ),
        data_source,
    )


def main() -> None:
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements-client.txt`.") from exc

    client_id = os.getenv("CLIENT_ID", "A")
    server_host = os.getenv("SERVER_HOST", "fl_server")
    server_port = int(os.getenv("SERVER_PORT", "8080"))
    max_retries = int(os.getenv("CLIENT_CONNECT_RETRIES", "30"))
    retry_seconds = float(os.getenv("CLIENT_CONNECT_SLEEP", "2"))
    client, data_source = build_client_from_env()

    print(
        {
            "event": "client_start",
            "client_id": client_id,
            "server": f"{server_host}:{server_port}",
            "data_source": data_source,
        },
        flush=True,
    )
    for attempt in range(1, max_retries + 1):
        try:
            fl.client.start_numpy_client(
                server_address=f"{server_host}:{server_port}",
                client=client.numpy_client,
            )
            print({"event": "client_done", "client_id": client_id}, flush=True)
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            print(
                {
                    "event": "client_connect_retry",
                    "client_id": client_id,
                    "attempt": attempt,
                    "reason": str(exc),
                },
                flush=True,
            )
            time.sleep(retry_seconds)


if __name__ == "__main__":
    main()
