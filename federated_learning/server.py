from __future__ import annotations


def build_strategy(strategy_name: str = "fedprox", min_clients: int = 3):
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements.txt`.") from exc

    if strategy_name.lower() == "fedavg":
        return fl.server.strategy.FedAvg(
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
        )
    if strategy_name.lower() == "fedprox":
        return fl.server.strategy.FedProx(
            proximal_mu=0.01,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
        )
    raise ValueError(f"Unknown strategy: {strategy_name}")


def main() -> None:
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements.txt`.") from exc

    strategy = build_strategy("fedprox")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
