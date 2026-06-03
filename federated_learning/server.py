from __future__ import annotations

import os

from federated_learning.aggregation.fednova import fednova


def round_config(server_round: int) -> dict[str, int]:
    return {"server_round": server_round}


def aggregate_weighted_loss(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"loss": 0.0}
    weighted_loss = sum(num_examples * float(values.get("loss", 0.0)) for num_examples, values in metrics)
    return {"loss": weighted_loss / total_examples}


def build_strategy(
    strategy_name: str = "fednova",
    min_clients: int = 3,
    robust_aggregator: str = "none",
    trim_ratio: float = 0.1,
    byzantine_clients: int = 1,
):
    try:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements.txt`.") from exc

    if strategy_name.lower() == "fedavg":
        return fl.server.strategy.FedAvg(
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            on_fit_config_fn=round_config,
            on_evaluate_config_fn=round_config,
            fit_metrics_aggregation_fn=aggregate_weighted_loss,
            evaluate_metrics_aggregation_fn=aggregate_weighted_loss,
        )
    if strategy_name.lower() == "fedprox":
        return fl.server.strategy.FedProx(
            proximal_mu=0.01,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            on_fit_config_fn=round_config,
            on_evaluate_config_fn=round_config,
            fit_metrics_aggregation_fn=aggregate_weighted_loss,
            evaluate_metrics_aggregation_fn=aggregate_weighted_loss,
        )
    if strategy_name.lower() == "fednova":
        class FedNovaStrategy(fl.server.strategy.FedAvg):
            def __init__(self):
                super().__init__(
                    min_fit_clients=min_clients,
                    min_available_clients=min_clients,
                    on_fit_config_fn=round_config,
                    on_evaluate_config_fn=round_config,
                    fit_metrics_aggregation_fn=aggregate_weighted_loss,
                    evaluate_metrics_aggregation_fn=aggregate_weighted_loss,
                )
                self.latest_global_parameters = None

            def configure_fit(self, server_round, parameters, client_manager):
                self.latest_global_parameters = parameters
                return super().configure_fit(server_round, parameters, client_manager)

            def aggregate_fit(self, server_round, results, failures):
                if not results:
                    return None, {}
                if failures and not self.accept_failures:
                    return None, {}
                if self.latest_global_parameters is None:
                    return super().aggregate_fit(server_round, results, failures)

                global_params = parameters_to_ndarrays(self.latest_global_parameters)
                client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                num_examples = [fit_res.num_examples for _, fit_res in results]
                local_steps = [
                    int(fit_res.metrics.get("local_steps", 1))
                    for _, fit_res in results
                ]
                aggregated = fednova(
                    global_params,
                    client_params,
                    num_examples,
                    local_steps,
                    robust_aggregator=robust_aggregator,
                    trim_ratio=trim_ratio,
                    byzantine_clients=byzantine_clients,
                )
                metrics = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
                    metrics = self.fit_metrics_aggregation_fn(fit_metrics)
                return ndarrays_to_parameters(aggregated), metrics

        return FedNovaStrategy()
    raise ValueError(f"Unknown strategy: {strategy_name}")


def main() -> None:
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements.txt`.") from exc

    num_rounds = int(os.getenv("FL_NUM_ROUNDS", "10"))
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "3"))
    strategy_name = os.getenv("FL_STRATEGY", "fednova")
    robust_aggregator = os.getenv("FL_ROBUST_AGGREGATOR", "none")
    trim_ratio = float(os.getenv("FL_TRIM_RATIO", "0.1"))
    byzantine_clients = int(os.getenv("FL_BYZANTINE_CLIENTS", "1"))
    strategy = build_strategy(
        strategy_name,
        min_clients=min_clients,
        robust_aggregator=robust_aggregator,
        trim_ratio=trim_ratio,
        byzantine_clients=byzantine_clients,
    )
    print(
        {
            "event": "fl_server_start",
            "strategy": strategy_name,
            "robust_aggregator": robust_aggregator,
            "num_rounds": num_rounds,
            "min_clients": min_clients,
        },
        flush=True,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
