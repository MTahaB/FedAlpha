from __future__ import annotations


class FedAlphaClientConfig:
    def __init__(self, client_id: str, target_epsilon: float = 1.0, local_epochs: int = 3):
        self.client_id = client_id
        self.target_epsilon = target_epsilon
        self.local_epochs = local_epochs


def main() -> None:
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower with `pip install -r requirements.txt`.") from exc

    raise NotImplementedError(
        "Wire a torch model and DataLoader here after Phase 1 feature matrices are saved."
    )


if __name__ == "__main__":
    main()
