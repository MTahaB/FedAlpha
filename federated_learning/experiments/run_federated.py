from __future__ import annotations

import numpy as np

from federated_learning.aggregation import fedavg, trimmed_mean_aggregate


def main() -> None:
    updates = [
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([0.9, 2.1]),
        np.array([1.0, 2.0]),
        np.array([10.0, -10.0]),
    ]
    print({"fedavg": fedavg(updates)[0].tolist(), "trimmed": trimmed_mean_aggregate(updates, 0.2).tolist()})


if __name__ == "__main__":
    main()
