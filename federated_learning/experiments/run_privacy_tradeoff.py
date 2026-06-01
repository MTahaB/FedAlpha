from __future__ import annotations

import numpy as np

from federated_learning.privacy import clip_and_add_noise


def main() -> None:
    gradient = np.array([0.1, 2.0, -3.0])
    for noise in [0.0, 0.5, 1.0, 2.0]:
        private_gradient = clip_and_add_noise(gradient, noise_multiplier=noise, seed=42)
        print({"noise_multiplier": noise, "private_gradient": private_gradient.round(4).tolist()})


if __name__ == "__main__":
    main()
