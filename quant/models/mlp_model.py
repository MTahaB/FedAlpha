from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install torch with `pip install -r requirements.txt`.") from exc
    return torch, nn


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.1):
    _, nn = require_torch()
    layers = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)
