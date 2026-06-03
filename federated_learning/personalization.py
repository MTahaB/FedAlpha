from __future__ import annotations

from copy import deepcopy


class DITTO:
    """Post-FL personalization with L2 regularization toward the global model."""

    def __init__(self, lambda_reg: float = 0.1, lr: float = 1e-3, n_steps: int = 100):
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.n_steps = n_steps

    def personalize(self, global_model, x_local, y_local, device: str = "cpu"):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("Install torch to use DITTO personalization.") from exc

        local_model = deepcopy(global_model).to(device)
        anchor_model = deepcopy(global_model).to(device)
        anchor_params = {
            name: param.detach().clone()
            for name, param in anchor_model.named_parameters()
        }
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        x = torch.as_tensor(x_local, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_local, dtype=torch.float32, device=device).reshape(-1)
        loss_fn = nn.MSELoss()

        for _ in range(self.n_steps):
            optimizer.zero_grad()
            pred = local_model(x).reshape(-1)
            local_loss = loss_fn(pred, y)
            reg = sum(
                torch.norm(param - anchor_params[name]) ** 2
                for name, param in local_model.named_parameters()
            )
            (local_loss + self.lambda_reg * reg).backward()
            optimizer.step()

        return local_model
