from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:

    class TCNBlock:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install torch with `pip install -r requirements.txt`.")

    class TCNRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install torch with `pip install -r requirements.txt`.")

else:

    class TCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
            self.norm = nn.LayerNorm(out_channels)
            self.activation = nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            if self.padding:
                out = out[:, :, : -self.padding]
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
            return self.activation(out)

    class TCNRegressor(nn.Module):
        def __init__(
            self,
            input_channels: int,
            channels: tuple[int, ...] = (32, 32, 64),
            kernel_size: int = 3,
        ):
            super().__init__()
            blocks = []
            prev = input_channels
            for i, out_channels in enumerate(channels):
                blocks.append(TCNBlock(prev, out_channels, kernel_size, dilation=2**i))
                prev = out_channels
            self.network = nn.Sequential(*blocks)
            self.head = nn.Linear(prev, 1)

        def forward(self, x):
            encoded = self.network(x)
            last_step = encoded[:, :, -1]
            return self.head(last_step).squeeze(-1)
