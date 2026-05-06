from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name_norm = str(name).strip().lower()
    if name_norm == "relu":
        return nn.ReLU()
    if name_norm == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [int(input_dim), *[int(x) for x in hidden_dims], int(output_dim)]
        layers: list[nn.Module] = []
        act_name = str(activation)
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = i == len(dims) - 2
            if not is_last:
                if bool(use_layernorm):
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(_activation(act_name))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class USAD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: tuple[int, int] = (512, 256),
        latent_dim: int = 96,
        decoder_hidden_dims: tuple[int, int] = (256, 512),
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = _MLP(
            input_dim=self.input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=self.latent_dim,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.decoder1 = _MLP(
            input_dim=self.latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.input_dim,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.decoder2 = _MLP(
            input_dim=self.latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.input_dim,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encode(w1))
        return w1, w2, w3

    @staticmethod
    def _per_sample_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((x - y) ** 2, dim=1)

    def compute_losses(self, x: torch.Tensor, epoch_index: int) -> dict[str, torch.Tensor]:
        n = float(max(int(epoch_index), 1))
        w1, w2, w3 = self.reconstruct(x)
        err1 = self._per_sample_mse(x, w1)
        err2 = self._per_sample_mse(x, w2)
        err3 = self._per_sample_mse(x, w3)
        loss1 = (1.0 / n) * err1.mean() + (1.0 - 1.0 / n) * err3.mean()
        loss2 = (1.0 / n) * err2.mean() - (1.0 - 1.0 / n) * err3.mean()
        return {
            "loss1": loss1,
            "loss2": loss2,
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "err1": err1,
            "err2": err2,
            "err3": err3,
        }

    def anomaly_score(self, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        alpha = float(alpha)
        w1, _, w3 = self.reconstruct(x)
        err1 = self._per_sample_mse(x, w1)
        err3 = self._per_sample_mse(x, w3)
        return alpha * err1 + (1.0 - alpha) * err3
