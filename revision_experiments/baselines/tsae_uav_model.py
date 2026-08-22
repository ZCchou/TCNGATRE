from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception2D(nn.Module):
    """Four-branch Inception block described in the TSAE-UAV paper."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        if channels % 4:
            raise ValueError("TSAE-UAV embedding channels must be divisible by four")
        branch = channels // 4
        self.branch_1 = nn.Conv2d(channels, branch, kernel_size=1)
        self.branch_3 = nn.Sequential(
            nn.Conv2d(channels, branch, kernel_size=1),
            nn.Conv2d(branch, branch, kernel_size=3, padding=1),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv2d(channels, branch, kernel_size=1),
            nn.Conv2d(branch, branch, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, branch, kernel_size=1),
        )
        self.dropout = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (
            self.branch_1(x),
            self.branch_3(x),
            self.branch_5(x),
            self.branch_pool(x),
        )
        return self.dropout(torch.cat(branches, dim=1))


class TimeSenseBlock(nn.Module):
    """FFT period selection, temporal 2-D variation modeling and residual fusion."""

    def __init__(self, channels: int, top_k: int = 3, dropout: float = 0.0):
        super().__init__()
        self.top_k = int(top_k)
        self.inception_1 = Inception2D(channels, dropout=dropout)
        self.activation = nn.GELU()
        self.inception_2 = Inception2D(channels, dropout=dropout)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def dominant_periods(x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch,time,channels], received {tuple(x.shape)}")
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=2)
        if spectrum.shape[1] <= 1:
            raise ValueError("TSAE-UAV requires a window with at least two frequency bins")
        spectrum = spectrum.clone()
        spectrum[:, 0] = -torch.inf
        count = min(int(top_k), spectrum.shape[1] - 1)
        amplitudes, frequencies = torch.topk(spectrum, k=count, dim=1)
        periods = torch.ceil(
            torch.as_tensor(x.shape[1], device=x.device, dtype=x.dtype)
            / frequencies.to(dtype=x.dtype)
        ).to(dtype=torch.long)
        return periods.clamp_min(1), amplitudes

    def _model_period(self, x: torch.Tensor, period: int) -> torch.Tensor:
        batch, length, channels = x.shape
        padded_length = int(math.ceil(length / int(period)) * int(period))
        if padded_length > length:
            x = F.pad(x, (0, 0, 0, padded_length - length))
        cycles = padded_length // int(period)
        image = x.reshape(batch, cycles, int(period), channels).permute(0, 3, 1, 2)
        encoded = self.inception_2(self.activation(self.inception_1(image)))
        restored = encoded.permute(0, 2, 3, 1).reshape(batch, padded_length, channels)
        return restored[:, :length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        periods, amplitudes = self.dominant_periods(x, self.top_k)
        weights = torch.softmax(amplitudes, dim=1)
        aggregate = torch.zeros_like(x)
        for rank in range(periods.shape[1]):
            rank_output = torch.zeros_like(x)
            for period_tensor in torch.unique(periods[:, rank], sorted=True):
                period = int(period_tensor.item())
                selected = periods[:, rank] == period_tensor
                rank_output[selected] = self._model_period(x[selected], period)
            aggregate = aggregate + weights[:, rank].view(-1, 1, 1) * rank_output
        return x + self.dropout(aggregate)


class TSAEUAV(nn.Module):
    """Paper-based, protocol-compatible TSAE-UAV reconstruction network."""

    def __init__(
        self,
        channels: int,
        d_model: int = 64,
        top_k: int = 3,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.embedding = nn.Linear(self.channels, int(d_model))
        self.blocks = nn.ModuleList(
            TimeSenseBlock(int(d_model), top_k=int(top_k), dropout=float(dropout))
            for _ in range(int(layers))
        )
        self.recovery = nn.Linear(int(d_model), self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"Expected [batch,time,{self.channels}], received {tuple(x.shape)}"
            )
        hidden = self.embedding(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.recovery(hidden)
