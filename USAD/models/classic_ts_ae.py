from __future__ import annotations

import torch
from torch import nn

from .recurrent_ae import RecurrentAutoencoder


class ConvAutoencoder1D(nn.Module):
    """
    Simple Conv1D autoencoder over [B, T, M] windows.
    Convolution runs along the temporal axis after transposing to [B, M, T].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        latent_dim: int = 48,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        padding = max(int(kernel_size) // 2, 0)
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, int(hidden_dim), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), int(latent_dim), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(int(latent_dim), int(hidden_dim), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), self.input_dim, kernel_size=int(kernel_size), padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        x = window.transpose(1, 2)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.transpose(1, 2).contiguous()

    def anomaly_components(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        recon = self.forward(window)
        err = (recon - window).pow(2)
        step_score = err.mean(dim=-1)
        final_score = step_score[:, -1]
        return {
            "reconstruction": recon,
            "error": err,
            "step_score": step_score,
            "final_score": final_score,
        }

    def loss(self, window: torch.Tensor) -> torch.Tensor:
        return (self.forward(window) - window).pow(2).mean()


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self.left_padding = int((kernel_size - 1) * dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.left_padding > 0:
            out = out[..., :-self.left_padding]
        return out


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.residual = nn.Identity() if int(in_channels) == int(out_channels) else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TCNAutoencoder(nn.Module):
    """
    Causal TCN autoencoder with dilated residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        latent_dim: int = 48,
        kernel_size: int = 3,
        num_levels: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        levels = max(int(num_levels), 1)
        enc_blocks: list[nn.Module] = []
        in_channels = self.input_dim
        for level in range(levels):
            out_channels = int(hidden_dim) if level < levels - 1 else int(latent_dim)
            enc_blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int(kernel_size),
                    dilation=2 ** level,
                    dropout=float(dropout),
                )
            )
            in_channels = out_channels
        dec_blocks: list[nn.Module] = []
        in_channels = int(latent_dim)
        for level in reversed(range(levels)):
            out_channels = int(hidden_dim) if level > 0 else self.input_dim
            dec_blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int(kernel_size),
                    dilation=2 ** level,
                    dropout=float(dropout),
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*enc_blocks)
        self.decoder = nn.Sequential(*dec_blocks)
        self.output_activation = nn.Sigmoid()

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        x = window.transpose(1, 2)
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = self.output_activation(recon)
        return recon.transpose(1, 2).contiguous()

    def anomaly_components(self, window: torch.Tensor) -> dict[str, torch.Tensor]:
        recon = self.forward(window)
        err = (recon - window).pow(2)
        step_score = err.mean(dim=-1)
        final_score = step_score[:, -1]
        return {
            "reconstruction": recon,
            "error": err,
            "step_score": step_score,
            "final_score": final_score,
        }

    def loss(self, window: torch.Tensor) -> torch.Tensor:
        return (self.forward(window) - window).pow(2).mean()


def build_classic_ts_autoencoder(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_layers: int,
    dropout: float,
    conv_kernel_size: int,
    tcn_kernel_size: int,
    tcn_num_levels: int,
) -> nn.Module:
    model_name = str(model_type).strip().lower()
    if model_name in {"lstm", "gru"}:
        return RecurrentAutoencoder(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            cell_type=model_name,
        )
    if model_name == "conv":
        return ConvAutoencoder1D(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            kernel_size=int(conv_kernel_size),
            dropout=float(dropout),
        )
    if model_name == "tcn":
        return TCNAutoencoder(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            kernel_size=int(tcn_kernel_size),
            num_levels=int(tcn_num_levels),
            dropout=float(dropout),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
