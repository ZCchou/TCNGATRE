from __future__ import annotations

import torch
from torch import nn


class RecurrentAutoencoder(nn.Module):
    """
    Classic recurrent autoencoder baseline with either LSTM or GRU cells.

    Input:
    - window: [B, T, M]

    Output:
    - reconstruction: [B, T, M]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        latent_dim: int = 48,
        num_layers: int = 1,
        dropout: float = 0.10,
        cell_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.cell_type = str(cell_type).strip().lower()
        if self.cell_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        rnn_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.encoder = rnn_cls(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        )
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.decoder_state_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.decoder = rnn_cls(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, window: torch.Tensor) -> torch.Tensor:
        if self.cell_type == "lstm":
            _, (h_n, _) = self.encoder(window)
        else:
            _, h_n = self.encoder(window)
        last_hidden = h_n[-1]
        return self.latent_proj(last_hidden)

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        repeated_input = self.decoder_input_proj(latent).unsqueeze(1).repeat(1, int(seq_len), 1)
        init_state = self.decoder_state_proj(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.cell_type == "lstm":
            dec_out, _ = self.decoder(repeated_input, (init_state, torch.zeros_like(init_state)))
        else:
            dec_out, _ = self.decoder(repeated_input, init_state)
        return self.output_proj(dec_out)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        latent = self.encode(window)
        return self.decode(latent=latent, seq_len=int(window.shape[1]))

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
