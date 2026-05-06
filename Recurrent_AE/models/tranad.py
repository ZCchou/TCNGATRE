from __future__ import annotations

import itertools

import torch
from torch import nn

from models.modules.positional_encoding import PositionalEncoding


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


class TranADDecoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedWindowBlock(nn.Module):
    """
    One TranAD-style masked window block:
    1. causal self-attention over the local window
    2. cross-attention into complete-sequence memory
    3. feed-forward + residual + layer norm
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, window_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        causal = _causal_mask(length=int(window_tokens.shape[1]), device=window_tokens.device)
        self_out, _ = self.self_attn(
            query=window_tokens,
            key=window_tokens,
            value=window_tokens,
            attn_mask=causal,
            need_weights=False,
        )
        x = self.norm1(window_tokens + self.dropout(self_out))

        cross_out, _ = self.cross_attn(
            query=x,
            key=memory_tokens,
            value=memory_tokens,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(cross_out))
        x = self.norm3(x + self.ffn(x))
        return x


class TranAD(nn.Module):
    """
    Minimal engineering-faithful TranAD baseline.

    Input:
    - window: [B, K, M]
    - context: [B, C, M], current implementation uses window-as-context

    Output:
    - phase1 O1 / O2
    - phase2 O2_hat using focus F = (O1 - W)^2
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
        decoder_hidden_dim: int = 128,
        use_positional_encoding: bool = True,
        eps_adv: float = 1.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.eps_adv = float(max(eps_adv, 1.0001))

        self.context_input_proj = nn.Linear(self.input_dim * 2, self.d_model)
        self.window_input_proj = nn.Linear(self.input_dim, self.d_model)
        self.context_pos = PositionalEncoding(self.d_model, dropout=dropout) if use_positional_encoding else nn.Identity()
        self.window_pos = PositionalEncoding(self.d_model, dropout=dropout) if use_positional_encoding else nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.complete_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_encoder_layers))
        self.window_blocks = nn.ModuleList(
            [
                MaskedWindowBlock(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(int(num_encoder_layers))
            ]
        )
        self.decoder1 = TranADDecoder(
            d_model=self.d_model,
            hidden_dim=int(decoder_hidden_dim),
            output_dim=self.input_dim,
            dropout=float(dropout),
        )
        self.decoder2 = TranADDecoder(
            d_model=self.d_model,
            hidden_dim=int(decoder_hidden_dim),
            output_dim=self.input_dim,
            dropout=float(dropout),
        )

    def shared_parameters(self):
        return itertools.chain(
            self.context_input_proj.parameters(),
            self.window_input_proj.parameters(),
            self.complete_encoder.parameters(),
            self.window_blocks.parameters(),
        )

    def params_for_loss1(self):
        return itertools.chain(self.shared_parameters(), self.decoder1.parameters())

    def params_for_loss2(self):
        return itertools.chain(self.shared_parameters(), self.decoder2.parameters())

    def _align_focus_to_context(self, focus: torch.Tensor, context_len: int) -> torch.Tensor:
        if focus.shape[1] == context_len:
            return focus
        if focus.shape[1] > context_len:
            return focus[:, -context_len:, :]
        pad_len = context_len - focus.shape[1]
        pad = torch.zeros((focus.shape[0], pad_len, focus.shape[2]), dtype=focus.dtype, device=focus.device)
        return torch.cat([pad, focus], dim=1)

    def _encode_context(self, context: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        focus_ctx = self._align_focus_to_context(focus=focus, context_len=int(context.shape[1]))
        ctx = torch.cat([context, focus_ctx], dim=-1)
        ctx = self.context_input_proj(ctx)
        ctx = self.context_pos(ctx)
        return self.complete_encoder(ctx)

    def _encode_window(self, window: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.window_input_proj(window)
        x = self.window_pos(x)
        for block in self.window_blocks:
            x = block(window_tokens=x, memory_tokens=memory)
        return x

    def forward_phase(
        self,
        window: torch.Tensor,
        context: torch.Tensor | None = None,
        focus: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context is None:
            context = window
        if focus is None:
            focus = torch.zeros_like(window)
        memory = self._encode_context(context=context, focus=focus)
        encoded_window = self._encode_window(window=window, memory=memory)
        return {
            "encoded_window": encoded_window,
            "memory": memory,
            "o1": self.decoder1(encoded_window),
            "o2": self.decoder2(encoded_window),
        }

    def forward(self, window: torch.Tensor, context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        phase1 = self.forward_phase(window=window, context=context, focus=torch.zeros_like(window))
        focus = (phase1["o1"] - window).pow(2)
        phase2 = self.forward_phase(window=window, context=context, focus=focus)
        return {
            "phase1_o1": phase1["o1"],
            "phase1_o2": phase1["o2"],
            "phase2_o1": phase2["o1"],
            "phase2_o2": phase2["o2"],
            "focus": focus,
        }

    def anomaly_components(self, window: torch.Tensor, context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.forward(window=window, context=context)
        phase1_err = (out["phase1_o1"] - window).pow(2)
        phase2_err = (out["phase2_o2"] - window).pow(2)
        step_score = 0.5 * phase1_err.mean(dim=-1) + 0.5 * phase2_err.mean(dim=-1)
        return {
            **out,
            "phase1_err": phase1_err,
            "phase2_err": phase2_err,
            "step_score": step_score,
            "final_score": step_score[:, -1],
        }

    def compute_losses(self, window: torch.Tensor, context: torch.Tensor | None = None, epoch_index: int = 1) -> dict[str, torch.Tensor]:
        out = self.forward(window=window, context=context)
        recon1 = (out["phase1_o1"] - window).pow(2).mean()
        recon2 = (out["phase1_o2"] - window).pow(2).mean()
        adv = (out["phase2_o2"] - window).pow(2).mean()
        alpha = float(self.eps_adv ** (-max(int(epoch_index), 1)))
        alpha = float(min(max(alpha, 1e-4), 0.9999))
        loss1 = alpha * recon1 + (1.0 - alpha) * adv
        loss2 = alpha * recon2 - (1.0 - alpha) * adv
        return {
            "loss1": loss1,
            "loss2": loss2,
            "recon1": recon1,
            "recon2": recon2,
            "adv": adv,
            "alpha": torch.tensor(alpha, dtype=window.dtype, device=window.device),
        }
