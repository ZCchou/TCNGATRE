from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .time_encoding import TimeMLPEncoding


def build_control_sensor_mask(
    sensor_names: Sequence[str] | None,
    sensor_patterns: Sequence[str],
    num_sensors: int,
) -> torch.Tensor:
    names = list(sensor_names or [f"sensor_{i:02d}" for i in range(int(num_sensors))])
    if len(names) < int(num_sensors):
        names.extend([f"sensor_{i:02d}" for i in range(len(names), int(num_sensors))])
    patterns = [str(x).strip().lower() for x in sensor_patterns if str(x).strip() != ""]
    mask = torch.zeros((int(num_sensors),), dtype=torch.float32)
    if len(patterns) <= 0:
        return mask
    for idx, name in enumerate(names[: int(num_sensors)]):
        lname = str(name).lower()
        if any(pat in lname for pat in patterns):
            mask[idx] = 1.0
    return mask


class PatchSensorEncoder(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        time_hidden: int,
        value_hidden: int,
        use_value_normalization: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.use_value_normalization = bool(use_value_normalization)
        self.norm_eps = float(norm_eps)

        self.sensor_emb = nn.Embedding(self.num_sensors, self.d_model)
        self.time_enc = TimeMLPEncoding(d_model=self.d_model, hidden=time_hidden)
        self.value_proj = nn.Sequential(
            nn.Linear(1, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, self.d_model),
        )
        self.sensor_query = nn.Parameter(torch.randn(1, self.num_sensors, self.d_model) * 0.02)
        self.global_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sensor_ln = nn.LayerNorm(self.d_model)
        self.global_ln = nn.LayerNorm(self.d_model)
        self.sensor_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.global_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.register_buffer("value_mean", torch.zeros(self.num_sensors, dtype=torch.float32))
        self.register_buffer("value_std", torch.ones(self.num_sensors, dtype=torch.float32))

    def set_value_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().float().view(-1)
        std = std.detach().float().view(-1)
        if mean.numel() != self.num_sensors or std.numel() != self.num_sensors:
            raise ValueError(
                f"Normalization stats shape mismatch: expect {self.num_sensors}, "
                f"got mean={mean.numel()}, std={std.numel()}"
            )
        self.value_mean.copy_(mean)
        self.value_std.copy_(torch.clamp(std, min=self.norm_eps))

    def _normalize_value(self, sensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if not self.use_value_normalization:
            return value
        mean = self.value_mean[sensor]
        std = torch.clamp(self.value_std[sensor], min=self.norm_eps)
        return (value - mean) / std

    def forward(
        self,
        dt: torch.Tensor,
        sensor: torch.Tensor,
        value: torch.Tensor,
        event_mask: torch.Tensor,
        regime_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = dt.shape
        value_norm = self._normalize_value(sensor, value)
        sensor_emb = self.sensor_emb(sensor)
        value_emb = self.value_proj(value_norm.unsqueeze(-1))
        regime_expand = (
            torch.zeros((batch_size, 1, self.d_model), device=dt.device, dtype=sensor_emb.dtype)
            if regime_context is None
            else regime_context.unsqueeze(1)
        )
        key = sensor_emb + self.time_enc(dt) + value_emb + regime_expand
        val = sensor_emb + value_emb + regime_expand
        key_padding_mask = ~event_mask

        sensor_q = self.sensor_query.expand(batch_size, -1, -1) + regime_expand
        sensor_out, _ = self.attn(sensor_q, key, val, key_padding_mask=key_padding_mask)
        sensor_token = self.sensor_ln(sensor_q + sensor_out)
        sensor_token = self.sensor_ln(sensor_token + self.sensor_ff(sensor_token))

        global_q = self.global_query.expand(batch_size, -1, -1) + regime_expand
        global_out, _ = self.attn(global_q, key, val, key_padding_mask=key_padding_mask)
        global_token = self.global_ln(global_q + global_out)
        global_token = self.global_ln(global_token + self.global_ff(global_token))
        return sensor_token, global_token.squeeze(1)


class _SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros((seq_len, self.d_model), device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class CausalTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.global_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_enc = _SinusoidalPosition(d_model=d_model)
        self.sensor_ln = nn.LayerNorm(d_model)
        self.global_ln = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def _gather_last_valid(self, seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        idx = torch.clamp(valid_mask.long().sum(dim=1) - 1, min=0)
        gather_index = idx.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        return seq.gather(dim=1, index=gather_index).squeeze(1)

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        win_mask: torch.Tensor,
        regime_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_sensors, d_model = sensor_tokens.shape
        pos = self.pos_enc(seq_len, sensor_tokens.device, sensor_tokens.dtype).view(1, seq_len, 1, d_model)
        regime_expand = (
            torch.zeros((batch_size, 1, d_model), device=sensor_tokens.device, dtype=sensor_tokens.dtype)
            if regime_context is None
            else regime_context.unsqueeze(1)
        )
        sensor_in = sensor_tokens + pos + global_tokens.unsqueeze(2) + regime_expand.unsqueeze(2)
        sensor_in = sensor_in.permute(0, 2, 1, 3).reshape(batch_size * num_sensors, seq_len, d_model)
        sensor_mask = (~win_mask).unsqueeze(1).expand(-1, num_sensors, -1).reshape(batch_size * num_sensors, seq_len)
        sensor_seq = self.sensor_encoder(
            sensor_in,
            mask=self._causal_mask(seq_len, sensor_tokens.device),
            src_key_padding_mask=sensor_mask,
        )
        sensor_seq = self.sensor_ln(sensor_seq)
        sensor_seq = sensor_seq.view(batch_size, num_sensors, seq_len, d_model).permute(0, 2, 1, 3)

        global_in = global_tokens + self.pos_enc(seq_len, global_tokens.device, global_tokens.dtype).unsqueeze(0) + regime_expand
        global_seq = self.global_encoder(
            global_in,
            mask=self._causal_mask(seq_len, global_tokens.device),
            src_key_padding_mask=~win_mask,
        )
        global_seq = self.global_ln(global_seq)

        sensor_last = self._gather_last_valid(
            sensor_seq.permute(0, 2, 1, 3).reshape(batch_size * num_sensors, seq_len, d_model),
            win_mask.unsqueeze(1).expand(-1, num_sensors, -1).reshape(batch_size * num_sensors, seq_len),
        ).view(batch_size, num_sensors, d_model)
        global_last = self._gather_last_valid(global_seq, win_mask)
        return sensor_last, global_last


class LSTMTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        del n_heads, ff_dim
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.num_layers = max(int(num_layers), 1)
        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.sensor_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.global_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.pos_enc = _SinusoidalPosition(d_model=d_model)
        self.sensor_ln = nn.LayerNorm(d_model)
        self.global_ln = nn.LayerNorm(d_model)

    def _encode_lstm(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        lstm: nn.LSTM,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = valid_mask.long().sum(dim=1)
        safe_lengths = lengths.clamp(min=1)
        seq_mask = valid_mask.unsqueeze(-1).to(dtype=seq.dtype)
        masked_seq = seq * seq_mask
        packed = pack_padded_sequence(
            masked_seq,
            lengths=safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = lstm(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq.shape[1],
        )
        last_idx = (safe_lengths - 1).to(device=seq.device)
        gather_index = last_idx.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        last = out.gather(dim=1, index=gather_index).squeeze(1)
        has_any = (lengths > 0).unsqueeze(-1)
        last = torch.where(has_any, last, torch.zeros_like(last))
        return out, last

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        win_mask: torch.Tensor,
        regime_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_sensors, d_model = sensor_tokens.shape
        pos = self.pos_enc(seq_len, sensor_tokens.device, sensor_tokens.dtype).view(1, seq_len, 1, d_model)
        regime_expand = (
            torch.zeros((batch_size, 1, d_model), device=sensor_tokens.device, dtype=sensor_tokens.dtype)
            if regime_context is None
            else regime_context.unsqueeze(1)
        )
        sensor_in = sensor_tokens + pos + global_tokens.unsqueeze(2) + regime_expand.unsqueeze(2)
        sensor_in = sensor_in.permute(0, 2, 1, 3).reshape(batch_size * num_sensors, seq_len, d_model)
        sensor_mask = win_mask.unsqueeze(1).expand(-1, num_sensors, -1).reshape(batch_size * num_sensors, seq_len)
        sensor_seq, sensor_last = self._encode_lstm(sensor_in, sensor_mask, self.sensor_lstm)
        sensor_seq = self.sensor_ln(sensor_seq)
        sensor_last = self.sensor_ln(sensor_last)
        sensor_seq = sensor_seq.view(batch_size, num_sensors, seq_len, d_model).permute(0, 2, 1, 3)
        sensor_last = sensor_last.view(batch_size, num_sensors, d_model)

        global_in = global_tokens + self.pos_enc(seq_len, global_tokens.device, global_tokens.dtype).unsqueeze(0) + regime_expand
        global_seq, global_last = self._encode_lstm(global_in, win_mask, self.global_lstm)
        global_seq = self.global_ln(global_seq)
        global_last = self.global_ln(global_last)
        return sensor_last, global_last


class MultiScaleHistoryFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.sensor_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.global_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.sensor_ln = nn.LayerNorm(self.d_model)
        self.global_ln = nn.LayerNorm(self.d_model)

    def forward(
        self,
        fine_sensor_history: torch.Tensor,
        fine_global_history: torch.Tensor,
        coarse_sensor_history: torch.Tensor,
        coarse_global_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sensor_cat = torch.cat([fine_sensor_history, coarse_sensor_history], dim=-1)
        global_cat = torch.cat([fine_global_history, coarse_global_history], dim=-1)
        sensor_out = self.sensor_ln(fine_sensor_history + self.sensor_fuse(sensor_cat))
        global_out = self.global_ln(fine_global_history + self.global_fuse(global_cat))
        return sensor_out, global_out


class SensorHistoryCurveEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.out_ln = nn.LayerNorm(d_model)

    def _forward_fill(
        self,
        value_seq: torch.Tensor,
        has_value_seq: torch.Tensor,
    ) -> torch.Tensor:
        filled = value_seq.clone()
        batch_size, seq_len, num_sensors = filled.shape
        for t in range(1, seq_len):
            prev = filled[:, t - 1]
            cur_mask = has_value_seq[:, t] > 0.5
            filled[:, t] = torch.where(cur_mask, filled[:, t], prev)
        has_any = (has_value_seq.sum(dim=1, keepdim=True) > 0.0).to(dtype=filled.dtype)
        return filled * has_any

    def forward(
        self,
        hist_patch_last_value_seq: torch.Tensor,
        hist_patch_has_value_seq: torch.Tensor,
    ) -> torch.Tensor:
        filled = self._forward_fill(hist_patch_last_value_seq, hist_patch_has_value_seq)
        delta = torch.zeros_like(filled)
        delta[:, 1:] = filled[:, 1:] - filled[:, :-1]
        feat = torch.stack(
            [
                filled,
                hist_patch_has_value_seq,
                delta,
            ],
            dim=-1,
        )
        batch_size, seq_len, num_sensors, feat_dim = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size * num_sensors, seq_len, feat_dim)
        hidden_in = self.input_proj(feat)
        _, hidden = self.gru(hidden_in)
        curve_context = self.out_ln(hidden[-1]).view(batch_size, num_sensors, self.d_model)
        has_any = (hist_patch_has_value_seq.sum(dim=1) > 0.0).to(dtype=curve_context.dtype).unsqueeze(-1)
        return curve_context * has_any


class SensorLocalHistoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = max(int(kernel_size), 3)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.input_proj = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.depthwise = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.hidden_dim,
        )
        self.pointwise = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.d_model,
            kernel_size=1,
        )
        self.out_ln = nn.LayerNorm(self.d_model)

    def _forward_fill(
        self,
        value_seq: torch.Tensor,
        has_value_seq: torch.Tensor,
    ) -> torch.Tensor:
        filled = value_seq.clone()
        for t in range(1, value_seq.shape[1]):
            filled[:, t] = torch.where(has_value_seq[:, t] > 0.5, filled[:, t], filled[:, t - 1])
        return filled

    def forward(
        self,
        hist_patch_last_value_seq: torch.Tensor,
        hist_patch_has_value_seq: torch.Tensor,
    ) -> torch.Tensor:
        filled = self._forward_fill(hist_patch_last_value_seq, hist_patch_has_value_seq)
        delta = torch.zeros_like(filled)
        delta[:, 1:] = filled[:, 1:] - filled[:, :-1]
        feat = torch.stack([filled, hist_patch_has_value_seq, delta], dim=-1)
        batch_size, seq_len, num_sensors, feat_dim = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size * num_sensors, seq_len, feat_dim)
        hidden = self.input_proj(feat)
        hidden = hidden.transpose(1, 2)
        hidden = self.depthwise(hidden)
        hidden = F.gelu(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        mask = hist_patch_has_value_seq.permute(0, 2, 1).reshape(batch_size * num_sensors, seq_len, 1)
        recency = torch.linspace(
            1.0,
            2.0,
            steps=seq_len,
            dtype=hidden.dtype,
            device=hidden.device,
        ).view(1, seq_len, 1)
        weight = mask.to(dtype=hidden.dtype) * recency
        weight_sum = weight.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * weight).sum(dim=1) / weight_sum
        pooled = self.out_ln(pooled).view(batch_size, num_sensors, self.d_model)
        has_any = (hist_patch_has_value_seq.sum(dim=1) > 0.0).to(dtype=pooled.dtype).unsqueeze(-1)
        return pooled * has_any


class PhaseAwareMoEHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phase_dim: int,
        output_dim: int = 1,
        num_experts: int = 4,
        gate_hidden: int = 96,
        expert_hidden: int = 128,
        dropout: float = 0.10,
        temperature: float = 1.0,
        residual_scale: float = 0.20,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.phase_dim = int(phase_dim)
        self.output_dim = int(output_dim)
        self.num_experts = max(int(num_experts), 1)
        self.temperature = max(float(temperature), 1e-3)
        self.residual_scale = max(float(residual_scale), 0.0)
        gate_hidden = max(int(gate_hidden), 8)
        expert_hidden = max(int(expert_hidden), self.output_dim)
        self.base_head = nn.Linear(self.input_dim, self.output_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim + self.phase_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, self.output_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.residual_log_scale = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, hidden: torch.Tensor, phase_context: torch.Tensor | None = None) -> torch.Tensor:
        base_out = self.base_head(hidden)
        if phase_context is None:
            phase_context = hidden.new_zeros(hidden.shape[:-1] + (self.phase_dim,))
        gate_in = torch.cat([hidden, phase_context], dim=-1)
        gate_logits = self.gate(gate_in) / self.temperature
        gate = torch.softmax(gate_logits, dim=-1)
        residual_out = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        residual_out = torch.einsum("...e,...eo->...o", gate, residual_out)
        residual_gain = self.residual_scale * torch.tanh(self.residual_log_scale)
        view_shape = (1,) * (base_out.dim() - 1) + (self.output_dim,)
        return base_out + residual_out * residual_gain.view(view_shape)


class SensorFutureDecoder(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        phase_emb_dim: int,
        head_variant: str = "full",
        use_sensor_local_branch: bool = False,
        num_future_bins: int = 0,
        bin_hidden: int | None = None,
        use_onset_head: bool = False,
        onset_hidden: int | None = None,
        use_curve_context: bool = False,
        use_change_pattern_branch: bool = False,
        change_branch_hidden: int | None = None,
        use_control_aux_head: bool = False,
        use_phase_aware_moe_heads: bool = False,
        phase_aware_moe_num_experts: int = 4,
        phase_aware_moe_gate_hidden: int = 96,
        phase_aware_moe_expert_hidden: int = 128,
        phase_aware_moe_dropout: float = 0.10,
        phase_aware_moe_temperature: float = 1.0,
        phase_aware_moe_residual_scale: float = 0.20,
        phase_aware_moe_targets: Sequence[str] = (),
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.phase_emb_dim = int(phase_emb_dim)
        self.head_variant = str(head_variant).strip().lower()
        if self.head_variant == "compact_aux":
            self.head_variant = "core_aux"
        if self.head_variant in {"vdm", "value_delta_mean"}:
            self.head_variant = "vdm_only"
        if self.head_variant not in {"full", "core_aux", "vdm_only", "value_only"}:
            raise ValueError(f"Unsupported forecast head variant: {head_variant}")
        self.core_aux_head = self.head_variant == "core_aux"
        self.vdm_only_head = self.head_variant == "vdm_only"
        self.value_only_head = self.head_variant == "value_only"
        self.use_sensor_local_branch = bool(use_sensor_local_branch)
        self.num_future_bins = 0 if (self.value_only_head or self.core_aux_head or self.vdm_only_head) else max(int(num_future_bins), 0)
        self.use_onset_head = bool(use_onset_head) and not (self.value_only_head or self.vdm_only_head)
        self.use_onset_mask_head = self.use_onset_head and not self.core_aux_head
        self.use_curve_context = bool(use_curve_context)
        self.use_change_pattern_branch = bool(use_change_pattern_branch) and not self.value_only_head
        self.use_control_aux_head = bool(use_control_aux_head) and not (self.value_only_head or self.vdm_only_head)
        self.use_phase_aware_moe_heads = bool(use_phase_aware_moe_heads)
        self.phase_aware_moe_num_experts = max(int(phase_aware_moe_num_experts), 1)
        self.phase_aware_moe_gate_hidden = max(int(phase_aware_moe_gate_hidden), 8)
        self.phase_aware_moe_expert_hidden = max(int(phase_aware_moe_expert_hidden), 8)
        self.phase_aware_moe_dropout = float(phase_aware_moe_dropout)
        self.phase_aware_moe_temperature = max(float(phase_aware_moe_temperature), 1e-3)
        self.phase_aware_moe_residual_scale = max(float(phase_aware_moe_residual_scale), 0.0)
        self.phase_aware_moe_targets = {str(x).strip().lower() for x in phase_aware_moe_targets if str(x).strip() != ""}
        self.sensor_emb = nn.Embedding(self.num_sensors, self.d_model)
        local_dim = self.d_model if self.use_sensor_local_branch else 0
        base_in_dim = self.d_model * 3 + local_dim + 2 + self.phase_emb_dim
        curve_dim = (self.d_model + 3) if self.use_curve_context else 0
        change_in_dim = base_in_dim + curve_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(base_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        change_hidden_dim = int(change_branch_hidden or hidden_dim)
        if self.use_change_pattern_branch:
            self.change_mlp = nn.Sequential(
                nn.Linear(change_in_dim, change_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(change_hidden_dim, change_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.change_mlp = None
            change_hidden_dim = hidden_dim

        self.value_head = self._build_head("value", hidden_dim, 1)
        if self.value_only_head:
            self.mask_head = None
            self.mean_head = None
            self.count_head = None
            self.delta_head = None
            self.range_head = None
            self.var_head = None
            self.constant_head = None
            self.abs_diff_head = None
            self.zero_diff_head = None
            self.slope_head = None
            self.control_change_head = None
        elif self.vdm_only_head:
            self.mask_head = None
            self.mean_head = self._build_head("mean", hidden_dim, 1)
            self.count_head = None
            self.delta_head = self._build_head("delta", change_hidden_dim, 1)
            self.range_head = None
            self.var_head = None
            self.constant_head = None
            self.abs_diff_head = None
            self.zero_diff_head = None
            self.slope_head = None
            self.control_change_head = None
        elif self.core_aux_head:
            self.mask_head = None
            self.mean_head = self._build_head("mean", hidden_dim, 1)
            self.count_head = None
            self.delta_head = self._build_head("delta", change_hidden_dim, 1)
            self.range_head = None
            self.var_head = None
            self.constant_head = None
            self.abs_diff_head = None
            self.zero_diff_head = None
            self.slope_head = None
            self.control_change_head = self._build_head("control_change", change_hidden_dim, 1) if self.use_control_aux_head else None
        else:
            self.mask_head = self._build_head("mask", hidden_dim, 1)
            self.mean_head = self._build_head("mean", hidden_dim, 1)
            self.count_head = self._build_head("count", hidden_dim, 1)
            self.delta_head = self._build_head("delta", change_hidden_dim, 1)
            self.range_head = self._build_head("range", change_hidden_dim, 1)
            self.var_head = self._build_head("var", change_hidden_dim, 1)
            self.constant_head = self._build_head("constant", change_hidden_dim, 1)
            self.abs_diff_head = self._build_head("abs_diff", change_hidden_dim, 1)
            self.zero_diff_head = self._build_head("zero_diff", change_hidden_dim, 1)
            self.slope_head = self._build_head("slope", change_hidden_dim, 1)
            self.control_change_head = self._build_head("control_change", change_hidden_dim, 1) if self.use_control_aux_head else None
        if self.num_future_bins > 0 and not self.value_only_head:
            bin_hidden_dim = int(bin_hidden or hidden_dim)
            self.bin_mlp = nn.Sequential(
                nn.Linear(base_in_dim + change_hidden_dim, bin_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bin_hidden_dim, bin_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.bin_mask_head = self._build_head("bin_mask", bin_hidden_dim, self.num_future_bins)
            self.bin_last_head = self._build_head("bin_last", bin_hidden_dim, self.num_future_bins)
            self.bin_mean_head = self._build_head("bin_mean", bin_hidden_dim, self.num_future_bins)
            self.bin_count_head = self._build_head("bin_count", bin_hidden_dim, self.num_future_bins)
        else:
            self.bin_mlp = None
            self.bin_mask_head = None
            self.bin_last_head = None
            self.bin_mean_head = None
            self.bin_count_head = None
        if self.use_onset_head:
            onset_hidden_dim = int(onset_hidden or hidden_dim)
            self.onset_mlp = nn.Sequential(
                nn.Linear(base_in_dim + change_hidden_dim, onset_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(onset_hidden_dim, onset_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.onset_mask_head = self._build_head("onset_mask", onset_hidden_dim, 1) if self.use_onset_mask_head else None
            self.onset_mean_head = self._build_head("onset_mean", onset_hidden_dim, 1)
            self.onset_delta_head = self._build_head("onset_delta", onset_hidden_dim, 1)
        else:
            self.onset_mlp = None
            self.onset_mask_head = None
            self.onset_mean_head = None
            self.onset_delta_head = None

    def _use_moe_for(self, name: str) -> bool:
        return self.use_phase_aware_moe_heads and str(name).strip().lower() in self.phase_aware_moe_targets

    def _build_head(self, name: str, input_dim: int, output_dim: int) -> nn.Module:
        if self._use_moe_for(name):
            return PhaseAwareMoEHead(
                input_dim=input_dim,
                phase_dim=self.phase_emb_dim,
                output_dim=output_dim,
                num_experts=self.phase_aware_moe_num_experts,
                gate_hidden=self.phase_aware_moe_gate_hidden,
                expert_hidden=self.phase_aware_moe_expert_hidden,
                dropout=self.phase_aware_moe_dropout,
                temperature=self.phase_aware_moe_temperature,
                residual_scale=self.phase_aware_moe_residual_scale,
            )
        return nn.Linear(input_dim, output_dim)

    def _apply_head(self, head: nn.Module, hidden: torch.Tensor, phase_context: torch.Tensor) -> torch.Tensor:
        if isinstance(head, PhaseAwareMoEHead):
            return head(hidden, phase_context)
        return head(hidden)

    def forward(
        self,
        sensor_history: torch.Tensor,
        global_history: torch.Tensor,
        hist_last_value: torch.Tensor,
        hist_has_value: torch.Tensor,
        phase_context: torch.Tensor,
        sensor_local_context: torch.Tensor | None = None,
        curve_context: torch.Tensor | None = None,
        recent_mean: torch.Tensor | None = None,
        recent_delta: torch.Tensor | None = None,
        recent_range: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_sensors, _ = sensor_history.shape
        sensor_ids = torch.arange(num_sensors, device=sensor_history.device, dtype=torch.long).view(1, num_sensors)
        sensor_ids = sensor_ids.expand(batch_size, -1)
        global_expand = global_history.unsqueeze(1).expand(-1, num_sensors, -1)
        phase_expand = phase_context.unsqueeze(1).expand(-1, num_sensors, -1)
        base_pieces = [
            sensor_history,
            global_expand,
            self.sensor_emb(sensor_ids),
        ]
        if self.use_sensor_local_branch:
            if sensor_local_context is None:
                raise ValueError("sensor local branch enabled but sensor_local_context was not provided")
            base_pieces.append(sensor_local_context)
        base_pieces.extend(
            [
                hist_last_value.unsqueeze(-1),
                hist_has_value.unsqueeze(-1),
                phase_expand,
            ]
        )
        base_decoder_in = torch.cat(base_pieces, dim=-1)
        base_hidden = self.base_mlp(base_decoder_in)
        if self.value_only_head:
            future_value_hat = self._apply_head(self.value_head, base_hidden, phase_expand).squeeze(-1)
            zero_scalar = torch.zeros_like(future_value_hat)
            zero_bins = future_value_hat.unsqueeze(-1).new_zeros((batch_size, num_sensors, 0))
            return {
                "future_mask_logit": zero_scalar,
                "future_value_hat": future_value_hat,
                "future_delta_hat": zero_scalar,
                "future_mean_hat": zero_scalar,
                "future_range_hat": zero_scalar,
                "future_var_hat": zero_scalar,
                "future_log_count_hat": zero_scalar,
                "future_constant_logit": zero_scalar,
                "future_abs_diff_hat": zero_scalar,
                "future_zero_diff_logit": zero_scalar,
                "future_slope_hat": zero_scalar,
                "future_control_change_hat": zero_scalar,
                "future_bin_mask_logit": zero_bins,
                "future_bin_last_hat": zero_bins,
                "future_bin_mean_hat": zero_bins,
                "future_bin_log_count_hat": zero_bins,
                "future_onset_mask_logit": zero_scalar,
                "future_onset_mean_hat": zero_scalar,
                "future_onset_delta_hat": zero_scalar,
            }

        change_hidden = base_hidden
        if self.use_change_pattern_branch:
            change_pieces = list(base_pieces)
            if self.use_curve_context:
                if curve_context is None or recent_mean is None or recent_delta is None or recent_range is None:
                    raise ValueError("curve context enabled but curve features were not provided")
                change_pieces.extend(
                    [
                        curve_context,
                        recent_mean.unsqueeze(-1),
                        recent_delta.unsqueeze(-1),
                        recent_range.unsqueeze(-1),
                    ]
                )
            change_decoder_in = torch.cat(change_pieces, dim=-1)
            change_hidden = self.change_mlp(change_decoder_in)
        elif self.use_curve_context:
            if curve_context is None or recent_mean is None or recent_delta is None or recent_range is None:
                raise ValueError("curve context enabled but curve features were not provided")
            change_hidden = base_hidden + 0.0 * (
                curve_context.mean(dim=-1, keepdim=True)
                + recent_mean.unsqueeze(-1)
                + recent_delta.unsqueeze(-1)
                + recent_range.unsqueeze(-1)
            )
        out = {
            "future_mask_logit": (
                self._apply_head(self.mask_head, base_hidden, phase_expand).squeeze(-1)
                if self.mask_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_value_hat": self._apply_head(self.value_head, base_hidden, phase_expand).squeeze(-1),
            "future_delta_hat": (
                self._apply_head(self.delta_head, change_hidden, phase_expand).squeeze(-1)
                if self.delta_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_mean_hat": (
                self._apply_head(self.mean_head, base_hidden, phase_expand).squeeze(-1)
                if self.mean_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_range_hat": (
                F.softplus(self._apply_head(self.range_head, change_hidden, phase_expand).squeeze(-1))
                if self.range_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_var_hat": (
                F.softplus(self._apply_head(self.var_head, change_hidden, phase_expand).squeeze(-1))
                if self.var_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_log_count_hat": (
                F.softplus(self._apply_head(self.count_head, base_hidden, phase_expand).squeeze(-1))
                if self.count_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_constant_logit": (
                self._apply_head(self.constant_head, change_hidden, phase_expand).squeeze(-1)
                if self.constant_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_abs_diff_hat": (
                F.softplus(self._apply_head(self.abs_diff_head, change_hidden, phase_expand).squeeze(-1))
                if self.abs_diff_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_zero_diff_logit": (
                self._apply_head(self.zero_diff_head, change_hidden, phase_expand).squeeze(-1)
                if self.zero_diff_head is not None
                else torch.zeros_like(hist_last_value)
            ),
            "future_slope_hat": (
                self._apply_head(self.slope_head, change_hidden, phase_expand).squeeze(-1)
                if self.slope_head is not None
                else torch.zeros_like(hist_last_value)
            ),
        }
        if self.use_control_aux_head and self.control_change_head is not None:
            out["future_control_change_hat"] = F.softplus(
                self._apply_head(self.control_change_head, change_hidden, phase_expand).squeeze(-1)
            )
        else:
            out["future_control_change_hat"] = torch.zeros_like(out["future_value_hat"])
        if self.num_future_bins > 0 and self.bin_mlp is not None:
            bin_hidden = self.bin_mlp(torch.cat([base_decoder_in, change_hidden], dim=-1))
            out["future_bin_mask_logit"] = self._apply_head(self.bin_mask_head, bin_hidden, phase_expand)
            out["future_bin_last_hat"] = self._apply_head(self.bin_last_head, bin_hidden, phase_expand)
            out["future_bin_mean_hat"] = self._apply_head(self.bin_mean_head, bin_hidden, phase_expand)
            out["future_bin_log_count_hat"] = F.softplus(self._apply_head(self.bin_count_head, bin_hidden, phase_expand))
        else:
            zeros = out["future_value_hat"].unsqueeze(-1).new_zeros((batch_size, num_sensors, 0))
            out["future_bin_mask_logit"] = zeros
            out["future_bin_last_hat"] = zeros
            out["future_bin_mean_hat"] = zeros
            out["future_bin_log_count_hat"] = zeros
        if self.use_onset_head and self.onset_mlp is not None:
            onset_hidden = self.onset_mlp(torch.cat([base_decoder_in, change_hidden], dim=-1))
            out["future_onset_mask_logit"] = (
                self._apply_head(self.onset_mask_head, onset_hidden, phase_expand).squeeze(-1)
                if self.onset_mask_head is not None
                else torch.zeros_like(out["future_value_hat"])
            )
            out["future_onset_mean_hat"] = (
                self._apply_head(self.onset_mean_head, onset_hidden, phase_expand).squeeze(-1)
                if self.onset_mean_head is not None
                else torch.zeros_like(out["future_value_hat"])
            )
            out["future_onset_delta_hat"] = (
                self._apply_head(self.onset_delta_head, onset_hidden, phase_expand).squeeze(-1)
                if self.onset_delta_head is not None
                else torch.zeros_like(out["future_value_hat"])
            )
        else:
            zeros = out["future_value_hat"].new_zeros((batch_size, num_sensors))
            out["future_onset_mask_logit"] = zeros
            out["future_onset_mean_hat"] = zeros
            out["future_onset_delta_hat"] = zeros
        return out


class IrregularFutureWindowForecaster(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        time_hidden: int,
        value_hidden: int,
        decoder_hidden: int,
        phase_num_clusters: int,
        phase_emb_dim: int,
        temporal_backbone: str = "transformer",
        use_phase_context: bool = True,
        head_variant: str = "full",
        use_curve_context: bool = False,
        use_multiscale_history: bool = False,
        multiscale_stride: int = 2,
        multiscale_num_layers: int = 2,
        multiscale_dropout: float = 0.10,
        curve_hidden: int = 96,
        use_sensor_local_branch: bool = False,
        sensor_local_hidden: int = 96,
        use_future_bin_decoder: bool = False,
        num_future_bins: int = 4,
        bin_hidden: int = 160,
        use_onset_head: bool = False,
        onset_hidden: int = 160,
        use_change_pattern_branch: bool = False,
        change_branch_hidden: int = 192,
        use_control_aux_head: bool = False,
        control_aux_hidden: int = 160,
        use_phase_aware_moe_heads: bool = False,
        phase_aware_moe_num_experts: int = 4,
        phase_aware_moe_gate_hidden: int = 96,
        phase_aware_moe_expert_hidden: int = 128,
        phase_aware_moe_dropout: float = 0.10,
        phase_aware_moe_temperature: float = 1.0,
        phase_aware_moe_residual_scale: float = 0.20,
        phase_aware_moe_targets: Sequence[str] = (),
        sensor_names: Sequence[str] | None = None,
        control_sensor_patterns: Sequence[str] = (),
        use_anchored_decoder: bool = True,
        bound_value_outputs: bool = True,
        horizon_scale: float = 0.5,
        value_residual_scale: float = 0.18,
        delta_residual_scale: float = 0.30,
        mean_residual_scale: float = 0.16,
        slope_residual_scale: float = 0.20,
        use_value_normalization: bool = True,
        norm_eps: float = 1e-6,
        use_huber_loss: bool = True,
        huber_beta: float = 1.0,
        loss_weight_mask: float = 0.5,
        loss_weight_value: float = 1.0,
        loss_weight_delta: float = 1.0,
        loss_weight_mean: float = 0.6,
        loss_weight_range: float = 0.6,
        loss_weight_var: float = 0.4,
        loss_weight_count: float = 0.3,
        loss_weight_constant: float = 0.4,
        loss_weight_abs_diff: float = 0.8,
        loss_weight_zero_diff: float = 0.8,
        loss_weight_slope: float = 0.7,
        loss_weight_control_change: float = 0.3,
        loss_weight_bin_mask: float = 0.0,
        loss_weight_bin_last: float = 0.0,
        loss_weight_bin_mean: float = 0.0,
        loss_weight_bin_count: float = 0.0,
        loss_weight_onset_mask: float = 0.0,
        loss_weight_onset_mean: float = 0.0,
        loss_weight_onset_delta: float = 0.0,
        score_weight_mask: float = 0.25,
        score_weight_value: float = 1.0,
        score_weight_delta: float = 1.0,
        score_weight_mean: float = 0.5,
        score_weight_range: float = 0.7,
        score_weight_var: float = 0.3,
        score_weight_count: float = 0.25,
        score_weight_constant: float = 0.8,
        score_weight_abs_diff: float = 1.0,
        score_weight_zero_diff: float = 1.0,
        score_weight_slope: float = 0.8,
        score_weight_control_change: float = 0.0,
        score_weight_onset_mean: float = 0.0,
        score_weight_onset_delta: float = 0.0,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.phase_num_clusters = int(phase_num_clusters)
        self.phase_emb_dim = int(phase_emb_dim)
        self.temporal_backbone = str(temporal_backbone).strip().lower()
        self.forecast_head_variant = str(head_variant).strip().lower()
        if self.forecast_head_variant == "compact_aux":
            self.forecast_head_variant = "core_aux"
        if self.forecast_head_variant in {"vdm", "value_delta_mean"}:
            self.forecast_head_variant = "vdm_only"
        if self.forecast_head_variant not in {"full", "core_aux", "vdm_only", "value_only"}:
            raise ValueError(f"Unsupported forecast head variant: {head_variant}")
        self.core_aux_head = self.forecast_head_variant == "core_aux"
        self.vdm_only_head = self.forecast_head_variant == "vdm_only"
        self.value_only_head = self.forecast_head_variant == "value_only"
        self.use_phase_context = bool(use_phase_context)
        self.use_curve_context = bool(use_curve_context)
        self.use_sensor_local_branch = bool(use_sensor_local_branch)
        self.use_future_bin_decoder = bool(use_future_bin_decoder) and not (self.value_only_head or self.core_aux_head or self.vdm_only_head)
        self.num_future_bins = max(int(num_future_bins), 0) if self.use_future_bin_decoder else 0
        self.use_onset_head = bool(use_onset_head) and not (self.value_only_head or self.vdm_only_head)
        self.use_multiscale_history = bool(use_multiscale_history)
        self.multiscale_stride = max(int(multiscale_stride), 2)
        self.use_change_pattern_branch = bool(use_change_pattern_branch)
        self.use_control_aux_head = bool(use_control_aux_head) and not (self.value_only_head or self.vdm_only_head)
        self.use_anchored_decoder = bool(use_anchored_decoder)
        self.bound_value_outputs = bool(bound_value_outputs)
        self.horizon_scale = float(horizon_scale)
        self.value_residual_scale = float(value_residual_scale)
        self.delta_residual_scale = float(delta_residual_scale)
        self.mean_residual_scale = float(mean_residual_scale)
        self.slope_residual_scale = float(slope_residual_scale)
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_beta = float(huber_beta)
        self.loss_weight_mask = float(loss_weight_mask)
        self.loss_weight_value = float(loss_weight_value)
        self.loss_weight_delta = float(loss_weight_delta)
        self.loss_weight_mean = float(loss_weight_mean)
        self.loss_weight_range = float(loss_weight_range)
        self.loss_weight_var = float(loss_weight_var)
        self.loss_weight_count = float(loss_weight_count)
        self.loss_weight_constant = float(loss_weight_constant)
        self.loss_weight_abs_diff = float(loss_weight_abs_diff)
        self.loss_weight_zero_diff = float(loss_weight_zero_diff)
        self.loss_weight_slope = float(loss_weight_slope)
        self.loss_weight_control_change = float(loss_weight_control_change)
        self.loss_weight_bin_mask = float(loss_weight_bin_mask)
        self.loss_weight_bin_last = float(loss_weight_bin_last)
        self.loss_weight_bin_mean = float(loss_weight_bin_mean)
        self.loss_weight_bin_count = float(loss_weight_bin_count)
        self.loss_weight_onset_mask = float(loss_weight_onset_mask)
        self.loss_weight_onset_mean = float(loss_weight_onset_mean)
        self.loss_weight_onset_delta = float(loss_weight_onset_delta)
        self.score_weight_mask = float(score_weight_mask)
        self.score_weight_value = float(score_weight_value)
        self.score_weight_delta = float(score_weight_delta)
        self.score_weight_mean = float(score_weight_mean)
        self.score_weight_range = float(score_weight_range)
        self.score_weight_var = float(score_weight_var)
        self.score_weight_count = float(score_weight_count)
        self.score_weight_constant = float(score_weight_constant)
        self.score_weight_abs_diff = float(score_weight_abs_diff)
        self.score_weight_zero_diff = float(score_weight_zero_diff)
        self.score_weight_slope = float(score_weight_slope)
        self.score_weight_control_change = float(score_weight_control_change)
        self.score_weight_onset_mean = float(score_weight_onset_mean)
        self.score_weight_onset_delta = float(score_weight_onset_delta)
        if self.value_only_head:
            self.loss_weight_mask = 0.0
            self.loss_weight_delta = 0.0
            self.loss_weight_mean = 0.0
            self.loss_weight_range = 0.0
            self.loss_weight_var = 0.0
            self.loss_weight_count = 0.0
            self.loss_weight_constant = 0.0
            self.loss_weight_abs_diff = 0.0
            self.loss_weight_zero_diff = 0.0
            self.loss_weight_slope = 0.0
            self.loss_weight_control_change = 0.0
            self.loss_weight_bin_mask = 0.0
            self.loss_weight_bin_last = 0.0
            self.loss_weight_bin_mean = 0.0
            self.loss_weight_bin_count = 0.0
            self.loss_weight_onset_mask = 0.0
            self.loss_weight_onset_mean = 0.0
            self.loss_weight_onset_delta = 0.0
            self.score_weight_mask = 0.0
            self.score_weight_delta = 0.0
            self.score_weight_mean = 0.0
            self.score_weight_range = 0.0
            self.score_weight_var = 0.0
            self.score_weight_count = 0.0
            self.score_weight_constant = 0.0
            self.score_weight_abs_diff = 0.0
            self.score_weight_zero_diff = 0.0
            self.score_weight_slope = 0.0
            self.score_weight_control_change = 0.0
            self.score_weight_onset_mean = 0.0
            self.score_weight_onset_delta = 0.0
        elif self.vdm_only_head:
            self.loss_weight_mask = 0.0
            self.loss_weight_range = 0.0
            self.loss_weight_var = 0.0
            self.loss_weight_count = 0.0
            self.loss_weight_constant = 0.0
            self.loss_weight_abs_diff = 0.0
            self.loss_weight_zero_diff = 0.0
            self.loss_weight_slope = 0.0
            self.loss_weight_control_change = 0.0
            self.loss_weight_bin_mask = 0.0
            self.loss_weight_bin_last = 0.0
            self.loss_weight_bin_mean = 0.0
            self.loss_weight_bin_count = 0.0
            self.loss_weight_onset_mask = 0.0
            self.loss_weight_onset_mean = 0.0
            self.loss_weight_onset_delta = 0.0
            self.score_weight_mask = 0.0
            self.score_weight_range = 0.0
            self.score_weight_var = 0.0
            self.score_weight_count = 0.0
            self.score_weight_constant = 0.0
            self.score_weight_abs_diff = 0.0
            self.score_weight_zero_diff = 0.0
            self.score_weight_slope = 0.0
            self.score_weight_control_change = 0.0
            self.score_weight_onset_mean = 0.0
            self.score_weight_onset_delta = 0.0
        elif self.core_aux_head:
            self.loss_weight_mask = 0.0
            self.loss_weight_range = 0.0
            self.loss_weight_var = 0.0
            self.loss_weight_count = 0.0
            self.loss_weight_constant = 0.0
            self.loss_weight_abs_diff = 0.0
            self.loss_weight_zero_diff = 0.0
            self.loss_weight_slope = 0.0
            self.loss_weight_bin_mask = 0.0
            self.loss_weight_bin_last = 0.0
            self.loss_weight_bin_mean = 0.0
            self.loss_weight_bin_count = 0.0
            self.loss_weight_onset_mask = 0.0
            self.score_weight_mask = 0.0
            self.score_weight_range = 0.0
            self.score_weight_var = 0.0
            self.score_weight_count = 0.0
            self.score_weight_constant = 0.0
            self.score_weight_abs_diff = 0.0
            self.score_weight_zero_diff = 0.0
            self.score_weight_slope = 0.0
        if not self.use_control_aux_head:
            self.loss_weight_control_change = 0.0
            self.score_weight_control_change = 0.0
        if not self.use_onset_head:
            self.loss_weight_onset_mask = 0.0
            self.loss_weight_onset_mean = 0.0
            self.loss_weight_onset_delta = 0.0
            self.score_weight_onset_mean = 0.0
            self.score_weight_onset_delta = 0.0
        if not self.use_future_bin_decoder:
            self.loss_weight_bin_mask = 0.0
            self.loss_weight_bin_last = 0.0
            self.loss_weight_bin_mean = 0.0
            self.loss_weight_bin_count = 0.0

        self.patch_encoder = PatchSensorEncoder(
            num_sensors=num_sensors,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            time_hidden=time_hidden,
            value_hidden=value_hidden,
            use_value_normalization=use_value_normalization,
            norm_eps=norm_eps,
        )
        self.temporal_encoder = self._make_temporal_encoder(
            num_sensors=num_sensors,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.coarse_temporal_encoder = (
            self._make_temporal_encoder(
                num_sensors=num_sensors,
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=max(1, int(multiscale_num_layers)),
                dropout=multiscale_dropout,
            )
            if self.use_multiscale_history
            else None
        )
        self.history_fusion = (
            MultiScaleHistoryFusion(d_model=d_model, dropout=multiscale_dropout)
            if self.use_multiscale_history
            else None
        )
        self.curve_encoder = (
            SensorHistoryCurveEncoder(
                d_model=d_model,
                hidden_dim=curve_hidden,
                dropout=dropout,
            )
            if self.use_curve_context
            else None
        )
        self.sensor_local_encoder = (
            SensorLocalHistoryEncoder(
                d_model=d_model,
                hidden_dim=sensor_local_hidden,
                dropout=dropout,
            )
            if self.use_sensor_local_branch
            else None
        )
        self.phase_proj = nn.Linear(self.phase_num_clusters, self.phase_emb_dim)
        self.phase_token_proj = nn.Linear(self.phase_num_clusters, d_model)
        self.decoder = SensorFutureDecoder(
            num_sensors=num_sensors,
            d_model=d_model,
            hidden_dim=decoder_hidden,
            dropout=dropout,
            phase_emb_dim=self.phase_emb_dim,
            head_variant=self.forecast_head_variant,
            use_sensor_local_branch=self.use_sensor_local_branch,
            num_future_bins=self.num_future_bins,
            bin_hidden=bin_hidden,
            use_onset_head=self.use_onset_head,
            onset_hidden=onset_hidden,
            use_curve_context=self.use_curve_context,
            use_change_pattern_branch=self.use_change_pattern_branch,
            change_branch_hidden=change_branch_hidden,
            use_control_aux_head=self.use_control_aux_head,
            use_phase_aware_moe_heads=use_phase_aware_moe_heads,
            phase_aware_moe_num_experts=phase_aware_moe_num_experts,
            phase_aware_moe_gate_hidden=phase_aware_moe_gate_hidden,
            phase_aware_moe_expert_hidden=phase_aware_moe_expert_hidden,
            phase_aware_moe_dropout=phase_aware_moe_dropout,
            phase_aware_moe_temperature=phase_aware_moe_temperature,
            phase_aware_moe_residual_scale=phase_aware_moe_residual_scale,
            phase_aware_moe_targets=phase_aware_moe_targets,
        )
        self.register_buffer(
            "control_sensor_mask",
            build_control_sensor_mask(
                sensor_names=sensor_names,
                sensor_patterns=control_sensor_patterns,
                num_sensors=num_sensors,
            ),
        )

    def _make_temporal_encoder(
        self,
        num_sensors: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        if self.temporal_backbone == "transformer":
            return CausalTemporalEncoder(
                num_sensors=num_sensors,
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        if self.temporal_backbone == "lstm":
            return LSTMTemporalEncoder(
                num_sensors=num_sensors,
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        raise ValueError(f"Unsupported temporal backbone: {self.temporal_backbone}")

    def _bounded(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bound_value_outputs:
            return x
        return torch.clamp(x, 0.0, 1.0)

    def _compute_history_curve_stats(
        self,
        hist_patch_last_value_seq: torch.Tensor,
        hist_patch_has_value_seq: torch.Tensor,
        hist_last_value: torch.Tensor,
        hist_has_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = hist_patch_last_value_seq.shape[1]
        x = torch.linspace(
            0.0,
            1.0,
            steps=seq_len,
            dtype=hist_patch_last_value_seq.dtype,
            device=hist_patch_last_value_seq.device,
        ).view(1, seq_len, 1)
        mask = hist_patch_has_value_seq.to(dtype=hist_patch_last_value_seq.dtype)
        weight_sum = mask.sum(dim=1)
        safe_weight_sum = weight_sum.clamp_min(1.0)

        mean_x = (mask * x).sum(dim=1) / safe_weight_sum
        mean_y = (mask * hist_patch_last_value_seq).sum(dim=1) / safe_weight_sum
        xc = x - mean_x.unsqueeze(1)
        yc = hist_patch_last_value_seq - mean_y.unsqueeze(1)
        numer = (mask * xc * yc).sum(dim=1)
        denom = (mask * xc * xc).sum(dim=1)
        valid_slope = (weight_sum >= 2.0) & (denom > 1e-6)
        trend_slope = torch.where(valid_slope, numer / denom.clamp_min(1e-6), torch.zeros_like(numer))

        recency = torch.linspace(
            1.0,
            2.0,
            steps=seq_len,
            dtype=hist_patch_last_value_seq.dtype,
            device=hist_patch_last_value_seq.device,
        ).view(1, seq_len, 1)
        recency_weight = mask * recency
        recency_sum = recency_weight.sum(dim=1)
        recent_mean = torch.where(
            recency_sum > 0.0,
            (recency_weight * hist_patch_last_value_seq).sum(dim=1) / recency_sum.clamp_min(1.0),
            hist_last_value,
        )
        recent_mean = torch.where(hist_has_value > 0.5, recent_mean, torch.zeros_like(recent_mean))
        trend_slope = trend_slope * (hist_has_value > 0.5).to(dtype=trend_slope.dtype)
        return trend_slope, recent_mean

    def _apply_prediction_structure(
        self,
        pred: Dict[str, torch.Tensor],
        hist_patch_last_value_seq: torch.Tensor,
        hist_patch_has_value_seq: torch.Tensor,
        hist_last_value: torch.Tensor,
        hist_has_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.value_only_head:
            pred["future_value_hat"] = self._bounded(pred["future_value_hat"])
            return pred
        if not self.use_anchored_decoder:
            pred["future_value_hat"] = self._bounded(pred["future_value_hat"])
            pred["future_mean_hat"] = self._bounded(pred["future_mean_hat"])
            if "future_onset_mean_hat" in pred:
                pred["future_onset_mean_hat"] = self._bounded(pred["future_onset_mean_hat"])
            if "future_bin_last_hat" in pred:
                pred["future_bin_last_hat"] = self._bounded(pred["future_bin_last_hat"])
            if "future_bin_mean_hat" in pred:
                pred["future_bin_mean_hat"] = self._bounded(pred["future_bin_mean_hat"])
            return pred

        raw_value = pred["future_value_hat"]
        raw_delta = pred["future_delta_hat"]
        raw_mean = pred["future_mean_hat"]
        raw_slope = pred["future_slope_hat"]

        trend_slope, recent_mean = self._compute_history_curve_stats(
            hist_patch_last_value_seq=hist_patch_last_value_seq,
            hist_patch_has_value_seq=hist_patch_has_value_seq,
            hist_last_value=hist_last_value,
            hist_has_value=hist_has_value,
        )
        trend_delta = self.horizon_scale * trend_slope
        delta_hat = trend_delta + self.delta_residual_scale * torch.tanh(raw_delta)
        value_anchor = hist_last_value + delta_hat
        value_hat = value_anchor + self.value_residual_scale * torch.tanh(raw_value)
        mean_anchor = 0.5 * (recent_mean + value_hat)
        mean_hat = mean_anchor + self.mean_residual_scale * torch.tanh(raw_mean)
        slope_hat = trend_slope + self.slope_residual_scale * torch.tanh(raw_slope)

        fallback_value = torch.sigmoid(raw_value)
        fallback_mean = torch.sigmoid(raw_mean)
        fallback_delta = torch.tanh(raw_delta)
        fallback_slope = torch.tanh(raw_slope)
        has_hist = hist_has_value > 0.5

        pred["future_delta_hat"] = torch.where(has_hist, delta_hat, fallback_delta)
        pred["future_value_hat"] = self._bounded(torch.where(has_hist, value_hat, fallback_value))
        pred["future_mean_hat"] = self._bounded(torch.where(has_hist, mean_hat, fallback_mean))
        pred["future_slope_hat"] = torch.where(has_hist, slope_hat, fallback_slope)
        if "future_onset_mean_hat" in pred:
            pred["future_onset_mean_hat"] = self._bounded(pred["future_onset_mean_hat"])
        if "future_bin_last_hat" in pred:
            pred["future_bin_last_hat"] = self._bounded(pred["future_bin_last_hat"])
        if "future_bin_mean_hat" in pred:
            pred["future_bin_mean_hat"] = self._bounded(pred["future_bin_mean_hat"])
        return pred

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.patch_encoder.set_value_normalization_stats(mean=mean, std=std)

    def encode_history(
        self,
        dt: torch.Tensor,
        sensor: torch.Tensor,
        value: torch.Tensor,
        event_mask: torch.Tensor,
        win_mask: torch.Tensor,
        regime_token_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_events = dt.shape
        flat_regime_context = None
        if regime_token_context is not None:
            flat_regime_context = regime_token_context.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
        sensor_tokens, global_tokens = self.patch_encoder(
            dt=dt.reshape(batch_size * seq_len, num_events),
            sensor=sensor.reshape(batch_size * seq_len, num_events),
            value=value.reshape(batch_size * seq_len, num_events),
            event_mask=event_mask.reshape(batch_size * seq_len, num_events),
            regime_context=flat_regime_context,
        )
        sensor_tokens = sensor_tokens.view(batch_size, seq_len, self.num_sensors, -1)
        global_tokens = global_tokens.view(batch_size, seq_len, -1)
        fine_sensor_history, fine_global_history = self.temporal_encoder(
            sensor_tokens=sensor_tokens,
            global_tokens=global_tokens,
            win_mask=win_mask,
            regime_context=regime_token_context,
        )
        if not self.use_multiscale_history or self.coarse_temporal_encoder is None or self.history_fusion is None:
            return fine_sensor_history, fine_global_history

        coarse_sensor_tokens = sensor_tokens[:, :: self.multiscale_stride]
        coarse_global_tokens = global_tokens[:, :: self.multiscale_stride]
        coarse_win_mask = win_mask[:, :: self.multiscale_stride]
        coarse_sensor_history, coarse_global_history = self.coarse_temporal_encoder(
            sensor_tokens=coarse_sensor_tokens,
            global_tokens=coarse_global_tokens,
            win_mask=coarse_win_mask,
            regime_context=regime_token_context,
        )
        return self.history_fusion(
            fine_sensor_history=fine_sensor_history,
            fine_global_history=fine_global_history,
            coarse_sensor_history=coarse_sensor_history,
            coarse_global_history=coarse_global_history,
        )

    def _safe_masked_mean(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weight = valid_mask.to(dtype=x.dtype)
        denom = weight.sum()
        if float(denom.item()) <= 0.0:
            return x.new_zeros(())
        return (x * weight).sum() / denom

    def _pointwise_regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_huber_loss:
            return F.smooth_l1_loss(pred, target, beta=self.huber_beta, reduction="none")
        return torch.abs(pred - target)

    def _phase_context(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        if self.use_phase_context and "phase_prob" in batch:
            phase_prob = batch["phase_prob"].to(device)
            return self.phase_proj(phase_prob)
        batch_size = batch["hist_last_value"].shape[0]
        return torch.zeros((batch_size, self.phase_emb_dim), dtype=torch.float32, device=device)

    def _phase_token_context(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        if self.use_phase_context and "phase_prob" in batch:
            phase_prob = batch["phase_prob"].to(device)
            return self.phase_token_proj(phase_prob)
        batch_size = batch["hist_last_value"].shape[0]
        return torch.zeros((batch_size, self.patch_encoder.d_model), dtype=torch.float32, device=device)

    def _compute_curve_summary(
        self,
        hist_patch_last_value_seq: torch.Tensor,
        hist_patch_has_value_seq: torch.Tensor,
        hist_last_value: torch.Tensor,
        hist_has_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_curve_context:
            zeros = torch.zeros_like(hist_last_value)
            return zeros, zeros, zeros

        mask = hist_patch_has_value_seq > 0.5
        filled = hist_patch_last_value_seq.clone()
        for t in range(1, hist_patch_last_value_seq.shape[1]):
            filled[:, t] = torch.where(mask[:, t], filled[:, t], filled[:, t - 1])
        has_any = mask.any(dim=1)
        recency = torch.linspace(
            1.0,
            2.0,
            steps=hist_patch_last_value_seq.shape[1],
            dtype=hist_patch_last_value_seq.dtype,
            device=hist_patch_last_value_seq.device,
        ).view(1, hist_patch_last_value_seq.shape[1], 1)
        weight = mask.to(dtype=hist_patch_last_value_seq.dtype) * recency
        weight_sum = weight.sum(dim=1)
        recent_mean = torch.where(
            weight_sum > 0.0,
            (weight * hist_patch_last_value_seq).sum(dim=1) / weight_sum.clamp_min(1.0),
            hist_last_value,
        )
        if hist_patch_last_value_seq.shape[1] >= 2:
            recent_delta = filled[:, -1] - filled[:, -2]
        else:
            recent_delta = torch.zeros_like(hist_last_value)

        seq_min = torch.where(mask, hist_patch_last_value_seq, torch.full_like(hist_patch_last_value_seq, float("inf"))).min(dim=1).values
        seq_max = torch.where(mask, hist_patch_last_value_seq, torch.full_like(hist_patch_last_value_seq, float("-inf"))).max(dim=1).values
        recent_range = torch.where(
            has_any,
            seq_max - seq_min,
            torch.zeros_like(hist_last_value),
        )
        recent_mean = torch.where(hist_has_value > 0.5, recent_mean, torch.zeros_like(recent_mean))
        recent_delta = torch.where(hist_has_value > 0.5, recent_delta, torch.zeros_like(recent_delta))
        recent_range = torch.where(hist_has_value > 0.5, recent_range, torch.zeros_like(recent_range))
        return recent_mean, recent_delta, recent_range

    def compute_losses_and_scores(
        self,
        pred: Dict[str, torch.Tensor],
        future_mask: torch.Tensor,
        future_last_value: torch.Tensor,
        future_delta_value: torch.Tensor,
        future_delta_mask: torch.Tensor,
        future_diff_mask: torch.Tensor,
        future_mean_value: torch.Tensor,
        future_var_value: torch.Tensor,
        future_range_value: torch.Tensor,
        future_log_count: torch.Tensor,
        future_is_constant: torch.Tensor,
        future_abs_diff_sum: torch.Tensor,
        future_zero_diff_ratio: torch.Tensor,
        future_slope: torch.Tensor,
        future_control_change: torch.Tensor,
        future_bin_mask: torch.Tensor,
        future_bin_last_value: torch.Tensor,
        future_bin_mean_value: torch.Tensor,
        future_bin_log_count: torch.Tensor,
        future_onset_mask: torch.Tensor,
        future_onset_mean_value: torch.Tensor,
        future_onset_delta_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        future_mask_logit = pred["future_mask_logit"]
        future_value_hat = pred["future_value_hat"]
        future_delta_hat = pred["future_delta_hat"]
        future_mean_hat = pred["future_mean_hat"]
        future_var_hat = pred["future_var_hat"]
        future_range_hat = pred["future_range_hat"]
        future_log_count_hat = pred["future_log_count_hat"]
        future_constant_logit = pred["future_constant_logit"]
        future_abs_diff_hat = pred["future_abs_diff_hat"]
        future_zero_diff_logit = pred["future_zero_diff_logit"]
        future_slope_hat = pred["future_slope_hat"]
        future_control_change_hat = pred["future_control_change_hat"]
        future_bin_mask_logit = pred["future_bin_mask_logit"]
        future_bin_last_hat = pred["future_bin_last_hat"]
        future_bin_mean_hat = pred["future_bin_mean_hat"]
        future_bin_log_count_hat = pred["future_bin_log_count_hat"]
        future_onset_mask_logit = pred["future_onset_mask_logit"]
        future_onset_mean_hat = pred["future_onset_mean_hat"]
        future_onset_delta_hat = pred["future_onset_delta_hat"]

        valid = future_mask > 0.5
        delta_valid = future_delta_mask > 0.5
        diff_valid = future_diff_mask > 0.5
        control_valid = valid & (self.control_sensor_mask.view(1, -1).to(device=future_mask.device) > 0.5)
        bin_valid = future_bin_mask > 0.5
        onset_valid = future_onset_mask > 0.5
        onset_delta_valid = onset_valid & valid
        loss_mask = F.binary_cross_entropy_with_logits(future_mask_logit, future_mask, reduction="mean")
        loss_value = self._safe_masked_mean(self._pointwise_regression_loss(future_value_hat, future_last_value), valid)
        loss_delta = self._safe_masked_mean(
            self._pointwise_regression_loss(future_delta_hat, future_delta_value),
            delta_valid,
        )
        loss_mean = self._safe_masked_mean(self._pointwise_regression_loss(future_mean_hat, future_mean_value), valid)
        loss_var = self._safe_masked_mean(
            self._pointwise_regression_loss(future_var_hat, future_var_value),
            diff_valid,
        )
        loss_range = self._safe_masked_mean(
            self._pointwise_regression_loss(future_range_hat, future_range_value),
            diff_valid,
        )
        loss_count = self._safe_masked_mean(self._pointwise_regression_loss(future_log_count_hat, future_log_count), valid)
        loss_constant = self._safe_masked_mean(
            F.binary_cross_entropy_with_logits(future_constant_logit, future_is_constant, reduction="none"),
            diff_valid,
        )
        loss_abs_diff = self._safe_masked_mean(
            self._pointwise_regression_loss(future_abs_diff_hat, future_abs_diff_sum),
            diff_valid,
        )
        loss_zero_diff = self._safe_masked_mean(
            self._pointwise_regression_loss(torch.sigmoid(future_zero_diff_logit), future_zero_diff_ratio),
            diff_valid,
        )
        loss_slope = self._safe_masked_mean(
            self._pointwise_regression_loss(future_slope_hat, future_slope),
            diff_valid,
        )
        loss_control_change = self._safe_masked_mean(
            self._pointwise_regression_loss(future_control_change_hat, future_control_change),
            control_valid,
        )
        loss_bin_mask = (
            F.binary_cross_entropy_with_logits(future_bin_mask_logit, future_bin_mask, reduction="mean")
            if future_bin_mask_logit.numel() > 0
            else future_mask_logit.new_zeros(())
        )
        loss_bin_last = self._safe_masked_mean(
            self._pointwise_regression_loss(future_bin_last_hat, future_bin_last_value),
            bin_valid,
        ) if future_bin_last_hat.numel() > 0 else future_mask_logit.new_zeros(())
        loss_bin_mean = self._safe_masked_mean(
            self._pointwise_regression_loss(future_bin_mean_hat, future_bin_mean_value),
            bin_valid,
        ) if future_bin_mean_hat.numel() > 0 else future_mask_logit.new_zeros(())
        loss_bin_count = self._safe_masked_mean(
            self._pointwise_regression_loss(future_bin_log_count_hat, future_bin_log_count),
            bin_valid,
        ) if future_bin_log_count_hat.numel() > 0 else future_mask_logit.new_zeros(())
        loss_onset_mask = (
            F.binary_cross_entropy_with_logits(future_onset_mask_logit, future_onset_mask, reduction="mean")
            if future_onset_mask_logit.numel() > 0
            else future_mask_logit.new_zeros(())
        )
        loss_onset_mean = self._safe_masked_mean(
            self._pointwise_regression_loss(future_onset_mean_hat, future_onset_mean_value),
            onset_valid,
        ) if future_onset_mean_hat.numel() > 0 else future_mask_logit.new_zeros(())
        loss_onset_delta = self._safe_masked_mean(
            self._pointwise_regression_loss(future_onset_delta_hat, future_onset_delta_value),
            onset_delta_valid,
        ) if future_onset_delta_hat.numel() > 0 else future_mask_logit.new_zeros(())
        loss = (
            self.loss_weight_mask * loss_mask
            + self.loss_weight_value * loss_value
            + self.loss_weight_delta * loss_delta
            + self.loss_weight_mean * loss_mean
            + self.loss_weight_var * loss_var
            + self.loss_weight_range * loss_range
            + self.loss_weight_count * loss_count
            + self.loss_weight_constant * loss_constant
            + self.loss_weight_abs_diff * loss_abs_diff
            + self.loss_weight_zero_diff * loss_zero_diff
            + self.loss_weight_slope * loss_slope
            + self.loss_weight_control_change * loss_control_change
            + self.loss_weight_bin_mask * loss_bin_mask
            + self.loss_weight_bin_last * loss_bin_last
            + self.loss_weight_bin_mean * loss_bin_mean
            + self.loss_weight_bin_count * loss_bin_count
            + self.loss_weight_onset_mask * loss_onset_mask
            + self.loss_weight_onset_mean * loss_onset_mean
            + self.loss_weight_onset_delta * loss_onset_delta
        )

        pred_mask_prob = torch.sigmoid(future_mask_logit)
        pred_constant_prob = torch.sigmoid(future_constant_logit)
        pred_zero_diff_ratio = torch.sigmoid(future_zero_diff_logit)
        pred_count = torch.expm1(future_log_count_hat).clamp_min(0.0)
        pred_bin_mask_prob = torch.sigmoid(future_bin_mask_logit) if future_bin_mask_logit.numel() > 0 else future_bin_mask_logit
        pred_bin_count = torch.expm1(future_bin_log_count_hat).clamp_min(0.0) if future_bin_log_count_hat.numel() > 0 else future_bin_log_count_hat
        pred_onset_mask_prob = torch.sigmoid(future_onset_mask_logit) if future_onset_mask_logit.numel() > 0 else future_onset_mask_logit
        resid_mask = torch.abs(pred_mask_prob - future_mask)
        resid_value = torch.abs(future_value_hat - future_last_value) * valid.to(dtype=future_value_hat.dtype)
        resid_delta = torch.abs(future_delta_hat - future_delta_value) * delta_valid.to(dtype=future_delta_hat.dtype)
        resid_mean = torch.abs(future_mean_hat - future_mean_value) * valid.to(dtype=future_mean_hat.dtype)
        resid_var = torch.abs(future_var_hat - future_var_value) * diff_valid.to(dtype=future_var_hat.dtype)
        resid_range = torch.abs(future_range_hat - future_range_value) * diff_valid.to(dtype=future_range_hat.dtype)
        resid_count = torch.abs(future_log_count_hat - future_log_count) * valid.to(dtype=future_log_count_hat.dtype)
        resid_constant = torch.abs(pred_constant_prob - future_is_constant) * diff_valid.to(dtype=pred_constant_prob.dtype)
        resid_abs_diff = torch.abs(future_abs_diff_hat - future_abs_diff_sum) * diff_valid.to(dtype=future_abs_diff_hat.dtype)
        resid_zero_diff = torch.abs(pred_zero_diff_ratio - future_zero_diff_ratio) * diff_valid.to(dtype=pred_zero_diff_ratio.dtype)
        resid_slope = torch.abs(future_slope_hat - future_slope) * diff_valid.to(dtype=future_slope_hat.dtype)
        resid_control_change = torch.abs(future_control_change_hat - future_control_change) * control_valid.to(dtype=future_control_change_hat.dtype)
        resid_bin_mask = (
            torch.abs(pred_bin_mask_prob - future_bin_mask) * future_bin_mask.new_ones(future_bin_mask.shape)
            if future_bin_mask_logit.numel() > 0
            else future_bin_mask_logit
        )
        resid_bin_last = (
            torch.abs(future_bin_last_hat - future_bin_last_value) * bin_valid.to(dtype=future_bin_last_hat.dtype)
            if future_bin_last_hat.numel() > 0
            else future_bin_last_hat
        )
        resid_bin_mean = (
            torch.abs(future_bin_mean_hat - future_bin_mean_value) * bin_valid.to(dtype=future_bin_mean_hat.dtype)
            if future_bin_mean_hat.numel() > 0
            else future_bin_mean_hat
        )
        resid_bin_count = (
            torch.abs(future_bin_log_count_hat - future_bin_log_count) * bin_valid.to(dtype=future_bin_log_count_hat.dtype)
            if future_bin_log_count_hat.numel() > 0
            else future_bin_log_count_hat
        )
        resid_onset_mask = torch.abs(pred_onset_mask_prob - future_onset_mask)
        resid_onset_mean = torch.abs(future_onset_mean_hat - future_onset_mean_value) * onset_valid.to(dtype=future_onset_mean_hat.dtype)
        resid_onset_delta = torch.abs(future_onset_delta_hat - future_onset_delta_value) * onset_delta_valid.to(dtype=future_onset_delta_hat.dtype)
        value_sensor_score = resid_value
        control_change_sensor_score = resid_control_change
        sensor_score = (
            self.score_weight_mask * resid_mask
            + self.score_weight_value * resid_value
            + self.score_weight_delta * resid_delta
            + self.score_weight_mean * resid_mean
            + self.score_weight_var * resid_var
            + self.score_weight_range * resid_range
            + self.score_weight_count * resid_count
            + self.score_weight_constant * resid_constant
            + self.score_weight_abs_diff * resid_abs_diff
            + self.score_weight_zero_diff * resid_zero_diff
            + self.score_weight_slope * resid_slope
            + self.score_weight_control_change * resid_control_change
            + self.score_weight_onset_mean * resid_onset_mean
            + self.score_weight_onset_delta * resid_onset_delta
        )
        denom = valid.to(dtype=sensor_score.dtype).sum(dim=1)
        total_score = torch.where(
            denom > 0,
            (sensor_score * valid.to(dtype=sensor_score.dtype)).sum(dim=1) / denom.clamp_min(1.0),
            torch.zeros_like(denom),
        )
        control_denom = control_valid.to(dtype=control_change_sensor_score.dtype).sum(dim=1)
        control_change_total_score = torch.where(
            control_denom > 0,
            (control_change_sensor_score * control_valid.to(dtype=control_change_sensor_score.dtype)).sum(dim=1)
            / control_denom.clamp_min(1.0),
            torch.zeros_like(control_denom),
        )
        value_total_score = torch.where(
            denom > 0,
            (value_sensor_score * valid.to(dtype=value_sensor_score.dtype)).sum(dim=1) / denom.clamp_min(1.0),
            torch.zeros_like(denom),
        )
        return {
            "loss": loss,
            "loss_mask": loss_mask,
            "loss_value": loss_value,
            "loss_delta": loss_delta,
            "loss_mean": loss_mean,
            "loss_var": loss_var,
            "loss_range": loss_range,
            "loss_count": loss_count,
            "loss_constant": loss_constant,
            "loss_abs_diff": loss_abs_diff,
            "loss_zero_diff": loss_zero_diff,
            "loss_slope": loss_slope,
            "loss_control_change": loss_control_change,
            "loss_bin_mask": loss_bin_mask,
            "loss_bin_last": loss_bin_last,
            "loss_bin_mean": loss_bin_mean,
            "loss_bin_count": loss_bin_count,
            "loss_onset_mask": loss_onset_mask,
            "loss_onset_mean": loss_onset_mean,
            "loss_onset_delta": loss_onset_delta,
            "pred_mask_prob": pred_mask_prob,
            "pred_constant_prob": pred_constant_prob,
            "pred_zero_diff_ratio": pred_zero_diff_ratio,
            "pred_count": pred_count,
            "pred_bin_mask_prob": pred_bin_mask_prob,
            "pred_bin_count": pred_bin_count,
            "pred_onset_mask_prob": pred_onset_mask_prob,
            "resid_mask": resid_mask,
            "resid_value": resid_value,
            "resid_delta": resid_delta,
            "resid_mean": resid_mean,
            "resid_var": resid_var,
            "resid_range": resid_range,
            "resid_count": resid_count,
            "resid_constant": resid_constant,
            "resid_abs_diff": resid_abs_diff,
            "resid_zero_diff": resid_zero_diff,
            "resid_slope": resid_slope,
            "resid_control_change": resid_control_change,
            "resid_bin_mask": resid_bin_mask,
            "resid_bin_last": resid_bin_last,
            "resid_bin_mean": resid_bin_mean,
            "resid_bin_count": resid_bin_count,
            "resid_onset_mask": resid_onset_mask,
            "resid_onset_mean": resid_onset_mean,
            "resid_onset_delta": resid_onset_delta,
            "value_sensor_score": value_sensor_score,
            "control_change_sensor_score": control_change_sensor_score,
            "sensor_score": sensor_score,
            "value_total_score": value_total_score,
            "control_change_total_score": control_change_total_score,
            "total_score": total_score,
        }

    def forward(self, batch: Dict[str, torch.Tensor], return_details: bool = False) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        dt = batch["dt"].to(device)
        sensor = batch["sensor"].to(device)
        value = batch["value"].to(device)
        event_mask = batch["event_mask"].to(device)
        win_mask = batch["win_mask"].to(device)
        hist_last_value = batch["hist_last_value"].to(device)
        hist_has_value = batch["hist_has_value"].to(device)
        hist_patch_last_value_seq = batch["hist_patch_last_value_seq"].to(device)
        hist_patch_has_value_seq = batch["hist_patch_has_value_seq"].to(device)
        future_mask = batch["future_mask"].to(device)
        future_last_value = batch["future_last_value"].to(device)
        future_delta_value = batch["future_delta_value"].to(device)
        future_delta_mask = batch["future_delta_mask"].to(device)
        future_diff_mask = batch["future_diff_mask"].to(device)
        future_mean_value = batch["future_mean_value"].to(device)
        future_var_value = batch["future_var_value"].to(device)
        future_range_value = batch["future_range_value"].to(device)
        future_log_count = batch["future_log_count"].to(device)
        future_is_constant = batch["future_is_constant"].to(device)
        future_abs_diff_sum = batch["future_abs_diff_sum"].to(device)
        future_zero_diff_ratio = batch["future_zero_diff_ratio"].to(device)
        future_slope = batch["future_slope"].to(device)
        future_bin_mask = batch["future_bin_mask"].to(device)
        future_bin_last_value = batch["future_bin_last_value"].to(device)
        future_bin_mean_value = batch["future_bin_mean_value"].to(device)
        future_bin_log_count = batch["future_bin_log_count"].to(device)
        future_onset_mask = batch["future_onset_mask"].to(device)
        future_onset_mean_value = batch["future_onset_mean_value"].to(device)
        future_onset_delta_value = batch["future_onset_delta_value"].to(device)
        future_control_change = (
            torch.abs(future_delta_value) * future_delta_mask
            + 0.50 * future_abs_diff_sum * future_diff_mask
            + 0.35 * torch.abs(future_slope) * future_diff_mask
            + 0.25 * future_range_value * future_diff_mask
        ) * self.control_sensor_mask.view(1, -1).to(device=device, dtype=future_mask.dtype)

        phase_context = self._phase_context(batch=batch, device=device)
        phase_token_context = self._phase_token_context(batch=batch, device=device)
        sensor_history, global_history = self.encode_history(
            dt=dt,
            sensor=sensor,
            value=value,
            event_mask=event_mask,
            win_mask=win_mask,
            regime_token_context=phase_token_context,
        )
        sensor_local_context = (
            self.sensor_local_encoder(
                hist_patch_last_value_seq=hist_patch_last_value_seq,
                hist_patch_has_value_seq=hist_patch_has_value_seq,
            )
            if self.sensor_local_encoder is not None
            else None
        )
        curve_context = (
            self.curve_encoder(hist_patch_last_value_seq=hist_patch_last_value_seq, hist_patch_has_value_seq=hist_patch_has_value_seq)
            if self.curve_encoder is not None
            else None
        )
        recent_mean, recent_delta, recent_range = self._compute_curve_summary(
            hist_patch_last_value_seq=hist_patch_last_value_seq,
            hist_patch_has_value_seq=hist_patch_has_value_seq,
            hist_last_value=hist_last_value,
            hist_has_value=hist_has_value,
        )
        pred = self.decoder(
            sensor_history=sensor_history,
            global_history=global_history,
            hist_last_value=hist_last_value,
            hist_has_value=hist_has_value,
            phase_context=phase_context,
            sensor_local_context=sensor_local_context,
            curve_context=curve_context,
            recent_mean=recent_mean,
            recent_delta=recent_delta,
            recent_range=recent_range,
        )
        pred = self._apply_prediction_structure(
            pred=pred,
            hist_patch_last_value_seq=hist_patch_last_value_seq,
            hist_patch_has_value_seq=hist_patch_has_value_seq,
            hist_last_value=hist_last_value,
            hist_has_value=hist_has_value,
        )
        out = self.compute_losses_and_scores(
            pred=pred,
            future_mask=future_mask,
            future_last_value=future_last_value,
            future_delta_value=future_delta_value,
            future_delta_mask=future_delta_mask,
            future_diff_mask=future_diff_mask,
            future_mean_value=future_mean_value,
            future_var_value=future_var_value,
            future_range_value=future_range_value,
            future_log_count=future_log_count,
            future_is_constant=future_is_constant,
            future_abs_diff_sum=future_abs_diff_sum,
            future_zero_diff_ratio=future_zero_diff_ratio,
            future_slope=future_slope,
            future_control_change=future_control_change,
            future_bin_mask=future_bin_mask,
            future_bin_last_value=future_bin_last_value,
            future_bin_mean_value=future_bin_mean_value,
            future_bin_log_count=future_bin_log_count,
            future_onset_mask=future_onset_mask,
            future_onset_mean_value=future_onset_mean_value,
            future_onset_delta_value=future_onset_delta_value,
        )
        out.update(pred)
        out["phase_context"] = phase_context
        out["phase_token_context"] = phase_token_context
        out["future_control_change_target"] = future_control_change
        if return_details:
            out["sensor_history"] = sensor_history
            out["global_history"] = global_history
        return out
