# -*- coding: utf-8 -*-
"""Spatial-Temporal Graph TCN model (copied from F:/STGTCN/model/st_graph_tcn.py).

Two variants:
- STGraphTCN                : TCN temporal encoder + SpatialEncoder (single fuse)
- STGraphTCNParallelCross   : TCN temporal branch + short-window spatial branch + cross attention fusion

Forward signature:
    y, aux = model(x, a_stat, m_mask, node_index=None, short_patch=5, zmask=None)
where x : (B, T, N, F), y : (B, horizon, N, F).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_bool_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((length, length), dtype=torch.bool, device=device).triu(1)


class PartialCausalConv1d(nn.Conv1d):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        self.ks = int(kernel_size)
        self.dl = int(dilation)
        pad = (self.ks - 1) * self.dl
        super().__init__(in_ch, out_ch, self.ks, padding=pad, dilation=self.dl, bias=True)
        self.register_buffer("ones_kernel", torch.ones(1, 1, self.ks))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = super().forward(x * m)
        denom = F.conv1d(m, self.ones_kernel, padding=self.padding[0], dilation=self.dilation[0])
        cut = (self.ks - 1) * self.dl
        if cut > 0:
            y = y[..., :-cut]
            denom = denom[..., :-cut]
        y = y / denom.clamp_min(1e-6)
        return y, (denom > 0).float()


class TCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "pconv": PartialCausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=2 ** i),
                "drop": nn.Dropout(dropout),
                "pw": nn.Conv1d(d_model, d_model, kernel_size=1),
            })
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2).contiguous()
        m = m_t.unsqueeze(1)
        for blk in self.layers:
            res = h
            y, m = blk["pconv"](h, m)
            y = F.relu_(y)
            y = blk["pw"](blk["drop"](y))
            h = F.relu_(y + res)
        h = h.transpose(1, 2).contiguous()
        return self.norm(h), m.squeeze(1)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        tcn_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TCNBlock(d_model, kernel_size=kernel_size, n_layers=tcn_layers, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.trans = nn.TransformerEncoder(enc, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_in: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = h_in.shape
        h, m_out = self.tcn(h_in, m_t)
        h = self.trans(
            h,
            mask=causal_bool_mask(steps, h_in.device),
            src_key_padding_mask=(m_out < 0.5),
        )
        return self.norm(h), m_out


class TemporalEncoderTCNOnly(nn.Module):
    def __init__(
        self,
        d_model: int,
        tcn_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size=kernel_size, n_layers=tcn_layers, dropout=dropout)
            for _ in range(max(1, int(num_blocks)))
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_in: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h_in
        m_out = m_t
        for block in self.blocks:
            h, m_out = block(h, m_out)
        return self.norm(h), m_out


class DynamicGraphAttention(nn.Module):
    def __init__(self, d_model: int, eta: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.eta = float(eta)

    def forward(self, h_nodes: torch.Tensor, a_stat: torch.Tensor, m_mask: torch.Tensor) -> torch.Tensor:
        _, num_nodes, d_model = h_nodes.shape
        logits = torch.matmul(self.q(h_nodes), self.k(h_nodes).transpose(-1, -2)) / math.sqrt(d_model)
        bias = torch.logit(a_stat.clamp(1e-3, 1 - 1e-3))
        logits = logits + self.eta * bias.unsqueeze(0)
        eye = torch.eye(num_nodes, device=h_nodes.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float("-inf"))
        logits = logits.masked_fill((m_mask <= 0).unsqueeze(0), float("-inf"))
        return torch.softmax(logits, dim=-1)


class SpatialEncoder(nn.Module):
    def __init__(self, d_model: int, eta: float = 1.0, beta: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.dyn = DynamicGraphAttention(d_model, eta=eta)
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.beta = float(beta)

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_graph_ctx: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_dyn = self.dyn(h_graph_ctx, a_stat, m_mask)
        a_fuse = self.beta * a_dyn + (1.0 - self.beta) * a_stat.unsqueeze(0)
        a_fuse = a_fuse * m_mask.unsqueeze(0)
        a_fuse = a_fuse / a_fuse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        h_msg = torch.matmul(a_fuse, h_nodes)
        h_out = self.norm(h_nodes + self.drop(self.lin(h_msg)))
        return h_out, a_fuse


class CrossAttentionFuse(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, weights = self.attn(query=query, key=key_value, value=key_value, need_weights=True)
        return self.norm(query + self.drop(out)), weights


class SpatialShortWindowEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        eta: float = 1.0,
        beta: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial = SpatialEncoder(d_model=d_model, eta=eta, beta=beta, dropout=dropout)
        self.temporal = TemporalEncoderTCNOnly(
            d_model=d_model,
            tcn_layers=2,
            kernel_size=3,
            dropout=dropout,
            num_blocks=1,
        )
        self.fuse_t2s = CrossAttentionFuse(d_model=d_model, nhead=nhead, dropout=dropout)
        self.fuse_s2t = CrossAttentionFuse(d_model=d_model, nhead=nhead, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_short: torch.Tensor,
        z_short: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, steps, num_nodes, d_model = h_short.shape
        spatial_seq = []
        a_seq = []
        for t in range(steps):
            h_t = h_short[:, t]
            h_spatial_t, a_fuse_t = self.spatial(h_t, h_t, a_stat, m_mask)
            spatial_seq.append(h_spatial_t)
            a_seq.append(a_fuse_t)
        h_spatial = torch.stack(spatial_seq, dim=1)
        a_fuse_t = torch.stack(a_seq, dim=1)

        z_short_exp = z_short.float().unsqueeze(-1)
        h_spatial_ctx = (h_spatial * z_short_exp).sum(dim=1) / z_short_exp.sum(dim=1).clamp_min(1.0)

        h_spatial_flat = h_spatial.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, d_model)
        m_short = z_short.reshape(batch * num_nodes, steps)
        h_spatial_temporal, _ = self.temporal(h_spatial_flat, m_short)
        h_spatial_temporal = h_spatial_temporal.view(batch, num_nodes, steps, d_model).permute(0, 2, 1, 3).contiguous()
        h_spatial_last = h_spatial_temporal[:, -1]

        return self.out_norm(h_spatial_ctx + h_spatial_last), {
            "A_fuse_t": a_fuse_t,
            "H_spatial_seq": h_spatial_temporal,
            "H_spatial_ctx": h_spatial_ctx,
        }


class STGraphTCN(nn.Module):
    def __init__(
        self,
        num_nodes_hint: Optional[int],
        in_feat: int,
        d_model: int = 64,
        short_kernel: int = 5,
        nhead: int = 4,
        tcn_layers: int = 5,
        dropout: float = 0.1,
        eta: float = 1.0,
        beta: float = 0.5,
        out_feat: Optional[int] = None,
        horizon: int = 1,
        temporal_encoder_type: str = "tcn_only",
    ):
        super().__init__()
        self.in_feat = int(in_feat)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.out_feat = int(out_feat) if out_feat is not None else self.in_feat
        self._num_nodes_hint = num_nodes_hint
        self._short_kernel = int(short_kernel)
        self.temporal_encoder_type = str(temporal_encoder_type).lower()

        self.in_proj = nn.Linear(self.in_feat, self.d_model)
        temporal_kernel = max(3, min(self._short_kernel, 9))
        if self.temporal_encoder_type == "tcn_transformer":
            self.temporal = TemporalEncoder(
                d_model=self.d_model,
                nhead=nhead,
                tcn_layers=tcn_layers,
                kernel_size=temporal_kernel,
                dropout=dropout,
            )
        else:
            self.temporal = TemporalEncoderTCNOnly(
                d_model=self.d_model,
                tcn_layers=tcn_layers,
                kernel_size=temporal_kernel,
                dropout=dropout,
                num_blocks=2,
            )
        self.spatial = SpatialEncoder(self.d_model, eta=eta, beta=beta, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon * self.out_feat),
        )

    def forward(
        self,
        x: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        node_index: Optional[torch.Tensor] = None,
        short_patch: int = 5,
        zmask: Optional[torch.Tensor] = None,
    ):
        del node_index
        batch, steps, num_nodes, _ = x.shape
        if zmask is None:
            zmask = torch.ones(batch, steps, num_nodes, device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        h_flat = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, self.d_model)
        m_t = zmask.float().reshape(batch * num_nodes, steps)
        h_flat, _ = self.temporal(h_flat, m_t)
        h = h_flat.view(batch, num_nodes, steps, self.d_model).permute(0, 2, 1, 3).contiguous()

        window = max(1, min(int(short_patch), steps))
        h_short = h[:, -window:]
        z_short = zmask[:, -window:].float().unsqueeze(-1)
        h_graph_ctx = (h_short * z_short).sum(dim=1) / z_short.sum(dim=1).clamp_min(1.0)
        h_last = h[:, -1]
        h_last, a_fuse = self.spatial(h_last, h_graph_ctx, a_stat, m_mask)
        y = self.pred_head(h_last)
        y = y.view(batch, num_nodes, self.horizon, self.out_feat).permute(0, 2, 1, 3).contiguous()

        aux: Dict[str, torch.Tensor] = {
            "A_fuse": a_fuse,
            "A_fuse_t": a_fuse.unsqueeze(1),
            "H_last": h_last,
            "H_graph_ctx": h_graph_ctx,
            "short_window_len": torch.tensor(window, device=x.device),
        }
        return y, aux


class STGraphTCNParallelCross(nn.Module):
    def __init__(
        self,
        num_nodes_hint: Optional[int],
        in_feat: int,
        d_model: int = 64,
        short_kernel: int = 5,
        nhead: int = 4,
        tcn_layers: int = 5,
        dropout: float = 0.1,
        eta: float = 1.0,
        beta: float = 0.5,
        out_feat: Optional[int] = None,
        horizon: int = 1,
        temporal_encoder_type: str = "parallel_cross_attn",
    ):
        super().__init__()
        del temporal_encoder_type
        self.in_feat = int(in_feat)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.out_feat = int(out_feat) if out_feat is not None else self.in_feat
        self._num_nodes_hint = num_nodes_hint
        self._short_kernel = int(short_kernel)

        self.in_proj = nn.Linear(self.in_feat, self.d_model)
        temporal_kernel = max(3, min(self._short_kernel, 9))
        self.temporal = TemporalEncoderTCNOnly(
            d_model=self.d_model,
            tcn_layers=tcn_layers,
            kernel_size=temporal_kernel,
            dropout=dropout,
            num_blocks=2,
        )
        self.spatial_short = SpatialShortWindowEncoder(
            d_model=self.d_model,
            nhead=nhead,
            eta=eta,
            beta=beta,
            dropout=dropout,
        )
        self.temporal_to_spatial = CrossAttentionFuse(d_model=self.d_model, nhead=nhead, dropout=dropout)
        self.spatial_to_temporal = CrossAttentionFuse(d_model=self.d_model, nhead=nhead, dropout=dropout)
        self.fuse_norm = nn.LayerNorm(self.d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon * self.out_feat),
        )

    def forward(
        self,
        x: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        node_index: Optional[torch.Tensor] = None,
        short_patch: int = 5,
        zmask: Optional[torch.Tensor] = None,
    ):
        del node_index
        batch, steps, num_nodes, _ = x.shape
        if zmask is None:
            zmask = torch.ones(batch, steps, num_nodes, device=x.device, dtype=x.dtype)

        h = self.in_proj(x)

        h_flat = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, self.d_model)
        m_t = zmask.float().reshape(batch * num_nodes, steps)
        h_time_flat, _ = self.temporal(h_flat, m_t)
        h_time = h_time_flat.view(batch, num_nodes, steps, self.d_model).permute(0, 2, 1, 3).contiguous()
        h_time_last = h_time[:, -1]

        window = max(1, steps // 4)
        h_short = h[:, -window:]
        z_short = zmask[:, -window:].float()
        spatial_out, spatial_aux = self.spatial_short(h_short, z_short, a_stat, m_mask)
        h_spatial_seq = spatial_aux["H_spatial_seq"]

        q_time = h_time_last.reshape(batch * num_nodes, 1, self.d_model)
        kv_spatial = h_spatial_seq.permute(0, 2, 1, 3).reshape(batch * num_nodes, window, self.d_model)
        h_time_cross, t2s_attn = self.temporal_to_spatial(q_time, kv_spatial)

        q_spatial = spatial_out.reshape(batch * num_nodes, 1, self.d_model)
        kv_time = h_time.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, self.d_model)
        h_spatial_cross, s2t_attn = self.spatial_to_temporal(q_spatial, kv_time)

        h_fuse = 0.5 * (
            h_time_cross.reshape(batch, num_nodes, self.d_model) +
            h_spatial_cross.reshape(batch, num_nodes, self.d_model)
        )
        h_fuse = self.fuse_norm(h_fuse + h_time_last + spatial_out)

        y = self.pred_head(h_fuse)
        y = y.view(batch, num_nodes, self.horizon, self.out_feat).permute(0, 2, 1, 3).contiguous()

        aux: Dict[str, torch.Tensor] = {
            "A_fuse": spatial_aux["A_fuse_t"][:, -1],
            "A_fuse_t": spatial_aux["A_fuse_t"],
            "H_last": h_fuse,
            "H_graph_ctx": spatial_aux["H_spatial_ctx"],
            "H_temporal_last": h_time_last,
            "H_spatial_last": spatial_out,
            "time_to_space_attn": t2s_attn.view(batch, num_nodes, 1, window),
            "space_to_time_attn": s2t_attn.view(batch, num_nodes, 1, steps),
            "short_window_len": torch.tensor(window, device=x.device),
        }
        return y, aux


def build_stgtcn(
    model_name: str,
    num_nodes: int,
    in_feat: int = 1,
    d_model: int = 64,
    short_kernel: int = 5,
    nhead: int = 4,
    tcn_layers: int = 5,
    dropout: float = 0.1,
    eta: float = 1.0,
    beta: float = 0.5,
    out_feat: Optional[int] = None,
    horizon: int = 1,
    temporal_encoder_type: str = "tcn_only",
) -> nn.Module:
    """Factory for STGTCN variants based on ``model_name``."""
    name = str(model_name or "").strip().lower()
    kwargs = dict(
        num_nodes_hint=int(num_nodes),
        in_feat=int(in_feat),
        d_model=int(d_model),
        short_kernel=int(short_kernel),
        nhead=int(nhead),
        tcn_layers=int(tcn_layers),
        dropout=float(dropout),
        eta=float(eta),
        beta=float(beta),
        out_feat=out_feat,
        horizon=int(horizon),
    )
    if name in {"parallel_cross_attn", "parallel", "parallel_cross", "stgtcn_parallel"}:
        return STGraphTCNParallelCross(**kwargs, temporal_encoder_type=temporal_encoder_type)
    resolved_temporal = str(temporal_encoder_type or name or "tcn_only").strip().lower()
    return STGraphTCN(**kwargs, temporal_encoder_type=resolved_temporal)


class STGTCNPatchForecaster(nn.Module):
    """Mainline-aligned STGTCN wrapper over patch-sequence inputs.

    Inputs are fixed to:
    - ``hist_patch_last_value_seq`` -> STGTCN value stream
    - ``hist_patch_has_value_seq`` -> observation mask

    Outputs are intentionally limited to the value sequence / mask sequence and
    the summary targets derived from them so that downstream scoring stays
    aligned with the shared future-window analysis path.
    """

    def __init__(
        self,
        *,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        model_name: str,
        num_nodes: int,
        horizon: int,
        target_patch: str,
        in_feat: int = 1,
        d_model: int = 64,
        short_kernel: int = 9,
        nhead: int = 8,
        tcn_layers: int = 5,
        dropout: float = 0.20,
        eta: float = 2.0,
        beta: float = 0.5,
        temporal_encoder_type: str = "tcn_only",
        short_patch: int = 5,
        use_huber_loss: bool = True,
        huber_beta: float = 1.0,
        loss_weight_value_seq: float = 1.0,
        loss_weight_mask_seq: float = 0.10,
        loss_weight_value: float = 1.2,
        loss_weight_delta: float = 0.25,
        loss_weight_mean: float = 0.15,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.target_patch = str(target_patch).strip().lower()
        self.short_patch = int(short_patch)
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_beta = float(huber_beta)
        self.loss_weight_value_seq = float(loss_weight_value_seq)
        self.loss_weight_mask_seq = float(loss_weight_mask_seq)
        self.loss_weight_value = float(loss_weight_value)
        self.loss_weight_delta = float(loss_weight_delta)
        self.loss_weight_mean = float(loss_weight_mean)

        self.core = build_stgtcn(
            model_name=model_name,
            num_nodes=int(num_nodes),
            in_feat=int(in_feat),
            d_model=int(d_model),
            short_kernel=int(short_kernel),
            nhead=int(nhead),
            tcn_layers=int(tcn_layers),
            dropout=float(dropout),
            eta=float(eta),
            beta=float(beta),
            out_feat=int(in_feat),
            horizon=int(horizon),
            temporal_encoder_type=temporal_encoder_type,
        )
        self.mask_head = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(horizon)),
        )
        self.register_buffer("a_stat", a_stat.float(), persistent=False)
        self.register_buffer("m_mask", m_mask.float(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.a_stat.device

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.float()
        denom = valid.sum().clamp_min(1.0)
        if self.use_huber_loss:
            elem = F.smooth_l1_loss(
                pred,
                target,
                reduction="none",
                beta=float(self.huber_beta),
            )
        else:
            elem = (pred - target) ** 2
        return (elem * valid).sum() / denom

    @staticmethod
    def _bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.float(), reduction="mean")

    def _derive_summary_predictions(
        self,
        future_value_seq_hat: torch.Tensor,
        future_value_mask_seq_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_prob = torch.sigmoid(future_value_mask_seq_logit)
        mode = self.target_patch
        if mode == "first":
            future_value_hat = future_value_seq_hat[:, 0]
            future_mean_hat = future_value_seq_hat[:, 0]
        elif mode == "last":
            future_value_hat = future_value_seq_hat[:, -1]
            future_mean_hat = future_value_seq_hat[:, -1]
        elif mode == "all":
            future_value_hat = future_value_seq_hat[:, -1]
            denom = mask_prob.sum(dim=1).clamp_min(1e-6)
            future_mean_hat = (future_value_seq_hat * mask_prob).sum(dim=1) / denom
        else:
            raise ValueError(f"Unsupported target patch mode: {self.target_patch}")
        return future_value_hat, future_mean_hat, mask_prob

    def forward(self, batch: Dict[str, torch.Tensor | object]) -> Dict[str, torch.Tensor]:
        x = self._to_device(batch["hist_patch_last_value_seq"]).float().unsqueeze(-1)
        zmask = self._to_device(batch["hist_patch_has_value_seq"]).float()
        future_value_seq = self._to_device(batch["future_value_seq"]).float()
        future_value_mask_seq = self._to_device(batch["future_value_mask_seq"]).float()
        hist_last_value = self._to_device(batch["hist_last_value"]).float()
        future_mask = self._to_device(batch["future_mask"]).float()
        future_last_value = self._to_device(batch["future_last_value"]).float()
        future_delta_value = self._to_device(batch["future_delta_value"]).float()
        future_delta_mask = self._to_device(batch["future_delta_mask"]).float()
        future_mean_value = self._to_device(batch["future_mean_value"]).float()

        future_value_seq_hat, aux = self.core(
            x,
            self.a_stat,
            self.m_mask,
            short_patch=self.short_patch,
            zmask=zmask,
        )
        future_value_seq_hat = future_value_seq_hat.squeeze(-1)
        if future_value_seq_hat.shape != future_value_seq.shape:
            raise ValueError(
                "Future value sequence shape mismatch: "
                f"pred={tuple(future_value_seq_hat.shape)} target={tuple(future_value_seq.shape)}"
            )

        future_value_mask_seq_logit = self.mask_head(aux["H_last"]).transpose(1, 2).contiguous()
        if future_value_mask_seq_logit.shape != future_value_mask_seq.shape:
            raise ValueError(
                "Future mask sequence shape mismatch: "
                f"pred={tuple(future_value_mask_seq_logit.shape)} target={tuple(future_value_mask_seq.shape)}"
            )

        future_value_hat, future_mean_hat, future_value_mask_seq_prob = self._derive_summary_predictions(
            future_value_seq_hat=future_value_seq_hat,
            future_value_mask_seq_logit=future_value_mask_seq_logit,
        )
        future_delta_hat = future_value_hat - hist_last_value

        loss_value_seq = self._regression_loss(
            pred=future_value_seq_hat,
            target=future_value_seq,
            valid_mask=future_value_mask_seq,
        )
        loss_mask_seq = self._bce_loss(
            logits=future_value_mask_seq_logit,
            target=future_value_mask_seq,
        )
        loss_value = self._regression_loss(
            pred=future_value_hat,
            target=future_last_value,
            valid_mask=future_mask,
        )
        loss_delta = self._regression_loss(
            pred=future_delta_hat,
            target=future_delta_value,
            valid_mask=future_delta_mask,
        )
        loss_mean = self._regression_loss(
            pred=future_mean_hat,
            target=future_mean_value,
            valid_mask=future_mask,
        )
        loss = (
            self.loss_weight_value_seq * loss_value_seq
            + self.loss_weight_mask_seq * loss_mask_seq
            + self.loss_weight_value * loss_value
            + self.loss_weight_delta * loss_delta
            + self.loss_weight_mean * loss_mean
        )

        resid_value = (future_value_hat - future_last_value).abs() * future_mask
        resid_delta = (future_delta_hat - future_delta_value).abs() * future_delta_mask
        resid_mean = (future_mean_hat - future_mean_value).abs() * future_mask

        return {
            "loss": loss,
            "loss_value_seq": loss_value_seq,
            "loss_mask_seq": loss_mask_seq,
            "loss_value": loss_value,
            "loss_delta": loss_delta,
            "loss_mean": loss_mean,
            "future_value_seq_hat": future_value_seq_hat,
            "future_value_mask_seq_logit": future_value_mask_seq_logit,
            "future_value_mask_seq_prob": future_value_mask_seq_prob,
            "future_value_hat": future_value_hat,
            "future_delta_hat": future_delta_hat,
            "future_mean_hat": future_mean_hat,
            "resid_value": resid_value,
            "resid_delta": resid_delta,
            "resid_mean": resid_mean,
        }
