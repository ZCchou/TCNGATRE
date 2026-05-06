# -*- coding: utf-8 -*-
# A4 NoGraph variant: STGraphTCN.__init__ sets _correction_positions=[] and
# graph_corrections=[] so no graph correction step fires anywhere.
# The model is a pure stack of TCN blocks.

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _effective_graph_mask(m_mask: torch.Tensor) -> torch.Tensor:
    if m_mask.ndim != 2 or m_mask.shape[0] != m_mask.shape[1]:
        raise ValueError(f"m_mask must be square [N, N], got {tuple(m_mask.shape)}")
    num_nodes = m_mask.shape[0]
    eye = torch.eye(num_nodes, device=m_mask.device, dtype=torch.bool)
    valid = (m_mask > 0) & (~eye)
    empty_rows = ~valid.any(dim=-1)
    valid = valid | (empty_rows.unsqueeze(-1) & eye)
    return valid


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=int(kernel_size), dilation=int(dilation), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "conv": CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=2 ** i),
                "drop": nn.Dropout(dropout),
                "pw": nn.Conv1d(d_model, d_model, kernel_size=1),
            })
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2).contiguous()
        for blk in self.layers:
            res = h
            y = blk["conv"](h)
            y = F.relu_(y)
            y = blk["pw"](blk["drop"](y))
            h = F.relu_(y + res)
        return self.norm(h.transpose(1, 2).contiguous())


# Stub kept so that the API signature matches; never instantiated in this variant.
class DynamicGraphAttention(nn.Module):
    def __init__(self, d_model: int, eta: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.eta = float(eta)

    def forward(self, h_ctx, a_stat, m_mask):
        raise RuntimeError("DynamicGraphAttention should not be called in NoGraph variant")


class MultiHopGraphCorrection(nn.Module):
    def __init__(self, d_model, eta=1.0, beta=0.5, dropout=0.1, topk=None, gate_init=0.15, num_hops=2):
        super().__init__()
        self.dyn = DynamicGraphAttention(d_model, eta=eta)
        self.beta = float(beta)
        self.num_hops = max(1, int(num_hops))
        self.msg_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        init_gate = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))

    def forward(self, h_nodes, h_ctx, a_stat, m_mask):
        raise RuntimeError("MultiHopGraphCorrection should not be called in NoGraph variant")


class STGraphTCN(nn.Module):
    """Pure-TCN variant: no graph correction modules allocated or called."""

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
        predict_logvar: bool = False,
        tcn_blocks: int = 4,
        graph_topk: Optional[int] = None,
        graph_gate_init: float = 0.15,
        interleave_every: int = 2,
        num_hops: int = 2,
    ):
        super().__init__()
        del nhead

        self.in_feat = int(in_feat)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.out_feat = int(out_feat) if out_feat is not None else self.in_feat
        self.predict_logvar = bool(predict_logvar)
        self.pred_out_feat = self.out_feat * (2 if self.predict_logvar else 1)
        self._num_nodes_hint = None if num_nodes_hint is None else int(num_nodes_hint)
        self._short_kernel = int(short_kernel)

        self.num_blocks = max(1, int(tcn_blocks))
        self.interleave_every = max(1, int(interleave_every))

        self.in_proj = nn.Linear(self.in_feat, self.d_model)

        temporal_kernel = max(3, min(self._short_kernel, 9))
        self.tcn_block_list = nn.ModuleList([
            TCNBlock(d_model=self.d_model, kernel_size=temporal_kernel,
                     n_layers=int(tcn_layers), dropout=dropout)
            for _ in range(self.num_blocks)
        ])

        # No graph corrections in this variant
        self._correction_positions: List[int] = []
        self.graph_corrections = nn.ModuleList([])

        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon * self.pred_out_feat),
        )

    def forward(self, x, a_stat, m_mask, node_index=None, short_patch=5):
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, F], got {tuple(x.shape)}")

        batch, steps, num_nodes, in_feat = x.shape
        if in_feat != self.in_feat:
            raise ValueError(f"x feature dim must be {self.in_feat}, got {in_feat}")

        h = self.in_proj(x)
        h_flat = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, self.d_model)

        window = max(1, min(int(short_patch), steps))

        for tcn_block in self.tcn_block_list:
            h_flat = tcn_block(h_flat)

        h_final = h_flat[:, -1, :].view(batch, num_nodes, self.d_model)
        y = self.pred_head(h_final)
        y = y.view(batch, num_nodes, self.horizon, self.pred_out_feat).permute(0, 2, 1, 3).contiguous()

        aux: Dict[str, torch.Tensor] = {
            "A_fuse": torch.zeros(1, device=x.device),
            "A_dyn": torch.zeros(1, device=x.device),
            "A_static": torch.zeros(1, device=x.device),
            "H_last": h_final,
            "short_window_len": torch.tensor(window, device=x.device),
            "temporal_window_len": torch.tensor(steps, device=x.device),
        }
        return y, aux
