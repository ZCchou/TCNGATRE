# -*- coding: utf-8 -*-
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
    """Residual causal TCN block. [B, T, D] -> [B, T, D]"""

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
        h = x.transpose(1, 2).contiguous()  # [B, D, T]
        for blk in self.layers:
            res = h
            y = blk["conv"](h)
            y = F.relu_(y)
            y = blk["pw"](blk["drop"](y))
            h = F.relu_(y + res)
        return self.norm(h.transpose(1, 2).contiguous())  # [B, T, D]


class DynamicGraphAttention(nn.Module):
    """QK dot-product attention with MIC static prior as additive logit bias. [B,N,D] -> [B,N,N]"""

    def __init__(self, d_model: int, eta: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.eta = float(eta)

    def forward(
        self, h_ctx: torch.Tensor, a_stat: torch.Tensor, m_mask: torch.Tensor
    ) -> torch.Tensor:
        _, _, d_model = h_ctx.shape
        valid_mask = _effective_graph_mask(m_mask)
        logits = torch.matmul(self.q(h_ctx), self.k(h_ctx).transpose(-1, -2)) / math.sqrt(d_model)
        stat_bias = torch.logit(a_stat.clamp(1e-4, 1.0 - 1e-4))
        logits = logits + self.eta * stat_bias.unsqueeze(0)
        logits = logits.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
        return torch.softmax(logits, dim=-1)


class MultiHopGraphCorrection(nn.Module):
    """
    Dynamic + static graph fusion with K rounds of gated message passing.
    """

    def __init__(
        self,
        d_model: int,
        eta: float = 1.0,
        beta: float = 0.5,
        dropout: float = 0.1,
        topk: Optional[int] = None,
        gate_init: float = 0.15,
        num_hops: int = 2,
    ):
        super().__init__()
        self.dyn = DynamicGraphAttention(d_model, eta=eta)
        self.beta = float(beta)
        self.topk = None if topk is None else int(topk)
        self.num_hops = max(1, int(num_hops))

        self.msg_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        init_gate = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))

    def _build_static_graph(self, a_stat: torch.Tensor, m_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = _effective_graph_mask(m_mask)
        a_static = a_stat.clamp_min(0.0).masked_fill(~valid_mask, 0.0)
        empty_rows = a_static.sum(dim=-1) <= 0
        if empty_rows.any():
            eye = torch.eye(a_static.shape[0], device=a_static.device, dtype=a_static.dtype)
            a_static = torch.where(empty_rows.unsqueeze(-1), eye, a_static)
        return a_static / a_static.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _topk_sparsify(self, a: torch.Tensor) -> torch.Tensor:
        if self.topk is None:
            return a
        n = a.shape[-1]
        if self.topk <= 0 or self.topk >= n:
            return a
        topv, topi = torch.topk(a, k=self.topk, dim=-1)
        out = torch.zeros_like(a)
        out.scatter_(-1, topi, topv)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_ctx: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a_dyn = self.dyn(h_ctx, a_stat, m_mask)                                     # [B, N, N]
        a_static = self._build_static_graph(a_stat, m_mask)                         # [N, N]
        a_fuse = self.beta * a_dyn + (1.0 - self.beta) * a_static.unsqueeze(0)
        a_fuse = self._topk_sparsify(a_fuse)

        h = h_nodes
        for _ in range(self.num_hops):
            h_msg = torch.matmul(a_fuse, self.msg_proj(h))                          # [B, N, D]
            delta = self.out_proj(h_msg)
            gate = torch.sigmoid(self.gate(torch.cat([h, h_msg], dim=-1)))          # [B, N, 1]
            h = self.norm(h + gate * self.drop(delta))

        return h, {
            "A_dyn": a_dyn,
            "A_static": a_static.unsqueeze(0).expand_as(a_fuse),
            "A_fuse": a_fuse,
        }


class STGraphTCN(nn.Module):
    """
    Interleaved TCN + multi-hop graph correction.
    """

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
            TCNBlock(
                d_model=self.d_model,
                kernel_size=temporal_kernel,
                n_layers=int(tcn_layers),
                dropout=dropout,
            )
            for _ in range(self.num_blocks)
        ])

        correction_positions: Set[int] = set()
        for i in range(self.num_blocks):
            if (i + 1) % self.interleave_every == 0:
                correction_positions.add(i)
        correction_positions.add(self.num_blocks - 1)
        self._correction_positions: List[int] = sorted(correction_positions)

        self.graph_corrections = nn.ModuleList([
            MultiHopGraphCorrection(
                d_model=self.d_model,
                eta=eta,
                beta=beta,
                dropout=dropout,
                topk=graph_topk,
                gate_init=graph_gate_init,
                num_hops=num_hops,
            )
            for _ in self._correction_positions
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon * self.pred_out_feat),
        )

    def _select_graph(
        self,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        node_index: Optional[torch.Tensor],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if a_stat.ndim != 2 or a_stat.shape[0] != a_stat.shape[1]:
            raise ValueError(f"a_stat must be square [N, N], got {tuple(a_stat.shape)}")
        if a_stat.shape != m_mask.shape:
            raise ValueError(
                f"a_stat and m_mask shape mismatch: {tuple(a_stat.shape)} vs {tuple(m_mask.shape)}"
            )
        if node_index is not None:
            node_index = node_index.to(device=a_stat.device, dtype=torch.long)
            a_stat = a_stat.index_select(0, node_index).index_select(1, node_index)
            m_mask = m_mask.index_select(0, node_index).index_select(1, node_index)
        if a_stat.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"graph shape must match x num_nodes {num_nodes}, got {tuple(a_stat.shape)}"
            )
        return a_stat, m_mask

    def forward(
        self,
        x: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        node_index: Optional[torch.Tensor] = None,
        short_patch: int = 5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, F], got {tuple(x.shape)}")

        batch, steps, num_nodes, in_feat = x.shape
        if in_feat != self.in_feat:
            raise ValueError(f"x feature dim must be {self.in_feat}, got {in_feat}")

        a_stat, m_mask = self._select_graph(
            a_stat.to(device=x.device),
            m_mask.to(device=x.device),
            node_index,
            num_nodes,
        )
        a_stat = a_stat.to(dtype=x.dtype)
        m_mask = m_mask.to(dtype=x.dtype)

        h = self.in_proj(x)
        h_flat = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, steps, self.d_model)

        window = max(1, min(int(short_patch), steps))
        correction_idx = 0
        last_aux: Dict[str, torch.Tensor] = {}

        for block_idx, tcn_block in enumerate(self.tcn_block_list):
            h_flat = tcn_block(h_flat)

            if block_idx in self._correction_positions:
                h_last = h_flat[:, -1, :].view(batch, num_nodes, self.d_model)
                h_ctx = h_flat[:, -window:, :].mean(dim=1).view(batch, num_nodes, self.d_model)

                h_corrected, graph_aux = self.graph_corrections[correction_idx](
                    h_last, h_ctx, a_stat, m_mask
                )
                correction_idx += 1
                last_aux = graph_aux

                h_flat = torch.cat(
                    [h_flat[:, :-1, :], h_corrected.view(batch * num_nodes, 1, self.d_model)],
                    dim=1,
                )

        h_final = h_flat[:, -1, :].view(batch, num_nodes, self.d_model)
        y = self.pred_head(h_final)
        y = y.view(batch, num_nodes, self.horizon, self.pred_out_feat).permute(0, 2, 1, 3).contiguous()

        aux: Dict[str, torch.Tensor] = {
            "A_fuse": last_aux.get("A_fuse", torch.zeros(1, device=x.device)),
            "A_dyn": last_aux.get("A_dyn", torch.zeros(1, device=x.device)),
            "A_static": last_aux.get("A_static", torch.zeros(1, device=x.device)),
            "H_last": h_final,
            "short_window_len": torch.tensor(window, device=x.device),
            "temporal_window_len": torch.tensor(steps, device=x.device),
        }
        return y, aux
