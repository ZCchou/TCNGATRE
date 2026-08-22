from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from revision_experiments.core.paths import ensure_import_paths

ensure_import_paths()

from model.tcngatre import MultiHopGraphCorrection, STGraphTCN  # noqa: E402


FUSION_BY_VARIANT = {
    "dynamic_only": "dynamic",
    "fusion_static": "static",
    "fusion_dynamic": "dynamic",
    "fusion_learned_scalar": "learned_scalar",
    "fusion_sample_gate": "sample_gate",
    "fusion_concat_mlp": "concat_mlp",
}


class LightweightStaticGraphCorrection(nn.Module):
    """Parameter-free, one-hop smoothing with a fixed MIC adjacency."""

    def __init__(self, residual_weight: float = 0.15):
        super().__init__()
        self.residual_weight = float(min(max(residual_weight, 0.0), 1.0))

    @staticmethod
    def _build_static_graph(a_stat: torch.Tensor, m_mask: torch.Tensor) -> torch.Tensor:
        valid = m_mask > 0
        a_static = a_stat.clamp_min(0.0).masked_fill(~valid, 0.0)
        empty_rows = a_static.sum(dim=-1) <= 0
        if empty_rows.any():
            eye = torch.eye(a_static.shape[0], device=a_static.device, dtype=a_static.dtype)
            a_static = torch.where(empty_rows.unsqueeze(-1), eye, a_static)
        return a_static / a_static.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_ctx: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del h_ctx
        a_static = self._build_static_graph(a_stat, m_mask)
        a_batch = a_static.unsqueeze(0).expand(
            h_nodes.shape[0], a_static.shape[0], a_static.shape[1]
        )
        neighbour_state = torch.matmul(a_batch, h_nodes)
        weight = self.residual_weight
        corrected = (1.0 - weight) * h_nodes + weight * neighbour_state
        return corrected, {
            "A_dyn": torch.zeros(1, device=h_nodes.device, dtype=h_nodes.dtype),
            "A_static": a_batch,
            "A_fuse": a_batch,
            "fusion_mix": torch.zeros(
                (h_nodes.shape[0], 1, 1), device=h_nodes.device, dtype=h_nodes.dtype
            ),
        }


class FlexibleGraphCorrection(MultiHopGraphCorrection):
    """Graph correction with an explicit, auditable static/dynamic fusion rule."""

    def __init__(self, *args, fusion: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion = str(fusion)
        d_model = int(self.msg_proj.in_features)
        if self.fusion == "static":
            # A true static-only ablation must not retain or execute the
            # sample-adaptive Q/K attention branch. Assigning None also
            # removes its parameters from state_dict() and the optimizer.
            self.dyn = None
        elif self.fusion == "learned_scalar":
            self.mix_logit = nn.Parameter(torch.tensor(0.0))
        elif self.fusion == "sample_gate":
            self.sample_gate = nn.Sequential(
                nn.Linear(d_model, max(4, d_model // 2)),
                nn.ReLU(),
                nn.Linear(max(4, d_model // 2), 1),
            )
            nn.init.zeros_(self.sample_gate[-1].weight)
            nn.init.zeros_(self.sample_gate[-1].bias)
        elif self.fusion == "concat_mlp":
            self.edge_fuser = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
            with torch.no_grad():
                self.edge_fuser[-1].weight.fill_(0.5)
                self.edge_fuser[-1].bias.zero_()

    def _fuse(
        self,
        a_dyn: torch.Tensor,
        a_static: torch.Tensor,
        h_ctx: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_b = a_static.unsqueeze(0).expand_as(a_dyn)
        if self.fusion == "static":
            mix = torch.zeros((a_dyn.shape[0], 1, 1), device=a_dyn.device, dtype=a_dyn.dtype)
            return static_b, mix
        if self.fusion == "dynamic":
            mix = torch.ones((a_dyn.shape[0], 1, 1), device=a_dyn.device, dtype=a_dyn.dtype)
            return a_dyn, mix
        if self.fusion == "learned_scalar":
            mix = torch.sigmoid(self.mix_logit).view(1, 1, 1).expand(a_dyn.shape[0], 1, 1)
            return mix * a_dyn + (1.0 - mix) * static_b, mix
        if self.fusion == "sample_gate":
            mix = torch.sigmoid(self.sample_gate(h_ctx.mean(dim=1))).view(-1, 1, 1)
            return mix * a_dyn + (1.0 - mix) * static_b, mix
        if self.fusion == "concat_mlp":
            features = torch.stack(
                [torch.log(a_dyn.clamp_min(1e-8)), torch.log(static_b.clamp_min(1e-8))], dim=-1
            )
            logits = self.edge_fuser(features).squeeze(-1)
            valid = (m_mask > 0).unsqueeze(0)
            logits = logits.masked_fill(~valid, float("-inf"))
            fused = torch.softmax(logits, dim=-1)
            mix = torch.full((a_dyn.shape[0], 1, 1), float("nan"), device=a_dyn.device)
            return fused, mix
        raise ValueError(f"Unsupported fusion: {self.fusion}")

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_ctx: torch.Tensor,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a_static = self._build_static_graph(a_stat, m_mask)
        if self.fusion == "static":
            a_fuse = a_static.unsqueeze(0).expand(
                h_nodes.shape[0], a_static.shape[0], a_static.shape[1]
            )
            mix = torch.zeros(
                (h_nodes.shape[0], 1, 1), device=h_nodes.device, dtype=h_nodes.dtype
            )
            # Keep the auxiliary interface stable without manufacturing a
            # dynamic graph that this ablation intentionally does not have.
            a_dyn_aux = torch.zeros(1, device=h_nodes.device, dtype=h_nodes.dtype)
        else:
            if self.dyn is None:
                raise RuntimeError(f"Dynamic graph branch is missing for fusion={self.fusion}")
            a_dyn = self.dyn(h_ctx, a_stat, m_mask)
            a_fuse, mix = self._fuse(a_dyn, a_static, h_ctx, m_mask)
            a_dyn_aux = a_dyn
        a_fuse = self._topk_sparsify(a_fuse)

        h = h_nodes
        for _ in range(self.num_hops):
            h_msg = torch.matmul(a_fuse, self.msg_proj(h))
            delta = self.out_proj(h_msg)
            gate = torch.sigmoid(self.gate(torch.cat([h, h_msg], dim=-1)))
            h = self.norm(h + gate * self.drop(delta))
        return h, {
            "A_dyn": a_dyn_aux,
            "A_static": a_static.unsqueeze(0).expand_as(a_fuse),
            "A_fuse": a_fuse,
            "fusion_mix": mix,
        }


def _base_model(cfg, num_nodes: int, num_hops: int | None = None, interleave_every: int | None = None):
    return STGraphTCN(
        num_nodes_hint=num_nodes,
        in_feat=1,
        d_model=cfg.d_model,
        short_kernel=cfg.short_kernel,
        tcn_layers=cfg.tcn_layers,
        tcn_blocks=cfg.tcn_blocks,
        dropout=cfg.dropout,
        eta=cfg.graph_eta,
        beta=cfg.graph_beta,
        out_feat=1,
        horizon=cfg.horizon_out,
        predict_logvar=False,
        graph_topk=None,
        graph_gate_init=cfg.graph_gate_init,
        interleave_every=cfg.interleave_every if interleave_every is None else interleave_every,
        num_hops=cfg.graph_num_hops if num_hops is None else num_hops,
    )


def build_revision_model(revision_cfg, legacy_cfg, num_nodes: int, device: torch.device) -> nn.Module:
    variant = revision_cfg.variant
    if variant in {"full", "prior_mic_fixed", "prior_identity_fixed", "prior_random_fixed", "no_cross_dim"}:
        return _base_model(legacy_cfg, num_nodes).to(device)
    if variant == "single_hop":
        return _base_model(legacy_cfg, num_nodes, num_hops=1).to(device)
    if variant == "late_graph":
        return _base_model(legacy_cfg, num_nodes, interleave_every=legacy_cfg.tcn_blocks + 1).to(device)
    if variant == "tcn_only":
        model = _base_model(legacy_cfg, num_nodes)
        model._correction_positions = []
        model.graph_corrections = nn.ModuleList()
        return model.to(device)
    if variant == "static_only":
        model = _base_model(legacy_cfg, num_nodes)
        # Deliberately simple static baseline: one parameter-free adjacency
        # smoothing step after the final TCN block only.
        model._correction_positions = [model.num_blocks - 1]
        model.graph_corrections = nn.ModuleList([
            LightweightStaticGraphCorrection(
                residual_weight=legacy_cfg.graph_gate_init,
            )
        ])
        return model.to(device)

    fusion = FUSION_BY_VARIANT.get(variant)
    if fusion is None:
        raise ValueError(f"No TCNGATRE model factory for variant={variant}")
    model = _base_model(legacy_cfg, num_nodes)
    model.graph_corrections = nn.ModuleList([
        FlexibleGraphCorrection(
            d_model=legacy_cfg.d_model,
            eta=legacy_cfg.graph_eta,
            beta=legacy_cfg.graph_beta,
            dropout=legacy_cfg.dropout,
            topk=None,
            gate_init=legacy_cfg.graph_gate_init,
            num_hops=legacy_cfg.graph_num_hops,
            fusion=fusion,
        )
        for _ in model._correction_positions
    ])
    return model.to(device)


def transform_graph_prior(
    a: torch.Tensor,
    m: torch.Tensor,
    variant: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if variant != "prior_identity_fixed" and variant != "prior_random_fixed":
        return a, m, {"prior": "mic", "seed": None}
    n = int(a.shape[0])
    if variant == "prior_identity_fixed":
        eye = torch.eye(n, dtype=a.dtype, device=a.device)
        return eye, eye.clone(), {"prior": "identity", "seed": None}

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n)
    idx = torch.as_tensor(permutation, dtype=torch.long, device=a.device)
    shuffled_a = a.index_select(0, idx).index_select(1, idx)
    shuffled_m = m.index_select(0, idx).index_select(1, idx)
    return shuffled_a, shuffled_m, {
        "prior": "node_permuted_mic",
        "seed": int(seed),
        "permutation": permutation.tolist(),
    }


def assert_graph_probabilities(aux: dict[str, torch.Tensor], atol: float = 1e-5) -> None:
    for key in ("A_dyn", "A_static", "A_fuse"):
        value = aux.get(key)
        if value is None or value.numel() <= 1:
            continue
        if not torch.isfinite(value).all():
            raise AssertionError(f"{key} contains non-finite values")
        checked = value.detach()
        if float(checked.min()) < -atol or float(checked.max()) > 1.0 + atol:
            raise AssertionError(f"{key} outside [0,1]")
        row_sum = value.sum(dim=-1)
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=atol, rtol=atol):
            raise AssertionError(f"{key} rows do not sum to one")
