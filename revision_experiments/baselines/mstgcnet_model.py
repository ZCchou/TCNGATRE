from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    batch, length, nodes, channels = x.shape
    flattened = x.permute(0, 2, 3, 1).reshape(batch * nodes, channels, length)
    averaged = F.avg_pool1d(
        F.pad(flattened, (int(kernel) - 1, 0)),
        kernel_size=int(kernel),
        stride=1,
    )
    return averaged.reshape(batch, nodes, channels, length).permute(0, 3, 1, 2)


class TemporalPatchExpert(nn.Module):
    def __init__(self, d_model: int, patch_size: int, dropout: float):
        super().__init__()
        self.patch_size = int(patch_size)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=self.patch_size, groups=d_model, bias=False
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, nodes, channels = x.shape
        flattened = x.permute(0, 2, 3, 1).reshape(batch * nodes, channels, length)
        filtered = self.depthwise(F.pad(flattened, (self.patch_size - 1, 0)))
        filtered = self.dropout(F.gelu(self.pointwise(filtered)))
        return filtered.reshape(batch, nodes, channels, length).permute(0, 3, 1, 2)


class CausalSpatialGraphAttention(nn.Module):
    """Same-time sensor attention with a learned sparse prior and no future access."""

    def __init__(self, nodes: int, d_model: int, heads: int, knn_k: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by the number of attention heads")
        self.nodes = int(nodes)
        self.heads = int(heads)
        self.head_dim = int(d_model) // int(heads)
        self.knn_k = min(max(int(knn_k), 1), self.nodes)
        node_dim = min(32, int(d_model))
        self.source_nodes = nn.Parameter(torch.randn(self.nodes, node_dim) * 0.02)
        self.target_nodes = nn.Parameter(torch.randn(self.nodes, node_dim) * 0.02)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(float(dropout))

    def sparse_prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.relu(
            self.source_nodes @ self.target_nodes.transpose(0, 1)
            / math.sqrt(self.source_nodes.shape[1])
        )
        logits = logits + torch.eye(self.nodes, device=logits.device, dtype=logits.dtype)
        indices = torch.topk(logits, k=self.knn_k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
        prior = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        return prior, mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, nodes, channels = x.shape
        if nodes != self.nodes:
            raise ValueError(f"Expected {self.nodes} graph nodes, received {nodes}")
        query = self.query(x).reshape(batch, length, nodes, self.heads, self.head_dim)
        key = self.key(x).reshape(batch, length, nodes, self.heads, self.head_dim)
        value = self.value(x).reshape(batch, length, nodes, self.heads, self.head_dim)
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        prior, mask = self.sparse_prior()
        scores = scores + torch.log(prior.clamp_min(1e-8)).view(1, 1, 1, nodes, nodes)
        scores = scores.masked_fill(~mask.view(1, 1, 1, nodes, nodes), -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attention), value)
        context = context.permute(0, 1, 3, 2, 4).reshape(batch, length, nodes, channels)
        return self.output(context), prior


class GraphMixtureLayer(nn.Module):
    def __init__(
        self,
        nodes: int,
        d_model: int,
        patch_sizes: list[int],
        top_k: int,
        heads: int,
        knn_k: int,
        trend_kernels: list[int],
        dropout: float,
    ):
        super().__init__()
        self.top_k = min(max(int(top_k), 1), len(patch_sizes))
        self.trend_kernels = [int(value) for value in trend_kernels]
        self.experts = nn.ModuleList(
            TemporalPatchExpert(d_model, int(patch), dropout) for patch in patch_sizes
        )
        self.router = nn.Linear(d_model, len(self.experts))
        self.trend_projection = nn.Linear(d_model, d_model)
        self.graph = CausalSpatialGraphAttention(nodes, d_model, heads, knn_k, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trends = torch.stack(
            [_causal_average(x, kernel) for kernel in self.trend_kernels], dim=0
        ).mean(dim=0)
        seasonal = x - trends
        router_probabilities = torch.softmax(self.router(seasonal.mean(dim=(1, 2))), dim=-1)
        selected = torch.topk(router_probabilities, k=self.top_k, dim=-1).indices
        selected_mask = torch.zeros_like(router_probabilities, dtype=torch.bool).scatter_(
            1, selected, True
        )
        sparse_probabilities = router_probabilities.masked_fill(~selected_mask, 0.0)
        sparse_probabilities = sparse_probabilities / sparse_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        mixture = torch.zeros_like(seasonal)
        for index, expert in enumerate(self.experts):
            mixture = mixture + sparse_probabilities[:, index].view(-1, 1, 1, 1) * expert(
                seasonal
            )
        mixture = mixture + self.trend_projection(trends)
        graph_output, adjacency = self.graph(mixture)
        output = self.norm(x + self.dropout(graph_output))
        importance = router_probabilities.mean(dim=0)
        balance = len(self.experts) * importance.square().sum() - 1.0
        return output, balance, adjacency


class MSTGCNetApprox(nn.Module):
    """Auditable engineering reconstruction of the incomplete MSTGCNet release."""

    def __init__(
        self,
        nodes: int,
        d_model: int = 64,
        patch_size_list: list[list[int]] | None = None,
        top_k: int = 3,
        heads: int = 2,
        knn_k: int = 5,
        trend_kernels: list[int] | None = None,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        patches = patch_size_list or [
            [16, 12, 8, 32],
            [12, 8, 6, 4],
            [8, 6, 4, 2],
        ]
        kernels = trend_kernels or [4, 8, 12]
        self.nodes = int(nodes)
        self.revin = bool(revin)
        self.embedding = nn.Linear(1, int(d_model))
        self.layers = nn.ModuleList(
            GraphMixtureLayer(
                self.nodes,
                int(d_model),
                [int(value) for value in layer_patches],
                int(top_k),
                int(heads),
                int(knn_k),
                [int(value) for value in kernels],
                float(dropout),
            )
            for layer_patches in patches
        )
        self.projection = nn.Linear(int(d_model), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.nodes:
            raise ValueError(f"Expected [batch,time,{self.nodes}], received {tuple(x.shape)}")
        if self.revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            normalized = (x - mean) / std
        else:
            mean = torch.zeros_like(x[:, :1])
            std = torch.ones_like(x[:, :1])
            normalized = x
        hidden = self.embedding(normalized.unsqueeze(-1))
        balance = x.new_zeros(())
        adjacencies: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, current_balance, adjacency = layer(hidden)
            balance = balance + current_balance
            adjacencies.append(adjacency)
        reconstructed = self.projection(hidden).squeeze(-1)
        if self.revin:
            reconstructed = reconstructed * std + mean
        return reconstructed, balance / max(len(self.layers), 1), adjacencies
