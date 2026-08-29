from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Length-preserving moving average without reading future positions."""
    batch, length, nodes, channels = x.shape
    flattened = x.permute(0, 2, 3, 1).reshape(batch * nodes, channels, length)
    averaged = F.avg_pool1d(
        F.pad(flattened, (int(kernel) - 1, 0), mode="replicate"),
        kernel_size=int(kernel),
        stride=1,
    )
    return averaged.reshape(batch, nodes, channels, length).permute(0, 3, 1, 2)


def _sinusoidal_positions(length: int, channels: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(channels, 1))
    )
    encoding = torch.zeros(length, channels, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class PaperInputEmbedding(nn.Module):
    """Temporal Conv1D projection plus time and sensor position embeddings."""

    def __init__(self, nodes: int, d_model: int, window: int, dropout: float):
        super().__init__()
        self.nodes = int(nodes)
        self.d_model = int(d_model)
        self.temporal_projection = nn.Conv1d(1, d_model, kernel_size=3, padding=1, bias=False)
        self.sensor_embedding = nn.Parameter(torch.empty(nodes, d_model))
        nn.init.normal_(self.sensor_embedding, std=0.02)
        self.register_buffer(
            "position_embedding", _sinusoidal_positions(window, d_model), persistent=False
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, nodes = x.shape
        flattened = x.permute(0, 2, 1).reshape(batch * nodes, 1, length)
        projected = self.temporal_projection(flattened)
        projected = projected.reshape(batch, nodes, self.d_model, length).permute(0, 3, 1, 2)
        hidden = self.scale * projected
        hidden = hidden + self.position_embedding[:length].view(1, length, 1, self.d_model)
        hidden = hidden + self.sensor_embedding.view(1, 1, nodes, self.d_model)
        return self.dropout(hidden)


class SeasonalTrendRouter(nn.Module):
    """Paper Eq. (6)-(12): FFT seasonal and multi-kernel trend routing."""

    def __init__(
        self,
        d_model: int,
        experts: int,
        top_k: int,
        seasonal_top_k: int,
        trend_kernels: list[int],
        noisy_gating: bool,
    ):
        super().__init__()
        self.top_k = min(max(int(top_k), 1), int(experts))
        self.seasonal_top_k = max(int(seasonal_top_k), 1)
        self.trend_kernels = [int(value) for value in trend_kernels]
        self.trend_weights = nn.Linear(d_model, len(self.trend_kernels))
        self.merge = nn.Linear(d_model, d_model)
        self.route = nn.Linear(d_model, experts)
        self.noise = nn.Linear(d_model, experts)
        self.noisy_gating = bool(noisy_gating)

    def _seasonal(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        amplitudes = spectrum.abs().mean(dim=(2, 3))
        if amplitudes.shape[1] <= 1:
            return torch.zeros_like(x)
        amplitudes = amplitudes.clone()
        amplitudes[:, 0] = -torch.inf
        selected = torch.topk(
            amplitudes, k=min(self.seasonal_top_k, amplitudes.shape[1] - 1), dim=1
        ).indices
        mask = torch.zeros_like(amplitudes, dtype=torch.bool).scatter_(1, selected, True)
        filtered = spectrum * mask[:, :, None, None]
        return torch.fft.irfft(filtered, n=x.shape[1], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=(1, 2))
        trend_candidates = torch.stack(
            [_causal_average(x, kernel) for kernel in self.trend_kernels], dim=1
        )
        trend_weight = torch.softmax(self.trend_weights(pooled), dim=-1)
        trend = (trend_candidates * trend_weight[:, :, None, None, None]).sum(dim=1)
        merged = self.merge(x + self._seasonal(x) + trend).mean(dim=(1, 2))
        logits = self.route(merged)
        if self.training and self.noisy_gating:
            scale = F.softplus(self.noise(merged)) + 1e-2
            logits = logits + torch.randn_like(logits) * scale
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.topk(probabilities, k=self.top_k, dim=-1).indices
        selected_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(
            1, selected, True
        )
        sparse = probabilities.masked_fill(~selected_mask, 0.0)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return sparse, probabilities


class CCSTGCNExpert(nn.Module):
    """Patch-level causal spatio-temporal graph expert from paper Eq. (13)-(22)."""

    def __init__(
        self,
        nodes: int,
        window: int,
        d_model: int,
        patch_size: int,
        heads: int,
        knn_k: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by the number of attention heads")
        self.nodes = int(nodes)
        self.window = int(window)
        self.patch_size = int(patch_size)
        self.patch_count = int(math.ceil(window / patch_size))
        self.graph_nodes = self.nodes * self.patch_count
        self.knn_k = min(max(int(knn_k), 1), self.graph_nodes)
        self.intrapatch_attention = nn.MultiheadAttention(
            d_model, heads, dropout=float(dropout), batch_first=True
        )
        embedding_dim = min(int(d_model), 32)
        self.graph_embedding = nn.Parameter(torch.empty(self.graph_nodes, embedding_dim))
        nn.init.normal_(self.graph_embedding, std=0.02)
        self.graph_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.graph_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))

        patch_index = torch.arange(self.patch_count).repeat(self.nodes)
        sensor_index = torch.arange(self.nodes).repeat_interleave(self.patch_count)
        self.register_buffer("patch_index", patch_index, persistent=False)
        self.register_buffer("sensor_index", sensor_index, persistent=False)

    def _causal_knn(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(self.graph_embedding, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        causal = self.patch_index[None, :] <= self.patch_index[:, None]
        similarity = similarity.masked_fill(~causal, -torch.inf)
        neighbor_index = torch.topk(similarity, k=self.knn_k, dim=-1).indices
        neighbor_logits = similarity.gather(1, neighbor_index)
        neighbor_weight = torch.softmax(neighbor_logits, dim=-1)
        return neighbor_index, neighbor_weight

    def _sensor_adjacency(
        self, neighbor_index: torch.Tensor, neighbor_weight: torch.Tensor
    ) -> torch.Tensor:
        target_sensor = self.sensor_index[:, None].expand_as(neighbor_index)
        source_sensor = self.sensor_index[neighbor_index]
        flattened_index = target_sensor * self.nodes + source_sensor
        adjacency = neighbor_weight.new_zeros(self.nodes * self.nodes)
        adjacency = adjacency.scatter_add(
            0, flattened_index.reshape(-1), neighbor_weight.reshape(-1)
        ).reshape(self.nodes, self.nodes)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, nodes, channels = x.shape
        if length != self.window or nodes != self.nodes:
            raise ValueError(
                f"Expected [batch,{self.window},{self.nodes},channels], received {tuple(x.shape)}"
            )
        padded_length = self.patch_count * self.patch_size
        temporal = x.permute(0, 2, 3, 1)
        if padded_length > length:
            temporal = F.pad(temporal, (0, padded_length - length), mode="replicate")
        patches = temporal.reshape(
            batch, nodes, channels, self.patch_count, self.patch_size
        ).permute(0, 1, 3, 4, 2)
        attention_input = patches.reshape(
            batch * nodes * self.patch_count, self.patch_size, channels
        )
        local, _ = self.intrapatch_attention(
            attention_input, attention_input, attention_input, need_weights=False
        )
        local = local.reshape(batch, nodes, self.patch_count, self.patch_size, channels)
        graph_input = local.mean(dim=3).reshape(batch, self.graph_nodes, channels)
        neighbor_index, neighbor_weight = self._causal_knn()
        neighbors = graph_input[:, neighbor_index, :]
        propagated = (neighbors * neighbor_weight[None, :, :, None]).sum(dim=2)
        graph_output = self.graph_norm(
            graph_input + self.dropout(self.graph_projection(propagated))
        )
        corrected = local + graph_output.reshape(
            batch, nodes, self.patch_count, 1, channels
        )
        corrected = self.output_norm(self.output_projection(corrected))
        restored = corrected.permute(0, 1, 4, 2, 3).reshape(
            batch, nodes, channels, padded_length
        )[..., :length]
        output = restored.permute(0, 3, 1, 2)
        return output, self._sensor_adjacency(neighbor_index, neighbor_weight)


class GraphMixtureLayer(nn.Module):
    def __init__(
        self,
        nodes: int,
        window: int,
        d_model: int,
        patch_sizes: list[int],
        top_k: int,
        heads: int,
        knn_k: int,
        seasonal_top_k: int,
        trend_kernels: list[int],
        dropout: float,
        noisy_gating: bool,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            CCSTGCNExpert(nodes, window, d_model, patch, heads, knn_k, dropout)
            for patch in patch_sizes
        )
        self.router = SeasonalTrendRouter(
            d_model=d_model,
            experts=len(self.experts),
            top_k=top_k,
            seasonal_top_k=seasonal_top_k,
            trend_kernels=trend_kernels,
            noisy_gating=noisy_gating,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse_probabilities, routing_probabilities = self.router(x)
        mixture = torch.zeros_like(x)
        adjacency = x.new_zeros(x.shape[2], x.shape[2])
        for index, expert in enumerate(self.experts):
            expert_output, expert_adjacency = expert(x)
            weight = sparse_probabilities[:, index].view(-1, 1, 1, 1)
            mixture = mixture + weight * expert_output
            adjacency = adjacency + sparse_probabilities[:, index].mean() * expert_adjacency
        output = self.norm(x + self.dropout(mixture))
        importance = routing_probabilities.mean(dim=0)
        balance = len(self.experts) * importance.square().sum() - 1.0
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return output, balance, adjacency


class MSTGCNetApprox(nn.Module):
    """Equation-aligned clean-room completion of the incomplete MSTGCNet release."""

    def __init__(
        self,
        nodes: int,
        window: int = 96,
        d_model: int = 64,
        patch_size_list: list[list[int]] | None = None,
        top_k: int = 3,
        heads: int = 2,
        knn_k: int = 5,
        seasonal_top_k: int = 3,
        trend_kernels: list[int] | None = None,
        dropout: float = 0.1,
        revin: bool = True,
        noisy_gating: bool = True,
    ):
        super().__init__()
        patches = patch_size_list or [
            [8, 12, 16, 32],
            [6, 8, 12, 16],
            [2, 6, 8, 12],
        ]
        kernels = trend_kernels or [4, 8, 12]
        self.nodes = int(nodes)
        self.window = int(window)
        self.revin = bool(revin)
        self.embedding = PaperInputEmbedding(nodes, d_model, window, dropout)
        self.layers = nn.ModuleList(
            GraphMixtureLayer(
                nodes=self.nodes,
                window=self.window,
                d_model=int(d_model),
                patch_sizes=[int(value) for value in layer_patches],
                top_k=int(top_k),
                heads=int(heads),
                knn_k=int(knn_k),
                seasonal_top_k=int(seasonal_top_k),
                trend_kernels=[int(value) for value in kernels],
                dropout=float(dropout),
                noisy_gating=bool(noisy_gating),
            )
            for layer_patches in patches
        )
        self.projection = nn.Linear(int(d_model), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3 or x.shape[1] != self.window or x.shape[-1] != self.nodes:
            raise ValueError(
                f"Expected [batch,{self.window},{self.nodes}], received {tuple(x.shape)}"
            )
        if self.revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            normalized = (x - mean) / std
        else:
            mean = torch.zeros_like(x[:, :1])
            std = torch.ones_like(x[:, :1])
            normalized = x
        hidden = self.embedding(normalized)
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
