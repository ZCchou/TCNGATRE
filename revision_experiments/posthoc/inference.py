from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from revision_experiments.core.paths import ensure_import_paths
from revision_experiments.scoring.aggregators import aggregate_dataframe

from .constants import EMA_ALPHA, PRIMARY_THRESHOLD
from .data import (
    ArrayWindowDataset,
    ChannelStatistics,
    FlightArray,
    collate_windows,
    fit_channel_statistics,
    load_normalized_flights,
)
from .io import read_json
from .source import SourceRun, native_config

ensure_import_paths()

from data.stgtcn_window_dataset import resolve_flight_splits  # noqa: E402
from tcngatre_infer_impl import load_checkpoint  # noqa: E402
from tcngatre_train_impl import load_graph  # noqa: E402
from utils.normalization import load_minmax_stats  # noqa: E402


def _json_vector(values: np.ndarray) -> str:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return json.dumps(clean.astype(float).tolist(), ensure_ascii=False)


@dataclass
class LoadedSourceModel:
    source: SourceRun
    cfg: Any
    device: torch.device
    nodes: list[str]
    adjacency: torch.Tensor
    mask: torch.Tensor
    normalization: dict
    train_flights: dict[str, FlightArray]
    validation_flights: dict[str, FlightArray]
    failure_flights: dict[str, FlightArray]
    channel_statistics: ChannelStatistics
    model: torch.nn.Module

    @classmethod
    def load(cls, source: SourceRun) -> "LoadedSourceModel":
        cfg = native_config(source)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
        nodes, adjacency, mask = load_graph(Path(source.graph_dir))
        adjacency, mask = adjacency.to(device), mask.to(device)
        normalization = load_minmax_stats(source.normalization_stats)
        train_paths, val_paths, failure_paths = resolve_flight_splits(
            dataset_root=Path(cfg.data_root), split_info_path=Path(cfg.split_info_path)
        )
        train = load_normalized_flights(train_paths, normalization, nodes, cfg.trim_leading_sec)
        validation = load_normalized_flights(val_paths, normalization, nodes, cfg.trim_leading_sec)
        failure = load_normalized_flights(failure_paths, normalization, nodes, cfg.trim_leading_sec)
        statistics = fit_channel_statistics(train)
        model = load_checkpoint(cfg, source.checkpoint, len(nodes), device)
        return cls(
            source=source, cfg=cfg, device=device, nodes=nodes,
            adjacency=adjacency, mask=mask, normalization=normalization,
            train_flights=train, validation_flights=validation, failure_flights=failure,
            channel_statistics=statistics, model=model,
        )

    def score(
        self,
        flights: dict[str, FlightArray],
        *,
        capture_graph: bool = False,
        description: str = "posthoc inference",
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
        dataset = ArrayWindowDataset(
            flights, self.cfg.lookback, self.cfg.horizon_out, self.cfg.sample_stride
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_windows,
        )
        rows: list[dict[str, Any]] = []
        graph_store: dict[str, list[np.ndarray]] = {"A_static": [], "A_dyn": [], "A_fuse": []}
        self.model.eval()
        global_row = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True, mininterval=1.0):
                x = batch["x"].to(self.device, non_blocking=True).float()
                y = batch["y"].to(self.device, non_blocking=True).float()
                prediction, aux = self.model(
                    x, self.adjacency, self.mask, short_patch=self.cfg.short_patch
                )
                pred = prediction[..., 0].detach().cpu().numpy().astype(np.float32)
                truth = y[..., 0].detach().cpu().numpy().astype(np.float32)
                value = np.mean(np.abs(pred - truth), axis=1)
                delta = (
                    np.mean(np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1)), axis=1)
                    if pred.shape[1] >= 2 else np.zeros_like(value)
                )
                if capture_graph:
                    for name in graph_store:
                        graph = aux.get(name)
                        if graph is None or graph.ndim != 3 or graph.shape[0] != len(batch["flight"]):
                            raise RuntimeError(f"Missing graph auxiliary output: {name}")
                        graph_store[name].append(graph.detach().cpu().numpy().astype(np.float32))
                for index, flight in enumerate(batch["flight"]):
                    rows.append({
                        "flight": str(flight),
                        "current_index": int(batch["current_index"][index].item()),
                        "t_start": float(batch["t_future_start"][index].item()),
                        "t_end": float(batch["t_future_end"][index].item()),
                        "t_mid": 0.5 * (
                            float(batch["t_future_start"][index].item())
                            + float(batch["t_future_end"][index].item())
                        ),
                        "value_residual_vec": _json_vector(value[index]),
                        "delta_residual_vec": _json_vector(delta[index]),
                        "sensor_score_vec": _json_vector(value[index] + 0.20 * delta[index]),
                        "graph_row": global_row + index if capture_graph else -1,
                    })
                global_row += len(batch["flight"])
        if not rows:
            raise RuntimeError("Posthoc scoring produced no windows")
        frame = pd.DataFrame(rows)
        graphs = None
        if capture_graph:
            graphs = {name: np.concatenate(parts, axis=0) for name, parts in graph_store.items()}
            if any(len(value) != len(frame) for value in graphs.values()):
                raise RuntimeError("Graph arrays and score rows are misaligned")
        return frame, graphs

    def aggregate(self, residuals: pd.DataFrame, method: str) -> pd.DataFrame:
        return aggregate_dataframe(residuals, method, EMA_ALPHA)


def primary_proxy() -> SimpleNamespace:
    return SimpleNamespace(threshold_method=PRIMARY_THRESHOLD)


def source_primary(source: SourceRun) -> dict[str, Any]:
    return read_json(source.primary_metrics)
