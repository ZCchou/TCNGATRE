from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.alfa_shared import build_flight_path_maps
from utils.normalization import apply_train_minmax, load_wide_flight_frame


def resolve_flight_splits(
    csv_root: Path | None = None,
    split_info_path: Path | None = None,
    dataset_root: Path | None = None,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    del split_info_path
    root = Path(dataset_root if dataset_root is not None else csv_root)
    train_paths, val_paths, failure_paths, _ = build_flight_path_maps(root)
    return train_paths, val_paths, failure_paths


def load_graph(graph_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    graph_dir = Path(graph_dir)
    keep_cols_path = graph_dir / "keep_columns.json"
    adj_path = graph_dir / "adjacency_dense.csv"
    if not keep_cols_path.exists() or not adj_path.exists():
        raise FileNotFoundError(f"Graph artefacts missing in {graph_dir}")
    node_names = list(json.loads(keep_cols_path.read_text(encoding="utf-8")))
    adj_df = pd.read_csv(adj_path, index_col=0)
    if list(adj_df.index) != list(adj_df.columns):
        raise ValueError("adjacency_dense.csv index/columns mismatch")
    a_stat = adj_df.loc[node_names, node_names].to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(a_stat, 0.0)
    a_max = float(a_stat.max()) if a_stat.size > 0 else 0.0
    if a_max > 0.0:
        a_stat = (a_stat / a_max).astype(np.float32, copy=False)
    np.fill_diagonal(a_stat, 0.0)
    m_mask = (a_stat > 0).astype(np.float32)
    np.fill_diagonal(m_mask, 1.0)
    edges_path = graph_dir / "edges_mic.csv"
    if edges_path.exists():
        edges = pd.read_csv(edges_path)
        if len(edges) > 0:
            m_mask = np.eye(len(node_names), dtype=np.float32)
            name_to_idx = {n: i for i, n in enumerate(node_names)}
            for _, row in edges.iterrows():
                src = str(row["src"])
                dst = str(row["dst"])
                if src in name_to_idx and dst in name_to_idx:
                    i, j = name_to_idx[src], name_to_idx[dst]
                    m_mask[i, j] = 1.0
                    m_mask[j, i] = 1.0
    offdiag_degree = m_mask.sum(axis=1) - 1.0
    isolated = np.where(offdiag_degree <= 0.0)[0].tolist()
    for idx in isolated:
        m_mask[idx, :] = 1.0
        m_mask[:, idx] = 1.0
        m_mask[idx, idx] = 1.0
    return node_names, a_stat, m_mask


def resolve_node_column_names(feature_names_all: list[str], node_names: list[str]) -> list[str]:
    available = {str(name) for name in feature_names_all}
    resolved: list[str] = []
    missing: list[str] = []
    for node_name in [str(x) for x in node_names]:
        candidates = [
            node_name,
            f"{node_name}_med_mm",
            f"{node_name}_med",
        ]
        matched = next((cand for cand in candidates if cand in available), None)
        if matched is None:
            missing.append(node_name)
        else:
            resolved.append(str(matched))
    if missing:
        raise ValueError(f"Nodes not found in feature columns: {missing[:5]} ...")
    return resolved


def project_node_features(
    x_raw: np.ndarray,
    feature_names_all: list[str],
    node_names: list[str],
) -> np.ndarray:
    resolved = resolve_node_column_names(
        feature_names_all=[str(x) for x in feature_names_all],
        node_names=[str(x) for x in node_names],
    )
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names_all)}
    indices = np.array([name_to_idx[name] for name in resolved], dtype=np.int64)
    return np.asarray(x_raw[:, indices], dtype=np.float32)


class STGTCNWindowDataset(Dataset):
    def __init__(
        self,
        csv_root: Path | None,
        flights: list[str] | None,
        normalization_stats: dict,
        node_names: list[str],
        history_len: int,
        horizon: int,
        stride: int = 1,
        trim_leading_sec: float = 0.0,
        use_replication_padding: bool = True,
        flight_paths: dict[str, Path] | None = None,
    ):
        self.csv_root = None if csv_root is None else Path(csv_root)
        self.history_len = max(int(history_len), 2)
        self.horizon = max(int(horizon), 1)
        self.stride = max(int(stride), 1)
        self.trim_leading_sec = float(trim_leading_sec)
        self.use_replication_padding = bool(use_replication_padding)
        self.normalization_stats = normalization_stats
        self.node_names = [str(x) for x in node_names]
        self.num_nodes = int(len(self.node_names))

        self.feature_names_all = [str(x) for x in normalization_stats["feature_names"]]

        self.flight_data: dict[str, dict[str, np.ndarray]] = {}
        self.refs: list[tuple[str, int]] = []

        if flight_paths is not None:
            items = [(str(flight), Path(path)) for flight, path in flight_paths.items()]
        else:
            if self.csv_root is None or flights is None:
                raise ValueError("STGTCNWindowDataset requires flight_paths or (csv_root, flights)")
            items = [(str(flight), self.csv_root / f"{flight}.csv") for flight in flights]

        for flight, csv_path in items:
            if not Path(csv_path).exists():
                continue
            t, x_raw, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=self.trim_leading_sec)
            x_nodes = project_node_features(
                x_raw=x_raw,
                feature_names_all=list(cols),
                node_names=self.node_names,
            )
            if x_raw.shape[0] <= 0:
                continue
            x_nodes = apply_train_minmax(x_nodes, stats=self.normalization_stats)
            self.flight_data[str(flight)] = {
                "t": t.astype(np.float32, copy=False),
                "x": x_nodes.astype(np.float32, copy=False),
            }
            start_idx = int(self.history_len - 1)
            end_exclusive = int(x_nodes.shape[0] - self.horizon)
            for current_idx in range(start_idx, end_exclusive, self.stride):
                self.refs.append((str(flight), int(current_idx)))

        self.flights = list(self.flight_data.keys())

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        flight, current_idx = self.refs[idx]
        seq = self.flight_data[str(flight)]
        t = seq["t"]
        x = seq["x"]
        start_idx = int(current_idx) - self.history_len + 1
        history = x[start_idx : int(current_idx) + 1].astype(np.float32, copy=False)
        future = x[int(current_idx) + 1 : int(current_idx) + 1 + self.horizon].astype(np.float32, copy=False)

        t_ref = float(t[int(current_idx)])
        t_hist0 = float(t[start_idx])
        future_start_idx = min(int(current_idx) + 1, int(t.shape[0]) - 1)
        future_end_idx = min(int(t.shape[0]) - 1, int(current_idx) + self.horizon)
        t_future_start = float(t[future_start_idx]) if t.shape[0] > 0 else t_ref
        t_future_end = float(t[future_end_idx]) if future_end_idx >= 0 else t_ref

        return {
            "x": torch.from_numpy(history[:, :, None]).float(),
            "y": torch.from_numpy(future[:, :, None]).float(),
            "flight": str(flight),
            "current_index": int(current_idx),
            "t": float(t_ref),
            "t_hist0": float(t_hist0),
            "t_future_start": float(t_future_start),
            "t_future_end": float(t_future_end),
        }


def collate_stgtcn_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "flight": [str(item["flight"]) for item in batch],
        "current_index": torch.tensor([int(item["current_index"]) for item in batch], dtype=torch.long),
        "t": torch.tensor([float(item["t"]) for item in batch], dtype=torch.float32),
        "t_hist0": torch.tensor([float(item["t_hist0"]) for item in batch], dtype=torch.float32),
        "t_future_start": torch.tensor([float(item["t_future_start"]) for item in batch], dtype=torch.float32),
        "t_future_end": torch.tensor([float(item["t_future_end"]) for item in batch], dtype=torch.float32),
    }
