"""STGTCN sliding-window forecasting dataset.

Each sample is a forecast of the next `horizon` steps given the previous
`history_len` steps. Both are tensors over all graph nodes (sensors).

Returned tensors:
  x        : (T_hist, N, in_feat=1)   history window values
  y        : (horizon, N, in_feat=1)  target future values
  y_mask   : (horizon, N)             1 if target step is observed in-range, else 0
  zmask    : (T_hist, N)              history observation mask (1 if in-range, else 0)
  flight   : str                      flight name
  t_ref    : float                    time (sec) of current step (last history step)
  t_hist0  : float                    time of first history step
  t_future_end : float                time of last forecast step
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.normalization import apply_train_minmax, load_wide_flight_frame


def load_graph(graph_dir: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load keep_columns + dense adjacency from a static graph directory.

    Returns:
      node_names : list[str]
      a_stat     : (N, N) float32 dense adjacency (self-loops set to 1.0)
      m_mask     : (N, N) float32 connectivity mask (1 where edge exists or diag)
    """
    graph_dir = Path(graph_dir)
    keep_cols_path = graph_dir / "keep_columns.json"
    adj_path = graph_dir / "adjacency_dense.csv"
    if not keep_cols_path.exists() or not adj_path.exists():
        raise FileNotFoundError(f"Graph artefacts missing in {graph_dir}")
    node_names = list(json.loads(keep_cols_path.read_text(encoding="utf-8")))
    adj_df = pd.read_csv(adj_path, index_col=0)
    if list(adj_df.index) != list(adj_df.columns):
        raise ValueError("adjacency_dense.csv index/columns mismatch")
    A = adj_df.loc[node_names, node_names].to_numpy(dtype=np.float32, copy=True)
    # Template convention: diagonal = 0, then normalize by max so a_stat is in [0,1].
    # The model masks self-loops internally in DynamicGraphAttention via an explicit
    # eye-mask, so self-mixing is handled by the residual connection in SpatialEncoder.
    np.fill_diagonal(A, 0.0)
    a_max = float(A.max()) if A.size > 0 else 0.0
    if a_max > 0.0:
        A = (A / a_max).astype(np.float32, copy=False)
    np.fill_diagonal(A, 0.0)
    # Connectivity mask: 1 where edge exists OR diagonal (self-loops allowed for residual).
    m_mask = (A > 0).astype(np.float32)
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
    return node_names, A, m_mask


class STGTCNWindowDataset(Dataset):
    """Sliding window dataset for (B, T, N, F=1) forecasting.

    Each reference index (flight, current_idx) defines a sample where history
    covers [current_idx - history_len + 1 ... current_idx] and target covers
    [current_idx + 1 ... current_idx + horizon]. Samples near the end of a flight
    (where future would run past the last observed row) are still produced, but
    the missing future rows are zero-filled and y_mask marks them as unobserved.
    """

    def __init__(
        self,
        csv_root: Path,
        flights: List[str],
        normalization_stats: dict,
        node_names: List[str],
        history_len: int,
        horizon: int,
        stride: int = 1,
        trim_leading_sec: float = 0.0,
        use_replication_padding: bool = True,
    ):
        self.csv_root = Path(csv_root)
        self.history_len = max(int(history_len), 2)
        self.horizon = max(int(horizon), 1)
        self.stride = max(int(stride), 1)
        self.trim_leading_sec = float(trim_leading_sec)
        self.use_replication_padding = bool(use_replication_padding)
        self.normalization_stats = normalization_stats
        self.node_names = [str(x) for x in node_names]
        self.num_nodes = int(len(self.node_names))

        self.feature_names_all = [str(x) for x in normalization_stats["feature_names"]]
        # Build index from node_names -> position in feature list
        name_to_idx = {n: i for i, n in enumerate(self.feature_names_all)}
        missing = [n for n in self.node_names if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"Nodes not found in normalization feature list: {missing[:5]} ..."
            )
        self._node_indices = np.array([name_to_idx[n] for n in self.node_names], dtype=np.int64)

        self.flight_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.refs: List[Tuple[str, int]] = []

        for flight in flights:
            csv_path = self.csv_root / f"{flight}.csv"
            if not csv_path.exists():
                continue
            t, x_raw, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=self.trim_leading_sec)
            if list(cols) != list(self.feature_names_all):
                raise ValueError(f"Feature names mismatch in {csv_path}")
            if x_raw.shape[0] <= 0:
                continue
            x = apply_train_minmax(x_raw, stats=self.normalization_stats)
            x_nodes = x[:, self._node_indices].astype(np.float32, copy=False)
            self.flight_data[str(flight)] = {
                "t": t.astype(np.float32, copy=False),
                "x": x_nodes,
            }
            for current_idx in range(0, int(x_nodes.shape[0]), self.stride):
                self.refs.append((str(flight), int(current_idx)))

        self.flights = list(self.flight_data.keys())

    def __len__(self) -> int:
        return len(self.refs)

    def _build_history(self, x: np.ndarray, current_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = max(0, int(current_idx) - self.history_len + 1)
        window = x[start:int(current_idx) + 1]
        z = np.ones((window.shape[0], self.num_nodes), dtype=np.float32)
        if window.shape[0] == self.history_len:
            return window, z
        pad_len = self.history_len - window.shape[0]
        if self.use_replication_padding and window.shape[0] > 0:
            pad_values = np.repeat(window[0:1], pad_len, axis=0)
        else:
            pad_values = np.zeros((pad_len, self.num_nodes), dtype=np.float32)
        pad_mask = np.zeros((pad_len, self.num_nodes), dtype=np.float32)
        history = np.concatenate([pad_values, window], axis=0).astype(np.float32, copy=False)
        mask = np.concatenate([pad_mask, z], axis=0).astype(np.float32, copy=False)
        return history, mask

    def _build_future(self, x: np.ndarray, current_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        end = int(current_idx) + 1 + self.horizon
        future = x[int(current_idx) + 1:end]
        y_mask = np.ones((future.shape[0], self.num_nodes), dtype=np.float32)
        if future.shape[0] == self.horizon:
            return future, y_mask
        pad_len = self.horizon - future.shape[0]
        pad = np.zeros((pad_len, self.num_nodes), dtype=np.float32)
        pad_mask = np.zeros((pad_len, self.num_nodes), dtype=np.float32)
        future = np.concatenate([future, pad], axis=0).astype(np.float32, copy=False)
        y_mask = np.concatenate([y_mask, pad_mask], axis=0).astype(np.float32, copy=False)
        return future, y_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        flight, current_idx = self.refs[idx]
        seq = self.flight_data[str(flight)]
        t = seq["t"]
        x = seq["x"]
        history, zmask = self._build_history(x=x, current_idx=int(current_idx))  # (T, N)
        future, y_mask = self._build_future(x=x, current_idx=int(current_idx))    # (H, N)

        # Shape to (T, N, 1) / (H, N, 1)
        x_t = torch.from_numpy(history[:, :, None]).float()
        y_t = torch.from_numpy(future[:, :, None]).float()
        zmask_t = torch.from_numpy(zmask).float()
        y_mask_t = torch.from_numpy(y_mask).float()

        t_ref = float(t[int(current_idx)])
        start_idx = max(0, int(current_idx) - self.history_len + 1)
        t_hist0 = float(t[start_idx])
        end_idx = min(int(t.shape[0]) - 1, int(current_idx) + self.horizon)
        t_future_end = float(t[end_idx]) if end_idx >= 0 else t_ref

        return {
            "x": x_t,                     # (T, N, 1)
            "y": y_t,                     # (H, N, 1)
            "zmask": zmask_t,            # (T, N)
            "y_mask": y_mask_t,          # (H, N)
            "flight": str(flight),
            "current_index": int(current_idx),
            "t": float(t_ref),
            "t_hist0": float(t_hist0),
            "t_future_end": float(t_future_end),
        }


def collate_stgtcn_windows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "zmask": torch.stack([b["zmask"] for b in batch], dim=0),
        "y_mask": torch.stack([b["y_mask"] for b in batch], dim=0),
        "flight": [str(b["flight"]) for b in batch],
        "current_index": torch.tensor([int(b["current_index"]) for b in batch], dtype=torch.long),
        "t": torch.tensor([float(b["t"]) for b in batch], dtype=torch.float32),
        "t_hist0": torch.tensor([float(b["t_hist0"]) for b in batch], dtype=torch.float32),
        "t_future_end": torch.tensor([float(b["t_future_end"]) for b in batch], dtype=torch.float32),
    }
