from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.normalization import apply_train_minmax, load_wide_flight_frame


def resolve_flight_splits(csv_root: Path, split_info_path: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    """
    Reuse the existing normal/failure split when available.
    Falls back to filename-based 80/20 split if the split JSON is missing.
    """
    csv_root = Path(csv_root)
    all_flights = sorted([path.stem for path in csv_root.glob("*.csv")])
    if split_info_path is not None and Path(split_info_path).exists():
        info = json.loads(Path(split_info_path).read_text(encoding="utf-8"))
        train_flights = [str(x) for x in info.get("train_flights", [])]
        val_flights = [str(x) for x in info.get("val_flights", [])]
        failure_flights = [x for x in all_flights if "no_failure" not in x.lower()]
        return train_flights, val_flights, failure_flights

    normal_flights = [x for x in all_flights if "no_failure" in x.lower()]
    failure_flights = [x for x in all_flights if "no_failure" not in x.lower()]
    n_val = max(1, int(round(0.2 * len(normal_flights)))) if len(normal_flights) > 1 else len(normal_flights)
    val_flights = normal_flights[:n_val]
    train_flights = normal_flights[n_val:]
    return train_flights, val_flights, failure_flights


class TranADWindowDataset(Dataset):
    """
    Sliding-window dataset over interpolated wide-flight sequences.

    Each sample corresponds to one current timestamp t and returns:
    - window: [K, M]
    - context: [K, M] for now, keeping the context input slot for later extension
    - metadata: flight / current index / current time
    """

    def __init__(
        self,
        csv_root: Path,
        flights: list[str],
        normalization_stats: dict,
        window_size: int,
        stride: int = 1,
        trim_leading_sec: float = 0.0,
        use_replication_padding: bool = True,
        context_mode: str = "window",
    ):
        self.csv_root = Path(csv_root)
        self.window_size = max(int(window_size), 1)
        self.stride = max(int(stride), 1)
        self.trim_leading_sec = float(trim_leading_sec)
        self.use_replication_padding = bool(use_replication_padding)
        self.context_mode = str(context_mode)
        self.normalization_stats = normalization_stats

        self.feature_names = [str(x) for x in normalization_stats["feature_names"]]
        self.num_features = int(len(self.feature_names))
        self.flight_data: dict[str, dict[str, np.ndarray]] = {}
        self.refs: list[tuple[str, int]] = []

        for flight in flights:
            csv_path = self.csv_root / f"{flight}.csv"
            if not csv_path.exists():
                continue
            t, x_raw, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=self.trim_leading_sec)
            if list(cols) != list(self.feature_names):
                raise ValueError(f"Feature names mismatch in {csv_path}")
            if x_raw.shape[0] <= 0:
                continue
            x = apply_train_minmax(x_raw, stats=self.normalization_stats)
            self.flight_data[str(flight)] = {
                "t": t.astype(np.float32, copy=False),
                "x": x.astype(np.float32, copy=False),
            }
            for current_idx in range(0, int(x.shape[0]), self.stride):
                self.refs.append((str(flight), int(current_idx)))

        self.flights = list(self.flight_data.keys())

    def __len__(self) -> int:
        return len(self.refs)

    def _build_window(self, x: np.ndarray, current_idx: int) -> np.ndarray:
        start = max(0, int(current_idx) - self.window_size + 1)
        window = x[start:int(current_idx) + 1]
        if window.shape[0] == self.window_size:
            return window.astype(np.float32, copy=False)
        if not self.use_replication_padding:
            pad = np.zeros((self.window_size - window.shape[0], self.num_features), dtype=np.float32)
            return np.concatenate([pad, window], axis=0).astype(np.float32, copy=False)
        pad_len = self.window_size - window.shape[0]
        pad_value = np.repeat(x[int(current_idx):int(current_idx) + 1], pad_len, axis=0)
        return np.concatenate([pad_value, window], axis=0).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        flight, current_idx = self.refs[idx]
        seq = self.flight_data[str(flight)]
        t = seq["t"]
        x = seq["x"]
        window = self._build_window(x=x, current_idx=int(current_idx))
        context = window.copy() if self.context_mode == "window" else window.copy()
        t_current = float(t[int(current_idx)])
        return {
            "window": torch.from_numpy(window).float(),
            "context": torch.from_numpy(context).float(),
            "flight": str(flight),
            "current_index": int(current_idx),
            "t": float(t_current),
        }


def collate_tranad_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "window": torch.stack([item["window"] for item in batch], dim=0),
        "context": torch.stack([item["context"] for item in batch], dim=0),
        "flight": [str(item["flight"]) for item in batch],
        "current_index": torch.tensor([int(item["current_index"]) for item in batch], dtype=torch.long),
        "t": torch.tensor([float(item["t"]) for item in batch], dtype=torch.float32),
    }
