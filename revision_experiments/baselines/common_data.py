from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from revision_experiments.core.paths import REPO_ROOT, RESULTS_ROOT
from revision_experiments.scoring.aggregators import ema


COMMON_DATA_ROOT = RESULTS_ROOT / "protocol_v1" / "_baseline_common_data"


@dataclass(frozen=True)
class FlightRecord:
    flight: str
    path: Path
    rows: int
    channels: int


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "source": "train_normal only"}


class CommonDataBundle:
    def __init__(self, dataset: str, root: Path = COMMON_DATA_ROOT):
        self.dataset = str(dataset)
        self.root = Path(root) / self.dataset
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Common baseline data is missing: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        from revision_experiments.baselines.export_common_data import validate_common_data

        payload = validate_common_data(self.dataset, root, verify_arrays=False)
        self.nodes = list(payload["nodes"])
        self.normalization_source = payload["normalization_source"]
        self.splits = {
            "train": self._records(payload["train"]),
            "validation": self._records(payload["validation"]),
            "failure": self._records(payload["failure"]),
        }

    @staticmethod
    def _resolve_path(raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return REPO_ROOT / candidate

    def _records(self, rows: list[dict]) -> list[FlightRecord]:
        records = []
        for row in rows:
            path = self._resolve_path(row["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            records.append(FlightRecord(
                flight=str(row["flight"]),
                path=path,
                rows=int(row["rows"]),
                channels=int(row["channels"]),
            ))
        return records

    def load(self, record: FlightRecord) -> tuple[np.ndarray, np.ndarray]:
        with np.load(record.path) as data:
            time = np.asarray(data["time"], dtype=np.float64)
            values = np.asarray(data["values"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.nodes):
            raise RuntimeError(f"Unexpected shape for {record.path}: {values.shape}")
        if len(time) != len(values):
            raise RuntimeError(f"Time/value length mismatch for {record.path}")
        return time, values

    def fit_standardizer(self) -> Standardizer:
        total = np.zeros(len(self.nodes), dtype=np.float64)
        total_sq = np.zeros(len(self.nodes), dtype=np.float64)
        count = 0
        for record in self.splits["train"]:
            _, values = self.load(record)
            total += values.sum(axis=0, dtype=np.float64)
            total_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
            count += len(values)
        if count < 2:
            raise RuntimeError("Not enough normal training rows to fit a standardizer")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        return Standardizer(mean.astype(np.float32), std.astype(np.float32))


def window_starts(length: int, window: int, stride: int, max_windows: int | None = None) -> np.ndarray:
    if length < window:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - window + 1, max(int(stride), 1), dtype=np.int64)
    if max_windows is not None and len(starts) > max_windows:
        select = np.linspace(0, len(starts) - 1, num=max_windows, dtype=np.int64)
        starts = starts[select]
    return np.unique(starts)


class FlightWindowDataset(Dataset):
    """Window normal flights without ever crossing a flight boundary."""

    def __init__(
        self,
        bundle: CommonDataBundle,
        split: str,
        standardizer: Standardizer,
        window: int,
        stride: int,
        max_windows_per_flight: int | None = None,
    ):
        self.window = int(window)
        self.entries: list[tuple[np.ndarray, int]] = []
        for record in bundle.splits[split]:
            _, values = bundle.load(record)
            values = standardizer.transform(values)
            starts = window_starts(len(values), self.window, stride, max_windows_per_flight)
            self.entries.extend((values, int(start)) for start in starts)
        if not self.entries:
            raise RuntimeError(f"No windows for {bundle.dataset}/{split}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> torch.Tensor:
        values, start = self.entries[index]
        return torch.from_numpy(values[start:start + self.window].copy())


def make_loader(
    bundle: CommonDataBundle,
    split: str,
    standardizer: Standardizer,
    window: int,
    stride: int,
    batch_size: int,
    max_windows_per_flight: int | None,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = FlightWindowDataset(
        bundle=bundle,
        split=split,
        standardizer=standardizer,
        window=window,
        stride=stride,
        max_windows_per_flight=max_windows_per_flight,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def iter_scoring_flights(
    bundle: CommonDataBundle,
    split: str,
    standardizer: Standardizer,
    window: int,
    stride: int,
    max_windows_per_flight: int | None,
) -> Iterator[tuple[FlightRecord, np.ndarray, np.ndarray, np.ndarray]]:
    for record in bundle.splits[split]:
        time, values = bundle.load(record)
        values = standardizer.transform(values)
        starts = window_starts(len(values), window, stride, max_windows_per_flight)
        if len(starts):
            yield record, time, values, starts


def batch_windows(values: np.ndarray, starts: np.ndarray, window: int, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for offset in range(0, len(starts), int(batch_size)):
        current = starts[offset:offset + int(batch_size)]
        windows = np.stack([values[int(start):int(start) + int(window)] for start in current], axis=0)
        yield current, windows.astype(np.float32, copy=False)


def score_split(
    bundle: CommonDataBundle,
    split: str,
    standardizer: Standardizer,
    window: int,
    stride: int,
    batch_size: int,
    max_windows_per_flight: int | None,
    score_batch: Callable[[torch.Tensor], tuple[np.ndarray, np.ndarray | None]],
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict] = []
    for record, time, values, starts in iter_scoring_flights(
        bundle, split, standardizer, window, stride, max_windows_per_flight
    ):
        for batch_starts, windows in batch_windows(values, starts, window, batch_size):
            total, channel = score_batch(torch.from_numpy(windows).to(device))
            total = np.asarray(total, dtype=np.float64).reshape(-1)
            if len(total) != len(batch_starts):
                raise RuntimeError("Score batch size mismatch")
            if channel is not None:
                channel = np.asarray(channel, dtype=np.float64)
                if channel.shape[0] != len(batch_starts):
                    raise RuntimeError("Channel score batch size mismatch")
            for index, start in enumerate(batch_starts):
                end = int(start) + int(window) - 1
                row = {
                    "flight": record.flight,
                    "current_index": end,
                    "t_start": float(time[end]),
                    "t_end": float(time[end]),
                    "t_mid": float(time[end]),
                    "raw_total_score": float(total[index]),
                }
                if channel is not None:
                    row["sensor_score_vec"] = json.dumps(channel[index].astype(np.float32).tolist())
                    row["valid_dim_count"] = int(channel.shape[1])
                rows.append(row)
    if not rows:
        raise RuntimeError(f"Scoring produced no rows for {bundle.dataset}/{split}")
    return pd.DataFrame(rows)


def apply_flightwise_ema(frame: pd.DataFrame, alpha: float, method: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("flight", sort=False):
        current = group.sort_values("t_start", kind="mergesort").reset_index(drop=True).copy()
        current["total_score"] = ema(current["raw_total_score"].to_numpy(dtype=np.float64), alpha)
        current["aggregation_method"] = method
        if "valid_dim_count" not in current:
            current["valid_dim_count"] = 1
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adapter_config_hash(cfg, baseline: str, parameters: dict, source_commit: str, adapter_file: Path) -> str:
    finalize_file = Path(adapter_file).with_name("finalize.py")
    manifest_file = COMMON_DATA_ROOT / cfg.dataset / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Common baseline manifest is missing: {manifest_file}")
    payload = {
        "revision_config": cfg.to_dict(),
        "baseline": baseline,
        "parameters": parameters,
        "source_commit": source_commit,
        "adapter_sha256": hashlib.sha256(Path(adapter_file).read_bytes()).hexdigest(),
        "common_data_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "common_data_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "finalize_sha256": hashlib.sha256(finalize_file.read_bytes()).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
