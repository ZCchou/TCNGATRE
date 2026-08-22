from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from revision_experiments.core.paths import ensure_import_paths

from .io import stable_seed

ensure_import_paths()

from data.stgtcn_window_dataset import project_node_features  # noqa: E402
from utils.normalization import apply_train_minmax, load_wide_flight_frame  # noqa: E402


@dataclass(frozen=True)
class FlightArray:
    flight: str
    time: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ChannelStatistics:
    median: np.ndarray
    std: np.ndarray
    robust_scale: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "median": self.median.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "robust_scale": self.robust_scale.astype(float).tolist(),
        }


def load_normalized_flights(
    flight_paths: dict[str, Path],
    normalization_stats: dict,
    node_names: list[str],
    trim_leading_sec: float = 0.0,
) -> dict[str, FlightArray]:
    result: dict[str, FlightArray] = {}
    for flight, path in flight_paths.items():
        time, raw, columns = load_wide_flight_frame(Path(path), trim_leading_sec)
        nodes = project_node_features(raw, list(columns), node_names)
        values = apply_train_minmax(nodes, normalization_stats)
        if len(time) == 0 or values.shape[0] == 0:
            continue
        result[str(flight)] = FlightArray(
            flight=str(flight),
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )
    if not result:
        raise RuntimeError("No normalized flight arrays were loaded")
    return result


def fit_channel_statistics(train_flights: dict[str, FlightArray]) -> ChannelStatistics:
    values = np.concatenate([item.values for item in train_flights.values()], axis=0).astype(np.float64)
    median = np.nanmedian(values, axis=0)
    std = np.nanstd(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    robust = (q75 - q25) / 1.349
    median = np.where(np.isfinite(median), median, 0.5)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1e-8)
    robust = np.where(np.isfinite(robust) & (robust > 1e-8), robust, std)
    return ChannelStatistics(
        median=median.astype(np.float32),
        std=std.astype(np.float32),
        robust_scale=robust.astype(np.float32),
    )


class ArrayWindowDataset(Dataset):
    """Boundary-safe window dataset backed by in-memory, already normalized flights."""

    def __init__(
        self,
        flights: dict[str, FlightArray],
        history: int,
        horizon: int,
        stride: int,
    ) -> None:
        self.flights = flights
        self.history = max(int(history), 2)
        self.horizon = max(int(horizon), 1)
        self.stride = max(int(stride), 1)
        self.refs: list[tuple[str, int]] = []
        for flight, item in flights.items():
            for current in range(self.history - 1, len(item.time) - self.horizon, self.stride):
                self.refs.append((flight, int(current)))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        flight, current = self.refs[index]
        item = self.flights[flight]
        start = current - self.history + 1
        future_start = current + 1
        future_end = current + self.horizon
        return {
            "x": torch.from_numpy(item.values[start : current + 1, :, None]).float(),
            "y": torch.from_numpy(item.values[future_start : future_end + 1, :, None]).float(),
            "flight": flight,
            "current_index": current,
            "t": float(item.time[current]),
            "t_hist0": float(item.time[start]),
            "t_future_start": float(item.time[future_start]),
            "t_future_end": float(item.time[future_end]),
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "flight": [str(item["flight"]) for item in batch],
        "current_index": torch.tensor([int(item["current_index"]) for item in batch]),
        "t": torch.tensor([float(item["t"]) for item in batch]),
        "t_hist0": torch.tensor([float(item["t_hist0"]) for item in batch]),
        "t_future_start": torch.tensor([float(item["t_future_start"]) for item in batch]),
        "t_future_end": torch.tensor([float(item["t_future_end"]) for item in batch]),
    }


def clone_flights(flights: dict[str, FlightArray]) -> dict[str, FlightArray]:
    return {
        name: FlightArray(name, item.time.copy(), item.values.copy())
        for name, item in flights.items()
    }


def _forward_fill(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    for channel in range(out.shape[1]):
        previous = float(fill[channel])
        for index in range(out.shape[0]):
            if np.isfinite(out[index, channel]):
                previous = float(out[index, channel])
            else:
                out[index, channel] = previous
    return out


def corrupt_full_flights(
    flights: dict[str, FlightArray],
    kind: str,
    level: float,
    statistics: ChannelStatistics,
    perturbation_seed: int,
) -> tuple[dict[str, FlightArray], list[dict[str, Any]]]:
    output: dict[str, FlightArray] = {}
    manifests: list[dict[str, Any]] = []
    for flight, item in flights.items():
        seed = stable_seed("ex08", perturbation_seed, kind, level, flight)
        rng = np.random.default_rng(seed)
        values = item.values.copy()
        metadata: dict[str, Any] = {
            "flight": flight, "kind": kind, "level": float(level), "seed": int(seed)
        }
        if kind == "gaussian":
            noise = rng.normal(0.0, 1.0, size=values.shape).astype(np.float32)
            values += noise * (float(level) * statistics.std[None, :])
        elif kind == "missing":
            mask = rng.random(values.shape) < float(level)
            values[mask] = np.nan
            values = _forward_fill(values, statistics.median)
            metadata["masked_fraction"] = float(mask.mean())
        elif kind == "channel_dropout":
            count = min(max(int(round(level)), 1), values.shape[1])
            channels = np.sort(rng.choice(values.shape[1], size=count, replace=False))
            values[:, channels] = statistics.median[channels]
            metadata["channels"] = channels.astype(int).tolist()
        elif kind == "downsample":
            factor = min(max(int(round(level)), 1), len(values))
            source = (np.arange(len(values)) // factor) * factor
            values = values[source]
            metadata["factor"] = factor
        else:
            raise ValueError(f"Unsupported corruption: {kind}")
        if values.shape != item.values.shape or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid corruption result for {flight}/{kind}_{level}")
        output[flight] = FlightArray(flight, item.time.copy(), values.astype(np.float32))
        manifests.append(metadata)
    return output, manifests


def inject_events(
    item: FlightArray,
    channels: int,
    kind: str,
    severity: float | None,
    duration: int,
    robust_scale: np.ndarray,
    scenario_seed: int,
) -> tuple[FlightArray, np.ndarray, list[dict[str, Any]]]:
    """Inject three deterministic, non-overlapping events into one normal flight."""
    values = item.values.copy()
    original = item.values.copy()
    labels = np.zeros_like(values, dtype=np.int8)
    rng = np.random.default_rng(int(scenario_seed))
    events: list[dict[str, Any]] = []
    duration = max(int(duration), 1)
    anchors = (0.30, 0.50, 0.70)
    for event_index, fraction in enumerate(anchors):
        start = int(round(fraction * max(len(values) - duration - 1, 1)))
        start = min(max(start, 1), max(len(values) - duration, 1))
        end = min(start + duration, len(values))
        selected = np.sort(rng.choice(values.shape[1], size=min(channels, values.shape[1]), replace=False))
        scale = np.maximum(robust_scale[selected], 1e-8)
        if kind == "bias":
            values[start:end, selected] += float(severity) * scale
        elif kind == "drift":
            ramp = np.linspace(0.0, float(severity), end - start, dtype=np.float32)[:, None]
            values[start:end, selected] += ramp * scale[None, :]
        elif kind == "freeze":
            values[start:end, selected] = values[start - 1, selected]
        elif kind == "noise":
            values[start:end, selected] += rng.normal(
                0.0, float(severity), size=(end - start, len(selected))
            ).astype(np.float32) * scale[None, :]
        else:
            raise ValueError(f"Unsupported local anomaly: {kind}")
        labels[start:end, selected] = 1
        events.append({
            "event": event_index,
            "start_index": start,
            "end_index_exclusive": end,
            "t_start": float(item.time[start]),
            "t_end": float(item.time[end - 1]),
            "channels": selected.astype(int).tolist(),
        })
    if np.array_equal(values, original):
        raise RuntimeError(f"Synthetic scenario produced no change: {item.flight}/{kind}")
    return FlightArray(item.flight, item.time.copy(), values.astype(np.float32)), labels, events


def parse_condition(condition: str) -> tuple[str, float]:
    prefix, raw = str(condition).rsplit("_", 1)
    return prefix, float(raw)
