from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def load_wide_flight_frame(csv_path: Path, trim_leading_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one wide-flight CSV and return trimmed time, values, and feature names."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    if "t" not in df.columns:
        raise ValueError(f"Missing 't' column in {csv_path}")

    df = df.copy()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t", kind="mergesort").reset_index(drop=True)
    if len(df) <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

    t = df["t"].to_numpy(dtype=np.float32, copy=True)
    if float(trim_leading_sec) > 0.0 and t.size > 0:
        keep = t >= float(t[0]) + float(trim_leading_sec)
        df = df.loc[keep].reset_index(drop=True)
        t = df["t"].to_numpy(dtype=np.float32, copy=True)

    feature_names = [str(col) for col in df.columns.tolist() if str(col) != "t"]
    if not feature_names:
        return t.astype(np.float32, copy=False), np.zeros((len(t), 0), dtype=np.float32), []
    x = df[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    return t.astype(np.float32, copy=False), x, feature_names


def fit_train_minmax(
    csv_root: Path | None = None,
    flights: list[str] | None = None,
    trim_leading_sec: float = 0.0,
    eps: float = 1e-8,
    flight_paths: Mapping[str, Path] | None = None,
) -> dict:
    """Fit per-feature min-max stats using only normal training flights."""
    train_min = None
    train_max = None
    feature_names: list[str] | None = None
    total_rows = 0

    if flight_paths is not None:
        items = [(str(flight), Path(path)) for flight, path in flight_paths.items()]
    else:
        if csv_root is None or flights is None:
            raise ValueError("fit_train_minmax requires either flight_paths or (csv_root, flights)")
        csv_root = Path(csv_root)
        items = [(str(flight), csv_root / f"{flight}.csv") for flight in flights]

    for flight, csv_path in items:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Training flight csv not found: {csv_path}")
        _, x, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=trim_leading_sec)
        if feature_names is None:
            feature_names = list(cols)
        elif list(cols) != list(feature_names):
            raise ValueError(f"Feature columns mismatch in {csv_path}")
        if x.size <= 0:
            continue
        cur_min = np.nanmin(x, axis=0)
        cur_max = np.nanmax(x, axis=0)
        train_min = cur_min if train_min is None else np.minimum(train_min, cur_min)
        train_max = cur_max if train_max is None else np.maximum(train_max, cur_max)
        total_rows += int(x.shape[0])

    if feature_names is None or train_min is None or train_max is None:
        raise ValueError("No valid training rows were available to fit normalization stats")

    all_nan_mask = ~np.isfinite(train_min) | ~np.isfinite(train_max)
    train_min = np.where(all_nan_mask, 0.0, train_min)
    train_max = np.where(all_nan_mask, 1.0, train_max)
    scale = np.maximum(train_max - train_min, float(eps))

    return {
        "feature_names": [str(x) for x in feature_names],
        "train_min": train_min.astype(np.float64).tolist(),
        "train_max": train_max.astype(np.float64).tolist(),
        "scale": scale.astype(np.float64).tolist(),
        "all_nan_mask": all_nan_mask.astype(np.int32).tolist(),
        "trim_leading_sec": float(trim_leading_sec),
        "num_train_flights": int(len(items)),
        "num_train_rows": int(total_rows),
        "eps": float(eps),
    }


def apply_train_minmax(x: np.ndarray, stats: dict) -> np.ndarray:
    """Apply saved train-only min-max stats and clamp non-finite outputs to 0."""
    arr = np.asarray(x, dtype=np.float32)
    train_min = np.asarray(stats["train_min"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    out = (arr - train_min.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), 1e-8)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


def save_minmax_stats(stats: dict, path: Path):
    Path(path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def load_minmax_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
