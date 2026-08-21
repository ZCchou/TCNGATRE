from __future__ import annotations

import json

import numpy as np
import pandas as pd


AGGREGATORS = ("mean", "max", "topk_1", "topk_3", "topk_5", "quantile_90", "quantile_95")


def aggregate_channels(sensor_scores: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(sensor_scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"sensor_scores must be [samples, channels], got {values.shape}")
    if method == "mean":
        return np.mean(values, axis=1)
    if method == "max":
        return np.max(values, axis=1)
    if method.startswith("topk_"):
        k = min(max(int(method.split("_")[1]), 1), values.shape[1])
        top = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
        return np.mean(top, axis=1)
    if method.startswith("quantile_"):
        q = int(method.split("_")[1]) / 100.0
        return np.quantile(values, q, axis=1)
    raise ValueError(f"Unknown aggregation method: {method}")


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    previous = np.nan
    for idx, value in enumerate(arr):
        previous = value if not np.isfinite(previous) else alpha * value + (1.0 - alpha) * previous
        out[idx] = previous
    return out


def _vectors(series: pd.Series) -> np.ndarray:
    rows = []
    for value in series.tolist():
        rows.append(np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64))
    return np.stack(rows, axis=0)


def aggregate_dataframe(df: pd.DataFrame, method: str, alpha: float = 0.25) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("flight", sort=False):
        current = group.sort_values("t_start", kind="mergesort").reset_index(drop=True).copy()
        if "sensor_score_vec" in current:
            sensor_scores = _vectors(current["sensor_score_vec"])
        else:
            value = _vectors(current["value_residual_vec"])
            delta = _vectors(current["delta_residual_vec"])
            sensor_scores = value + 0.20 * delta
        raw = aggregate_channels(sensor_scores, method)
        current["sensor_score_vec"] = [
            json.dumps(row.astype(np.float32).tolist(), ensure_ascii=False) for row in sensor_scores
        ]
        current["raw_total_score"] = raw
        current["total_score"] = ema(raw, alpha)
        current["aggregation_method"] = method
        current["valid_dim_count"] = sensor_scores.shape[1]
        parts.append(current)
    return pd.concat(parts, ignore_index=True)
