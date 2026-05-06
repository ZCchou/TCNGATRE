from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


def build_phase_features(
    hist_last_value: np.ndarray,
    hist_has_value: np.ndarray,
    history_event_count: np.ndarray,
    history_valid_ratio: np.ndarray,
    win_mask: np.ndarray,
) -> np.ndarray:
    last = np.asarray(hist_last_value, dtype=np.float32).reshape(-1)
    has = np.asarray(hist_has_value, dtype=np.float32).reshape(-1)
    event_count = np.asarray(history_event_count, dtype=np.float32).reshape(-1)
    valid_ratio = np.asarray(history_valid_ratio, dtype=np.float32).reshape(-1)
    win_mask = np.asarray(win_mask, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            last * has,
            has,
            np.log1p(np.maximum(event_count, 0.0)),
            valid_ratio,
            win_mask,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def fit_pseudo_phase_clusterer(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
) -> dict[str, np.ndarray | int]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError(f"features must be non-empty 2D, got shape={x.shape}")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    x_norm = (x - mean) / std
    k = max(2, min(int(num_clusters), int(x_norm.shape[0])))
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=int(seed))
    kmeans.fit(x_norm)
    return {
        "num_clusters": int(k),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "centers": np.asarray(kmeans.cluster_centers_, dtype=np.float32),
    }


def phase_probabilities(
    phase_feature: np.ndarray,
    clusterer: dict[str, np.ndarray | int],
    temperature: float,
) -> np.ndarray:
    feat = np.asarray(phase_feature, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[None, :]
    mean = np.asarray(clusterer["feature_mean"], dtype=np.float32)
    std = np.asarray(clusterer["feature_std"], dtype=np.float32)
    centers = np.asarray(clusterer["centers"], dtype=np.float32)
    x_norm = (feat - mean) / np.maximum(std, 1e-6)
    diff = x_norm[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 1e-12))
    logits = -dist / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs.astype(np.float32)


def save_pseudo_phase_clusterer(clusterer: dict[str, np.ndarray | int], path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        num_clusters=np.asarray([int(clusterer["num_clusters"])], dtype=np.int64),
        feature_mean=np.asarray(clusterer["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(clusterer["feature_std"], dtype=np.float32),
        centers=np.asarray(clusterer["centers"], dtype=np.float32),
    )


def load_pseudo_phase_clusterer(path: str | Path) -> dict[str, np.ndarray | int]:
    data = np.load(str(path))
    return {
        "num_clusters": int(np.asarray(data["num_clusters"]).reshape(-1)[0]),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
        "centers": np.asarray(data["centers"], dtype=np.float32),
    }
