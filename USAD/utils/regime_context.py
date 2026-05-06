from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None


def _safe_float_array(x: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _fit_numpy_kmeans(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    max_iter: int = 100,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError(f"features must be non-empty 2D, got shape={x.shape}")

    rng = np.random.default_rng(int(seed))
    num_samples = int(x.shape[0])
    k = max(1, min(int(num_clusters), num_samples))

    # KMeans++ style seeding to keep the fallback stable and reasonably close
    # to sklearn's behavior when that dependency is unavailable.
    first_idx = int(rng.integers(0, num_samples))
    centers = [x[first_idx].copy()]
    while len(centers) < k:
        cur = np.stack(centers, axis=0)
        dist_sq = np.min(np.sum((x[:, None, :] - cur[None, :, :]) ** 2, axis=-1), axis=1)
        total = float(np.sum(dist_sq))
        if total <= 1e-12:
            remaining = [idx for idx in range(num_samples) if not any(np.allclose(x[idx], c) for c in centers)]
            if not remaining:
                break
            centers.append(x[int(remaining[0])].copy())
            continue
        probs = dist_sq / total
        pick = int(rng.choice(num_samples, p=probs))
        centers.append(x[pick].copy())

    if len(centers) < k:
        while len(centers) < k:
            centers.append(x[len(centers) % num_samples].copy())
    centers_arr = np.stack(centers[:k], axis=0).astype(np.float32, copy=False)

    for _ in range(max(1, int(max_iter))):
        dist_sq = np.sum((x[:, None, :] - centers_arr[None, :, :]) ** 2, axis=-1)
        assign = np.argmin(dist_sq, axis=1)
        new_centers = centers_arr.copy()
        for cluster_idx in range(k):
            mask = assign == cluster_idx
            if np.any(mask):
                new_centers[cluster_idx] = np.mean(x[mask], axis=0, dtype=np.float32)
            else:
                new_centers[cluster_idx] = x[int(rng.integers(0, num_samples))]
        if np.allclose(new_centers, centers_arr, atol=1e-5, rtol=0.0):
            centers_arr = new_centers
            break
        centers_arr = new_centers
    return centers_arr.astype(np.float32, copy=False)


def uses_regime_features(
    forecast_use_regime_context: bool,
    score_use_regime_calibration: bool,
) -> bool:
    return bool(forecast_use_regime_context) or bool(score_use_regime_calibration)


def make_dummy_regime_clusterer(num_clusters: int) -> dict[str, np.ndarray | int]:
    k = max(2, int(num_clusters))
    return {
        "num_clusters": int(k),
        "feature_mean": np.zeros((1, 1), dtype=np.float32),
        "feature_std": np.ones((1, 1), dtype=np.float32),
        "centers": np.zeros((int(k), 1), dtype=np.float32),
    }


def build_regime_features(
    hist_last_value: np.ndarray,
    hist_has_value: np.ndarray,
    history_event_count: np.ndarray,
    history_valid_ratio: np.ndarray,
    win_mask: np.ndarray,
    hist_patch_last_value_seq: np.ndarray | None = None,
    hist_patch_has_value_seq: np.ndarray | None = None,
) -> np.ndarray:
    last = _safe_float_array(hist_last_value).reshape(-1)
    has = _safe_float_array(hist_has_value).reshape(-1)
    event_count = _safe_float_array(history_event_count).reshape(-1)
    valid_ratio = _safe_float_array(history_valid_ratio).reshape(-1)
    win_mask = _safe_float_array(win_mask).reshape(-1)

    pieces: list[np.ndarray] = [
        last * has,
        has,
        np.log1p(np.maximum(event_count, 0.0)),
        valid_ratio,
        win_mask,
        np.asarray(
            [
                float(np.mean(event_count)) if event_count.size > 0 else 0.0,
                float(np.std(event_count)) if event_count.size > 0 else 0.0,
                float(np.mean(valid_ratio)) if valid_ratio.size > 0 else 0.0,
                float(np.std(valid_ratio)) if valid_ratio.size > 0 else 0.0,
                float(np.mean(win_mask)) if win_mask.size > 0 else 0.0,
            ],
            dtype=np.float32,
        ),
    ]

    if hist_patch_last_value_seq is not None and hist_patch_has_value_seq is not None:
        value_seq = _safe_float_array(hist_patch_last_value_seq)
        has_seq = _safe_float_array(hist_patch_has_value_seq)
        if value_seq.ndim != 2 or has_seq.shape != value_seq.shape:
            raise ValueError(
                f"hist patch seq shape mismatch: value_seq={value_seq.shape}, has_seq={has_seq.shape}"
            )
        filled = value_seq.copy()
        for t in range(1, filled.shape[0]):
            filled[t] = np.where(has_seq[t] > 0.5, filled[t], filled[t - 1])
        if filled.shape[0] >= 2:
            delta_seq = filled[1:] - filled[:-1]
            valid_delta_mask = (has_seq[1:] > 0.5) & (has_seq[:-1] > 0.5)
        else:
            delta_seq = np.zeros((0, filled.shape[1]), dtype=np.float32)
            valid_delta_mask = np.zeros((0, filled.shape[1]), dtype=bool)

        active_sensor_ratio = np.mean(has_seq, axis=1).astype(np.float32, copy=False)
        level_mean = np.where(has > 0.5, np.mean(filled, axis=0), 0.0).astype(np.float32, copy=False)
        level_range = np.where(
            np.sum(has_seq, axis=0) > 0.0,
            np.max(np.where(has_seq > 0.5, filled, -np.inf), axis=0)
            - np.min(np.where(has_seq > 0.5, filled, np.inf), axis=0),
            0.0,
        ).astype(np.float32, copy=False)

        if delta_seq.shape[0] > 0:
            delta_mean = np.where(
                np.sum(valid_delta_mask, axis=0) > 0,
                np.sum(delta_seq * valid_delta_mask, axis=0) / np.maximum(np.sum(valid_delta_mask, axis=0), 1.0),
                0.0,
            ).astype(np.float32, copy=False)
            delta_abs_mean = np.where(
                np.sum(valid_delta_mask, axis=0) > 0,
                np.sum(np.abs(delta_seq) * valid_delta_mask, axis=0) / np.maximum(np.sum(valid_delta_mask, axis=0), 1.0),
                0.0,
            ).astype(np.float32, copy=False)
            delta_zero_ratio = np.where(
                np.sum(valid_delta_mask, axis=0) > 0,
                np.sum((np.abs(delta_seq) <= 2e-3) * valid_delta_mask, axis=0) / np.maximum(np.sum(valid_delta_mask, axis=0), 1.0),
                0.0,
            ).astype(np.float32, copy=False)
        else:
            delta_mean = np.zeros((value_seq.shape[1],), dtype=np.float32)
            delta_abs_mean = np.zeros((value_seq.shape[1],), dtype=np.float32)
            delta_zero_ratio = np.zeros((value_seq.shape[1],), dtype=np.float32)

        pieces.extend(
            [
                level_mean * has,
                level_range * has,
                delta_mean * has,
                delta_abs_mean * has,
                delta_zero_ratio * has,
                active_sensor_ratio,
                np.asarray(
                    [
                        float(np.mean(active_sensor_ratio)) if active_sensor_ratio.size > 0 else 0.0,
                        float(np.std(active_sensor_ratio)) if active_sensor_ratio.size > 0 else 0.0,
                        float(np.mean(np.abs(delta_seq))) if delta_seq.size > 0 else 0.0,
                        float(np.mean(delta_zero_ratio)) if delta_zero_ratio.size > 0 else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    feat = np.concatenate([p.reshape(-1) for p in pieces], axis=0)
    return feat.astype(np.float32, copy=False)


def fit_regime_clusterer(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
) -> dict[str, np.ndarray | int]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError(f"features must be non-empty 2D, got shape={x.shape}")
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    x_norm = (x - mean) / std
    k = max(2, min(int(num_clusters), int(x_norm.shape[0])))
    if KMeans is not None:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=int(seed))
        kmeans.fit(x_norm)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    else:
        centers = _fit_numpy_kmeans(
            features=x_norm,
            num_clusters=k,
            seed=seed,
            max_iter=100,
        )
    return {
        "num_clusters": int(k),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "centers": centers,
    }


def _align_regime_feature_dims(
    feat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align runtime regime features with persisted clusterer statistics.

    Older experiment directories may carry clusterers fitted with a slightly
    different regime feature layout. For replay and threshold refits, we keep
    the shared feature prefix and fill any missing tail dimensions with the
    cluster mean so the normalized value defaults to zero contribution.
    """

    feat_arr = np.asarray(feat, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    centers_arr = np.asarray(centers, dtype=np.float32)

    if mean_arr.ndim == 1:
        mean_arr = mean_arr[None, :]
    if std_arr.ndim == 1:
        std_arr = std_arr[None, :]

    target_dim = int(mean_arr.shape[-1])
    cur_dim = int(feat_arr.shape[-1])
    if cur_dim == target_dim:
        return feat_arr, mean_arr, std_arr, centers_arr

    if cur_dim > target_dim:
        return feat_arr[:, :target_dim], mean_arr, std_arr, centers_arr

    pad = np.broadcast_to(mean_arr[:, cur_dim:target_dim], (feat_arr.shape[0], target_dim - cur_dim)).copy()
    feat_arr = np.concatenate([feat_arr, pad], axis=1)
    return feat_arr, mean_arr, std_arr, centers_arr


def regime_probabilities(
    regime_feature: np.ndarray,
    clusterer: dict[str, np.ndarray | int],
    temperature: float,
) -> np.ndarray:
    feat = np.asarray(regime_feature, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[None, :]
    mean = np.asarray(clusterer["feature_mean"], dtype=np.float32)
    std = np.asarray(clusterer["feature_std"], dtype=np.float32)
    centers = np.asarray(clusterer["centers"], dtype=np.float32)
    feat, mean, std, centers = _align_regime_feature_dims(feat=feat, mean=mean, std=std, centers=centers)
    x_norm = (feat - mean) / np.maximum(std, 1e-6)
    diff = x_norm[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 1e-12))
    logits = -dist / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs.astype(np.float32)


def save_regime_clusterer(clusterer: dict[str, np.ndarray | int], path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        num_clusters=np.asarray([int(clusterer["num_clusters"])], dtype=np.int64),
        feature_mean=np.asarray(clusterer["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(clusterer["feature_std"], dtype=np.float32),
        centers=np.asarray(clusterer["centers"], dtype=np.float32),
    )


def load_regime_clusterer(path: str | Path) -> dict[str, np.ndarray | int]:
    data = np.load(str(path))
    return {
        "num_clusters": int(np.asarray(data["num_clusters"]).reshape(-1)[0]),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
        "centers": np.asarray(data["centers"], dtype=np.float32),
    }
