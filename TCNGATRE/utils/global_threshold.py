from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_single_global_threshold(
    scores: np.ndarray,
    mode: str = "val_normal_sigma3",
    sigma_k: float = 3.0,
    mad_k: float = 4.0,
    quantile: float = 0.995,
) -> dict[str, float | str | int]:
    x = np.asarray(scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        raise ValueError("Cannot fit threshold on empty score array")
    mode_norm = str(mode).strip().lower()

    if mode_norm.endswith("sigma3") or mode_norm.endswith("sigma"):
        mean = float(np.mean(x))
        std = float(np.std(x))
        threshold = mean + float(sigma_k) * std
        return {
            "fit_mode": mode_norm,
            "threshold": float(threshold),
            "mean": mean,
            "std": std,
            "sigma_k": float(sigma_k),
            "num_samples": int(x.size),
        }
    if mode_norm.endswith("mad"):
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        robust_scale = max(1.4826 * mad, 1e-8)
        threshold = median + float(mad_k) * robust_scale
        return {
            "fit_mode": mode_norm,
            "threshold": float(threshold),
            "median": median,
            "mad": mad,
            "robust_scale": robust_scale,
            "mad_k": float(mad_k),
            "num_samples": int(x.size),
        }
    if mode_norm.endswith("quantile") or mode_norm.endswith("q"):
        q = float(np.clip(quantile, 0.5, 0.999999))
        threshold = float(np.quantile(x, q))
        return {
            "fit_mode": mode_norm,
            "threshold": threshold,
            "quantile": q,
            "num_samples": int(x.size),
        }
    raise ValueError(f"Unsupported threshold fit mode: {mode}")


def save_global_threshold(payload: dict, path: Path):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
