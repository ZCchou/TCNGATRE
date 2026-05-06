from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def compute_regime_component_calibration(
    residual: np.ndarray,
    valid_mask: np.ndarray,
    regime_prob: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    resid = np.asarray(residual, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    prob = np.asarray(regime_prob, dtype=np.float32)
    if resid.shape != valid.shape:
        raise ValueError(f"Residual and valid mask shape mismatch: {resid.shape} vs {valid.shape}")
    if resid.ndim != 2 or prob.ndim != 2 or resid.shape[0] != prob.shape[0]:
        raise ValueError(f"Expected residual (N,D), regime_prob (N,K), got {resid.shape} and {prob.shape}")

    num_regimes = int(prob.shape[1])
    num_sensors = int(resid.shape[1])
    mean = np.zeros((num_regimes, num_sensors), dtype=np.float32)
    scale = np.ones((num_regimes, num_sensors), dtype=np.float32)

    valid_f = valid.astype(np.float32)
    for k in range(num_regimes):
        weight_k = prob[:, [k]] * valid_f
        weight_sum = np.sum(weight_k, axis=0)
        safe_weight = np.maximum(weight_sum, float(eps))
        mean_k = np.sum(weight_k * resid, axis=0) / safe_weight
        var_k = np.sum(weight_k * np.square(resid - mean_k[None, :]), axis=0) / safe_weight
        mean[k] = mean_k.astype(np.float32, copy=False)
        scale[k] = np.sqrt(np.maximum(var_k, float(eps))).astype(np.float32, copy=False)
    return {"mean": mean, "scale": np.maximum(scale, float(eps)).astype(np.float32, copy=False)}


def compute_regime_residual_calibration(
    residual_map: Dict[str, np.ndarray],
    valid_mask: np.ndarray | Dict[str, np.ndarray],
    regime_prob: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, dict[str, np.ndarray]]:
    if isinstance(valid_mask, dict):
        return {
            str(name): compute_regime_component_calibration(
                residual=resid,
                valid_mask=valid_mask.get(str(name), valid_mask.get("default")),
                regime_prob=regime_prob,
                eps=eps,
            )
            for name, resid in residual_map.items()
        }
    return {
        str(name): compute_regime_component_calibration(
            residual=resid,
            valid_mask=valid_mask,
            regime_prob=regime_prob,
            eps=eps,
        )
        for name, resid in residual_map.items()
    }


def apply_regime_component_calibration(
    residual: np.ndarray,
    calibration: dict[str, np.ndarray],
    regime_prob: np.ndarray,
    valid_mask: np.ndarray | None = None,
    clip: float = 10.0,
) -> np.ndarray:
    resid = np.asarray(residual, dtype=np.float32)
    prob = np.asarray(regime_prob, dtype=np.float32)
    mean = np.asarray(calibration["mean"], dtype=np.float32)
    scale = np.asarray(calibration["scale"], dtype=np.float32)
    expected_mean = prob @ mean
    expected_scale = np.maximum(prob @ scale, 1e-6)
    out = np.maximum((resid - expected_mean) / expected_scale, 0.0)
    out = np.clip(out, 0.0, float(clip))
    if valid_mask is not None:
        out = out * np.asarray(valid_mask, dtype=np.float32)
    return out.astype(np.float32, copy=False)


def save_regime_residual_calibration(
    calibration: dict[str, dict[str, np.ndarray]],
    path: str | Path,
):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, item in calibration.items():
        payload[f"{name}__mean"] = np.asarray(item["mean"], dtype=np.float32)
        payload[f"{name}__scale"] = np.asarray(item["scale"], dtype=np.float32)
    np.savez(p, **payload)


def load_regime_residual_calibration(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(str(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if "__" not in key:
            continue
        name, suffix = key.split("__", 1)
        out.setdefault(name, {})[suffix] = np.asarray(data[key], dtype=np.float32)
    return out
