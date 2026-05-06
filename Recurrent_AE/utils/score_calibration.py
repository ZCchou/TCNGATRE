from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _safe_scale(x: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float32), float(eps))


def compute_component_calibration(
    residual: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    resid = np.asarray(residual, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if resid.shape != valid.shape:
        raise ValueError(f"Residual and valid mask shape mismatch: {resid.shape} vs {valid.shape}")

    num_sensors = int(resid.shape[1])
    median = np.zeros((num_sensors,), dtype=np.float32)
    scale = np.ones((num_sensors,), dtype=np.float32)

    for d in range(num_sensors):
        vals = resid[:, d][valid[:, d]]
        if vals.size <= 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        median[d] = med
        scale[d] = max(1.4826 * mad, float(eps))
    return {"median": median, "scale": scale}


def compute_residual_calibration(
    residual_map: Dict[str, np.ndarray],
    valid_mask: np.ndarray | Dict[str, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, dict[str, np.ndarray]]:
    if isinstance(valid_mask, dict):
        return {
            str(name): compute_component_calibration(
                residual=resid,
                valid_mask=valid_mask.get(str(name), valid_mask.get("default")),
                eps=eps,
            )
            for name, resid in residual_map.items()
        }
    return {
        str(name): compute_component_calibration(residual=resid, valid_mask=valid_mask, eps=eps)
        for name, resid in residual_map.items()
    }


def apply_component_calibration(
    residual: np.ndarray,
    calibration: dict[str, np.ndarray],
    valid_mask: np.ndarray | None = None,
    clip: float = 10.0,
) -> np.ndarray:
    resid = np.asarray(residual, dtype=np.float32)
    median = np.asarray(calibration["median"], dtype=np.float32).reshape(1, -1)
    scale = _safe_scale(calibration["scale"], eps=1e-6).reshape(1, -1)
    out = np.maximum((resid - median) / scale, 0.0)
    out = np.clip(out, 0.0, float(clip))
    if valid_mask is not None:
        out = out * np.asarray(valid_mask, dtype=np.float32)
    return out.astype(np.float32, copy=False)


def save_residual_calibration(
    calibration: dict[str, dict[str, np.ndarray]],
    path: str | Path,
):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, item in calibration.items():
        payload[f"{name}__median"] = np.asarray(item["median"], dtype=np.float32)
        payload[f"{name}__scale"] = np.asarray(item["scale"], dtype=np.float32)
    np.savez(p, **payload)


def load_residual_calibration(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(str(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if "__" not in key:
            continue
        name, suffix = key.split("__", 1)
        out.setdefault(name, {})[suffix] = np.asarray(data[key], dtype=np.float32)
    return out
