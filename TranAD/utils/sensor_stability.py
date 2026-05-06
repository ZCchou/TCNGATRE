from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_sensor_stability(
    residual: np.ndarray,
    valid_mask: np.ndarray,
    sensor_names: list[str] | tuple[str, ...],
    *,
    q95_quantile: float = 0.85,
    mad_quantile: float = 0.85,
    max_false_alarm_rate: float = 0.15,
    z_threshold: float = 3.0,
    min_valid_count: int = 32,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    resid = np.asarray(residual, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if resid.shape != valid.shape:
        raise ValueError(f"Residual and valid mask shape mismatch: {resid.shape} vs {valid.shape}")

    num_sensors = int(resid.shape[1])
    names = list(sensor_names)
    if len(names) != num_sensors:
        raise ValueError(f"Expected {num_sensors} sensor names, got {len(names)}")

    valid_count = np.zeros((num_sensors,), dtype=np.int32)
    median = np.zeros((num_sensors,), dtype=np.float32)
    mad = np.zeros((num_sensors,), dtype=np.float32)
    robust_scale = np.ones((num_sensors,), dtype=np.float32)
    q95 = np.zeros((num_sensors,), dtype=np.float32)
    q99 = np.zeros((num_sensors,), dtype=np.float32)
    mean = np.zeros((num_sensors,), dtype=np.float32)
    std = np.zeros((num_sensors,), dtype=np.float32)
    false_alarm_rate = np.ones((num_sensors,), dtype=np.float32)

    for d in range(num_sensors):
        vals = resid[:, d][valid[:, d]]
        valid_count[d] = int(vals.size)
        if vals.size <= 0:
            continue
        med = float(np.median(vals))
        mad_d = float(np.median(np.abs(vals - med)))
        scale = max(1.4826 * mad_d, float(eps))
        median[d] = med
        mad[d] = mad_d
        robust_scale[d] = scale
        q95[d] = float(np.quantile(vals, 0.95))
        q99[d] = float(np.quantile(vals, 0.99))
        mean[d] = float(np.mean(vals))
        std[d] = float(np.std(vals))
        z = np.maximum((vals - med) / scale, 0.0)
        false_alarm_rate[d] = float(np.mean(z > float(z_threshold)))

    valid_sensors = valid_count >= int(min_valid_count)
    if bool(valid_sensors.any()):
        q95_cut = float(np.quantile(q95[valid_sensors], float(q95_quantile)))
        mad_cut = float(np.quantile(mad[valid_sensors], float(mad_quantile)))
    else:
        q95_cut = float("inf")
        mad_cut = float("inf")

    stable_mask = (
        valid_sensors
        & (q95 <= q95_cut)
        & (mad <= mad_cut)
        & (false_alarm_rate <= float(max_false_alarm_rate))
    )
    if not bool(stable_mask.any()):
        stable_mask = valid_sensors.copy()
    if not bool(stable_mask.any()):
        stable_mask = np.ones((num_sensors,), dtype=bool)

    unstable_mask = ~stable_mask
    return {
        "sensor_names": np.asarray(names, dtype=object),
        "valid_count": valid_count.astype(np.int32, copy=False),
        "median": median.astype(np.float32, copy=False),
        "mad": mad.astype(np.float32, copy=False),
        "robust_scale": robust_scale.astype(np.float32, copy=False),
        "q95": q95.astype(np.float32, copy=False),
        "q99": q99.astype(np.float32, copy=False),
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "false_alarm_rate": false_alarm_rate.astype(np.float32, copy=False),
        "stable_mask": stable_mask.astype(np.bool_, copy=False),
        "unstable_mask": unstable_mask.astype(np.bool_, copy=False),
        "q95_cut": np.asarray([q95_cut], dtype=np.float32),
        "mad_cut": np.asarray([mad_cut], dtype=np.float32),
        "max_false_alarm_rate": np.asarray([max_false_alarm_rate], dtype=np.float32),
        "z_threshold": np.asarray([z_threshold], dtype=np.float32),
        "min_valid_count": np.asarray([min_valid_count], dtype=np.int32),
    }


def sensor_stability_to_frame(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor": stats["sensor_names"].tolist(),
            "valid_count": np.asarray(stats["valid_count"], dtype=np.int32),
            "median": np.asarray(stats["median"], dtype=np.float32),
            "mad": np.asarray(stats["mad"], dtype=np.float32),
            "robust_scale": np.asarray(stats["robust_scale"], dtype=np.float32),
            "mean": np.asarray(stats["mean"], dtype=np.float32),
            "std": np.asarray(stats["std"], dtype=np.float32),
            "q95": np.asarray(stats["q95"], dtype=np.float32),
            "q99": np.asarray(stats["q99"], dtype=np.float32),
            "false_alarm_rate": np.asarray(stats["false_alarm_rate"], dtype=np.float32),
            "stable": np.asarray(stats["stable_mask"], dtype=bool),
            "unstable": np.asarray(stats["unstable_mask"], dtype=bool),
        }
    )


def save_sensor_stability(stats: dict[str, np.ndarray], path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in stats.items():
        if key == "sensor_names":
            payload[key] = np.asarray(value, dtype="<U128")
        else:
            payload[key] = np.asarray(value)
    np.savez(p, **payload)


def load_sensor_stability(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        out[key] = data[key]
    if "stable_mask" in out:
        out["stable_mask"] = np.asarray(out["stable_mask"], dtype=bool)
    if "unstable_mask" in out:
        out["unstable_mask"] = np.asarray(out["unstable_mask"], dtype=bool)
    if "sensor_names" in out:
        out["sensor_names"] = np.asarray(out["sensor_names"]).astype(str)
    return out
