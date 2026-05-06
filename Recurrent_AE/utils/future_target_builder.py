from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class PatchObservation:
    t: np.ndarray
    x: np.ndarray
    mask: np.ndarray


def load_patch_npz(npz_path: str | Path) -> PatchObservation:
    data = np.load(str(npz_path))
    required = {"t", "X", "mask"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Patch npz missing keys: {sorted(missing)} | path={npz_path}")
    t = np.asarray(data["t"], dtype=np.float32)
    x = np.asarray(data["X"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.uint8)
    if x.ndim != 2 or mask.shape != x.shape:
        raise ValueError(f"Invalid X/mask shape in {npz_path}: X={x.shape}, mask={mask.shape}")
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError(f"Invalid t shape in {npz_path}: t={t.shape}, X={x.shape}")
    return PatchObservation(t=t, x=x, mask=mask)


def infer_num_sensors_from_npz(npz_path: str | Path) -> int:
    obs = load_patch_npz(npz_path)
    return int(obs.x.shape[1])


def select_future_target_paths(
    future_npz_paths: Sequence[str | Path],
    mode: str = "last",
) -> list[str | Path]:
    paths = list(future_npz_paths)
    if len(paths) <= 0:
        raise ValueError("future_npz_paths must not be empty")
    mode_norm = str(mode).strip().lower()
    if mode_norm == "last":
        return [paths[-1]]
    if mode_norm == "first":
        return [paths[0]]
    if mode_norm == "all":
        return paths
    raise ValueError(f"Unsupported future target mode: {mode}")


def build_future_patch_sequence_targets(
    future_npz_paths: Sequence[str | Path],
) -> dict[str, np.ndarray]:
    if len(future_npz_paths) <= 0:
        raise ValueError("future_npz_paths must not be empty")

    future_obs = [load_patch_npz(p) for p in future_npz_paths]
    num_sensors = int(future_obs[0].x.shape[1])
    future_len = int(len(future_obs))

    future_value_seq = np.zeros((future_len, num_sensors), dtype=np.float32)
    future_value_mask_seq = np.zeros((future_len, num_sensors), dtype=np.float32)
    future_patch_mid_t = np.zeros((future_len,), dtype=np.float32)

    for t_idx, obs in enumerate(future_obs):
        if int(obs.x.shape[1]) != num_sensors:
            raise ValueError("Future patches have inconsistent sensor dimensions")
        if obs.t.size > 0:
            future_patch_mid_t[t_idx] = float(0.5 * (obs.t[0] + obs.t[-1]))
        idx_l, idx_d = np.where(obs.mask == 1)
        if idx_l.size <= 0:
            continue
        last_local_idx = np.full((num_sensors,), -1, dtype=np.int64)
        for l, d in zip(idx_l.tolist(), idx_d.tolist()):
            if int(l) >= int(last_local_idx[d]):
                last_local_idx[d] = int(l)
                future_value_seq[t_idx, d] = float(obs.x[l, d])
                future_value_mask_seq[t_idx, d] = 1.0

    return {
        "future_value_seq": future_value_seq.astype(np.float32),
        "future_value_mask_seq": future_value_mask_seq.astype(np.float32),
        "future_patch_mid_t": future_patch_mid_t.astype(np.float32),
    }


def build_last_observation_targets(
    history_npz_paths: Sequence[str | Path],
    future_npz_paths: Sequence[str | Path],
    constant_eps: float = 1e-3,
    zero_diff_eps: float | None = None,
    num_future_bins: int = 0,
) -> dict[str, np.ndarray]:
    if len(history_npz_paths) <= 0:
        raise ValueError("history_npz_paths must not be empty")
    if len(future_npz_paths) <= 0:
        raise ValueError("future_npz_paths must not be empty")

    history_obs = [load_patch_npz(p) for p in history_npz_paths]
    future_obs = [load_patch_npz(p) for p in future_npz_paths]
    num_sensors = int(history_obs[0].x.shape[1])

    hist_last_value = np.zeros((num_sensors,), dtype=np.float32)
    hist_has_value = np.zeros((num_sensors,), dtype=np.float32)
    future_mask = np.zeros((num_sensors,), dtype=np.float32)
    future_last_value = np.zeros((num_sensors,), dtype=np.float32)
    future_mean_value = np.zeros((num_sensors,), dtype=np.float32)
    future_count = np.zeros((num_sensors,), dtype=np.float32)
    future_delta_value = np.zeros((num_sensors,), dtype=np.float32)
    future_delta_mask = np.zeros((num_sensors,), dtype=np.float32)
    future_diff_mask = np.zeros((num_sensors,), dtype=np.float32)
    future_var_value = np.zeros((num_sensors,), dtype=np.float32)
    future_range_value = np.zeros((num_sensors,), dtype=np.float32)
    future_log_count = np.zeros((num_sensors,), dtype=np.float32)
    future_is_constant = np.zeros((num_sensors,), dtype=np.float32)
    future_abs_diff_sum = np.zeros((num_sensors,), dtype=np.float32)
    future_zero_diff_ratio = np.zeros((num_sensors,), dtype=np.float32)
    future_slope = np.zeros((num_sensors,), dtype=np.float32)
    num_future_bins = max(int(num_future_bins), 0)
    future_bin_mask = np.zeros((num_sensors, num_future_bins), dtype=np.float32)
    future_bin_last_value = np.zeros((num_sensors, num_future_bins), dtype=np.float32)
    future_bin_mean_value = np.zeros((num_sensors, num_future_bins), dtype=np.float32)
    future_bin_count = np.zeros((num_sensors, num_future_bins), dtype=np.float32)
    future_bin_log_count = np.zeros((num_sensors, num_future_bins), dtype=np.float32)

    hist_last_index = np.full((num_sensors,), -1, dtype=np.int64)
    future_last_index = np.full((num_sensors,), -1, dtype=np.int64)
    future_sum = np.zeros((num_sensors,), dtype=np.float64)
    future_sq_sum = np.zeros((num_sensors,), dtype=np.float64)
    future_counter = np.zeros((num_sensors,), dtype=np.int64)
    future_min = np.full((num_sensors,), np.inf, dtype=np.float64)
    future_max = np.full((num_sensors,), -np.inf, dtype=np.float64)
    future_value_seq: list[list[float]] = [[] for _ in range(num_sensors)]
    future_time_seq: list[list[float]] = [[] for _ in range(num_sensors)]
    zero_eps = float(constant_eps if zero_diff_eps is None else zero_diff_eps)

    hist_offset = 0
    for obs in history_obs:
        if int(obs.x.shape[1]) != num_sensors:
            raise ValueError("History patches have inconsistent sensor dimensions")
        idx_l, idx_d = np.where(obs.mask == 1)
        for l, d in zip(idx_l.tolist(), idx_d.tolist()):
            global_idx = hist_offset + int(l)
            if global_idx >= hist_last_index[d]:
                hist_last_index[d] = global_idx
                hist_last_value[d] = float(obs.x[l, d])
                hist_has_value[d] = 1.0
        hist_offset += int(obs.x.shape[0])

    future_offset = 0
    for obs in future_obs:
        if int(obs.x.shape[1]) != num_sensors:
            raise ValueError("Future patches have inconsistent sensor dimensions")
        idx_l, idx_d = np.where(obs.mask == 1)
        for l, d in zip(idx_l.tolist(), idx_d.tolist()):
            global_idx = future_offset + int(l)
            value = float(obs.x[l, d])
            future_mask[d] = 1.0
            future_sum[d] += value
            future_sq_sum[d] += value * value
            future_counter[d] += 1
            future_min[d] = min(future_min[d], value)
            future_max[d] = max(future_max[d], value)
            future_value_seq[d].append(value)
            future_time_seq[d].append(float(obs.t[l]))
            if global_idx >= future_last_index[d]:
                future_last_index[d] = global_idx
                future_last_value[d] = value
        future_offset += int(obs.x.shape[0])

    valid_future = future_counter > 0
    future_count[valid_future] = future_counter[valid_future].astype(np.float32)
    future_mean_value[valid_future] = (future_sum[valid_future] / future_counter[valid_future]).astype(np.float32)
    future_log_count[valid_future] = np.log1p(future_count[valid_future]).astype(np.float32)
    future_var = np.zeros((num_sensors,), dtype=np.float64)
    future_var[valid_future] = (
        future_sq_sum[valid_future] / future_counter[valid_future]
        - (future_sum[valid_future] / future_counter[valid_future]) ** 2
    )
    future_var = np.maximum(future_var, 0.0)
    future_var_value[valid_future] = future_var[valid_future].astype(np.float32)
    future_range_value[valid_future] = (future_max[valid_future] - future_min[valid_future]).astype(np.float32)
    future_is_constant[valid_future] = (future_range_value[valid_future] <= float(constant_eps)).astype(np.float32)

    valid_delta = valid_future & (hist_has_value > 0.5)
    future_delta_mask[valid_delta] = 1.0
    future_delta_value[valid_delta] = future_last_value[valid_delta] - hist_last_value[valid_delta]
    valid_diff = future_counter >= 2
    future_diff_mask[valid_diff] = 1.0

    for d in np.where(valid_diff)[0].tolist():
        vals = np.asarray(future_value_seq[d], dtype=np.float32)
        ts = np.asarray(future_time_seq[d], dtype=np.float32)
        diffs = np.diff(vals)
        future_abs_diff_sum[d] = float(np.sum(np.abs(diffs)))
        future_zero_diff_ratio[d] = float(np.mean(np.abs(diffs) <= zero_eps))

        ts = ts - float(ts[0])
        span = float(ts[-1]) if ts.size > 0 else 0.0
        if span <= 1e-6:
            future_slope[d] = 0.0
            continue
        x = ts / span
        x_center = x - float(np.mean(x))
        y_center = vals - float(np.mean(vals))
        denom = float(np.sum(x_center * x_center))
        future_slope[d] = 0.0 if denom <= 1e-8 else float(np.sum(x_center * y_center) / denom)

    if num_future_bins > 0:
        non_empty_obs = [obs for obs in future_obs if obs.t.size > 0]
        if len(non_empty_obs) > 0:
            global_t_start = float(non_empty_obs[0].t[0])
            global_t_end = float(non_empty_obs[-1].t[-1])
        else:
            global_t_start = 0.0
            global_t_end = 0.0
        if global_t_end <= global_t_start:
            global_t_end = global_t_start + float(max(len(future_obs), 1))
        edges = np.linspace(global_t_start, global_t_end, num=num_future_bins + 1, dtype=np.float32)
        bin_sum = np.zeros((num_sensors, num_future_bins), dtype=np.float64)
        bin_counter = np.zeros((num_sensors, num_future_bins), dtype=np.int64)
        bin_last_time = np.full((num_sensors, num_future_bins), -np.inf, dtype=np.float64)

        for d in np.where(valid_future)[0].tolist():
            vals = np.asarray(future_value_seq[d], dtype=np.float32)
            ts = np.asarray(future_time_seq[d], dtype=np.float32)
            if vals.size <= 0:
                continue
            bin_idx = np.searchsorted(edges[1:], ts, side="right")
            bin_idx = np.clip(bin_idx, 0, num_future_bins - 1)
            for value, t_value, k in zip(vals.tolist(), ts.tolist(), bin_idx.tolist()):
                future_bin_mask[d, k] = 1.0
                future_bin_count[d, k] += 1.0
                bin_sum[d, k] += float(value)
                if float(t_value) >= float(bin_last_time[d, k]):
                    bin_last_time[d, k] = float(t_value)
                    future_bin_last_value[d, k] = float(value)
            positive = future_bin_count[d] > 0.0
            future_bin_mean_value[d, positive] = (bin_sum[d, positive] / future_bin_count[d, positive]).astype(np.float32)
            future_bin_log_count[d, positive] = np.log1p(future_bin_count[d, positive]).astype(np.float32)

    return {
        "hist_last_value": hist_last_value.astype(np.float32),
        "hist_has_value": hist_has_value.astype(np.float32),
        "future_mask": future_mask.astype(np.float32),
        "future_last_value": future_last_value.astype(np.float32),
        "future_delta_value": future_delta_value.astype(np.float32),
        "future_delta_mask": future_delta_mask.astype(np.float32),
        "future_diff_mask": future_diff_mask.astype(np.float32),
        "future_mean_value": future_mean_value.astype(np.float32),
        "future_var_value": future_var_value.astype(np.float32),
        "future_range_value": future_range_value.astype(np.float32),
        "future_count": future_count.astype(np.float32),
        "future_log_count": future_log_count.astype(np.float32),
        "future_is_constant": future_is_constant.astype(np.float32),
        "future_abs_diff_sum": future_abs_diff_sum.astype(np.float32),
        "future_zero_diff_ratio": future_zero_diff_ratio.astype(np.float32),
        "future_slope": future_slope.astype(np.float32),
        "future_bin_mask": future_bin_mask.astype(np.float32),
        "future_bin_last_value": future_bin_last_value.astype(np.float32),
        "future_bin_mean_value": future_bin_mean_value.astype(np.float32),
        "future_bin_count": future_bin_count.astype(np.float32),
        "future_bin_log_count": future_bin_log_count.astype(np.float32),
    }
