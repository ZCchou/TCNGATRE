from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd

from utils.regime_score_calibration import apply_regime_component_calibration
from utils.score_calibration import apply_component_calibration


def safe_json_list(x: np.ndarray) -> str:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return json.dumps([float(v) for v in arr.tolist()], ensure_ascii=False)


def score_is_pure_resid_value(cfg) -> bool:
    return (
        float(cfg.score_weight_value) > 0.0
        and float(cfg.score_weight_mask) == 0.0
        and float(cfg.score_weight_delta) == 0.0
        and float(cfg.score_weight_mean) == 0.0
        and float(cfg.score_weight_var) == 0.0
        and float(cfg.score_weight_range) == 0.0
        and float(cfg.score_weight_count) == 0.0
        and float(cfg.score_weight_constant) == 0.0
        and float(cfg.score_weight_abs_diff) == 0.0
        and float(cfg.score_weight_zero_diff) == 0.0
        and float(cfg.score_weight_slope) == 0.0
        and float(cfg.score_weight_onset_mean) == 0.0
        and float(cfg.score_weight_onset_delta) == 0.0
        and float(cfg.score_weight_control_change) == 0.0
        and not bool(cfg.score_use_residual_calibration)
        and not bool(cfg.score_use_squared_residual)
    )


def aggregate_sensor_score(
    sensor_score: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "mean",
    topk_k: int = 3,
    fallback_to_all_valid: bool = True,
) -> np.ndarray:
    score = np.asarray(sensor_score, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if score.ndim != 2 or valid.shape != score.shape:
        raise ValueError(f"aggregate_sensor_score expects [N, D], got {score.shape} vs {valid.shape}")

    mode_norm = str(mode).strip().lower()
    out = np.zeros((score.shape[0],), dtype=np.float32)
    for i in range(score.shape[0]):
        cur = score[i, valid[i]]
        if cur.size <= 0:
            out[i] = 0.0
            continue
        if mode_norm == "mean":
            out[i] = float(cur.mean())
            continue
        if mode_norm == "topk":
            k = min(max(int(topk_k), 1), cur.size)
            if cur.size < int(topk_k) and not bool(fallback_to_all_valid):
                out[i] = 0.0
                continue
            out[i] = float(np.mean(np.partition(cur, -k)[-k:]))
            continue
        raise ValueError(f"Unsupported score aggregation mode: {mode}")
    return out


def compute_calibrated_scores(
    cfg,
    calibration: dict[str, dict[str, np.ndarray]],
    regime_calibration: dict[str, dict[str, np.ndarray]],
    regime_prob: np.ndarray,
    valid_mask: np.ndarray,
    residual_map: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if cfg.score_use_regime_calibration and len(regime_calibration) > 0:
        calibrated = {
            name: apply_regime_component_calibration(
                residual=residual_map[name],
                calibration=regime_calibration[name],
                regime_prob=regime_prob,
                valid_mask=valid_mask,
                clip=cfg.score_calibration_clip,
            )
            for name in regime_calibration.keys()
            if name in residual_map
        }
    elif cfg.score_use_residual_calibration and len(calibration) > 0:
        calibrated = {
            name: apply_component_calibration(
                residual=residual_map[name],
                calibration=calibration[name],
                valid_mask=valid_mask,
                clip=cfg.score_calibration_clip,
            )
            for name in calibration.keys()
            if name in residual_map
        }
    else:
        calibrated = {
            name: (np.asarray(residual_map[name], dtype=np.float32) * valid_mask.astype(np.float32))
            for name in residual_map.keys()
        }
    if cfg.score_use_squared_residual:
        calibrated = {
            name: np.square(np.asarray(value, dtype=np.float32), dtype=np.float32)
            for name, value in calibrated.items()
        }

    zeros = np.zeros_like(valid_mask, dtype=np.float32)
    sensor_score = (
        cfg.score_weight_mask * calibrated.get("mask", zeros)
        + cfg.score_weight_value * calibrated.get("value", zeros)
        + cfg.score_weight_delta * calibrated.get("delta", zeros)
        + cfg.score_weight_mean * calibrated.get("mean", zeros)
        + cfg.score_weight_var * calibrated.get("var", zeros)
        + cfg.score_weight_range * calibrated.get("range", zeros)
        + cfg.score_weight_count * calibrated.get("count", zeros)
        + cfg.score_weight_constant * calibrated.get("constant", zeros)
        + cfg.score_weight_abs_diff * calibrated.get("abs_diff", zeros)
        + cfg.score_weight_zero_diff * calibrated.get("zero_diff", zeros)
        + cfg.score_weight_slope * calibrated.get("slope", zeros)
        + cfg.score_weight_onset_mean * calibrated.get("onset_mean", zeros)
        + cfg.score_weight_onset_delta * calibrated.get("onset_delta", zeros)
        + cfg.score_weight_control_change * calibrated.get("control_change", zeros)
    ).astype(np.float32, copy=False)
    total_score = aggregate_sensor_score(
        sensor_score=sensor_score,
        valid_mask=valid_mask,
        mode=cfg.score_aggregation_mode,
        topk_k=cfg.score_topk_k,
        fallback_to_all_valid=cfg.score_topk_fallback_to_all_valid,
    )
    return sensor_score, total_score.astype(np.float32, copy=False), calibrated


def ema_smooth_matrix(
    values: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
    carry_forward_on_missing: bool = False,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if arr.ndim != 2 or valid.shape != arr.shape:
        raise ValueError(f"EMA smooth expects 2D arrays with same shape, got {arr.shape} vs {valid.shape}")
    for d in range(arr.shape[1]):
        prev: float | None = None
        for t in range(arr.shape[0]):
            if not valid[t, d]:
                if carry_forward_on_missing and prev is not None:
                    out[t, d] = prev
                else:
                    out[t, d] = 0.0
                    prev = None
                continue
            x = float(arr[t, d])
            prev = x if prev is None else alpha * x + (1.0 - alpha) * prev
            out[t, d] = prev
    return out


def forward_fill_valid_mask(valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"forward-fill valid mask expects 2D array, got {valid.shape}")
    out = np.zeros_like(valid, dtype=bool)
    for d in range(valid.shape[1]):
        seen = False
        for t in range(valid.shape[0]):
            if valid[t, d]:
                seen = True
            out[t, d] = seen
    return out


def apply_temporal_smoothing(flight_df: pd.DataFrame, cfg) -> pd.DataFrame:
    if not cfg.score_use_temporal_smoothing or flight_df.empty:
        return flight_df

    out_df = flight_df.sort_values("t_start", kind="mergesort").reset_index(drop=True).copy()
    valid_mask = np.stack(out_df["future_mask_vec"].apply(json.loads).to_list(), axis=0).astype(np.float32) > 0.5
    resid_value = np.stack(out_df["resid_value_vec"].apply(json.loads).to_list(), axis=0).astype(np.float32)
    sensor_score = np.stack(out_df["sensor_score_vec"].apply(json.loads).to_list(), axis=0).astype(np.float32)
    control_valid_mask = valid_mask.copy()
    if "future_control_change_vec" in out_df.columns:
        future_control_change = np.stack(out_df["future_control_change_vec"].apply(json.loads).to_list(), axis=0).astype(np.float32)
        control_valid_mask = future_control_change > 0.0

    score_valid_mask = forward_fill_valid_mask(valid_mask) if cfg.score_fill_missing_with_previous else valid_mask
    score_control_valid_mask = (
        forward_fill_valid_mask(control_valid_mask) if cfg.score_fill_missing_with_previous else control_valid_mask
    )

    smoothed_resid_value = ema_smooth_matrix(
        values=resid_value,
        valid_mask=valid_mask,
        alpha=cfg.score_temporal_smooth_alpha,
        carry_forward_on_missing=cfg.score_fill_missing_with_previous,
    )
    if score_is_pure_resid_value(cfg):
        smoothed_sensor_score = smoothed_resid_value.copy()
    else:
        smoothed_sensor_score = ema_smooth_matrix(
            values=sensor_score,
            valid_mask=valid_mask,
            alpha=cfg.score_temporal_smooth_alpha,
            carry_forward_on_missing=cfg.score_fill_missing_with_previous,
        )

    smoothed_total_score = aggregate_sensor_score(
        sensor_score=smoothed_sensor_score,
        valid_mask=score_valid_mask,
        mode=cfg.score_aggregation_mode,
        topk_k=cfg.score_topk_k,
        fallback_to_all_valid=cfg.score_topk_fallback_to_all_valid,
    )
    smoothed_value_total_score = aggregate_sensor_score(
        sensor_score=smoothed_resid_value,
        valid_mask=score_valid_mask,
        mode=cfg.score_aggregation_mode,
        topk_k=cfg.score_topk_k,
        fallback_to_all_valid=cfg.score_topk_fallback_to_all_valid,
    )

    if "control_change_sensor_score_vec" in out_df.columns:
        control_sensor_score = np.stack(out_df["control_change_sensor_score_vec"].apply(json.loads).to_list(), axis=0).astype(np.float32)
        smoothed_control_sensor_score = ema_smooth_matrix(
            values=control_sensor_score,
            valid_mask=control_valid_mask,
            alpha=cfg.score_temporal_smooth_alpha,
            carry_forward_on_missing=cfg.score_fill_missing_with_previous,
        )
        smoothed_control_total_score = aggregate_sensor_score(
            sensor_score=smoothed_control_sensor_score,
            valid_mask=score_control_valid_mask,
            mode=cfg.score_aggregation_mode,
            topk_k=cfg.score_topk_k,
            fallback_to_all_valid=cfg.score_topk_fallback_to_all_valid,
        )
        out_df["control_change_sensor_score_vec"] = [safe_json_list(x) for x in smoothed_control_sensor_score]
        out_df["control_change_total_score"] = np.nan_to_num(
            smoothed_control_total_score, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64)

    out_df["smoothed_resid_value_vec"] = [safe_json_list(x) for x in smoothed_resid_value]
    out_df["smoothed_sensor_score_vec"] = [safe_json_list(x) for x in smoothed_sensor_score]
    out_df["sensor_score_vec"] = [safe_json_list(x) for x in smoothed_sensor_score]
    out_df["value_sensor_score_vec"] = [safe_json_list(x) for x in smoothed_resid_value]
    out_df["value_total_score"] = np.nan_to_num(smoothed_value_total_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    out_df["total_score"] = np.nan_to_num(smoothed_total_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    return out_df


def backfill_then_forward_fill_display(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 1:
        raise ValueError(f"display fill expects 1D array, got {arr.shape}")
    next_val = np.nan
    for i in range(arr.shape[0] - 1, -1, -1):
        if np.isfinite(arr[i]):
            next_val = arr[i]
        elif np.isfinite(next_val):
            arr[i] = next_val
    prev_val = np.nan
    for i in range(arr.shape[0]):
        if np.isfinite(arr[i]):
            prev_val = arr[i]
        elif np.isfinite(prev_val):
            arr[i] = prev_val
    return arr
