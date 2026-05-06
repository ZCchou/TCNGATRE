from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-12
STATIC_METHOD = "static_f1_oracle"
STATIC_VAL_SIGMA_METHOD = "static_val_sigma3"
DYNAMIC_METHOD = "dynamic_history"
SPOT_METHOD = "spot"


def ema_smooth(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially smooth a 1D score sequence."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size <= 0:
        return x.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for idx in range(1, x.size):
        out[idx] = alpha * x[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def count_exceedance_segments(mask: np.ndarray) -> int:
    """Count contiguous True runs in a boolean exceedance mask."""
    x = np.asarray(mask, dtype=bool)
    if x.size <= 0:
        return 0
    starts = x & np.concatenate(([True], ~x[:-1]))
    return int(starts.sum())


def compute_binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(pred, dtype=np.int32)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, EPS))
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    out: dict[str, float | int] = {
        "num_samples": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "auroc": float("nan"),
        "average_precision": float("nan"),
    }
    if len(y) <= 0:
        return out
    if positives > 0:
        out["average_precision"] = float(average_precision_score(y, s))
    if positives > 0 and negatives > 0:
        out["auroc"] = float(roc_auc_score(y, s))
    return out


def finite_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def robust_median_mad(values: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """Return finite-only median and MAD with a small positive floor."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(mad, float(eps))


def fit_static_threshold(scores_smooth: np.ndarray, gt_labels: np.ndarray | None, p: int = 1000) -> dict[str, Any]:
    """Fit static oracle threshold by maximizing F1 over q / p * max(score)."""
    scores = np.asarray(scores_smooth, dtype=np.float64)
    finite = np.isfinite(scores)
    p = max(int(p), 1)
    max_score = float(np.max(scores[finite])) if bool(finite.any()) else 0.0
    thresholds = (np.arange(1, p + 1, dtype=np.float64) / float(p)) * max_score

    if gt_labels is None:
        return {
            "fit_mode": STATIC_METHOD,
            "best_threshold": float(thresholds[-1]) if thresholds.size else 0.0,
            "best_f1": float("nan"),
            "best_q": int(p),
            "p": int(p),
            "max_score": max_score,
            "thresholds_static": thresholds.tolist(),
            "used_gt_labels": False,
        }

    labels = np.asarray(gt_labels, dtype=np.float64)
    valid = finite & np.isfinite(labels)
    if not bool(valid.any()):
        return {
            "fit_mode": STATIC_METHOD,
            "best_threshold": float(thresholds[-1]) if thresholds.size else 0.0,
            "best_f1": float("nan"),
            "best_q": int(p),
            "p": int(p),
            "max_score": max_score,
            "thresholds_static": thresholds.tolist(),
            "used_gt_labels": False,
        }

    y = (labels[valid] > 0.5).astype(np.int32)
    s = scores[valid]
    best_f1 = -1.0
    best_threshold = float(thresholds[0]) if thresholds.size else 0.0
    best_q = 1
    for q, threshold in enumerate(thresholds, start=1):
        pred = (s > float(threshold)).astype(np.int32)
        f1 = float(compute_binary_metrics(y, pred)["f1"])
        if f1 > best_f1 + EPS:
            best_f1 = f1
            best_threshold = float(threshold)
            best_q = int(q)
    return {
        "fit_mode": STATIC_METHOD,
        "best_threshold": best_threshold,
        "best_f1": float(best_f1),
        "best_q": best_q,
        "p": int(p),
        "max_score": max_score,
        "thresholds_static": thresholds.tolist(),
        "used_gt_labels": True,
        "num_valid": int(valid.sum()),
    }


def apply_dynamic_threshold(
    scores_smooth: np.ndarray,
    history: int,
    z_values: list[int | float],
    mad_k: float = 4.0,
    warmup_pred: int = 0,
) -> dict[str, np.ndarray]:
    """Apply causal rolling median + MAD thresholding to one score sequence."""
    scores = np.asarray(scores_smooth, dtype=np.float64)
    h = max(int(history), 1)
    need = max(8, h // 4)
    k = float(mad_k) if math.isfinite(float(mad_k)) and float(mad_k) > 0 else 4.0
    warmup = int(1 if int(warmup_pred) else 0)

    thresholds = np.full(scores.shape, np.nan, dtype=np.float64)
    labels = np.full(scores.shape, warmup, dtype=np.int32)
    medians = np.full(scores.shape, np.nan, dtype=np.float64)
    mads = np.full(scores.shape, np.nan, dtype=np.float64)
    valid_history: list[float] = []

    for idx, value in enumerate(scores):
        if len(valid_history) >= need:
            hist = np.asarray(valid_history[-h:], dtype=np.float64)
            med, mad = robust_median_mad(hist)
            threshold = float(med + k * mad)
            thresholds[idx] = threshold
            medians[idx] = med
            mads[idx] = mad
            labels[idx] = int(np.isfinite(value) and value > threshold)
        if np.isfinite(value):
            valid_history.append(float(value))

    return {
        "thresholds_dynamic": thresholds,
        "labels_dynamic": labels,
        "dynamic_median": medians,
        "dynamic_mad": mads,
    }


def fit_static_val_sigma_threshold(
    val_scores_smooth: np.ndarray,
    sigma_k: float = 3.0,
) -> dict[str, Any]:
    """Fit a non-oracle static threshold from normal validation scores."""
    scores = np.asarray(val_scores_smooth, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    k = float(sigma_k) if math.isfinite(float(sigma_k)) and float(sigma_k) > 0 else 3.0
    if finite.size <= 0:
        return {
            "fit_mode": STATIC_VAL_SIGMA_METHOD,
            "rule": "mean_plus_k_std",
            "available": False,
            "skip_reason": "no finite validation normal scores",
            "sigma_k": k,
            "mean": float("nan"),
            "std": float("nan"),
            "threshold": float("nan"),
            "best_threshold": float("nan"),
            "num_samples": 0,
            "used_gt_labels": False,
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0))
    threshold = float(mean + k * std)
    return {
        "fit_mode": STATIC_VAL_SIGMA_METHOD,
        "rule": "mean_plus_k_std",
        "available": True,
        "sigma_k": k,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "best_threshold": threshold,
        "num_samples": int(finite.size),
        "used_gt_labels": False,
    }


def _fit_gpd_moments(excesses: np.ndarray) -> tuple[float, float]:
    """Fit GPD shape/scale with method-of-moments, avoiding SciPy dependency."""
    x = np.asarray(excesses, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size <= 0:
        return 0.0, 1.0
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if mean <= EPS:
        return 0.0, 1.0
    if var <= EPS or var <= mean * mean:
        return 0.0, max(mean, EPS)
    shape = 0.5 * (1.0 - (mean * mean / var))
    shape = float(np.clip(shape, -0.45, 0.45))
    scale = 0.5 * mean * (1.0 + (mean * mean / var))
    return shape, max(float(scale), EPS)


def _spot_threshold(init_threshold: float, excesses: list[float], seen_count: int, q: float) -> float:
    peaks = np.asarray(excesses, dtype=np.float64)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    if peaks.size <= 0 or seen_count <= 0:
        return float(init_threshold)
    shape, scale = _fit_gpd_moments(peaks)
    tail_prob = float(np.clip(q, 1e-9, 0.5))
    ratio = max(float(seen_count) * tail_prob / float(peaks.size), EPS)
    if abs(shape) <= 1e-8:
        threshold = float(init_threshold + scale * math.log(1.0 / ratio))
    else:
        threshold = float(init_threshold + (scale / shape) * (math.pow(ratio, -shape) - 1.0))
    return max(float(init_threshold), threshold)


def apply_spot_threshold(
    scores_smooth: np.ndarray,
    history: int,
    q: float = 1e-3,
    init_quantile: float = 0.98,
    warmup_pred: int = 0,
) -> dict[str, np.ndarray]:
    """Apply a lightweight causal SPOT threshold to one score sequence."""
    scores = np.asarray(scores_smooth, dtype=np.float64)
    h = max(int(history), 1)
    quantile = float(np.clip(init_quantile, 0.5, 0.999))
    warmup = int(1 if int(warmup_pred) else 0)

    thresholds = np.full(scores.shape, np.nan, dtype=np.float64)
    labels = np.full(scores.shape, warmup, dtype=np.int32)
    init_thresholds = np.full(scores.shape, np.nan, dtype=np.float64)
    shapes = np.full(scores.shape, np.nan, dtype=np.float64)
    scales = np.full(scores.shape, np.nan, dtype=np.float64)

    finite_init: list[float] = []
    init_threshold = float("nan")
    excesses: list[float] = []
    seen_count = 0

    for idx, value in enumerate(scores):
        finite_value = np.isfinite(value)
        if not math.isfinite(init_threshold):
            if finite_value:
                finite_init.append(float(value))
            if len(finite_init) >= h:
                init_values = np.asarray(finite_init, dtype=np.float64)
                init_threshold = float(np.quantile(init_values, quantile))
                excesses = [float(v - init_threshold) for v in init_values if v > init_threshold]
                seen_count = len(init_values)
            continue

        threshold = _spot_threshold(init_threshold, excesses, seen_count, q)
        thresholds[idx] = threshold
        init_thresholds[idx] = init_threshold
        shape, scale = _fit_gpd_moments(np.asarray(excesses, dtype=np.float64))
        shapes[idx] = shape
        scales[idx] = scale
        is_alarm = bool(finite_value and float(value) > threshold)
        labels[idx] = int(is_alarm)
        if finite_value:
            seen_count += 1
            if (not is_alarm) and float(value) > init_threshold:
                excesses.append(float(value - init_threshold))

    return {
        "thresholds_spot": thresholds,
        "labels_spot": labels,
        "spot_init_threshold": init_thresholds,
        "spot_shape": shapes,
        "spot_scale": scales,
    }


def _score_source_col(scored_df: pd.DataFrame) -> str:
    return "raw_total_score" if "raw_total_score" in scored_df.columns else "total_score"


def _load_val_scores_df(
    val_scores_df: pd.DataFrame | None,
    val_score_path: str | Path | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if val_scores_df is not None:
        return val_scores_df.copy(), {"source": "dataframe"}
    if val_score_path is None:
        return None, {"source": "", "skip_reason": "val_score_path not provided"}
    path = Path(val_score_path)
    if not path.exists():
        return None, {"source": str(path), "skip_reason": "val_score_path missing"}
    return pd.read_csv(path), {"source": str(path)}


def _smooth_threshold_scores(scored_df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, str]:
    if scored_df.empty:
        return np.asarray([], dtype=np.float64), ""
    score_col = _score_source_col(scored_df)
    df = scored_df.copy()
    if "flight" in df.columns and "current_index" in df.columns:
        df = df.sort_values(["flight", "current_index"], kind="mergesort").reset_index(drop=True)
    elif "flight" in df.columns:
        df = df.sort_values(["flight"], kind="mergesort").reset_index(drop=True)
    if "flight" in df.columns:
        smooth_parts = [
            ema_smooth(group[score_col].to_numpy(dtype=np.float64), alpha=alpha)
            for _, group in df.groupby("flight", sort=False)
        ]
        return (
            np.concatenate(smooth_parts).astype(np.float64, copy=False)
            if smooth_parts
            else np.asarray([], dtype=np.float64)
        ), score_col
    return ema_smooth(df[score_col].to_numpy(dtype=np.float64), alpha=alpha), score_col


def apply_threshold_methods(
    scored_df: pd.DataFrame,
    alpha: float,
    static_p: int,
    static_label_col: str,
    dynamic_history: int,
    dynamic_z_values: list[int | float],
    dynamic_warmup_pred: int = 0,
    dynamic_mad_k: float = 4.0,
    spot_q: float = 1e-3,
    spot_init_quantile: float = 0.98,
    val_scores_df: pd.DataFrame | None = None,
    val_score_path: str | Path | None = None,
    static_val_sigma_k: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Add smoothed scores plus static and dynamic threshold outputs."""
    df = scored_df.copy()
    if "flight" in df.columns and "current_index" in df.columns:
        df = df.sort_values(["flight", "current_index"], kind="mergesort").reset_index(drop=True)
    elif "flight" in df.columns:
        df = df.sort_values(["flight"], kind="mergesort").reset_index(drop=True)

    score_col = _score_source_col(df)
    df["scores_smooth"] = np.nan
    df["threshold_dynamic"] = np.nan
    df["pred_dynamic"] = int(dynamic_warmup_pred)
    df["dynamic_median"] = np.nan
    df["dynamic_mad"] = np.nan
    df["threshold_spot"] = np.nan
    df["pred_spot"] = int(dynamic_warmup_pred)
    df["spot_init_threshold"] = np.nan
    df["spot_shape"] = np.nan
    df["spot_scale"] = np.nan
    df["threshold_static_val_sigma3"] = np.nan
    df["pred_static_val_sigma3"] = np.nan

    if "flight" in df.columns:
        groups = df.groupby("flight", sort=False)
    else:
        groups = [(None, df)]
    for _, group in groups:
        idx = group.index.to_numpy()
        raw_scores = group[score_col].to_numpy(dtype=np.float64)
        smooth = ema_smooth(raw_scores, alpha=alpha)
        dynamic = apply_dynamic_threshold(
            smooth,
            history=dynamic_history,
            z_values=dynamic_z_values,
            mad_k=dynamic_mad_k,
            warmup_pred=dynamic_warmup_pred,
        )
        spot = apply_spot_threshold(
            smooth,
            history=dynamic_history,
            q=spot_q,
            init_quantile=spot_init_quantile,
            warmup_pred=dynamic_warmup_pred,
        )
        df.loc[idx, "scores_smooth"] = smooth
        df.loc[idx, "threshold_dynamic"] = dynamic["thresholds_dynamic"]
        df.loc[idx, "pred_dynamic"] = dynamic["labels_dynamic"]
        df.loc[idx, "dynamic_median"] = dynamic["dynamic_median"]
        df.loc[idx, "dynamic_mad"] = dynamic["dynamic_mad"]
        df.loc[idx, "threshold_spot"] = spot["thresholds_spot"]
        df.loc[idx, "pred_spot"] = spot["labels_spot"]
        df.loc[idx, "spot_init_threshold"] = spot["spot_init_threshold"]
        df.loc[idx, "spot_shape"] = spot["spot_shape"]
        df.loc[idx, "spot_scale"] = spot["spot_scale"]

    labels = df[static_label_col].to_numpy(dtype=np.float64) if static_label_col in df.columns else None
    static_payload = fit_static_threshold(
        scores_smooth=df["scores_smooth"].to_numpy(dtype=np.float64),
        gt_labels=labels,
        p=static_p,
    )
    threshold_static = float(static_payload["best_threshold"])
    df["threshold_static"] = threshold_static
    df["pred_static"] = (df["scores_smooth"].to_numpy(dtype=np.float64) > threshold_static).astype(np.int32)
    df["pred_label"] = df["pred_static"].astype(np.int32)
    df["global_threshold"] = threshold_static

    val_df, val_source = _load_val_scores_df(val_scores_df=val_scores_df, val_score_path=val_score_path)
    if val_df is None:
        static_val_payload = fit_static_val_sigma_threshold(np.asarray([], dtype=np.float64), sigma_k=static_val_sigma_k)
        static_val_payload.update(val_source)
    else:
        val_smooth, val_source_score_col = _smooth_threshold_scores(val_df, alpha=alpha)
        static_val_payload = fit_static_val_sigma_threshold(val_smooth, sigma_k=static_val_sigma_k)
        static_val_payload.update(
            {
                **val_source,
                "alpha": float(alpha),
                "score_col": "scores_smooth",
                "source_score_col": val_source_score_col,
            }
        )
        threshold_static_val = float(static_val_payload["threshold"])
        if math.isfinite(threshold_static_val):
            df["threshold_static_val_sigma3"] = threshold_static_val
            df["pred_static_val_sigma3"] = (
                df["scores_smooth"].to_numpy(dtype=np.float64) > threshold_static_val
            ).astype(np.int32)

    static_payload.update(
        {
            "alpha": float(alpha),
            "score_col": "scores_smooth",
            "source_score_col": score_col,
            "label_col": static_label_col,
            STATIC_VAL_SIGMA_METHOD: static_val_payload,
        }
    )
    dynamic_payload = {
        "fit_mode": DYNAMIC_METHOD,
        "rule": "rolling_median_plus_mad",
        "h": int(max(dynamic_history, 1)),
        "mad_k": float(dynamic_mad_k),
        "min_history": int(max(8, max(dynamic_history, 1) // 4)),
        "alpha": float(alpha),
        "score_col": "scores_smooth",
        "source_score_col": score_col,
        "warmup_pred": int(dynamic_warmup_pred),
        "spot": {
            "fit_mode": SPOT_METHOD,
            "rule": "streaming_peaks_over_threshold",
            "h": int(max(dynamic_history, 1)),
            "q": float(spot_q),
            "init_quantile": float(spot_init_quantile),
            "score_col": "scores_smooth",
            "source_score_col": score_col,
            "warmup_pred": int(dynamic_warmup_pred),
        },
    }
    return df, static_payload, dynamic_payload


def summarize_threshold_methods(
    scored_df: pd.DataFrame,
    label_cols: list[str],
    score_col: str = "scores_smooth",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [
        (STATIC_METHOD, "pred_static", "threshold_static"),
        (STATIC_VAL_SIGMA_METHOD, "pred_static_val_sigma3", "threshold_static_val_sigma3"),
        (DYNAMIC_METHOD, "pred_dynamic", "threshold_dynamic"),
        (SPOT_METHOD, "pred_spot", "threshold_spot"),
    ]
    for method, pred_col, threshold_col in methods:
        if pred_col not in scored_df.columns:
            continue
        for label_col in label_cols:
            if label_col not in scored_df.columns:
                continue
            valid = (
                pd.to_numeric(scored_df[label_col], errors="coerce").notna()
                & pd.to_numeric(scored_df[score_col], errors="coerce").notna()
                & pd.to_numeric(scored_df[pred_col], errors="coerce").notna()
            ).to_numpy()
            if not bool(valid.any()):
                continue
            y = scored_df.loc[valid, label_col].to_numpy(dtype=np.int32)
            s = scored_df.loc[valid, score_col].to_numpy(dtype=np.float64)
            p = scored_df.loc[valid, pred_col].to_numpy(dtype=np.int32)
            threshold_values = (
                scored_df.loc[valid, threshold_col].to_numpy(dtype=np.float64)
                if threshold_col in scored_df.columns
                else np.asarray([], dtype=np.float64)
            )
            row: dict[str, Any] = {
                "threshold_method": method,
                "score_col": score_col,
                "pred_col": pred_col,
                "threshold_col": threshold_col,
                "label_col": label_col,
                "threshold": float(threshold_values[0])
                if method in {STATIC_METHOD, STATIC_VAL_SIGMA_METHOD} and threshold_values.size
                else float("nan"),
                "threshold_mean": finite_mean(threshold_values),
            }
            row.update(compute_ranking_metrics(y_true=y, score=s))
            row.update(compute_binary_metrics(y_true=y, pred=p))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_flight_threshold_methods(
    scored_df: pd.DataFrame,
    label_cols: list[str],
    score_col: str = "scores_smooth",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [
        (STATIC_METHOD, "pred_static", "threshold_static"),
        (STATIC_VAL_SIGMA_METHOD, "pred_static_val_sigma3", "threshold_static_val_sigma3"),
        (DYNAMIC_METHOD, "pred_dynamic", "threshold_dynamic"),
        (SPOT_METHOD, "pred_spot", "threshold_spot"),
    ]
    groups = scored_df.groupby("flight", sort=True) if "flight" in scored_df.columns else [(None, scored_df)]
    for flight, group in groups:
        for method, pred_col, threshold_col in methods:
            if pred_col not in group.columns:
                continue
            for label_col in label_cols:
                if label_col not in group.columns:
                    continue
                valid = (
                    pd.to_numeric(group[label_col], errors="coerce").notna()
                    & pd.to_numeric(group[score_col], errors="coerce").notna()
                    & pd.to_numeric(group[pred_col], errors="coerce").notna()
                ).to_numpy()
                if not bool(valid.any()):
                    continue
                y = group.loc[valid, label_col].to_numpy(dtype=np.int32)
                s = group.loc[valid, score_col].to_numpy(dtype=np.float64)
                p = group.loc[valid, pred_col].to_numpy(dtype=np.int32)
                threshold_values = (
                    group.loc[valid, threshold_col].to_numpy(dtype=np.float64)
                    if threshold_col in group.columns
                    else np.asarray([], dtype=np.float64)
                )
                row: dict[str, Any] = {
                    "flight": "" if flight is None else str(flight),
                    "threshold_method": method,
                    "score_col": score_col,
                    "pred_col": pred_col,
                    "threshold_col": threshold_col,
                    "label_col": label_col,
                    "threshold": float(threshold_values[0])
                    if method in {STATIC_METHOD, STATIC_VAL_SIGMA_METHOD} and threshold_values.size
                    else float("nan"),
                    "threshold_mean": finite_mean(threshold_values),
                }
                row.update(compute_ranking_metrics(y_true=y, score=s))
                row.update(compute_binary_metrics(y_true=y, pred=p))
                rows.append(row)
    return pd.DataFrame(rows)
