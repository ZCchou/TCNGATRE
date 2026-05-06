from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _display_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    if not bool(valid.any()):
        return out
    lo = float(np.nanmin(x[valid]))
    hi = float(np.nanmax(x[valid]))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        out[valid] = 0.0
        return out
    out[valid] = (x[valid] - lo) / (hi - lo)
    return out


def _load_labels(labels_root: Path | None, flight: str, time_offset_sec: float = 0.0) -> pd.DataFrame | None:
    if labels_root is None:
        return None
    path = Path(labels_root) / f"{flight}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "t" not in df.columns or "anomaly_label" not in df.columns:
        return None
    out = df[["t", "anomaly_label"]].copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce") + float(time_offset_sec)
    out["anomaly_label"] = pd.to_numeric(out["anomaly_label"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["t"]).sort_values("t", kind="mergesort").reset_index(drop=True)
    return out if len(out) > 0 else None


def _add_anomaly_background(ax, labels_df: pd.DataFrame | None, t_min: float, t_max: float) -> None:
    if labels_df is None or labels_df.empty:
        return
    t = labels_df["t"].to_numpy(dtype=np.float64)
    y = (labels_df["anomaly_label"].to_numpy(dtype=np.float64) > 0.5).astype(np.int32)
    if len(t) <= 0 or not bool(y.any()):
        return
    in_region = False
    start = float(t_min)
    for idx, value in enumerate(y):
        if value and not in_region:
            in_region = True
            start = float(t[idx])
        if in_region and (not value or idx == len(y) - 1):
            end_idx = idx if value and idx == len(y) - 1 else max(idx - 1, 0)
            end = float(t[end_idx])
            ax.axvspan(max(start, t_min), min(end, t_max), color="tab:red", alpha=0.12)
            in_region = False


def _set_dynamic_ylim(ax, *series: np.ndarray) -> None:
    vals = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in series])
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if abs(hi - lo) <= 1e-12:
        pad = 0.05 if abs(hi) <= 1e-12 else abs(hi) * 0.05
    else:
        pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)


def plot_pred_true_resid_score_timelines(
    scored_df: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
    model_label: str,
    labels_root: Path | None = None,
    max_flights: int = 0,
    time_offset_sec: float = 0.0,
    true_prefix: str = "last_true__",
    pred_prefix: str = "last_pred__",
    err_prefix: str = "last_err__",
    score_prefix: str = "last_sensor_score__",
    residual_is_squared: bool = True,
) -> int:
    """Plot STGTCN-style per-flight top-10 true/recon/residual/score timelines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scored_df.empty:
        return 0

    features = [
        str(name)
        for name in feature_names
        if f"{true_prefix}{name}" in scored_df.columns
        and f"{pred_prefix}{name}" in scored_df.columns
        and f"{err_prefix}{name}" in scored_df.columns
    ]
    if not features:
        return 0

    flights = list(scored_df["flight"].drop_duplicates())
    if int(max_flights) > 0:
        flights = flights[: int(max_flights)]

    written = 0
    for flight in flights:
        group = scored_df.loc[scored_df["flight"] == flight].sort_values("current_index", kind="mergesort")
        if len(group) <= 0:
            continue
        ranked_features = sorted(
            features,
            key=lambda name: float(pd.to_numeric(group[f"{err_prefix}{name}"], errors="coerce").mean()),
            reverse=True,
        )[: min(10, len(features))]
        if not ranked_features:
            continue

        t_mid = group["t_mid"].to_numpy(dtype=np.float64)
        labels_df = _load_labels(labels_root=labels_root, flight=str(flight), time_offset_sec=time_offset_sec)
        fig, axes = plt.subplots(len(ranked_features), 2, figsize=(16, 3.2 * len(ranked_features)), squeeze=False)
        for row_idx, feature in enumerate(ranked_features):
            true_value = pd.to_numeric(group[f"{true_prefix}{feature}"], errors="coerce").to_numpy(dtype=np.float64)
            pred_value = pd.to_numeric(group[f"{pred_prefix}{feature}"], errors="coerce").to_numpy(dtype=np.float64)
            err_value = pd.to_numeric(group[f"{err_prefix}{feature}"], errors="coerce").to_numpy(dtype=np.float64)
            if residual_is_squared:
                resid_value = np.sqrt(np.maximum(err_value, 0.0))
            else:
                resid_value = np.abs(err_value)
            score_col = f"{score_prefix}{feature}"
            score_value = (
                pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=np.float64)
                if score_col in group.columns
                else err_value
            )

            ax0 = axes[row_idx, 0]
            _add_anomaly_background(ax0, labels_df, float(np.nanmin(t_mid)), float(np.nanmax(t_mid)))
            ax0.plot(t_mid, true_value, label="true_last_value", linewidth=1.1)
            ax0.plot(t_mid, pred_value, label="recon_or_pred_last_value", linewidth=1.1)
            ax0.set_title(f"{flight} | {feature} | {model_label} true vs recon/pred")
            ax0.set_xlabel("t (sec)")
            _set_dynamic_ylim(ax0, true_value, pred_value)
            ax0.grid(True, alpha=0.25)
            ax0.legend(loc="best", fontsize=8)

            ax1 = axes[row_idx, 1]
            _add_anomaly_background(ax1, labels_df, float(np.nanmin(t_mid)), float(np.nanmax(t_mid)))
            ax1.plot(t_mid, _display_normalize(resid_value), label="abs residual (display)", linewidth=1.1)
            ax1.plot(t_mid, _display_normalize(score_value), label="sensor_score (display)", linewidth=1.1)
            ax1.set_title(f"{flight} | {feature} | {model_label} residual vs score")
            ax1.set_xlabel("t (sec)")
            ax1.set_ylim(0.0, 1.0)
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__top10_pred_true_resid_score_timeline.png", dpi=140)
        plt.close(fig)
        written += 1
    return written
