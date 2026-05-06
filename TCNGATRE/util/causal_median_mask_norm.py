# -*- coding: utf-8 -*-
"""
causal_median_mask_norm.py

Processes set_a_raw data with a split-aware dense causal median + min-max pipeline.

The current model feature space expands 4 angle-like channels into sin/cos pairs
after causal median filtering:
  roll, pitch, yaw, heading  ->  sin/cos
This keeps the data causal because the transform is applied pointwise on the
current filtered value only.

  Stage 1 – Raw forward fill to make every selected feature dense
  Stage 2 – Strictly causal median filter on the dense sequence
  Stage 3 – Binary observation mask
  Stage 4 – Min-max normalization fitted ONLY on No_Failure data

═══════════════════════════════════════════════════════════════════════════════
Why time-based window (W seconds) vs. K-observation window?
─────────────────────────────────────────────────────────────────────────────
  Time-based (W sec):
    + Semantically meaningful: "what did the sensor tell us in the last W s?"
    + Consistent lag regardless of per-channel sample rate
    − Low-rate channels (vfr_hud ~3 Hz) may have very few obs in a small W
    → Choose W large enough to include ≥3 obs even for the slowest channel

  K-observation window:
    + Always exactly K past real obs → stable median quality
    + Adapts to each channel's actual rate: fast channels get a short lookback
    − Lag in real-time seconds varies: for vfr_hud K=5 spans ~1.7 s
    − For a slow channel early in the flight K points can span the whole file

  Decision (default):
    K = 16 obs  (fixed observation window)
      nav_info  ~21 Hz → ~0.76 s effective lookback
      vfr_hud    ~3 Hz → ~5.33 s effective lookback
    Optional legacy mode: --window_sec W  (time-based)

═══════════════════════════════════════════════════════════════════════════════
Output columns (per base feature f):
    t               time axis (seconds)
    {f}             x_obs_ffill – dense raw value after forward fill / head fill
    {f}_mask        0/1 int – 1 = real observation, 0 = missing
    {f}_med         x_med   – causal-median-filtered dense value

Output columns (per model feature g):
    {g}_med         model-space value after optional angle sin/cos expansion
    {g}_med_mm      min-max normalized model-space value

Output layout:
    <out_root>/
        Failure/*.csv
        No_Failure/*.csv
        norm_stats.json      ← {feature: {min, max}}

Usage
─────
    python util/causal_median_mask_norm.py
    python util/causal_median_mask_norm.py --window_k 16
    python util/causal_median_mask_norm.py --window_sec 0.5
    python util/causal_median_mask_norm.py \\
        --in_dir  dataset/alfa/wide_flights_set_a_raw \\
        --out_dir dataset/alfa/wide_flights_set_a_causal \\
        --window_k 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

SUBDIRS = ["Failure", "No_Failure"]

SET_A_BASE_FEATS: List[str] = [
    "mavros-nav_info-roll:field.measured",
    "mavros-nav_info-pitch:field.measured",
    "mavros-nav_info-yaw:field.measured",
    "mavros-vfr_hud:field.groundspeed",
    "mavros-vfr_hud:field.heading",
    "mavros-vfr_hud:field.throttle",
    "mavros-vfr_hud:field.climb",
    "mavros-vfr_hud:field.altitude",
]

ANGLE_BASE_FEATS: List[str] = [
    "mavros-nav_info-roll:field.measured",
    "mavros-nav_info-pitch:field.measured",
    "mavros-nav_info-yaw:field.measured",
    "mavros-vfr_hud:field.heading",
]


def build_model_feats(base_feats: List[str]) -> List[str]:
    model_feats: List[str] = []
    angle_set = set(ANGLE_BASE_FEATS)
    for feat in base_feats:
        if feat in angle_set:
            model_feats.append(f"{feat}__sin")
            model_feats.append(f"{feat}__cos")
        else:
            model_feats.append(feat)
    return model_feats


SET_A_MODEL_FEATS: List[str] = build_model_feats(SET_A_BASE_FEATS)


# ── causal median filters ──────────────────────────────────────────────────

def dense_forward_fill_raw(x_arr: np.ndarray) -> np.ndarray:
    """
    Make a raw feature dense using forward fill.

    If the sequence starts with missing rows, the leading gap is filled with
    the first valid observation so downstream values stay dense.
    """
    x = np.asarray(x_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D raw feature, got shape={x.shape}")
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    dense = pd.Series(x).ffill().bfill().to_numpy(dtype=np.float64)
    return dense.astype(np.float32)


def causal_median_time_window_dense(
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    W: float,
) -> np.ndarray:
    """
    Strictly causal time-based median filter on a dense sequence.
    """
    n = len(t_arr)
    t = np.asarray(t_arr, dtype=np.float64)
    x = np.asarray(x_arr, dtype=np.float64)
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    x_med = np.empty(n, dtype=np.float64)
    left = 0
    for ii in range(n):
        while t[left] < t[ii] - W:
            left += 1
        x_med[ii] = float(np.median(x[left : ii + 1]))

    return x_med.astype(np.float32)


def causal_median_k_dense(
    x_arr: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Strictly causal K-step median filter on a dense sequence.
    """
    x = np.asarray(x_arr, dtype=np.float64)
    if len(x) == 0:
        return np.empty((0,), dtype=np.float32)
    x_med = np.empty(len(x), dtype=np.float64)

    for ii in range(len(x)):
        lo = max(0, ii - K + 1)
        x_med[ii] = float(np.median(x[lo : ii + 1]))

    return x_med.astype(np.float32)


# ── per-file processing ────────────────────────────────────────────────────

def process_file(
    fp: Path,
    feat_cols: List[str],
    window_sec: Optional[float],
    window_k: Optional[int],
    crop_head_sec: float = 0.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        t, {f}, {f}_mask, {f}_med  for each feature f
    Min-max columns are added later.
    """
    df = pd.read_csv(fp, low_memory=False, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    if "t" not in df.columns:
        raise ValueError(f"Missing required timeline column 't' in: {fp}")

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.loc[np.isfinite(df["t"].to_numpy(dtype=np.float64))].copy()
    if len(df) <= 0:
        return pd.DataFrame({"t": np.empty((0,), dtype=np.float64)})

    # Crop logic is intentionally disabled so preprocessing preserves the full timeline.
    _ = crop_head_sec

    t_arr = df["t"].values.astype(np.float64)
    out = pd.DataFrame({"t": t_arr})

    for f in feat_cols:
        if f not in df.columns:
            out[f] = 0.0
            out[f"{f}_mask"] = 0
            out[f"{f}_med"] = 0.0
            continue

        x_arr = df[f].to_numpy(dtype=np.float64)
        x_dense = dense_forward_fill_raw(x_arr)
        out[f] = x_dense

        # mask: 1 where real observation, 0 otherwise
        mask = (~np.isnan(x_arr)).astype(np.int8)
        out[f"{f}_mask"] = mask

        # causal median filter over dense values
        if window_sec is not None:
            x_med = causal_median_time_window_dense(t_arr, x_dense, W=window_sec)
        else:
            x_med = causal_median_k_dense(x_dense, K=window_k)  # type: ignore[arg-type]
        out[f"{f}_med"] = x_med

    return out


def augment_model_feature_space(df: pd.DataFrame, base_feats: List[str]) -> pd.DataFrame:
    """
    Add model-space `{g}_med` columns.

    Angle-like channels are converted from degrees to sin/cos pairs after the
    strictly causal median filter has already been applied.
    """
    out = df.copy()
    angle_set = set(ANGLE_BASE_FEATS)
    for feat in base_feats:
        med_col = f"{feat}_med"
        if med_col not in out.columns:
            continue
        x = pd.to_numeric(out[med_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if feat in angle_set:
            rad = np.deg2rad(x)
            out[f"{feat}__sin_med"] = np.sin(rad).astype(np.float32)
            out[f"{feat}__cos_med"] = np.cos(rad).astype(np.float32)
    return out


# ── normalization helpers ──────────────────────────────────────────────────

NormStats = Dict[str, Dict[str, float]]   # {feat: {"min": ..., "max": ...}}


def fit_norm_stats(
    nofail_frames: List[pd.DataFrame],
    feat_cols: List[str],
) -> NormStats:
    """
    Fit min-max stats from No_Failure x_med values (ignoring NaN).
    """
    stats: NormStats = {}
    for f in feat_cols:
        med_col = f"{f}_med"
        all_vals: List[np.ndarray] = []
        for df in nofail_frames:
            if med_col in df.columns:
                v = df[med_col].dropna().values
                if len(v) > 0:
                    all_vals.append(v)
        if all_vals:
            combined = np.concatenate(all_vals).astype(np.float64)
            mu = float(np.min(combined))
            sd = float(np.max(combined))
            if not np.isfinite(mu):
                mu = 0.0
            if not np.isfinite(sd) or sd <= mu + 1e-12:
                sd = mu + 1.0
        else:
            mu, sd = 0.0, 1.0
        stats[f] = {"min": mu, "max": sd}
    return stats


def apply_minmax(df: pd.DataFrame, feat_cols: List[str], stats: NormStats) -> pd.DataFrame:
    """
    Add {f}_med_mm columns using:
        x_med -> min-max to [0, 1]
    """
    df = df.copy()
    for f in feat_cols:
        med_col = f"{f}_med"
        out_col = f"{f}_med_mm"
        if med_col not in df.columns:
            df[out_col] = 0.0
            continue
        vmin = stats[f]["min"]
        vmax = stats[f]["max"]
        x = pd.to_numeric(df[med_col], errors="coerce").fillna(vmin).to_numpy(dtype=np.float64)
        mm = (x - vmin) / max(vmax - vmin, 1e-8)
        mm = np.clip(mm, 0.0, 1.0)
        df[out_col] = mm.astype(np.float32)
    return df


# ── main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dense causal median filter + mask + min-max without timeline cropping."
    )
    ap.add_argument("--in_dir",  default="dataset/alfa/wide_flights_set_a_raw")
    ap.add_argument("--out_dir", default="dataset/alfa/wide_flights_set_a_causal")
    ap.add_argument(
        "--window_sec", type=float, default=None,
        help="Legacy time-based causal median window in seconds. "
             "Ignored when --window_k is set. If omitted, the default is fixed --window_k 16.",
    )
    ap.add_argument(
        "--window_k", type=int, default=16,
        help="K-observation causal median window. Default 16. Overrides --window_sec.",
    )
    ap.add_argument(
        "--crop_head_sec_failure",
        type=float,
        default=0.0,
        help="Deprecated. Crop logic is disabled and this value is ignored.",
    )
    ap.add_argument(
        "--crop_head_sec_no_failure",
        type=float,
        default=0.0,
        help="Deprecated. Crop logic is disabled and this value is ignored.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args    = parse_args()
    in_dir  = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    use_k      = args.window_k is not None
    window_sec = None if use_k else (float(args.window_sec) if args.window_sec is not None else None)
    window_k   = int(args.window_k) if use_k else None
    win_desc   = f"K={window_k} obs" if use_k else f"W={window_sec}s"
    crop_head_sec_failure = 0.0
    crop_head_sec_no_failure = 0.0

    if not in_dir.exists():
        print(f"[ERROR] --in_dir not found: {in_dir}", file=sys.stderr)
        return 2

    # ── collect files ──────────────────────────────────────────────────────
    all_files: List[Tuple[Path, str]] = []
    for sub in SUBDIRS:
        sd = in_dir / sub
        if sd.is_dir():
            for fp in sorted(sd.glob("*.csv")):
                all_files.append((fp, sub))

    if not all_files:
        print(f"[ERROR] no CSVs found under {in_dir}", file=sys.stderr)
        return 2

    base_feat_cols = SET_A_BASE_FEATS
    model_feat_cols = SET_A_MODEL_FEATS
    print(f"[INFO] {len(all_files)} files | causal median {win_desc}")
    print(f"[INFO] base features: {len(base_feat_cols)}")
    print(f"[INFO] model features: {len(model_feat_cols)}")
    print("[INFO] crop_head_sec Failure=0.0 (disabled)")
    print("[INFO] crop_head_sec No_Failure=0.0 (disabled)")
    print(f"[INFO] output  → {out_dir}")

    # ── Stage 1: dense causal median filter + mask (all files) ────────────
    print("\n[Stage 1] Dense causal median filter + mask …")
    processed: Dict[str, pd.DataFrame] = {}      # key: "sub/fname"

    for fp, sub in tqdm(all_files, desc="filter", unit="file"):
        key = f"{sub}/{fp.name}"
        crop_head_sec = crop_head_sec_failure if sub == "Failure" else crop_head_sec_no_failure
        df_out = process_file(fp, base_feat_cols, window_sec, window_k, crop_head_sec=crop_head_sec)
        df_out = augment_model_feature_space(df_out, base_feat_cols)
        processed[key] = df_out

        # print per-feature obs & median coverage stats
        n = len(df_out)
        if n <= 0:
            tqdm.write(f"[WARN] {key} produced 0 rows after preprocessing")
            continue
        lines = []
        for f in base_feat_cols:
            mc = f"{f}_mask"
            mm = f"{f}_med"
            n_obs = int(df_out[mc].sum()) if mc in df_out.columns else 0
            n_med = int(df_out[mm].notna().sum()) if mm in df_out.columns else 0
            lines.append(f"{f.split(':')[-1]}:obs={n_obs},med={n_med}")
        tqdm.write(f"[OK] {key} ({n} rows) | crop_head_sec={crop_head_sec} | " + "  ".join(lines))

    # ── Stage 2: fit min-max on No_Failure x_med ───────────────────────────
    print("\n[Stage 2] Fitting min-max from No_Failure x_med …")
    nofail_frames = [
        processed[f"{sub}/{fp.name}"]
        for fp, sub in all_files
        if sub == "No_Failure"
    ]
    if not nofail_frames:
        print("[WARN] No No_Failure frames found; min-max stats will be 0/1.")
    norm_stats = fit_norm_stats(nofail_frames, model_feat_cols)

    print("  Feature min-max stats:")
    for f, s in norm_stats.items():
        print(
            f"    {f.split(':')[-1]:30s}  min={s['min']:+.4f}  max={s['max']:.4f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(norm_stats, fh, indent=2)
    print(f"  Saved → {stats_path}")

    # ── Stage 3: apply min-max & write CSVs ────────────────────────────────
    print("\n[Stage 3] Applying min-max normalization & saving CSVs …")
    n_ok = n_fail = 0

    for fp, sub in tqdm(all_files, desc="save", unit="file"):
        key = f"{sub}/{fp.name}"
        dst = out_dir / sub / fp.name
        if dst.exists() and not args.overwrite:
            tqdm.write(f"[SKIP] {key}")
            n_ok += 1
            continue
        try:
            df_out = apply_minmax(processed[key], model_feat_cols, norm_stats)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(dst, index=False, encoding="utf-8")
            tqdm.write(f"[SAVED] {key}  shape={df_out.shape}")
            n_ok += 1
        except Exception as exc:
            import traceback
            tqdm.write(f"[FAIL] {key}: {exc}\n{traceback.format_exc()}")
            n_fail += 1

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n[DONE] ok={n_ok}  fail={n_fail}")
    print(f"Output columns per base feature f:")
    print(f"  {{f}}          = x_obs_ffill  (dense raw after forward fill)")
    print(f"  {{f}}_mask     = 0/1    (observation mask)")
    print(f"  {{f}}_med      = x_med  (dense causal median {win_desc})")
    print(f"Output columns per model feature g:")
    print(f"  {{g}}_med      = model-space value (sin/cos for roll/pitch/yaw/heading)")
    print(f"  {{g}}_med_mm   = min-max normalized model-space value (fit on No_Failure only)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
