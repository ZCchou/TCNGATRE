from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SOURCE_DATASET = "alfa"
TARGET_DATASET = "alfa_despiked"
TARGET_WIDE_ROOT = "wide_flights_set_a_despiked_offline_strong"
TARGET_LABELS_ROOT = "wide_flights_failure_labels"
MANIFEST_NAME = "dataset_manifest.json"

BASE_FEATURES = [
    "mavros-nav_info-roll:field.measured",
    "mavros-nav_info-pitch:field.measured",
    "mavros-nav_info-yaw:field.measured",
    "mavros-vfr_hud:field.groundspeed",
    "mavros-vfr_hud:field.heading",
    "mavros-vfr_hud:field.throttle",
    "mavros-vfr_hud:field.climb",
    "mavros-vfr_hud:field.altitude",
]
ANGLE_BASE_FEATURES = {
    "mavros-nav_info-roll:field.measured",
    "mavros-nav_info-pitch:field.measured",
    "mavros-nav_info-yaw:field.measured",
    "mavros-vfr_hud:field.heading",
}
HAMPEL_WINDOW = 21
HAMPEL_SIGMA = 6.0
SMOOTH_WINDOW = 7
COMPARE_PAGE_SIZE = 4
COMPARE_DPI = 140


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_root() -> Path:
    return repo_root() / "dataset" / SOURCE_DATASET


def target_root() -> Path:
    return repo_root() / "dataset" / TARGET_DATASET


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_wide_root(dataset_root: Path) -> Path:
    manifest = load_manifest(dataset_root / MANIFEST_NAME)
    explicit = str(manifest.get("wide_root_dirname", "")).strip()
    if explicit:
        return dataset_root / explicit
    candidates = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
        and (path / "No_Failure").is_dir()
        and (path / "Failure").is_dir()
        and any((path / "No_Failure").glob("*.csv"))
        and any((path / "Failure").glob("*.csv"))
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one source wide root under {dataset_root}, found {len(candidates)}")
    return candidates[0]


def build_model_feature_names(base_features: list[str]) -> list[str]:
    names: list[str] = []
    for feature in base_features:
        if feature in ANGLE_BASE_FEATURES:
            names.append(f"{feature}__sin_med")
            names.append(f"{feature}__cos_med")
        else:
            names.append(f"{feature}_med")
    return names


MODEL_FEATURES = build_model_feature_names(BASE_FEATURES)


def centered_rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rolling(window=int(window), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64, copy=False)
    )


def centered_rolling_mad(values: np.ndarray, med: np.ndarray, window: int) -> np.ndarray:
    resid = np.abs(values - med)
    return (
        pd.Series(resid, copy=False)
        .rolling(window=int(window), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64, copy=False)
    )


def despike_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    med = centered_rolling_median(x, HAMPEL_WINDOW)
    mad = centered_rolling_mad(x, med, HAMPEL_WINDOW)
    thresh = np.maximum(HAMPEL_SIGMA * 1.4826 * mad, 1e-9)
    spike_mask = np.abs(x - med) > thresh
    replaced = np.where(spike_mask, med, x)
    smoothed = centered_rolling_median(replaced, SMOOTH_WINDOW)
    return smoothed.astype(np.float64, copy=False), spike_mask.astype(bool, copy=False)


def add_model_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in BASE_FEATURES:
        med_col = f"{feature}_med"
        x = pd.to_numeric(out[med_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if feature in ANGLE_BASE_FEATURES:
            rad = np.deg2rad(x)
            out[f"{feature}__sin_med"] = np.sin(rad).astype(np.float32)
            out[f"{feature}__cos_med"] = np.cos(rad).astype(np.float32)
        else:
            out[med_col] = x.astype(np.float32)
    return out


def fit_minmax_stats(no_failure_frames: list[pd.DataFrame]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for feature in MODEL_FEATURES:
        all_values: list[np.ndarray] = []
        for frame in no_failure_frames:
            if feature in frame.columns:
                vals = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy(dtype=np.float64)
                if vals.size > 0:
                    all_values.append(vals)
        if all_values:
            merged = np.concatenate(all_values).astype(np.float64, copy=False)
            vmin = float(np.nanmin(merged))
            vmax = float(np.nanmax(merged))
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin + 1e-12:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        stats[feature] = {"min": vmin, "max": vmax}
    return stats


def apply_minmax_columns(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for feature in MODEL_FEATURES:
        mm_col = f"{feature}_mm"
        vals = pd.to_numeric(out[feature], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        vmin = float(stats[feature]["min"])
        vmax = float(stats[feature]["max"])
        scaled = (vals - vmin) / max(vmax - vmin, 1e-8)
        out[mm_col] = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return out


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    out = df.copy()
    stats_rows: list[dict[str, float]] = []
    for feature in BASE_FEATURES:
        original = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=np.float64)
        cleaned, spike_mask = despike_series(original)
        out[feature] = cleaned.astype(np.float32)
        out[f"{feature}_med"] = cleaned.astype(np.float32)
        stats_rows.append(
            {
                "feature": feature,
                "replaced_points": int(np.sum(spike_mask)),
                "replaced_ratio": float(np.mean(spike_mask.astype(np.float64))),
                "max_abs_delta": float(np.max(np.abs(cleaned - original))) if original.size > 0 else 0.0,
                "mean_abs_delta": float(np.mean(np.abs(cleaned - original))) if original.size > 0 else 0.0,
                "p95_abs_delta": float(np.quantile(np.abs(cleaned - original), 0.95)) if original.size > 0 else 0.0,
            }
        )
    out = add_model_feature_columns(out)
    return out, stats_rows


def copy_labels_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_labels(labels_root: Path, flight: str, time_offset_sec: float) -> pd.DataFrame | None:
    path = labels_root / f"{flight}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if "t" not in df.columns or "anomaly_label" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["t"], errors="coerce"),
            "anomaly_label": pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["t"])
    if out.empty:
        return None
    if float(time_offset_sec) != 0.0:
        out["t"] = out["t"] + float(time_offset_sec)
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)


def compute_anomaly_spans(labels_df: pd.DataFrame | None, t_min: float, t_max: float) -> list[tuple[float, float]]:
    if labels_df is None or labels_df.empty:
        return []
    t = labels_df["t"].to_numpy(dtype=float)
    y = (labels_df["anomaly_label"].to_numpy(dtype=float) > 0.5).astype(np.int32)
    spans: list[tuple[float, float]] = []
    in_run = False
    start = 0.0
    for i in range(len(t)):
        if y[i] == 1 and not in_run:
            start = float(t[i])
            in_run = True
        if in_run and (i == len(t) - 1 or y[i + 1] == 0):
            end = float(t[i] if i == len(t) - 1 else t[i + 1])
            left = max(start, t_min)
            right = min(end, t_max)
            if right > left:
                spans.append((left, right))
            in_run = False
    return spans


def plot_compare_page(
    flight: str,
    split_name: str,
    source_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    feature_names: list[str],
    spans: list[tuple[float, float]],
    page_index: int,
    total_pages: int,
    out_path: Path,
) -> None:
    t = pd.to_numeric(source_df["t"], errors="coerce").to_numpy(dtype=np.float64)
    n_axes = len(feature_names)
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(18.0, max(3.0, 2.6 * n_axes + 1.2)),
        sharex=True,
    )
    if n_axes == 1:
        axes = [axes]
    for ax, feature in zip(axes, feature_names):
        source_vals = pd.to_numeric(source_df[feature], errors="coerce").to_numpy(dtype=np.float64)
        cleaned_vals = pd.to_numeric(cleaned_df[feature], errors="coerce").to_numpy(dtype=np.float64)
        for left, right in spans:
            ax.axvspan(left, right, color="tab:red", alpha=0.12, linewidth=0.0)
        ax.plot(t, source_vals, color="#577590", linewidth=0.8, label="before")
        ax.plot(t, cleaned_vals, color="#E76F51", linewidth=0.9, label="after")
        ax.set_title(feature, fontsize=9, loc="left", pad=2.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (sec)")
    fig.suptitle(f"{split_name} | {flight} | before vs after | page {page_index + 1}/{total_pages}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out_path, dpi=COMPARE_DPI)
    plt.close(fig)


def render_compare_visuals(
    src_wide_root: Path,
    dst_wide_root: Path,
    labels_root: Path,
    label_offset_sec: float,
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    all_files: list[tuple[str, Path, Path]] = []
    for split_name in ("No_Failure", "Failure"):
        src_split = src_wide_root / split_name
        dst_split = dst_wide_root / split_name
        for src_path in sorted(src_split.glob("*.csv")):
            dst_path = dst_split / src_path.name
            if dst_path.exists():
                all_files.append((split_name, src_path, dst_path))

    for split_name, src_path, dst_path in tqdm(all_files, desc="render alfa_despiked compare", unit="flight"):
        flight = src_path.stem
        source_df = pd.read_csv(src_path, low_memory=False)
        cleaned_df = pd.read_csv(dst_path, low_memory=False)
        feature_names = list(BASE_FEATURES)
        labels_df = load_labels(
            labels_root=labels_root,
            flight=flight,
            time_offset_sec=label_offset_sec if split_name == "Failure" else 0.0,
        )
        t = pd.to_numeric(source_df["t"], errors="coerce").to_numpy(dtype=np.float64)
        spans = compute_anomaly_spans(labels_df, float(np.nanmin(t)), float(np.nanmax(t))) if t.size > 0 else []
        total_pages = int(math.ceil(len(feature_names) / COMPARE_PAGE_SIZE))
        flight_dir = out_root / split_name / flight
        flight_dir.mkdir(parents=True, exist_ok=True)
        for page_index in range(total_pages):
            start = page_index * COMPARE_PAGE_SIZE
            end = min(len(feature_names), start + COMPARE_PAGE_SIZE)
            page_features = feature_names[start:end]
            out_path = flight_dir / f"page_{page_index + 1:02d}.png"
            plot_compare_page(
                flight=flight,
                split_name=split_name,
                source_df=source_df,
                cleaned_df=cleaned_df,
                feature_names=page_features,
                spans=spans,
                page_index=page_index,
                total_pages=total_pages,
                out_path=out_path,
            )
        index_rows.append(
            {
                "split": split_name,
                "flight": flight,
                "source_csv": str(src_path),
                "cleaned_csv": str(dst_path),
                "output_dir": str(flight_dir),
                "num_pages": total_pages,
                "num_rows": int(len(source_df)),
            }
        )

    pd.DataFrame(index_rows).sort_values(["split", "flight"], kind="mergesort").to_csv(
        out_root / "index.csv",
        index=False,
        encoding="utf-8",
    )


def main() -> int:
    src_root = source_root()
    dst_root = target_root()
    src_manifest = load_manifest(src_root / MANIFEST_NAME)
    src_wide_root = discover_wide_root(src_root)
    src_labels_root = src_root / str(src_manifest.get("labels_dirname", TARGET_LABELS_ROOT))
    if not src_labels_root.exists():
        raise FileNotFoundError(f"Missing source labels root: {src_labels_root}")

    dst_wide_root = dst_root / TARGET_WIDE_ROOT
    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_wide_root / "Failure").mkdir(parents=True, exist_ok=True)
    (dst_wide_root / "No_Failure").mkdir(parents=True, exist_ok=True)

    processed_frames: dict[tuple[str, str], pd.DataFrame] = {}
    stats_rows: list[dict[str, object]] = []
    no_failure_frames: list[pd.DataFrame] = []
    all_source_files: list[tuple[str, Path]] = []
    for split_name in ("No_Failure", "Failure"):
        all_source_files.extend((split_name, path) for path in sorted((src_wide_root / split_name).glob("*.csv")))

    for split_name, src_path in tqdm(all_source_files, desc="build alfa_despiked", unit="flight"):
        source_df = pd.read_csv(src_path, low_memory=False)
        cleaned_df, per_feature_stats = clean_frame(source_df)
        processed_frames[(split_name, src_path.name)] = cleaned_df
        if split_name == "No_Failure":
            no_failure_frames.append(cleaned_df)
        for row in per_feature_stats:
            stats_rows.append(
                {
                    "split": split_name,
                    "flight": src_path.stem,
                    **row,
                }
            )

    norm_stats = fit_minmax_stats(no_failure_frames)
    (dst_wide_root / "norm_stats.json").write_text(
        json.dumps(norm_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for (split_name, filename), cleaned_df in tqdm(processed_frames.items(), desc="save alfa_despiked", unit="flight"):
        final_df = apply_minmax_columns(cleaned_df, norm_stats)
        dst_path = dst_wide_root / split_name / filename
        final_df.to_csv(dst_path, index=False, encoding="utf-8")

    copy_labels_tree(src_labels_root, dst_root / TARGET_LABELS_ROOT)

    dst_manifest = dict(src_manifest)
    dst_manifest["wide_root_dirname"] = TARGET_WIDE_ROOT
    dst_manifest["labels_dirname"] = TARGET_LABELS_ROOT
    dst_manifest["source_dataset"] = SOURCE_DATASET
    dst_manifest["physical_despike"] = True
    dst_manifest["despike_mode"] = "offline_strong_centered_hampel_plus_median"
    dst_manifest["despike_base_features"] = list(BASE_FEATURES)
    dst_manifest["despike_params"] = {
        "hampel_window": HAMPEL_WINDOW,
        "hampel_sigma": HAMPEL_SIGMA,
        "smooth_window": SMOOTH_WINDOW,
    }
    (dst_root / MANIFEST_NAME).write_text(
        json.dumps(dst_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats_df = pd.DataFrame(stats_rows).sort_values(["split", "flight", "feature"], kind="mergesort").reset_index(drop=True)
    stats_df.to_csv(dst_root / "despike_stats.csv", index=False, encoding="utf-8")
    summary = {
        "source_dataset": SOURCE_DATASET,
        "target_dataset": TARGET_DATASET,
        "wide_root": str(dst_wide_root),
        "labels_root": str(dst_root / TARGET_LABELS_ROOT),
        "num_processed_files": int(len(processed_frames)),
        "num_stat_rows": int(len(stats_df)),
        "total_replaced_points": int(stats_df["replaced_points"].sum()) if len(stats_df) > 0 else 0,
    }
    (dst_root / "despike_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    render_compare_visuals(
        src_wide_root=src_wide_root,
        dst_wide_root=dst_wide_root,
        labels_root=dst_root / TARGET_LABELS_ROOT,
        label_offset_sec=float(dst_manifest.get("failure_label_time_offset_sec", 0.0)),
        out_root=dst_root / "dataviz_compare",
    )

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
