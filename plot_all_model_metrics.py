from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import summarize_all_model_results as summarize_results


BUNDLE_ROOT = Path(__file__).resolve().parent

MODEL_ORDER = [
    "USAD",
    "Recurrent_AE",
    "TranAD",
    "OmniAnomaly",
    "BeatGAN",
    "TCNGATRE",
]
MODEL_DISPLAY = {
    "USAD": "USAD",
    "Recurrent_AE": "Recurrent AE",
    "TranAD": "TranAD",
    "OmniAnomaly": "OmniAnomaly",
    "BeatGAN": "BeatGAN",
    "TCNGATRE": "TCNGATRE",
}
MODEL_COLORS = {
    "USAD": "#2D6A4F",
    "Recurrent_AE": "#457B9D",
    "TranAD": "#E76F51",
    "OmniAnomaly": "#7B2CBF",
    "BeatGAN": "#BC6C25",
    "TCNGATRE": "#3A86FF",
}
MODEL_RUN_PREFIX = {
    "USAD": "usad",
    "Recurrent_AE": "recurrent_ae",
    "TranAD": "tranad",
    "OmniAnomaly": "omni_anomaly",
    "BeatGAN": "beatgan",
    "TCNGATRE": "tcngatre",
}

DATASET_ORDER = ["alfa", "alfa4hz", "simulate", "gpsdata"]
LABEL_COLS = ["label_any", "label_mid"]
THRESHOLD_METHODS = ["static_f1_oracle", "static_val_sigma3", "dynamic_history", "spot", "legacy"]
AVERAGE_KINDS = ["macro", "micro"]

METRIC_DISPLAY = {
    "auroc": "AUROC",
    "average_precision": "AP",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "fpr": "FPR",
}
METRIC_ORDER = list(METRIC_DISPLAY)


@dataclass(frozen=True)
class PlotSpec:
    average_kind: str
    dataset: str
    label_col: str
    threshold_method: str
    output_path: Path
    rows: int


@dataclass(frozen=True)
class ThresholdPlotSpec:
    model_key: str
    dataset: str
    flight: str
    label_col: str
    output_path: Path
    rows: int
    source_path: Path


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bar charts from all-model metric summaries. "
            "Each PNG keeps one dataset and one threshold method together."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        help="Models to plot.",
    )
    parser.add_argument(
        "--stat-dir",
        default="statistic",
        help="Directory containing all_model_dataset_metrics_*.csv. Relative paths are resolved from bundle root.",
    )
    parser.add_argument(
        "--out-dir",
        default="statistic/plots",
        help="Directory for PNG outputs. Relative paths are resolved from bundle root.",
    )
    parser.add_argument(
        "--averages",
        nargs="+",
        choices=AVERAGE_KINDS,
        default=AVERAGE_KINDS,
        help="Which summary levels to plot.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_ORDER,
        help="Datasets to plot.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=LABEL_COLS,
        help="Label columns to plot, for example label_any label_mid.",
    )
    parser.add_argument(
        "--threshold-methods",
        nargs="+",
        default=["static_f1_oracle", "static_val_sigma3", "dynamic_history", "spot"],
        help="Threshold methods to plot.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRIC_ORDER,
        help="Metric columns to include.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Do not draw macro mean +/- std error bars.",
    )
    parser.add_argument(
        "--no-threshold-plots",
        action="store_true",
        help="Only draw metric bar charts; skip score/threshold time-series plots.",
    )
    parser.add_argument(
        "--max-flights-per-run",
        type=int,
        default=0,
        help="Maximum flights plotted for each model/dataset. Use 0 to plot every flight.",
    )
    parser.add_argument(
        "--threshold-downsample",
        type=int,
        default=1,
        help="Plot every Nth point in threshold visualizations to keep large PNGs readable.",
    )
    parser.add_argument(
        "--skip-auto-summarize",
        action="store_true",
        help="Do not refresh statistic CSV files before plotting.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return BUNDLE_ROOT / path


def ordered_subset(values: Iterable[str], canonical_order: list[str]) -> list[str]:
    requested = list(values)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in canonical_order:
        if item in requested and item not in seen:
            ordered.append(item)
            seen.add(item)
    for item in requested:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def read_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def relative_to_bundle(path: Path) -> str:
    try:
        return path.resolve().relative_to(BUNDLE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def status_summary_lines(stat_dir: Path) -> list[str]:
    status_path = stat_dir / "all_model_dataset_metrics_status.csv"
    if not status_path.exists():
        return []
    try:
        df = pd.read_csv(status_path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty or "status" not in df.columns:
        return []
    counts = (
        df["status"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )
    return [f"{status}={int(count)}" for status, count in counts.items()]


def safe_write_text(path: Path, text: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except PermissionError as exc:
        print(f"[WARN] could not write {relative_to_bundle(path)}: {type(exc).__name__}: {exc}")
        return False


def metric_value(row: pd.Series, metric: str, average_kind: str) -> float:
    column = f"{metric}_mean" if average_kind == "macro" else metric
    if column not in row.index:
        return math.nan
    return pd.to_numeric(row[column], errors="coerce")


def metric_std(row: pd.Series, metric: str) -> float:
    column = f"{metric}_std"
    if column not in row.index:
        return math.nan
    return pd.to_numeric(row[column], errors="coerce")


def prepare_group(
    df: pd.DataFrame,
    average_kind: str,
    dataset: str,
    label_col: str,
    threshold_method: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    required = {"model_key", "dataset", "label_col", "threshold_method"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    group = df[
        (df["dataset"].astype(str) == dataset)
        & (df["label_col"].astype(str) == label_col)
        & (df["threshold_method"].astype(str) == threshold_method)
    ].copy()
    if group.empty:
        return group

    group["_model_order"] = group["model_key"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    group["_model_order"] = group["_model_order"].fillna(len(MODEL_ORDER)).astype(int)
    group = group.sort_values(["_model_order", "model_key"])
    group = group.drop_duplicates(["model_key"], keep="first")
    return group


def draw_grouped_metric_bars(
    group: pd.DataFrame,
    average_kind: str,
    dataset: str,
    label_col: str,
    threshold_method: str,
    metrics: list[str],
    output_path: Path,
    dpi: int,
    show_error_bars: bool,
) -> PlotSpec | None:
    if group.empty:
        return None

    available_metrics = [
        metric
        for metric in metrics
        if (
            (f"{metric}_mean" in group.columns if average_kind == "macro" else metric in group.columns)
            and metric in METRIC_DISPLAY
        )
    ]
    if not available_metrics:
        return None

    model_keys = [
        model for model in MODEL_ORDER if model in set(group["model_key"].astype(str))
    ]
    model_keys += [
        model
        for model in group["model_key"].astype(str).tolist()
        if model not in model_keys
    ]
    if not model_keys:
        return None

    values_by_model: dict[str, list[float]] = {}
    std_by_model: dict[str, list[float]] = {}
    for model in model_keys:
        row = group[group["model_key"].astype(str) == model].iloc[0]
        values_by_model[model] = [
            float(metric_value(row, metric, average_kind)) for metric in available_metrics
        ]
        std_by_model[model] = [
            float(metric_std(row, metric)) if average_kind == "macro" else math.nan
            for metric in available_metrics
        ]

    fig_width = max(10.5, 1.25 * len(available_metrics) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6.6), constrained_layout=True)

    x = np.arange(len(available_metrics), dtype=float)
    width = min(0.18, 0.72 / max(len(model_keys), 1))
    offsets = (np.arange(len(model_keys), dtype=float) - (len(model_keys) - 1) / 2.0) * width

    for model_index, model in enumerate(model_keys):
        values = np.asarray(values_by_model[model], dtype=float)
        yerr = None
        if show_error_bars and average_kind == "macro":
            std_values = np.asarray(std_by_model[model], dtype=float)
            if np.isfinite(std_values).any():
                yerr = np.where(np.isfinite(std_values), std_values, 0.0)
        bars = ax.bar(
            x + offsets[model_index],
            values,
            width=width,
            label=MODEL_DISPLAY.get(model, model),
            color=MODEL_COLORS.get(model, "#555555"),
            edgecolor="white",
            linewidth=0.8,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            alpha=0.92,
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.08),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                rotation=90,
                color="#222222",
            )

    title_method = threshold_method.replace("_", " ")
    title_label = label_col.replace("_", " ")
    title_average = "Macro mean" if average_kind == "macro" else "Micro"
    ax.set_title(
        f"{title_average} metrics | dataset={dataset} | threshold={title_method} | label={title_label}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY[metric] for metric in available_metrics], fontsize=10)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Metric value", fontsize=10)
    ax.grid(axis="y", color="#D8DEE9", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=min(len(model_keys), 4),
        frameon=False,
    )
    ax.text(
        0.0,
        -0.19,
        "Lower is better for FPR; higher is better for other metrics.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4A5568",
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except PermissionError as exc:
        print(f"[WARN] could not write metric plot {relative_to_bundle(output_path)}: {type(exc).__name__}: {exc}")
        plt.close(fig)
        return None
    plt.close(fig)

    return PlotSpec(
        average_kind=average_kind,
        dataset=dataset,
        label_col=label_col,
        threshold_method=threshold_method,
        output_path=output_path,
        rows=int(len(group)),
    )


def sanitize_filename(text: str) -> str:
    safe_chars: list[str] = []
    for char in str(text):
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._")
    return safe or "unknown"


def run_root_for(model_key: str, dataset: str) -> Path:
    run_prefix = MODEL_RUN_PREFIX.get(model_key, model_key.lower())
    return BUNDLE_ROOT / model_key / "runs" / f"{run_prefix}_{dataset}"


def candidate_score_threshold_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    candidates: list[Path] = []
    direct = run_root / "score_threshold_analysis"
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(path for path in run_root.rglob("score_threshold_analysis") if path.is_dir())
    unique: dict[str, Path] = {str(path.resolve()): path for path in candidates}

    def rank(path: Path) -> tuple[int, int, int, float]:
        summary = int((path / "summary_metrics.csv").exists())
        per_flight = int(
            (path / "per_flight_total_score_threshold_methods.csv").exists()
            or (path / "per_flight_total_score_single_global_threshold.csv").exists()
        )
        threshold = int(
            (path / "threshold_static.json").exists()
            or (path / "single_global_threshold_total_score.json").exists()
            or (path / "threshold.json").exists()
        )
        sequence = int((path / "sequence_scores_with_labels.csv").exists())
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (summary + per_flight + threshold + sequence, summary + per_flight, sequence + threshold, mtime)

    return sorted(unique.values(), key=rank, reverse=True)


def best_analysis_dir(run_root: Path) -> Path | None:
    candidates = candidate_score_threshold_dirs(run_root)
    return candidates[0] if candidates else None


def analysis_dir_for(model_key: str, dataset: str) -> Path | None:
    return best_analysis_dir(run_root_for(model_key, dataset))


def sequence_scores_path_for(model_key: str, dataset: str) -> Path | None:
    analysis_dir = analysis_dir_for(model_key, dataset)
    if analysis_dir is None:
        return None
    return analysis_dir / "sequence_scores_with_labels.csv"


def choose_x_col(df: pd.DataFrame) -> str | None:
    for candidate in ("t_mid", "t_end", "current_t", "t", "current_index", "sample_index"):
        if candidate in df.columns:
            return candidate
    return None


def choose_score_col(df: pd.DataFrame) -> str | None:
    for candidate in ("scores_smooth", "raw_total_score", "total_score"):
        if candidate in df.columns:
            return candidate
    return None


def choose_threshold_col(df: pd.DataFrame, kind: str) -> str | None:
    if kind == "static":
        candidates = ("threshold_static", "threshold_global", "single_global_threshold")
    elif kind == "static_val_sigma3":
        candidates = ("threshold_static_val_sigma3",)
    elif kind == "dynamic":
        candidates = ("threshold_dynamic", "threshold_adapt", "threshold_adaptive")
    elif kind == "spot":
        candidates = ("threshold_spot",)
    else:
        candidates = ()
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def contiguous_segments(x_values: np.ndarray, mask_values: np.ndarray) -> list[tuple[float, float]]:
    if len(x_values) == 0 or len(mask_values) == 0:
        return []
    finite = np.isfinite(x_values)
    if not finite.any():
        return []

    x = x_values[finite]
    mask = mask_values[finite].astype(bool)
    if len(x) == 0:
        return []

    if len(x) >= 2:
        step_candidates = np.diff(x)
        step_candidates = step_candidates[np.isfinite(step_candidates) & (step_candidates > 0)]
        half_step = float(np.median(step_candidates) / 2.0) if len(step_candidates) else 0.5
    else:
        half_step = 0.5

    segments: list[tuple[float, float]] = []
    start: float | None = None
    prev_x: float | None = None
    for current_x, is_active in zip(x, mask):
        current_x = float(current_x)
        if is_active and start is None:
            start = current_x - half_step
        if not is_active and start is not None and prev_x is not None:
            segments.append((start, float(prev_x) + half_step))
            start = None
        prev_x = current_x
    if start is not None and prev_x is not None:
        segments.append((start, float(prev_x) + half_step))
    return segments


def finite_ylim(*arrays: np.ndarray) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            values.append(arr)
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    low = float(np.nanmin(merged))
    high = float(np.nanmax(merged))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if low == high:
        pad = max(abs(high) * 0.1, 1e-6)
        return low - pad, high + pad
    pad = (high - low) * 0.12
    return low - pad, high + pad


def draw_threshold_timeseries(
    flight_df: pd.DataFrame,
    model_key: str,
    dataset: str,
    flight: str,
    label_col: str,
    source_path: Path,
    output_path: Path,
    dpi: int,
    downsample: int,
) -> ThresholdPlotSpec | None:
    x_col = choose_x_col(flight_df)
    score_col = choose_score_col(flight_df)
    threshold_static_col = choose_threshold_col(flight_df, "static")
    threshold_static_val_col = choose_threshold_col(flight_df, "static_val_sigma3")
    threshold_dynamic_col = choose_threshold_col(flight_df, "dynamic")
    threshold_spot_col = choose_threshold_col(flight_df, "spot")
    if x_col is None or score_col is None or threshold_static_col is None or threshold_dynamic_col is None:
        return None
    if label_col not in flight_df.columns:
        return None

    sort_cols = [x_col]
    if "current_index" in flight_df.columns and "current_index" not in sort_cols:
        sort_cols.append("current_index")
    df = flight_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    stride = max(int(downsample), 1)
    plot_df = df.iloc[::stride].copy()

    x_all = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    x = pd.to_numeric(plot_df[x_col], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(plot_df[score_col], errors="coerce").to_numpy(dtype=float)
    threshold_static = pd.to_numeric(plot_df[threshold_static_col], errors="coerce").to_numpy(dtype=float)
    threshold_static_val = (
        pd.to_numeric(plot_df[threshold_static_val_col], errors="coerce").to_numpy(dtype=float)
        if threshold_static_val_col is not None
        else np.asarray([], dtype=float)
    )
    threshold_dynamic = pd.to_numeric(plot_df[threshold_dynamic_col], errors="coerce").to_numpy(dtype=float)
    threshold_spot = (
        pd.to_numeric(plot_df[threshold_spot_col], errors="coerce").to_numpy(dtype=float)
        if threshold_spot_col is not None
        else np.asarray([], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(14.5, 6.4), constrained_layout=True)

    for start, end in contiguous_segments(x_all, labels):
        ax.axvspan(start, end, color="#E63946", alpha=0.13, lw=0)

    ax.plot(x, score, color="#1D3557", linewidth=1.05, label=f"{score_col}")
    ax.plot(
        x,
        threshold_static,
        color="#F4A261",
        linewidth=1.35,
        linestyle="--",
        label=f"static threshold ({threshold_static_col})",
    )
    if threshold_static_val_col is not None:
        ax.plot(
            x,
            threshold_static_val,
            color="#8D99AE",
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            alpha=0.95,
            label=f"val 3sigma threshold ({threshold_static_val_col})",
        )
    ax.plot(
        x,
        threshold_dynamic,
        color="#2A9D8F",
        linewidth=1.25,
        linestyle="-.",
        alpha=0.92,
        label=f"dynamic threshold ({threshold_dynamic_col})",
    )
    if threshold_spot_col is not None:
        ax.plot(
            x,
            threshold_spot,
            color="#6A4C93",
            linewidth=1.15,
            linestyle=":",
            alpha=0.95,
            label=f"SPOT threshold ({threshold_spot_col})",
        )

    for pred_col, color, marker, label in (
        ("pred_static", "#F77F00", "^", "static pred"),
        ("pred_static_val_sigma3", "#6C757D", "v", "val 3sigma pred"),
        ("pred_dynamic", "#D62828", "x", "dynamic pred"),
        ("pred_spot", "#6A4C93", "o", "SPOT pred"),
    ):
        if pred_col not in df.columns:
            continue
        pred_mask = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        if pred_mask.any():
            pred_x = x_all[pred_mask]
            pred_y = pd.to_numeric(df.loc[pred_mask, score_col], errors="coerce").to_numpy(dtype=float)
            ax.scatter(
                pred_x,
                pred_y,
                s=16 if marker != "x" else 20,
                color=color,
                marker=marker,
                linewidths=0.9,
                alpha=0.82,
                label=label,
                zorder=4,
            )

    y_low, y_high = finite_ylim(score, threshold_static, threshold_static_val, threshold_dynamic, threshold_spot)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(x_col)
    ax.set_ylabel("score / threshold")
    ax.grid(axis="y", color="#D8DEE9", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = (
        f"{MODEL_DISPLAY.get(model_key, model_key)} | dataset={dataset} | flight={flight} | "
        f"label={label_col}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    ax.text(
        0.0,
        -0.15,
        "Red background = ground-truth anomaly window. Markers show points predicted anomalous by each threshold method.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4A5568",
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except PermissionError as exc:
        print(f"[WARN] could not write threshold plot {relative_to_bundle(output_path)}: {type(exc).__name__}: {exc}")
        plt.close(fig)
        return None
    plt.close(fig)

    return ThresholdPlotSpec(
        model_key=model_key,
        dataset=dataset,
        flight=flight,
        label_col=label_col,
        output_path=output_path,
        rows=int(len(df)),
        source_path=source_path,
    )


def refresh_statistics_if_needed(args) -> None:
    if bool(args.skip_auto_summarize):
        return

    stat_dir = resolve_path(args.stat_dir)
    models = ordered_subset(args.models, MODEL_ORDER)
    datasets = ordered_subset(args.datasets, DATASET_ORDER)
    label_cols = ordered_subset(args.label_cols, LABEL_COLS)
    argv: list[str] = [
        "--out-dir",
        str(stat_dir),
        "--models",
        *models,
        "--datasets",
        *datasets,
        "--label-cols",
        *label_cols,
    ]
    try:
        summarize_results.main(argv)
    except Exception as exc:
        print(
            "[WARN] auto summarize failed; using existing statistic CSV files if available: "
            f"{type(exc).__name__}: {exc}"
        )


def plot_threshold_timeseries(args) -> list[ThresholdPlotSpec]:
    if bool(args.no_threshold_plots):
        return []

    out_dir = resolve_path(args.out_dir) / "threshold_timeseries"
    models = ordered_subset(args.models, MODEL_ORDER)
    datasets = ordered_subset(args.datasets, DATASET_ORDER)
    label_cols = ordered_subset(args.label_cols, LABEL_COLS)
    max_flights = max(int(args.max_flights_per_run), 0)
    downsample = max(int(args.threshold_downsample), 1)

    created: list[ThresholdPlotSpec] = []
    for dataset in datasets:
        for model_key in models:
            sequence_path = sequence_scores_path_for(model_key, dataset)
            if sequence_path is None or not sequence_path.exists():
                continue
            try:
                df = pd.read_csv(sequence_path)
            except pd.errors.EmptyDataError:
                continue
            if df.empty or "flight" not in df.columns:
                continue

            flights = sorted(df["flight"].dropna().astype(str).unique().tolist())
            if max_flights > 0:
                flights = flights[:max_flights]

            for flight in flights:
                flight_df = df[df["flight"].astype(str) == flight].copy()
                for label_col in label_cols:
                    filename = (
                        f"{sanitize_filename(model_key)}__{sanitize_filename(dataset)}__"
                        f"{sanitize_filename(flight)}__{sanitize_filename(label_col)}.png"
                    )
                    spec = draw_threshold_timeseries(
                        flight_df=flight_df,
                        model_key=model_key,
                        dataset=dataset,
                        flight=flight,
                        label_col=label_col,
                        source_path=sequence_path,
                        output_path=out_dir / sanitize_filename(model_key) / sanitize_filename(dataset) / filename,
                        dpi=int(args.dpi),
                        downsample=downsample,
                    )
                    if spec is not None:
                        created.append(spec)
    return created


def plot_all(args) -> tuple[list[PlotSpec], list[ThresholdPlotSpec]]:
    refresh_statistics_if_needed(args)
    stat_dir = resolve_path(args.stat_dir)
    out_dir = resolve_path(args.out_dir)
    models = ordered_subset(args.models, MODEL_ORDER)
    averages = ordered_subset(args.averages, AVERAGE_KINDS)
    datasets = ordered_subset(args.datasets, DATASET_ORDER)
    label_cols = ordered_subset(args.label_cols, LABEL_COLS)
    threshold_methods = ordered_subset(args.threshold_methods, THRESHOLD_METHODS)
    metrics = ordered_subset(args.metrics, METRIC_ORDER)

    csv_by_average = {
        "macro": stat_dir / "all_model_dataset_metrics_macro.csv",
        "micro": stat_dir / "all_model_dataset_metrics_micro.csv",
    }
    data_by_average = {
        average: read_summary_csv(csv_by_average[average])
        for average in averages
    }

    created: list[PlotSpec] = []
    for average_kind in averages:
        df = data_by_average.get(average_kind, pd.DataFrame())
        for dataset in datasets:
            for threshold_method in threshold_methods:
                for label_col in label_cols:
                    group = prepare_group(
                        df=df,
                        average_kind=average_kind,
                        dataset=dataset,
                        label_col=label_col,
                        threshold_method=threshold_method,
                    )
                    if not group.empty and "model_key" in group.columns:
                        group = group[group["model_key"].astype(str).isin(models)].copy()
                    filename = (
                        f"{average_kind}__{dataset}__{threshold_method}__{label_col}.png"
                    )
                    spec = draw_grouped_metric_bars(
                        group=group,
                        average_kind=average_kind,
                        dataset=dataset,
                        label_col=label_col,
                        threshold_method=threshold_method,
                        metrics=metrics,
                        output_path=out_dir / filename,
                        dpi=int(args.dpi),
                        show_error_bars=not bool(args.no_error_bars),
                    )
                    if spec is not None:
                        created.append(spec)
    threshold_created = plot_threshold_timeseries(args)
    write_plot_index(created, threshold_created, out_dir, data_by_average, csv_by_average)
    return created, threshold_created


def write_plot_index(
    created: list[PlotSpec],
    threshold_created: list[ThresholdPlotSpec],
    out_dir: Path,
    data_by_average: dict[str, pd.DataFrame],
    csv_by_average: dict[str, Path],
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        print(f"[WARN] could not create plot directory {relative_to_bundle(out_dir)}: {type(exc).__name__}: {exc}")
        return
    lines: list[str] = [
        "# All Model Metric and Threshold Plots",
        "",
        "Metric PNGs fix one dataset, one threshold method, one label column, and one average type.",
        "Threshold PNGs fix one model, one dataset, one flight, and one label column.",
        "",
        "## Sources",
        "",
    ]
    for average, path in csv_by_average.items():
        rows = 0 if data_by_average.get(average, pd.DataFrame()).empty else len(data_by_average[average])
        lines.append(f"- `{average}`: `{relative_to_bundle(path)}` (`{rows}` rows)")
    lines.extend(["", "## Generated Metric PNGs", ""])

    if not created:
        lines.append(
            "No metric plots were generated. Run model inference/evaluation first, then run "
            "`summarize_all_model_results.py`, and finally rerun this plotting script."
        )
    else:
        for spec in created:
            rel_path = relative_to_bundle(spec.output_path)
            lines.append(
                f"- `{spec.average_kind}` | `{spec.dataset}` | `{spec.threshold_method}` | "
                f"`{spec.label_col}`: `{rel_path}`"
            )

    lines.extend(["", "## Generated Threshold PNGs", ""])
    if not threshold_created:
        lines.append(
            "No threshold plots were generated. They require each run's "
            "`score_threshold_analysis/sequence_scores_with_labels.csv`."
        )
    else:
        for spec in threshold_created:
            rel_path = relative_to_bundle(spec.output_path)
            rel_source = relative_to_bundle(spec.source_path)
            lines.append(
                f"- `{spec.model_key}` | `{spec.dataset}` | `{spec.flight}` | "
                f"`{spec.label_col}`: `{rel_path}` (source: `{rel_source}`)"
            )

    safe_write_text(out_dir / "plot_index.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    created, threshold_created = plot_all(args)
    out_dir = resolve_path(args.out_dir)
    stat_dir = resolve_path(args.stat_dir)
    print(f"[OK] plot_dir={relative_to_bundle(out_dir)}")
    print(f"[OK] metric_png_count={len(created)}")
    print(f"[OK] threshold_png_count={len(threshold_created)}")
    if len(created) == 0:
        print("[WARN] no metric plots generated; summary CSV files may be empty or missing requested groups")
    if len(threshold_created) == 0 and not bool(args.no_threshold_plots):
        print("[WARN] no threshold plots generated; sequence_scores_with_labels.csv files may not exist yet")
    for spec in created:
        print(
            "[METRIC PNG] "
            f"{spec.average_kind} {spec.dataset} {spec.threshold_method} {spec.label_col} "
            f"rows={spec.rows} path={relative_to_bundle(spec.output_path)}"
        )
    for spec in threshold_created:
        print(
            "[THRESHOLD PNG] "
            f"{spec.model_key} {spec.dataset} {spec.flight} {spec.label_col} "
            f"rows={spec.rows} path={relative_to_bundle(spec.output_path)}"
        )
    status_lines = status_summary_lines(stat_dir)
    if status_lines:
        print(f"[STATUS] {'; '.join(status_lines)}")
    print(f"[OK] index={relative_to_bundle(out_dir / 'plot_index.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
