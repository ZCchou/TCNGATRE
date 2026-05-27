from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_all_model_metrics as base_plot


BUNDLE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DetectionThresholdPlotSpec:
    model_key: str
    dataset: str
    flight: str
    label_col: str
    output_path: Path
    rows: int
    source_path: Path


THRESHOLD_SPECS = [
    {
        "name": "static_f1_oracle",
        "threshold_kind": "static",
        "pred_col": "pred_static",
        "threshold_color": "#F4A261",
        "pred_color": "#F77F00",
        "linestyle": "--",
        "marker": "^",
        "label": "static F1 oracle",
    },
    {
        "name": "static_val_sigma3",
        "threshold_kind": "static_val_sigma3",
        "pred_col": "pred_static_val_sigma3",
        "threshold_color": "#8D99AE",
        "pred_color": "#6C757D",
        "linestyle": (0, (4, 2)),
        "marker": "v",
        "label": "val 3sigma",
    },
    {
        "name": "dynamic_history",
        "threshold_kind": "dynamic",
        "pred_col": "pred_dynamic",
        "threshold_color": "#2A9D8F",
        "pred_color": "#D62828",
        "linestyle": "-.",
        "marker": "x",
        "label": "dynamic history",
    },
    {
        "name": "spot",
        "threshold_kind": "spot",
        "pred_col": "pred_spot",
        "threshold_color": "#6A4C93",
        "pred_color": "#6A4C93",
        "linestyle": ":",
        "marker": "o",
        "label": "SPOT",
    },
]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Draw standalone anomaly-detection time-series plots for different threshold methods. "
            "The legend is placed on the left and axis/legend fonts are enlarged for publication-style inspection."
        )
    )
    parser.add_argument(
        "--model",
        default="TCNGATRE",
        choices=list(base_plot.MODEL_ORDER),
        help="Model key used for auto-discovery. Defaults to TCNGATRE.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        choices=list(base_plot.DATASET_ORDER),
        help="Single dataset name for backward compatibility. If omitted, --datasets is used.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(base_plot.DATASET_ORDER),
        choices=list(base_plot.DATASET_ORDER),
        help="Datasets to render. Defaults to all datasets.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=list(base_plot.LABEL_COLS),
        help="Label columns to render.",
    )
    parser.add_argument(
        "--threshold-methods",
        nargs="+",
        default=[spec["name"] for spec in THRESHOLD_SPECS],
        choices=[spec["name"] for spec in THRESHOLD_SPECS],
        help="Threshold methods to overlay.",
    )
    parser.add_argument(
        "--source-csv",
        default="",
        help=(
            "Optional explicit path to sequence_scores_with_labels.csv. "
            "If omitted, the best analysis dir is auto-discovered from the selected model/dataset."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="statistic/plots/threshold_detection_results",
        help="Output directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument("--downsample", type=int, default=1, help="Plot every Nth point.")
    parser.add_argument("--max-flights", type=int, default=0, help="0 means all flights.")
    parser.add_argument("--fig-width", type=float, default=15.5, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=7.0, help="Figure height in inches.")
    parser.add_argument("--title-fontsize", type=float, default=20.0, help="Title font size.")
    parser.add_argument("--axis-fontsize", type=float, default=18.0, help="X/Y axis label font size.")
    parser.add_argument("--tick-fontsize", type=float, default=15.0, help="Tick label font size.")
    parser.add_argument("--legend-fontsize", type=float, default=15.0, help="Legend font size.")
    parser.add_argument("--note-fontsize", type=float, default=13.0, help="Bottom note font size.")
    parser.add_argument(
        "--legend-loc",
        default="upper left",
        help="Matplotlib legend location. Defaults to upper left.",
    )
    parser.add_argument(
        "--legend-outside-left",
        action="store_true",
        help="Place legend outside the left side of the axes instead of inside upper-left.",
    )
    parser.add_argument("--y-min", type=float, default=None, help="Optional fixed lower y limit.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional fixed upper y limit.")
    parser.add_argument(
        "--clip-y",
        action="store_true",
        help="Force clipping to [--y-min, --y-max]. If omitted, gpsdata is clipped to [0,3] automatically.",
    )
    parser.add_argument(
        "--no-gps-clip",
        action="store_true",
        help="Disable the reference-style automatic [0,3] clipping for gpsdata.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return BUNDLE_ROOT / path


def selected_threshold_specs(methods: list[str]) -> list[dict]:
    selected = set(str(method) for method in methods)
    return [spec for spec in THRESHOLD_SPECS if str(spec["name"]) in selected]


def maybe_clip(values: np.ndarray, y_min: float | None, y_max: float | None, enabled: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if not bool(enabled):
        return arr
    finite = np.isfinite(arr)
    low = -np.inf if y_min is None else float(y_min)
    high = np.inf if y_max is None else float(y_max)
    arr[finite] = np.clip(arr[finite], low, high)
    return arr


def fixed_or_auto_ylim(
    arrays: list[np.ndarray],
    y_min: float | None,
    y_max: float | None,
) -> tuple[float, float]:
    if y_min is not None and y_max is not None:
        return float(y_min), float(y_max)
    auto_low, auto_high = base_plot.finite_ylim(*arrays)
    low = auto_low if y_min is None else float(y_min)
    high = auto_high if y_max is None else float(y_max)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return 0.0, 1.0
    return low, high


def choose_time_x_col(df: pd.DataFrame) -> str | None:
    for candidate in ("t", "current_t", "t_end", "current_index", "sample_index", "t_mid"):
        if candidate in df.columns:
            return candidate
    return None


def dataset_clip_settings(
    dataset: str,
    y_min: float | None,
    y_max: float | None,
    force_clip: bool,
    no_gps_clip: bool,
) -> tuple[float | None, float | None, bool]:
    """Match the clipped reference script only for gpsdata by default."""
    if bool(force_clip):
        return y_min, y_max, True
    if str(dataset) == "gpsdata" and not bool(no_gps_clip):
        low = 0.0 if y_min is None else float(y_min)
        high = 3.0 if y_max is None else float(y_max)
        return low, high, True
    return y_min, y_max, False


def draw_detection_threshold_plot(
    flight_df: pd.DataFrame,
    model_key: str,
    dataset: str,
    flight: str,
    label_col: str,
    source_path: Path,
    output_path: Path,
    threshold_specs: list[dict],
    dpi: int,
    downsample: int,
    fig_width: float,
    fig_height: float,
    title_fontsize: float,
    axis_fontsize: float,
    tick_fontsize: float,
    legend_fontsize: float,
    note_fontsize: float,
    legend_loc: str,
    legend_outside_left: bool,
    y_min: float | None,
    y_max: float | None,
    clip_y: bool,
) -> DetectionThresholdPlotSpec | None:
    x_col = choose_time_x_col(flight_df)
    score_col = base_plot.choose_score_col(flight_df)
    if x_col is None or score_col is None or label_col not in flight_df.columns:
        return None

    available_specs: list[dict] = []
    for spec in threshold_specs:
        threshold_col = base_plot.choose_threshold_col(flight_df, str(spec["threshold_kind"]))
        if threshold_col is None:
            continue
        available_specs.append({**spec, "threshold_col": threshold_col})
    if not available_specs:
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
    score = maybe_clip(
        pd.to_numeric(plot_df[score_col], errors="coerce").to_numpy(dtype=float),
        y_min=y_min,
        y_max=y_max,
        enabled=clip_y,
    )

    y_arrays = [score]
    fig, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)), constrained_layout=True)

    for start, end in base_plot.contiguous_segments(x_all, labels):
        ax.axvspan(start, end, color="#E63946", alpha=0.13, lw=0)

    ax.plot(x, score, color="#1D3557", linewidth=1.6, label=score_col)

    for spec in available_specs:
        threshold_col = str(spec["threshold_col"])
        threshold = maybe_clip(
            pd.to_numeric(plot_df[threshold_col], errors="coerce").to_numpy(dtype=float),
            y_min=y_min,
            y_max=y_max,
            enabled=clip_y,
        )
        y_arrays.append(threshold)
        ax.plot(
            x,
            threshold,
            color=str(spec["threshold_color"]),
            linewidth=1.8,
            linestyle=spec["linestyle"],
            alpha=0.96,
            label=f"{spec['label']} threshold ({threshold_col})",
        )

    for spec in available_specs:
        pred_col = str(spec["pred_col"])
        if pred_col not in df.columns:
            continue
        pred_mask = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        if not bool(pred_mask.any()):
            continue
        pred_y = maybe_clip(
            pd.to_numeric(df.loc[pred_mask, score_col], errors="coerce").to_numpy(dtype=float),
            y_min=y_min,
            y_max=y_max,
            enabled=clip_y,
        )
        y_arrays.append(pred_y)
        ax.scatter(
            x_all[pred_mask],
            pred_y,
            s=42 if spec["marker"] != "x" else 50,
            color=str(spec["pred_color"]),
            marker=spec["marker"],
            linewidths=1.3,
            alpha=0.86,
            label=f"{spec['label']} anomaly",
            zorder=4,
        )

    y_low, y_high = fixed_or_auto_ylim(y_arrays, y_min=y_min, y_max=y_max)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("t", fontsize=axis_fontsize)
    ax.set_ylabel("score / threshold", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", color="#D8DEE9", linestyle="--", linewidth=0.9, alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = (
        f"{base_plot.MODEL_DISPLAY.get(model_key, model_key)} | dataset={dataset} | "
        f"flight={flight} | label={label_col}"
    )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=14)
    if legend_outside_left:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(-0.02, 1.0),
            frameon=True,
            framealpha=0.94,
            fontsize=legend_fontsize,
            borderaxespad=0.0,
        )
    else:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.94, fontsize=legend_fontsize)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    except PermissionError as exc:
        print(
            f"[WARN] could not write threshold detection plot "
            f"{base_plot.relative_to_bundle(output_path)}: {type(exc).__name__}: {exc}"
        )
        plt.close(fig)
        return None
    plt.close(fig)

    return DetectionThresholdPlotSpec(
        model_key=model_key,
        dataset=dataset,
        flight=flight,
        label_col=label_col,
        output_path=output_path,
        rows=int(len(df)),
        source_path=source_path,
    )


def resolve_source_csv(model_key: str, dataset: str, source_csv: str) -> Path:
    if str(source_csv).strip():
        path = resolve_path(source_csv)
        if not path.exists():
            raise FileNotFoundError(f"Explicit source csv not found: {path}")
        return path
    path = base_plot.sequence_scores_path_for(model_key, dataset)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Could not auto-discover sequence_scores_with_labels.csv for "
            f"model={model_key}, dataset={dataset}. Please pass --source-csv explicitly."
        )
    return path


def resolve_datasets(dataset: str, datasets: list[str]) -> list[str]:
    if str(dataset).strip():
        return [str(dataset)]
    ordered = []
    seen = set()
    for item in datasets:
        name = str(item)
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_key = str(args.model)
    datasets = resolve_datasets(dataset=str(args.dataset), datasets=list(args.datasets))
    if str(args.source_csv).strip() and len(datasets) != 1:
        raise ValueError("--source-csv can only be used with one dataset. Pass --dataset when using an explicit CSV.")
    threshold_specs = selected_threshold_specs(list(args.threshold_methods))

    created = 0
    skipped = 0
    for dataset in datasets:
        try:
            source_path = resolve_source_csv(model_key=model_key, dataset=dataset, source_csv=str(args.source_csv))
        except FileNotFoundError as exc:
            skipped += 1
            print(f"[WARN] skip dataset={dataset}: {exc}")
            continue
        out_root = resolve_path(str(args.out_dir)) / model_key / dataset
        df = pd.read_csv(source_path)
        if df.empty:
            skipped += 1
            print(f"[WARN] skip empty source CSV for dataset={dataset}: {source_path}")
            continue
        if "flight" not in df.columns:
            raise ValueError(f"Missing 'flight' column in: {source_path}")

        y_min, y_max, clip_y = dataset_clip_settings(
            dataset=dataset,
            y_min=args.y_min,
            y_max=args.y_max,
            force_clip=bool(args.clip_y),
            no_gps_clip=bool(args.no_gps_clip),
        )
        clip_suffix = "__clipped_03" if clip_y and y_min == 0.0 and y_max == 3.0 else ""

        flights = list(df["flight"].astype(str).drop_duplicates())
        if int(args.max_flights) > 0:
            flights = flights[: int(args.max_flights)]

        for label_col in args.label_cols:
            if label_col not in df.columns:
                print(f"[WARN] skip missing label column for dataset={dataset}: {label_col}")
                continue
            for flight in flights:
                flight_df = df.loc[df["flight"].astype(str) == str(flight)].copy()
                if flight_df.empty:
                    continue
                suffix = f"__threshold_detection{clip_suffix}.png"
                output_path = out_root / str(label_col) / f"{base_plot.sanitize_filename(flight)}{suffix}"
                spec = draw_detection_threshold_plot(
                    flight_df=flight_df,
                    model_key=model_key,
                    dataset=dataset,
                    flight=str(flight),
                    label_col=str(label_col),
                    source_path=source_path,
                    output_path=output_path,
                    threshold_specs=threshold_specs,
                    dpi=int(args.dpi),
                    downsample=int(args.downsample),
                    fig_width=float(args.fig_width),
                    fig_height=float(args.fig_height),
                    title_fontsize=float(args.title_fontsize),
                    axis_fontsize=float(args.axis_fontsize),
                    tick_fontsize=float(args.tick_fontsize),
                    legend_fontsize=float(args.legend_fontsize),
                    note_fontsize=float(args.note_fontsize),
                    legend_loc=str(args.legend_loc),
                    legend_outside_left=bool(args.legend_outside_left),
                    y_min=y_min,
                    y_max=y_max,
                    clip_y=clip_y,
                )
                if spec is not None:
                    created += 1
                    print(f"[OK] {base_plot.relative_to_bundle(spec.output_path)}")

    print(f"[DONE] threshold_detection_png_count={created} skipped_datasets={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
