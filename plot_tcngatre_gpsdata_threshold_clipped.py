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
class ClippedThresholdPlotSpec:
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
            "Draw TCNGATRE gpsdata threshold time-series plots clipped to [0, 3]. "
            "This keeps the original plot style but zooms into the 0-3 range."
        )
    )
    parser.add_argument(
        "--model",
        default="TCNGATRE",
        choices=list(base_plot.MODEL_ORDER),
        help="Model key. Defaults to TCNGATRE.",
    )
    parser.add_argument(
        "--dataset",
        default="gpsdata",
        choices=list(base_plot.DATASET_ORDER),
        help="Dataset name. Defaults to gpsdata.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=list(base_plot.LABEL_COLS),
        help="Label columns to render.",
    )
    parser.add_argument(
        "--source-csv",
        default="",
        help=(
            "Optional explicit path to sequence_scores_with_labels.csv. "
            "If omitted, the script auto-discovers the best analysis dir like plot_all_model_metrics.py."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="statistic/plots/threshold_timeseries_clipped_03",
        help="Output directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Plot every Nth point, same semantics as threshold-downsample.",
    )
    parser.add_argument(
        "--max-flights",
        type=int,
        default=0,
        help="Maximum number of flights to plot. 0 means all flights.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return BUNDLE_ROOT / path


def clip03(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    arr[finite] = np.clip(arr[finite], 0.0, 3.0)
    return arr


def draw_threshold_timeseries_clipped(
    flight_df: pd.DataFrame,
    model_key: str,
    dataset: str,
    flight: str,
    label_col: str,
    source_path: Path,
    output_path: Path,
    dpi: int,
    downsample: int,
) -> ClippedThresholdPlotSpec | None:
    x_col = base_plot.choose_x_col(flight_df)
    score_col = base_plot.choose_score_col(flight_df)
    threshold_static_col = base_plot.choose_threshold_col(flight_df, "static")
    threshold_static_val_col = base_plot.choose_threshold_col(flight_df, "static_val_sigma3")
    threshold_dynamic_col = base_plot.choose_threshold_col(flight_df, "dynamic")
    threshold_spot_col = base_plot.choose_threshold_col(flight_df, "spot")
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
    score = clip03(pd.to_numeric(plot_df[score_col], errors="coerce").to_numpy(dtype=float))
    threshold_static = clip03(pd.to_numeric(plot_df[threshold_static_col], errors="coerce").to_numpy(dtype=float))
    threshold_static_val = (
        clip03(pd.to_numeric(plot_df[threshold_static_val_col], errors="coerce").to_numpy(dtype=float))
        if threshold_static_val_col is not None
        else np.asarray([], dtype=float)
    )
    threshold_dynamic = clip03(pd.to_numeric(plot_df[threshold_dynamic_col], errors="coerce").to_numpy(dtype=float))
    threshold_spot = (
        clip03(pd.to_numeric(plot_df[threshold_spot_col], errors="coerce").to_numpy(dtype=float))
        if threshold_spot_col is not None
        else np.asarray([], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(14.5, 6.4), constrained_layout=True)

    for start, end in base_plot.contiguous_segments(x_all, labels):
        ax.axvspan(start, end, color="#E63946", alpha=0.13, lw=0)

    ax.plot(x, score, color="#1D3557", linewidth=1.05, label=f"{score_col} (clipped)")
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
            pred_y = clip03(pd.to_numeric(df.loc[pred_mask, score_col], errors="coerce").to_numpy(dtype=float))
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

    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(x_col)
    ax.set_ylabel("score / threshold (clipped to [0, 3])")
    ax.grid(axis="y", color="#D8DEE9", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = (
        f"{base_plot.MODEL_DISPLAY.get(model_key, model_key)} | dataset={dataset} | "
        f"flight={flight} | label={label_col} | clipped view [0,3]"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    ax.text(
        0.0,
        -0.15,
        "Red background = ground-truth anomaly window. Curves and markers are clipped to [0,3] for zoomed inspection.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4A5568",
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except PermissionError as exc:
        print(
            f"[WARN] could not write clipped threshold plot "
            f"{base_plot.relative_to_bundle(output_path)}: {type(exc).__name__}: {exc}"
        )
        plt.close(fig)
        return None
    plt.close(fig)

    return ClippedThresholdPlotSpec(
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
            f"model={model_key}, dataset={dataset}. "
            f"Please pass --source-csv explicitly."
        )
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_key = str(args.model)
    dataset = str(args.dataset)
    source_path = resolve_source_csv(model_key=model_key, dataset=dataset, source_csv=str(args.source_csv))
    out_root = resolve_path(str(args.out_dir)) / model_key / dataset
    df = pd.read_csv(source_path)
    if df.empty:
        raise ValueError(f"Source CSV is empty: {source_path}")
    if "flight" not in df.columns:
        raise ValueError(f"Missing 'flight' column in: {source_path}")

    flights = list(df["flight"].astype(str).drop_duplicates())
    if int(args.max_flights) > 0:
        flights = flights[: int(args.max_flights)]

    created = 0
    for label_col in args.label_cols:
        if label_col not in df.columns:
            print(f"[WARN] skip missing label column: {label_col}")
            continue
        for flight in flights:
            flight_df = df.loc[df["flight"].astype(str) == str(flight)].copy()
            if flight_df.empty:
                continue
            output_path = out_root / label_col / f"{base_plot.sanitize_filename(flight)}__clipped_03.png"
            spec = draw_threshold_timeseries_clipped(
                flight_df=flight_df,
                model_key=model_key,
                dataset=dataset,
                flight=str(flight),
                label_col=str(label_col),
                source_path=source_path,
                output_path=output_path,
                dpi=int(args.dpi),
                downsample=int(args.downsample),
            )
            if spec is not None:
                created += 1
                print(f"[OK] {base_plot.relative_to_bundle(spec.output_path)}")

    print(f"[DONE] clipped_threshold_png_count={created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
