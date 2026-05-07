from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


MANIFEST_NAME = "dataset_manifest.json"
PAGE_SIZE = 8
FIGURE_WIDTH = 18.0
FIGURE_HEIGHT_PER_AXIS = 2.4
SAVE_DPI = 140


def load_manifest(dataset_root: Path) -> dict:
    manifest_path = Path(dataset_root) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def discover_wide_root(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)
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
        raise ValueError(f"Expected exactly one wide root under {dataset_root}, found {len(candidates)}")
    return candidates[0]


def discover_labels_root(dataset_root: Path, manifest: dict) -> Path:
    labels_root = Path(dataset_root) / str(manifest.get("labels_dirname", "wide_flights_failure_labels"))
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels root: {labels_root}")
    return labels_root


def load_labels(labels_root: Path, flight: str, time_offset_sec: float) -> pd.DataFrame | None:
    path = Path(labels_root) / f"{flight}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "t" not in df.columns or "anomaly_label" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["t"], errors="coerce"),
            "anomaly_label": pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["t"])
    if len(out) <= 0:
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


def load_flight_frame(csv_path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    df = pd.read_csv(Path(csv_path), low_memory=False)
    if "t" not in df.columns:
        raise ValueError(f"Missing 't' column in {csv_path}")
    df = df.copy()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t", kind="mergesort").reset_index(drop=True)
    feature_names = [str(col) for col in df.columns if str(col) != "t"]
    return df["t"].to_numpy(dtype=float), df, feature_names


def plot_feature_page(
    flight: str,
    split_name: str,
    t: np.ndarray,
    df: pd.DataFrame,
    feature_names: list[str],
    spans: list[tuple[float, float]],
    page_index: int,
    total_pages: int,
    out_path: Path,
):
    n_axes = len(feature_names)
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(FIGURE_WIDTH, max(3.0, FIGURE_HEIGHT_PER_AXIS * n_axes + 1.2)),
        sharex=True,
    )
    if n_axes == 1:
        axes = [axes]
    for ax, feature_name in zip(axes, feature_names):
        values = pd.to_numeric(df[feature_name], errors="coerce").to_numpy(dtype=float)
        for left, right in spans:
            ax.axvspan(left, right, color="tab:red", alpha=0.12, linewidth=0.0)
        ax.plot(t, values, color="#1f4e79", linewidth=0.8)
        ax.set_title(feature_name, fontsize=9, loc="left", pad=2.0)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("t (sec)")
    fig.suptitle(f"{split_name} | {flight} | page {page_index + 1}/{total_pages}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def main():
    dataset_root = Path(__file__).resolve().parent
    manifest = load_manifest(dataset_root)
    wide_root = discover_wide_root(dataset_root)
    labels_root = discover_labels_root(dataset_root, manifest=manifest)
    time_offset_sec = float(manifest.get("failure_label_time_offset_sec", 0.0))

    out_root = dataset_root / "dataviz"
    out_root.mkdir(parents=True, exist_ok=True)

    render_config = {
        "dataset_root": str(dataset_root),
        "wide_root": str(wide_root),
        "labels_root": str(labels_root),
        "failure_label_time_offset_sec": time_offset_sec,
        "page_size": PAGE_SIZE,
        "save_dpi": SAVE_DPI,
    }
    (out_root / "render_config.json").write_text(
        json.dumps(render_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_csvs: list[tuple[str, Path]] = []
    for split_name in ["No_Failure", "Failure"]:
        split_root = wide_root / split_name
        all_csvs.extend((split_name, path) for path in sorted(split_root.glob("*.csv")))

    index_rows: list[dict] = []
    progress = tqdm(all_csvs, desc="render dataset dataviz", unit="flight")
    for split_name, csv_path in progress:
        flight = csv_path.stem
        progress.set_postfix_str(f"{split_name}/{flight}")
        t, df, feature_names = load_flight_frame(csv_path)
        if len(feature_names) <= 0 or len(t) <= 0:
            continue
        labels_df = load_labels(
            labels_root=labels_root,
            flight=flight,
            time_offset_sec=time_offset_sec if split_name == "Failure" else 0.0,
        )
        spans = compute_anomaly_spans(labels_df=labels_df, t_min=float(np.min(t)), t_max=float(np.max(t)))
        total_pages = int(math.ceil(len(feature_names) / PAGE_SIZE))

        flight_dir = out_root / split_name / flight
        flight_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(total_pages):
            start = int(page_index * PAGE_SIZE)
            end = int(min(len(feature_names), start + PAGE_SIZE))
            page_features = feature_names[start:end]
            out_path = flight_dir / f"page_{page_index + 1:02d}.png"
            plot_feature_page(
                flight=flight,
                split_name=split_name,
                t=t,
                df=df,
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
                "source_csv": str(csv_path),
                "num_rows": int(len(df)),
                "num_features": int(len(feature_names)),
                "num_pages": int(total_pages),
                "has_labels": bool(labels_df is not None),
                "num_anomaly_spans": int(len(spans)),
                "output_dir": str(flight_dir),
            }
        )

    index_df = pd.DataFrame(index_rows).sort_values(["split", "flight"], kind="mergesort").reset_index(drop=True)
    index_df.to_csv(out_root / "index.csv", index=False, encoding="utf-8")

    summary = {
        "num_flights": int(len(index_df)),
        "num_no_failure": int((index_df["split"] == "No_Failure").sum()) if len(index_df) > 0 else 0,
        "num_failure": int((index_df["split"] == "Failure").sum()) if len(index_df) > 0 else 0,
        "total_pages": int(index_df["num_pages"].sum()) if len(index_df) > 0 else 0,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
