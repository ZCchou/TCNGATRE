from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HPARAM_ROOT = Path(__file__).resolve().parent
MODEL_ORDER = [
    "H1_Default",
    "H2_Small",
    "H3_Large",
    "H4_LongWindow",
    "H5_DenseGraph",
    "H6_Conservative",
]
DATASET_ORDER = ["alfa", "simulate", "gpsdata"]
LABEL_COLS = ["label_any", "label_mid"]
THRESHOLD_METHODS = ["static_f1_oracle", "static_val_sigma3", "dynamic_history", "spot"]
METRIC_COLS = [
    "auroc",
    "average_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
]
MICRO_COLUMNS = [
    "model",
    "model_key",
    "dataset",
    "label_col",
    "threshold_method",
    "score_col",
    "pred_col",
    "num_samples",
    "positives",
    "negatives",
    "auroc",
    "average_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "tp",
    "fp",
    "tn",
    "fn",
    "threshold",
    "analysis_dir",
    "source_path",
]
MACRO_COLUMNS = [
    "model",
    "model_key",
    "dataset",
    "label_col",
    "threshold_method",
    "threshold",
    "threshold_mean",
    "num_flights",
    *[name for metric in METRIC_COLS for name in (f"{metric}_mean", f"{metric}_std")],
    "analysis_dir",
    "source_path",
]
STATUS_COLUMNS = [
    "model",
    "model_key",
    "dataset",
    "status",
    "run_root",
    "analysis_dir",
    "summary_path",
    "per_flight_path",
    "threshold_path",
    "threshold_fit_mode",
    "threshold",
    "micro_rows",
    "macro_rows",
]

MODEL_DISPLAY = {
    "H1_Default": "H1 Default",
    "H2_Small": "H2 Small",
    "H3_Large": "H3 Large",
    "H4_LongWindow": "H4 Long Window",
    "H5_DenseGraph": "H5 Dense Graph",
    "H6_Conservative": "H6 Conservative",
}
RUN_PREFIX = {
    "H1_Default": "tcngatre_h1_default",
    "H2_Small": "tcngatre_h2_small",
    "H3_Large": "tcngatre_h3_large",
    "H4_LongWindow": "tcngatre_h4_longwindow",
    "H5_DenseGraph": "tcngatre_h5_densegraph",
    "H6_Conservative": "tcngatre_h6_conservative",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_dir: str
    run_prefix: str


MODEL_SPECS: dict[str, ModelSpec] = {
    key: ModelSpec(
        key=key,
        display_name=MODEL_DISPLAY[key],
        model_dir=key,
        run_prefix=RUN_PREFIX[key],
    )
    for key in MODEL_ORDER
}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Summarize TCNGATRE hyperparameter metrics across all variants and datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Models to include.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Datasets to include.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        choices=LABEL_COLS,
        default=LABEL_COLS,
        help="Label columns to include.",
    )
    parser.add_argument(
        "--out-dir",
        default="statistic",
        help="Output directory. Relative paths resolve from hparam root.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    if not path.is_absolute():
        path = HPARAM_ROOT / path
    return path


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(HPARAM_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def metric_value(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def format_float(value: Any) -> str:
    out = metric_value(value)
    if not np.isfinite(out):
        return "N/A"
    return f"{out:.4f}"


def format_macro(mean_value: Any, std_value: Any) -> str:
    mean = metric_value(mean_value)
    std = metric_value(std_value)
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def ordered_subset(values: list[str], canonical: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in canonical:
        if item in values and item not in seen:
            ordered.append(item)
            seen.add(item)
    for item in values:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def run_root_for(spec: ModelSpec, dataset: str) -> Path:
    return HPARAM_ROOT / spec.model_dir / "runs" / f"{spec.run_prefix}_{dataset}"


def candidate_score_threshold_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    candidates = []
    direct = run_root / "score_threshold_analysis"
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(p for p in run_root.rglob("score_threshold_analysis") if p.is_dir())
    unique: dict[str, Path] = {str(p.resolve()): p for p in candidates}

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
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (summary + per_flight + threshold, summary + per_flight, threshold, mtime)

    return sorted(unique.values(), key=rank, reverse=True)


def best_analysis_dir(run_root: Path) -> Path | None:
    candidates = candidate_score_threshold_dirs(run_root)
    return candidates[0] if candidates else None


def read_threshold(analysis_dir: Path | None, run_root: Path) -> tuple[float, str, str]:
    candidates: list[Path] = []
    if analysis_dir is not None:
        candidates.extend(
            [
                analysis_dir / "threshold_static.json",
                analysis_dir / "single_global_threshold_total_score.json",
                analysis_dir / "threshold.json",
            ]
        )
    candidates.append(run_root / "global_threshold.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ["single_global_threshold", "threshold", "best_threshold"]:
            if key in payload:
                return metric_value(payload.get(key)), relative_path(path), str(payload.get("fit_mode", ""))
    return float("nan"), "", ""


def read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def extract_micro_rows(
    summary_df: pd.DataFrame | None,
    spec: ModelSpec,
    dataset: str,
    label_cols: list[str],
    threshold: float,
    source_path: Path | None,
    analysis_dir: Path | None,
) -> list[dict[str, Any]]:
    if summary_df is None or "label_col" not in summary_df.columns:
        return []
    work = summary_df.copy()
    if "threshold_method" not in work.columns:
        work["threshold_method"] = "legacy"
    rows: list[dict[str, Any]] = []
    for threshold_method in list(dict.fromkeys(work["threshold_method"].astype(str).tolist())):
        method_df = work.loc[work["threshold_method"].astype(str) == threshold_method]
        for label_col in label_cols:
            matches = method_df.loc[method_df["label_col"].astype(str) == label_col]
            if matches.empty:
                continue
            item = matches.iloc[0].to_dict()
            row = {
                "model": spec.display_name,
                "model_key": spec.key,
                "dataset": dataset,
                "label_col": label_col,
                "threshold_method": threshold_method,
                "threshold": metric_value(item.get("threshold", threshold)),
                "analysis_dir": relative_path(analysis_dir),
                "source_path": relative_path(source_path),
            }
            for col in ["score_col", "pred_col", "num_samples", "positives", "negatives", "tp", "fp", "tn", "fn"]:
                row[col] = item.get(col, "")
            for col in METRIC_COLS:
                row[col] = metric_value(item.get(col))
            rows.append(row)
    return rows


def extract_macro_rows(
    per_flight_df: pd.DataFrame | None,
    spec: ModelSpec,
    dataset: str,
    label_cols: list[str],
    threshold: float,
    source_path: Path | None,
    analysis_dir: Path | None,
) -> list[dict[str, Any]]:
    if per_flight_df is None or "label_col" not in per_flight_df.columns:
        return []
    work = per_flight_df.copy()
    if "threshold_method" not in work.columns:
        work["threshold_method"] = "legacy"
    rows: list[dict[str, Any]] = []
    for threshold_method in list(dict.fromkeys(work["threshold_method"].astype(str).tolist())):
        method_df = work.loc[work["threshold_method"].astype(str) == threshold_method]
        for label_col in label_cols:
            g = method_df.loc[method_df["label_col"].astype(str) == label_col].copy()
            if g.empty:
                continue
            threshold_values = (
                pd.to_numeric(g["threshold"], errors="coerce").dropna().to_numpy(dtype=float)
                if "threshold" in g.columns
                else np.asarray([], dtype=float)
            )
            threshold_mean_values = (
                pd.to_numeric(g["threshold_mean"], errors="coerce").dropna().to_numpy(dtype=float)
                if "threshold_mean" in g.columns
                else np.asarray([], dtype=float)
            )
            row: dict[str, Any] = {
                "model": spec.display_name,
                "model_key": spec.key,
                "dataset": dataset,
                "label_col": label_col,
                "threshold_method": threshold_method,
                "threshold": float(np.mean(threshold_values)) if len(threshold_values) else threshold,
                "threshold_mean": float(np.mean(threshold_mean_values)) if len(threshold_mean_values) else float("nan"),
                "num_flights": int(g["flight"].nunique()) if "flight" in g.columns else int(len(g)),
                "analysis_dir": relative_path(analysis_dir),
                "source_path": relative_path(source_path),
            }
            for metric in METRIC_COLS:
                if metric not in g.columns:
                    row[f"{metric}_mean"] = float("nan")
                    row[f"{metric}_std"] = float("nan")
                    continue
                values = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            rows.append(row)
    return rows


def summarize_one(spec: ModelSpec, dataset: str, label_cols: list[str]) -> tuple[list[dict], list[dict], dict]:
    run_root = run_root_for(spec, dataset)
    analysis_dir = best_analysis_dir(run_root)
    threshold, threshold_path, threshold_fit_mode = read_threshold(analysis_dir, run_root)
    summary_path = None if analysis_dir is None else analysis_dir / "summary_metrics.csv"
    if analysis_dir is None:
        per_flight_path = None
    else:
        method_path = analysis_dir / "per_flight_total_score_threshold_methods.csv"
        legacy_path = analysis_dir / "per_flight_total_score_single_global_threshold.csv"
        per_flight_path = method_path if method_path.exists() else legacy_path

    summary_df = read_csv(summary_path)
    per_flight_df = read_csv(per_flight_path)
    micro_rows = extract_micro_rows(
        summary_df=summary_df,
        spec=spec,
        dataset=dataset,
        label_cols=label_cols,
        threshold=threshold,
        source_path=summary_path,
        analysis_dir=analysis_dir,
    )
    macro_rows = extract_macro_rows(
        per_flight_df=per_flight_df,
        spec=spec,
        dataset=dataset,
        label_cols=label_cols,
        threshold=threshold,
        source_path=per_flight_path,
        analysis_dir=analysis_dir,
    )

    if not run_root.exists():
        status = "missing_run_root"
    elif analysis_dir is None:
        status = "missing_analysis"
    elif summary_df is None and per_flight_df is None:
        status = "missing_metric_files"
    elif summary_df is None:
        status = "missing_summary"
    elif per_flight_df is None:
        status = "missing_per_flight"
    else:
        status = "ok"

    status_row = {
        "model": spec.display_name,
        "model_key": spec.key,
        "dataset": dataset,
        "status": status,
        "run_root": relative_path(run_root),
        "analysis_dir": relative_path(analysis_dir),
        "summary_path": relative_path(summary_path) if summary_path and summary_path.exists() else "",
        "per_flight_path": relative_path(per_flight_path) if per_flight_path and per_flight_path.exists() else "",
        "threshold_path": threshold_path,
        "threshold_fit_mode": threshold_fit_mode,
        "threshold": threshold,
        "micro_rows": int(len(micro_rows)),
        "macro_rows": int(len(macro_rows)),
    }
    return macro_rows, micro_rows, status_row


def row_lookup(
    rows: list[dict[str, Any]],
    label_col: str,
    model_key: str,
    dataset: str,
    threshold_method: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("model_key") == model_key
            and row.get("dataset") == dataset
            and row.get("label_col") == label_col
            and row.get("threshold_method") == threshold_method
        ):
            return row
    return None


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def threshold_methods_for(rows: list[dict[str, Any]]) -> list[str]:
    seen = {str(row.get("threshold_method", "")) for row in rows if str(row.get("threshold_method", "")).strip()}
    ordered = [method for method in THRESHOLD_METHODS if method in seen]
    for method in sorted(seen):
        if method not in ordered:
            ordered.append(method)
    return ordered


def build_micro_table(
    micro_rows: list[dict[str, Any]],
    models: list[str],
    datasets: list[str],
    label_col: str,
) -> str:
    headers = ["Model", "Dataset", "Method", "AUROC", "AP", "Accuracy", "Precision", "Recall", "F1", "FPR"]
    table_rows: list[list[str]] = []
    methods = threshold_methods_for(micro_rows)
    for model_key in models:
        spec = MODEL_SPECS[model_key]
        for dataset in datasets:
            for method in methods:
                row = row_lookup(micro_rows, label_col, model_key, dataset, method)
                table_rows.append(
                    [
                        spec.display_name,
                        dataset,
                        method,
                        format_float(None if row is None else row.get("auroc")),
                        format_float(None if row is None else row.get("average_precision")),
                        format_float(None if row is None else row.get("accuracy")),
                        format_float(None if row is None else row.get("precision")),
                        format_float(None if row is None else row.get("recall")),
                        format_float(None if row is None else row.get("f1")),
                        format_float(None if row is None else row.get("fpr")),
                    ]
                )
    return markdown_table(headers, table_rows)


def build_macro_table(
    macro_rows: list[dict[str, Any]],
    models: list[str],
    datasets: list[str],
    label_col: str,
) -> str:
    headers = ["Model", "Dataset", "Method", "AUROC", "AP", "Accuracy", "Precision", "Recall", "F1", "FPR"]
    table_rows: list[list[str]] = []
    methods = threshold_methods_for(macro_rows)
    for model_key in models:
        spec = MODEL_SPECS[model_key]
        for dataset in datasets:
            for method in methods:
                row = row_lookup(macro_rows, label_col, model_key, dataset, method)
                table_rows.append(
                    [
                        spec.display_name,
                        dataset,
                        method,
                        format_macro(None if row is None else row.get("auroc_mean"), None if row is None else row.get("auroc_std")),
                        format_macro(None if row is None else row.get("average_precision_mean"), None if row is None else row.get("average_precision_std")),
                        format_macro(None if row is None else row.get("accuracy_mean"), None if row is None else row.get("accuracy_std")),
                        format_macro(None if row is None else row.get("precision_mean"), None if row is None else row.get("precision_std")),
                        format_macro(None if row is None else row.get("recall_mean"), None if row is None else row.get("recall_std")),
                        format_macro(None if row is None else row.get("f1_mean"), None if row is None else row.get("f1_std")),
                        format_macro(None if row is None else row.get("fpr_mean"), None if row is None else row.get("fpr_std")),
                    ]
                )
    return markdown_table(headers, table_rows)


def build_status_table(status_rows: list[dict[str, Any]]) -> str:
    headers = ["Model", "Dataset", "Status", "Run Root", "Analysis Dir", "Summary", "Per-Flight", "Threshold"]
    rows: list[list[str]] = []
    for row in status_rows:
        rows.append(
            [
                str(row.get("model", "")),
                str(row.get("dataset", "")),
                str(row.get("status", "")),
                str(row.get("run_root", "")) or "N/A",
                str(row.get("analysis_dir", "")) or "N/A",
                str(row.get("summary_path", "")) or "N/A",
                str(row.get("per_flight_path", "")) or "N/A",
                str(row.get("threshold_path", "")) or "N/A",
            ]
        )
    return markdown_table(headers, rows)


def build_markdown(
    macro_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    models: list[str],
    datasets: list[str],
    label_cols: list[str],
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# TCNGATRE Hyperparameter Study Metrics",
        "",
        f"Generated at: `{generated}`",
        "",
        "All source paths below are relative to the hparam directory.",
        "",
        "## Main Table: label_any Macro Average",
        "",
        build_macro_table(macro_rows=macro_rows, models=models, datasets=datasets, label_col="label_any"),
        "",
        "Macro values are flight-level mean +/- std. Missing results are shown as `N/A`.",
        "",
        "## Supplement: label_any Micro Average",
        "",
        build_micro_table(micro_rows=micro_rows, models=models, datasets=datasets, label_col="label_any"),
        "",
    ]
    if "label_mid" in label_cols:
        lines.extend(
            [
                "## Appendix: label_mid Macro Average",
                "",
                build_macro_table(macro_rows=macro_rows, models=models, datasets=datasets, label_col="label_mid"),
                "",
                "## Appendix: label_mid Micro Average",
                "",
                build_micro_table(micro_rows=micro_rows, models=models, datasets=datasets, label_col="label_mid"),
                "",
            ]
        )
    lines.extend(
        [
            "## Result File Status",
            "",
            build_status_table(status_rows),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = ordered_subset(list(args.models), MODEL_ORDER)
    datasets = ordered_subset(list(args.datasets), DATASET_ORDER)
    label_cols = ordered_subset(list(args.label_cols), LABEL_COLS)
    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for model_key in models:
        spec = MODEL_SPECS[model_key]
        for dataset in datasets:
            one_macro, one_micro, one_status = summarize_one(spec=spec, dataset=dataset, label_cols=label_cols)
            macro_rows.extend(one_macro)
            micro_rows.extend(one_micro)
            status_rows.append(one_status)

    macro_df = pd.DataFrame(macro_rows, columns=MACRO_COLUMNS) if macro_rows else pd.DataFrame(columns=MACRO_COLUMNS)
    micro_df = pd.DataFrame(micro_rows, columns=MICRO_COLUMNS) if micro_rows else pd.DataFrame(columns=MICRO_COLUMNS)
    status_df = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)

    macro_csv = out_dir / "hparam_metrics_macro.csv"
    micro_csv = out_dir / "hparam_metrics_micro.csv"
    status_csv = out_dir / "hparam_metrics_status.csv"
    markdown_path = out_dir / "hparam_metrics.md"

    macro_df.to_csv(macro_csv, index=False, encoding="utf-8")
    micro_df.to_csv(micro_csv, index=False, encoding="utf-8")
    status_df.to_csv(status_csv, index=False, encoding="utf-8")
    markdown_path.write_text(
        build_markdown(
            macro_rows=macro_rows,
            micro_rows=micro_rows,
            status_rows=status_rows,
            models=models,
            datasets=datasets,
            label_cols=label_cols,
        ),
        encoding="utf-8",
    )

    print(f"[OK] markdown: {relative_path(markdown_path)}")
    print(f"[OK] macro csv: {relative_path(macro_csv)}")
    print(f"[OK] micro csv: {relative_path(micro_csv)}")
    print(f"[OK] status csv: {relative_path(status_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
