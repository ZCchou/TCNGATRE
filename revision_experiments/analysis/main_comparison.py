from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INFER_OUTPUT_NAMES = {
    "USAD": "infer_usad_global_threshold",
    "Recurrent_AE": "infer_recurrent_ae_failure",
    "TranAD": "infer_tranad_failure",
    "OmniAnomaly": "infer_future_window_failure",
    "BeatGAN": "infer_beatgan_failure",
    "TCNGATRE": "infer_tcngatre_failure",
}
PRIMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "fpr",
    "auroc",
    "average_precision",
]
COUNT_METRICS = ["num_samples", "positives", "negatives", "tp", "fp", "tn", "fn"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _status_for_run(run_dir: Path) -> str:
    if (run_dir / "DONE.json").is_file():
        return "complete"
    if (run_dir / "FAILED.json").is_file():
        return "failed"
    if (run_dir / "PARTIAL.json").is_file() or run_dir.exists():
        return "partial"
    return "missing"


def _micro_primary_metrics(run: dict) -> tuple[dict | None, str | None]:
    run_dir = Path(run["run_root"])
    infer_name = INFER_OUTPUT_NAMES[str(run["model"])]
    metric_path = (
        run_dir
        / infer_name
        / "score_threshold_analysis"
        / "summary_metrics.csv"
    )
    if not metric_path.is_file():
        return None, f"missing micro summary metrics: {metric_path}"
    frame = pd.read_csv(metric_path)
    required = {"threshold_method", "label_col", *PRIMARY_METRICS}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        return None, f"missing metric columns: {missing_columns}"
    selected = frame.loc[
        (frame["threshold_method"].astype(str).str.lower() == "spot")
        & (frame["label_col"].astype(str) == "label_any")
    ].copy()
    if selected.empty:
        return None, "SPOT + label_any micro summary row is missing"
    if len(selected) != 1:
        return None, f"expected one SPOT + label_any micro row, found {len(selected)}"

    source = selected.iloc[0]
    row: dict[str, Any] = {
        "run_id": run["run_id"],
        "dataset": run["dataset"],
        "model": run["model"],
        "seed": int(run["seed"]),
        "aggregation": "micro_over_all_windows",
        "threshold_method": "spot",
        "label_col": "label_any",
        "source": str(metric_path),
    }
    for metric in PRIMARY_METRICS:
        value = pd.to_numeric(pd.Series([source[metric]]), errors="coerce").iloc[0]
        row[metric] = float(value) if np.isfinite(value) else float("nan")
    for metric in COUNT_METRICS:
        if metric in selected.columns:
            value = pd.to_numeric(pd.Series([source[metric]]), errors="coerce").iloc[0]
            row[metric] = int(value) if np.isfinite(value) else None
    return row, None


def _seed_summary(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame(columns=["dataset", "model", "seed_count"])
    rows: list[dict[str, Any]] = []
    for (dataset, model), group in primary.groupby(["dataset", "model"], sort=True):
        row: dict[str, Any] = {
            "dataset": str(dataset),
            "model": str(model),
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(value)) for value in sorted(group["seed"].unique())),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if finite.size >= 2 else float("nan")
            )
            row[f"{metric}_count"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_main_comparison(result_root: Path, expected_runs: list[dict]) -> dict:
    result_root = Path(result_root)
    summary_root = result_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    run_ids = [str(row["run_id"]) for row in expected_runs]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    status_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for run in expected_runs:
        run_dir = Path(run["run_root"])
        status = _status_for_run(run_dir)
        primary, metric_error = _micro_primary_metrics(run) if status == "complete" else (None, None)
        if primary is not None:
            primary_path = run_dir / "primary_metrics.json"
            _write_json(primary_path, primary)
            primary_rows.append(primary)
        effective_status = status if primary is not None or status != "complete" else "invalid_metrics"
        status_rows.append(
            {
                **run,
                "status": effective_status,
                "metric_error": metric_error,
                "done_marker": str(run_dir / "DONE.json"),
            }
        )
        if effective_status != "complete":
            missing_rows.append(
                {
                    "run_id": run["run_id"],
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "seed": int(run["seed"]),
                    "status": effective_status,
                    "reason": metric_error or effective_status,
                }
            )

    status_frame = pd.DataFrame(status_rows)
    primary_columns = [
        "run_id", "dataset", "model", "seed", "aggregation", "threshold_method",
        "label_col", "source",
        *PRIMARY_METRICS,
        *COUNT_METRICS,
    ]
    primary_frame = pd.DataFrame(primary_rows, columns=primary_columns)
    seed_frame = _seed_summary(primary_frame)
    missing_frame = pd.DataFrame(
        missing_rows,
        columns=["run_id", "dataset", "model", "seed", "status", "reason"],
    )
    status_frame.to_csv(result_root / "run_status.csv", index=False, encoding="utf-8-sig")
    primary_frame.to_csv(
        summary_root / "primary_metrics_all_runs.csv", index=False, encoding="utf-8-sig"
    )
    seed_frame.to_csv(
        summary_root / "primary_metrics_seed_summary.csv", index=False, encoding="utf-8-sig"
    )
    missing_frame.to_csv(
        summary_root / "missing_experiment_cells.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "status": "complete" if not missing_rows and not duplicates else "incomplete",
        "expected_runs": int(len(expected_runs)),
        "complete_runs": int(len(primary_rows)),
        "missing_runs": int(len(missing_rows)),
        "duplicate_run_ids": duplicates,
        "primary_protocol": "label_any + causal EMA + flightwise SPOT; micro over all windows",
        "all_runs_csv": str(summary_root / "primary_metrics_all_runs.csv"),
        "seed_summary_csv": str(summary_root / "primary_metrics_seed_summary.csv"),
        "missing_cells_csv": str(summary_root / "missing_experiment_cells.csv"),
    }
    _write_json(summary_root / "summary_status.json", payload)
    if duplicates:
        raise RuntimeError(f"Duplicate run IDs found: {duplicates}")
    return payload
