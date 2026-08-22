from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import AGGREGATORS, ROBUSTNESS_CONDITIONS
from .io import finite_frame, write_json


def _mean_std(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    usable = [column for column in metrics if column in frame.columns]
    if frame.empty or not usable:
        return pd.DataFrame()
    numeric = frame.copy()
    for column in usable:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    summary = numeric.groupby(groups, dropna=False)[usable].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    counts = numeric.groupby(groups, dropna=False).size().rename("row_count").reset_index()
    seeds = numeric.groupby(groups, dropna=False)["seed"].nunique().rename("seed_count").reset_index()
    return summary.merge(counts, on=groups).merge(seeds, on=groups)


def _read_csv(path: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for key, value in metadata.items():
        if key not in frame:
            frame[key] = value
    return frame


def summarize_posthoc(
    output_root: Path,
    experiments: Iterable[str],
    datasets: Iterable[str],
    seeds: Iterable[int],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    selected_experiments = list(experiments)
    selected_datasets = list(datasets)
    selected_seeds = [int(seed) for seed in seeds]
    missing: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    ex05_real: list[pd.DataFrame] = []
    ex05_synthetic: list[pd.DataFrame] = []
    ex07_metrics: list[pd.DataFrame] = []
    ex08_metrics: list[dict[str, Any]] = []

    for experiment in selected_experiments:
        for dataset in selected_datasets:
            for seed in selected_seeds:
                if experiment in {"ex05", "ex07"}:
                    run_dir = output_root / experiment / dataset / f"seed_{seed}"
                    done = run_dir / "DONE.json"
                    run_id = f"{experiment}/{dataset}/seed_{seed}"
                    if not done.is_file():
                        missing.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "reason": "DONE.json missing"})
                        continue
                    if experiment == "ex05":
                        required = [run_dir / "real_failure_aggregation.csv", run_dir / "synthetic_event_metrics.csv"]
                        if not all(path.is_file() for path in required):
                            missing.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "reason": "EX-05 result CSV missing"})
                            continue
                        real = _read_csv(run_dir / "real_failure_aggregation.csv", {"dataset": dataset, "seed": seed})
                        synthetic = _read_csv(run_dir / "synthetic_event_metrics.csv", {"dataset": dataset, "seed": seed})
                        real_methods = set(real["aggregation_method"].astype(str))
                        synthetic_cells = set(zip(
                            synthetic["scenario_id"].astype(str),
                            synthetic["aggregation_method"].astype(str),
                        ))
                        finite_real = finite_frame(real, ["precision", "recall", "f1", "fpr", "auroc", "average_precision"])
                        finite_synthetic = finite_frame(synthetic, ["precision", "recall", "f1", "average_precision", "event_recall", "event_miss_rate", "mean_detection_delay", "channel_hit_at_k"])
                        if real_methods != set(AGGREGATORS) or synthetic["scenario_id"].nunique() != 42 or len(synthetic_cells) != 42 * len(AGGREGATORS) or not finite_real or not finite_synthetic:
                            missing.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "reason": "EX-05 matrix is not 42 scenarios x 7 aggregators"})
                            continue
                        ex05_real.append(real)
                        ex05_synthetic.append(synthetic)
                    else:
                        required = [run_dir / "per_flight_graph_metrics.csv", run_dir / "flight_period_coverage.csv"]
                        if not all(path.is_file() for path in required):
                            missing.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "reason": "EX-07 result CSV missing"})
                            continue
                        graph = _read_csv(run_dir / "per_flight_graph_metrics.csv", {"dataset": dataset, "seed": seed})
                        coverage = pd.read_csv(run_dir / "flight_period_coverage.csv")
                        expected_flights = {"alfa": 16, "gpsdata": 2, "simulate": 2}[dataset]
                        finite_graph = finite_frame(graph, ["frobenius_change", "mean_absolute_change", "top20_jaccard", "entropy_change", "graph_score_spearman_during"])
                        if coverage["flight"].astype(str).nunique() != expected_flights or graph.empty or not finite_graph:
                            missing.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "reason": f"EX-07 does not cover {expected_flights} failure flights"})
                            continue
                        ex07_metrics.append(graph)
                    run_rows.append({"run_id": run_id, "experiment": experiment, "dataset": dataset, "seed": seed, "condition": "", "status": "complete"})
                else:
                    for condition in ROBUSTNESS_CONDITIONS:
                        run_dir = output_root / "ex08" / dataset / condition / f"seed_{seed}"
                        done = run_dir / "DONE.json"
                        metrics = run_dir / "condition_metrics.json"
                        run_id = f"ex08/{dataset}/{condition}/seed_{seed}"
                        if not done.is_file() or not metrics.is_file():
                            missing.append({"run_id": run_id, "experiment": "ex08", "dataset": dataset, "seed": seed, "condition": condition, "reason": "DONE.json or condition_metrics.json missing"})
                            continue
                        run_rows.append({"run_id": run_id, "experiment": "ex08", "dataset": dataset, "seed": seed, "condition": condition, "status": "complete"})
                        ex08_metrics.append(json.loads(metrics.read_text(encoding="utf-8")))

    status = pd.DataFrame(run_rows, columns=["run_id", "experiment", "dataset", "seed", "condition", "status"])
    missing_frame = pd.DataFrame(missing, columns=["run_id", "experiment", "dataset", "seed", "condition", "reason"])
    if not status.empty and status["run_id"].duplicated().any():
        raise RuntimeError("Duplicate posthoc run IDs found")
    status.to_csv(summary_dir / "run_status.csv", index=False, encoding="utf-8-sig")
    missing_frame.to_csv(summary_dir / "missing_experiment_cells.csv", index=False, encoding="utf-8-sig")

    if ex05_real:
        real = pd.concat(ex05_real, ignore_index=True)
        real.to_csv(summary_dir / "ex05_real_failure_all_runs.csv", index=False, encoding="utf-8-sig")
        real_summary = _mean_std(
            real,
            ["dataset", "aggregation_method"],
            ["precision", "recall", "f1", "fpr", "auroc", "average_precision"],
        )
        real_summary.to_csv(summary_dir / "ex05_real_failure_seed_summary.csv", index=False, encoding="utf-8-sig")
        observed = set(real["aggregation_method"].astype(str))
        if observed != set(AGGREGATORS):
            raise RuntimeError(f"EX-05 aggregator mismatch: observed={sorted(observed)}")
    if ex05_synthetic:
        synthetic = pd.concat(ex05_synthetic, ignore_index=True)
        synthetic.to_csv(summary_dir / "ex05_synthetic_all_runs.csv", index=False, encoding="utf-8-sig")
        _mean_std(
            synthetic,
            ["dataset", "aggregation_method"],
            ["precision", "recall", "f1", "average_precision", "event_recall", "event_miss_rate", "mean_detection_delay", "channel_hit_at_k"],
        ).to_csv(summary_dir / "ex05_synthetic_seed_summary.csv", index=False, encoding="utf-8-sig")
    if ex07_metrics:
        graph = pd.concat(ex07_metrics, ignore_index=True)
        graph.to_csv(summary_dir / "ex07_per_flight_all_runs.csv", index=False, encoding="utf-8-sig")
        _mean_std(
            graph,
            ["dataset", "graph", "comparison"],
            ["frobenius_change", "mean_absolute_change", "top20_jaccard", "entropy_change", "graph_score_spearman_during"],
        ).to_csv(summary_dir / "ex07_graph_seed_summary.csv", index=False, encoding="utf-8-sig")
    if ex08_metrics:
        robustness = pd.DataFrame(ex08_metrics)
        primary_columns = ["precision", "recall", "f1", "fpr", "auroc", "average_precision", "absolute_f1_drop", "relative_f1_drop", "auprc_retention"]
        if not finite_frame(robustness, [column for column in primary_columns if column in robustness]):
            raise RuntimeError("EX-08 contains non-finite primary metrics")
        robustness.to_csv(summary_dir / "ex08_all_runs.csv", index=False, encoding="utf-8-sig")
        robust_summary = _mean_std(
            robustness,
            ["dataset", "condition"],
            primary_columns + ["flight_miss_rate", "mean_detection_delay"],
        )
        robust_summary.to_csv(summary_dir / "ex08_seed_summary.csv", index=False, encoding="utf-8-sig")
        worst = robustness.sort_values(["dataset", "seed", "f1"], kind="mergesort").groupby(
            ["dataset", "seed"], as_index=False
        ).first()
        worst.to_csv(summary_dir / "ex08_worst_condition_per_seed.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "complete" if not missing else "incomplete",
        "experiments": selected_experiments,
        "datasets": selected_datasets,
        "seeds": selected_seeds,
        "completed_units": len(status),
        "missing_units": len(missing_frame),
        "summary_dir": str(summary_dir),
    }
    write_json(summary_dir / "summary.json", payload)
    if require_complete and missing:
        raise RuntimeError(f"Posthoc matrix is incomplete: {len(missing)} missing units; see {summary_dir / 'missing_experiment_cells.csv'}")
    return payload
