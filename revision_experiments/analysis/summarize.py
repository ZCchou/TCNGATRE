from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revision_experiments.analysis.statistics import (
    hierarchical_paired_bootstrap,
    holm_adjust,
    paired_sign_permutation,
    rank_biserial,
)
from revision_experiments.core.config import make_config
from revision_experiments.core.engine import data_protocol_payload
from revision_experiments.core.paths import RESULTS_ROOT


SEED_PATTERN = re.compile(r"seed_(\d+)$")
PRIMARY_METRICS = [
    "precision", "recall", "f1", "fpr", "auroc", "average_precision",
]
COUNT_METRICS = ["num_samples", "positives", "negatives", "tp", "fp", "tn", "fn"]
ANALYSIS_RELATIVE = Path("infer_tcngatre_failure") / "score_threshold_analysis"
SIGNIFICANCE_COLUMNS = [
    "dataset", "reference_experiment", "reference_variant", "comparator_experiment",
    "comparator_variant", "metric", "status", "expected_seeds", "paired_observations",
    "expected_paired_observations", "n_flights", "reference_mean", "comparator_mean",
    "mean_difference_reference_minus_comparator", "ci95_low", "ci95_high", "p_value",
    "p_value_holm", "permutation_exact", "rank_biserial", "bootstrap_resamples",
    "significant_holm_0_05",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _expected_protocol(dataset: str, seed: int = 0) -> dict[str, Any]:
    cfg = make_config("ex01", dataset, "full", seed)
    return data_protocol_payload(cfg.to_legacy())


def _read_completed_run(
    run_dir: Path,
    *,
    expected_config_hash: str,
    expected_data_protocol_hash: str,
) -> tuple[dict[str, Any] | None, str | None]:
    done_path = run_dir / "DONE.json"
    metric_path = run_dir / ANALYSIS_RELATIVE / "primary_metrics.json"
    if not done_path.is_file():
        failed = run_dir / "FAILED.json"
        return None, "failed_marker_present" if failed.is_file() else "missing_DONE.json"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"unreadable_DONE.json: {exc!r}"
    if done.get("status") != "complete":
        return None, f"DONE_status={done.get('status')!r}"
    if done.get("config_hash") != expected_config_hash:
        return None, "config_hash_mismatch"
    if done.get("data_protocol_hash") != expected_data_protocol_hash:
        return None, "data_protocol_hash_mismatch"
    if not metric_path.is_file():
        return None, "missing_primary_metrics.json"
    try:
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"unreadable_primary_metrics.json: {exc!r}"
    if str(payload.get("threshold_method", "")).lower() != "spot":
        return None, "primary_threshold_is_not_spot"
    if payload.get("label_col") != "label_any":
        return None, "primary_label_is_not_label_any"
    for metric in PRIMARY_METRICS:
        try:
            value = float(payload[metric])
        except (KeyError, TypeError, ValueError):
            return None, f"invalid_primary_metric={metric}"
        if not math.isfinite(value):
            return None, f"non_finite_primary_metric={metric}"
        payload[metric] = value
    return payload, None


def _seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["experiment", "dataset", "variant", "seed_count", "seeds"]
    columns.extend(
        f"{metric}_{suffix}"
        for metric in PRIMARY_METRICS
        for suffix in ("mean", "std", "count")
    )
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (experiment, dataset, variant), group in frame.groupby(
        ["experiment", "dataset", "variant"], sort=True
    ):
        row: dict[str, Any] = {
            "experiment": experiment,
            "dataset": dataset,
            "variant": variant,
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
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
    return pd.DataFrame(rows, columns=columns)


def _read_per_flight(run_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    path = run_dir / ANALYSIS_RELATIVE / "per_flight_total_score_threshold_methods.csv"
    if not path.is_file():
        return None, "missing_per_flight_metrics"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return None, f"unreadable_per_flight_metrics: {exc!r}"
    required = {"flight", "threshold_method", "label_col", "f1"}
    if not required.issubset(frame.columns):
        return None, f"missing_per_flight_columns={sorted(required.difference(frame.columns))}"
    selected = frame.loc[
        (frame["threshold_method"].astype(str).str.lower() == "spot")
        & (frame["label_col"].astype(str) == "label_any")
    ].copy()
    if selected.empty:
        return None, "missing_per_flight_spot_label_any_rows"
    if selected["flight"].duplicated().any():
        return None, "duplicate_per_flight_spot_label_any_rows"
    selected["f1"] = pd.to_numeric(selected["f1"], errors="coerce")
    if not np.isfinite(selected["f1"].to_numpy(dtype=float)).all():
        return None, "non_finite_per_flight_f1"
    return selected, None


def _paired_significance(
    per_flight: pd.DataFrame,
    selection: dict[str, list[str]],
    datasets: list[str],
    seeds: list[int],
    reference: tuple[str, str],
    n_resamples: int,
    expected_flight_counts: dict[str, int],
) -> pd.DataFrame:
    reference_experiment, reference_variant = reference
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        reference_frame = per_flight.loc[
            (per_flight["dataset"] == dataset)
            & (per_flight["experiment"] == reference_experiment)
            & (per_flight["variant"] == reference_variant),
            ["flight", "seed", "f1"],
        ].rename(columns={"f1": "reference_f1"})
        for experiment, variants in selection.items():
            for variant in variants:
                if (experiment, variant) == reference:
                    continue
                comparator = per_flight.loc[
                    (per_flight["dataset"] == dataset)
                    & (per_flight["experiment"] == experiment)
                    & (per_flight["variant"] == variant),
                    ["flight", "seed", "f1"],
                ].rename(columns={"f1": "comparator_f1"})
                pairs = reference_frame.merge(
                    comparator, on=["flight", "seed"], how="inner", validate="one_to_one"
                )
                expected_pair_count = int(expected_flight_counts[dataset] * len(seeds))
                base = {
                    "dataset": dataset,
                    "reference_experiment": reference_experiment,
                    "reference_variant": reference_variant,
                    "comparator_experiment": experiment,
                    "comparator_variant": variant,
                    "metric": "f1",
                    "expected_seeds": len(seeds),
                    "paired_observations": int(len(pairs)),
                    "expected_paired_observations": expected_pair_count,
                }
                if (
                    pairs.empty
                    or sorted(pairs["seed"].unique().tolist()) != sorted(seeds)
                    or len(pairs) != expected_pair_count
                ):
                    rows.append({**base, "status": "incomplete_pairs"})
                    continue
                bootstrap = hierarchical_paired_bootstrap(
                    pairs,
                    "reference_f1",
                    "comparator_f1",
                    n_resamples=n_resamples,
                )
                flight_differences = pairs.assign(
                    difference=pairs["reference_f1"] - pairs["comparator_f1"]
                ).groupby("flight", sort=False)["difference"].mean().to_numpy(dtype=float)
                permutation = paired_sign_permutation(
                    flight_differences, n_resamples=n_resamples
                )
                rows.append(
                    {
                        **base,
                        "status": "complete",
                        "n_flights": int(pairs["flight"].nunique()),
                        "reference_mean": float(pairs["reference_f1"].mean()),
                        "comparator_mean": float(pairs["comparator_f1"].mean()),
                        "mean_difference_reference_minus_comparator": bootstrap["mean_difference"],
                        "ci95_low": bootstrap["ci95_low"],
                        "ci95_high": bootstrap["ci95_high"],
                        "p_value": permutation["p_value"],
                        "permutation_exact": permutation["exact"],
                        "rank_biserial": rank_biserial(flight_differences),
                        "bootstrap_resamples": n_resamples,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["p_value_holm"] = np.nan
        completed = frame.loc[frame["status"] == "complete"]
        for _, indices in completed.groupby("dataset").groups.items():
            positions = list(indices)
            adjusted = holm_adjust(frame.loc[positions, "p_value"].astype(float).tolist())
            frame.loc[positions, "p_value_holm"] = adjusted
        frame["significant_holm_0_05"] = frame["p_value_holm"] < 0.05
    return frame.reindex(columns=SIGNIFICANCE_COLUMNS)


def summarize_experiment_matrix(
    protocol_name: str,
    selection: dict[str, list[str]],
    datasets: list[str],
    seeds: list[int],
    *,
    results_root: Path = RESULTS_ROOT,
    preset_name: str = "custom",
    reference: tuple[str, str] = ("ex01", "full"),
    n_resamples: int = 10000,
) -> dict[str, Any]:
    if reference[0] not in selection or reference[1] not in selection[reference[0]]:
        raise ValueError(f"Reference {reference} is not included in the selected matrix")
    root = Path(results_root) / protocol_name
    output = root / "summary" / preset_name
    output.mkdir(parents=True, exist_ok=True)
    protocol_payloads = {dataset: _expected_protocol(dataset) for dataset in datasets}
    protocol_hashes = {
        dataset: str(payload["data_protocol_hash"])
        for dataset, payload in protocol_payloads.items()
    }
    expected_flight_counts = {
        dataset: len(payload["failure_flights"])
        for dataset, payload in protocol_payloads.items()
    }

    expected_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    per_flight_rows: list[pd.DataFrame] = []
    missing_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for experiment, variants in selection.items():
        for dataset in datasets:
            for variant in variants:
                for seed in seeds:
                    cfg = make_config(experiment, dataset, variant, seed)
                    run_id = f"{experiment}/{dataset}/{variant}/seed_{seed}"
                    run_ids.append(run_id)
                    run_dir = root / experiment / dataset / variant / f"seed_{seed}"
                    expected = {
                        "run_id": run_id,
                        "experiment": experiment,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": int(seed),
                        "config_hash": cfg.config_hash,
                        "data_protocol_hash": protocol_hashes[dataset],
                        "run_dir": str(run_dir),
                    }
                    metrics, reason = _read_completed_run(
                        run_dir,
                        expected_config_hash=cfg.config_hash,
                        expected_data_protocol_hash=protocol_hashes[dataset],
                    )
                    expected["status"] = "complete" if metrics is not None else "missing_or_invalid"
                    expected["reason"] = reason
                    expected["per_flight_status"] = "not_checked"
                    expected_rows.append(expected)
                    if metrics is None:
                        missing_rows.append(expected.copy())
                        continue
                    metric_row = {
                        **{key: expected[key] for key in ("run_id", "experiment", "dataset", "variant", "seed")},
                        "aggregation": "micro_over_all_windows",
                        "threshold_method": "spot",
                        "label_col": "label_any",
                        **{metric: metrics.get(metric) for metric in PRIMARY_METRICS + COUNT_METRICS},
                        "source": str(run_dir / ANALYSIS_RELATIVE / "primary_metrics.json"),
                    }
                    primary_rows.append(metric_row)
                    flight_frame, flight_error = _read_per_flight(run_dir)
                    if flight_frame is None:
                        expected["per_flight_status"] = flight_error
                    else:
                        expected["per_flight_status"] = "complete"
                        flight_frame.insert(0, "seed", int(seed))
                        flight_frame.insert(0, "variant", variant)
                        flight_frame.insert(0, "dataset", dataset)
                        flight_frame.insert(0, "experiment", experiment)
                        per_flight_rows.append(flight_frame)

    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    expected_frame = pd.DataFrame(expected_rows)
    primary_columns = [
        "run_id", "experiment", "dataset", "variant", "seed", "aggregation",
        "threshold_method", "label_col", *PRIMARY_METRICS, *COUNT_METRICS, "source",
    ]
    primary_frame = pd.DataFrame(primary_rows, columns=primary_columns)
    missing_frame = pd.DataFrame(missing_rows, columns=expected_frame.columns)
    per_flight_frame = (
        pd.concat(per_flight_rows, ignore_index=True)
        if per_flight_rows else pd.DataFrame(
            columns=["experiment", "dataset", "variant", "seed", "flight", "f1"]
        )
    )
    seed_frame = _seed_summary(primary_frame)
    significance = (
        _paired_significance(
            per_flight_frame, selection, datasets, seeds, reference, n_resamples,
            expected_flight_counts,
        )
        if not per_flight_frame.empty else pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)
    )
    per_flight_invalid_runs = sum(
        row.get("status") == "complete" and row.get("per_flight_status") != "complete"
        for row in expected_rows
    )

    expected_frame.to_csv(output / "run_status.csv", index=False, encoding="utf-8-sig")
    primary_frame.to_csv(output / "primary_metrics_all_runs.csv", index=False, encoding="utf-8-sig")
    seed_frame.to_csv(output / "primary_metrics_seed_summary.csv", index=False, encoding="utf-8-sig")
    missing_frame.to_csv(output / "missing_experiment_cells.csv", index=False, encoding="utf-8-sig")
    per_flight_frame.to_csv(output / "per_flight_primary_all_runs.csv", index=False, encoding="utf-8-sig")
    significance.to_csv(output / "paired_significance.csv", index=False, encoding="utf-8-sig")

    incomplete_significance = (
        int((significance.get("status", pd.Series(dtype=str)) != "complete").sum())
        if not significance.empty else 0
    )
    payload = {
        "status": "complete" if (
            not missing_rows
            and not duplicates
            and per_flight_invalid_runs == 0
            and incomplete_significance == 0
        ) else "incomplete",
        "preset": preset_name,
        "selection": selection,
        "datasets": datasets,
        "seeds": seeds,
        "expected_runs": len(expected_rows),
        "complete_runs": len(primary_rows),
        "missing_or_invalid_runs": len(missing_rows),
        "duplicate_run_ids": duplicates,
        "per_flight_invalid_runs": int(per_flight_invalid_runs),
        "incomplete_significance_rows": incomplete_significance,
        "primary_protocol": "label_any + causal EMA + flightwise SPOT; Micro over all windows",
        "reference": {"experiment": reference[0], "variant": reference[1]},
        "output_dir": str(output),
        "files": {
            "run_status": str(output / "run_status.csv"),
            "all_runs": str(output / "primary_metrics_all_runs.csv"),
            "seed_summary": str(output / "primary_metrics_seed_summary.csv"),
            "missing_cells": str(output / "missing_experiment_cells.csv"),
            "per_flight": str(output / "per_flight_primary_all_runs.csv"),
            "significance": str(output / "paired_significance.csv"),
        },
    }
    _write_json(output / "summary.json", payload)
    if duplicates:
        raise RuntimeError(f"Duplicate run IDs found: {duplicates}")
    return payload


def collect_primary_metrics(protocol: str, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Backward-compatible unrestricted collector for existing callers."""
    rows: list[dict[str, Any]] = []
    root = Path(results_root) / protocol
    for path in root.glob(
        "*/*/*/seed_*/infer_tcngatre_failure/score_threshold_analysis/primary_metrics.json"
    ):
        seed_dir = path.parents[2]
        match = SEED_PATTERN.match(seed_dir.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": path.parents[5].name,
                "dataset": path.parents[4].name,
                "variant": path.parents[3].name,
                "seed": int(match.group(1)),
                **payload,
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)
