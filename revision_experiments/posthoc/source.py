from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revision_experiments.core.paths import REPO_ROOT, ensure_import_paths
from revision_experiments.scoring.aggregators import aggregate_dataframe

from .constants import DATASETS, MODEL_SEEDS
from .io import sha256_file
from .parity import SCORE_RTOL, compare_scores

ensure_import_paths()

from data.stgtcn_window_dataset import resolve_flight_splits  # noqa: E402
from tcngatreconfig import TCNGATREConfig  # noqa: E402


INFER_NAME = "infer_tcngatre_failure"
RAW_SCORE_ATOL = 5e-6
SMOOTH_SCORE_ATOL = 5e-5


@dataclass(frozen=True)
class SourceRun:
    dataset: str
    seed: int
    source_root: Path
    run_dir: Path
    checkpoint: Path
    normalization_stats: Path
    config_path: Path
    split_path: Path
    failure_residuals: Path
    sequence_scores: Path
    primary_metrics: Path
    per_flight_metrics: Path
    graph_dir: Path
    checkpoint_sha256: str
    source_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }


def _resolve_graph_dir(source_root: Path, dataset: str, payload: dict) -> Path:
    candidates = []
    if payload.get("graph_dir"):
        candidates.append(Path(payload["graph_dir"]))
    candidates.extend([
        Path(source_root) / "_shared" / "tcngatre_graph" / dataset,
        REPO_ROOT / "revision_results" / "protocol_v1" / "_cache" / dataset / "graph",
    ])
    for path in candidates:
        if (path / "keep_columns.json").is_file() and (path / "adjacency_dense.csv").is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"No compatible TCNGATRE graph found for {dataset}; candidates={candidates}"
    )


def _resolve_primary_metrics(run_dir: Path, analysis_dir: Path) -> Path:
    """Accept both native evaluation and main-comparison summary locations."""
    candidates = (
        Path(analysis_dir) / "primary_metrics.json",
        Path(run_dir) / "primary_metrics.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _canonicalize_sensor_vectors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "sensor_score_vec" not in output:
        return output
    values = []
    for value in output["sensor_score_vec"]:
        if isinstance(value, str):
            try:
                parsed = np.asarray(json.loads(value), dtype=np.float64).reshape(-1)
            except json.JSONDecodeError:
                parsed = np.fromstring(value.strip().strip("[]").replace(",", " "), sep=" ")
        else:
            parsed = np.asarray(value, dtype=np.float64).reshape(-1)
        if parsed.size == 0 or not np.isfinite(parsed).all():
            raise ValueError(f"Invalid source sensor_score_vec: {str(value)[:80]!r}")
        values.append(json.dumps(parsed.astype(float).tolist()))
    output["sensor_score_vec"] = values
    return output


def resolve_source_run(source_root: Path, dataset: str, seed: int) -> SourceRun:
    dataset = str(dataset).lower()
    seed = int(seed)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if seed < 0:
        raise ValueError(f"Invalid seed: {seed}")
    source_root = Path(source_root).expanduser().resolve()
    run_dir = source_root / dataset / "TCNGATRE" / f"seed_{seed}"
    infer = run_dir / INFER_NAME
    analysis = infer / "score_threshold_analysis"
    paths = {
        "checkpoint": run_dir / "best.pt",
        "normalization_stats": run_dir / "normalization_stats.json",
        "config_path": run_dir / "config.json",
        "split_path": run_dir / "split_flights.json",
        "failure_residuals": infer / "all_failure_window_forecast_residual.csv",
        "sequence_scores": infer / "sequence_scores.csv",
        "primary_metrics": _resolve_primary_metrics(run_dir, analysis),
        "per_flight_metrics": analysis / "per_flight_total_score_threshold_methods.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Main-comparison TCNGATRE source is incomplete for {dataset}/seed_{seed}: {missing}"
        )
    config_payload = json.loads(paths["config_path"].read_text(encoding="utf-8"))
    graph_dir = _resolve_graph_dir(source_root, dataset, config_payload)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    signature_inputs = [
        checkpoint_hash,
        sha256_file(paths["normalization_stats"]),
        sha256_file(paths["config_path"]),
        sha256_file(paths["split_path"]),
        sha256_file(graph_dir / "keep_columns.json"),
        sha256_file(graph_dir / "adjacency_dense.csv"),
    ]
    import hashlib

    signature = hashlib.sha256("|".join(signature_inputs).encode("utf-8")).hexdigest()
    return SourceRun(
        dataset=dataset,
        seed=seed,
        source_root=source_root,
        run_dir=run_dir,
        graph_dir=graph_dir,
        checkpoint_sha256=checkpoint_hash,
        source_signature=signature,
        **paths,
    )


def native_config(source: SourceRun) -> TCNGATREConfig:
    payload = json.loads(source.config_path.read_text(encoding="utf-8"))
    # Native main-comparison configs already use the target names.  These
    # aliases also make older isolated RevisionConfig smoke artefacts auditable.
    for old_name, native_name in {
        "horizon": "horizon_out",
        "stride": "sample_stride",
        "epochs": "num_epochs",
        "data_split_seed": "split_seed",
    }.items():
        if native_name not in payload and old_name in payload:
            payload[native_name] = payload[old_name]
    cfg = TCNGATREConfig(
        dataset_name=source.dataset,
        run_root=source.run_dir,
        graph_dir=source.graph_dir,
        normalization_stats_path=source.normalization_stats,
    )
    path_fields = {
        "data_root", "labels_root", "run_root", "split_info_path",
        "graph_dir", "normalization_stats_path",
    }
    valid = {item.name for item in fields(TCNGATREConfig)}
    for name, value in payload.items():
        if name not in valid or name in path_fields or value is None:
            continue
        current = getattr(cfg, name)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, name, value)
    cfg.run_root = source.run_dir
    cfg.graph_dir = source.graph_dir
    cfg.normalization_stats_path = source.normalization_stats
    cfg.plot_scores = False
    cfg.num_workers = 0
    return cfg


def audit_source(source: SourceRun) -> dict[str, Any]:
    cfg = native_config(source)
    train, validation, failure = resolve_flight_splits(
        dataset_root=Path(cfg.data_root), split_info_path=Path(cfg.split_info_path)
    )
    split = json.loads(source.split_path.read_text(encoding="utf-8"))
    declared_train = sorted(str(item) for item in split.get("train_flights", []))
    declared_val = sorted(
        str(item) for item in split.get("val_flights", split.get("validation_flights", []))
    )
    actual_train = sorted(str(item) for item in train)
    actual_val = sorted(str(item) for item in validation)
    errors = []
    if declared_train != actual_train:
        errors.append("train_flight_mismatch")
    if declared_val != actual_val:
        errors.append("validation_flight_mismatch")
    expected = {
        "alfa": (29, 1, 16),
        "gpsdata": (1, 1, 2),
        "simulate": (8, 2, 2),
    }[source.dataset]
    actual = (len(train), len(validation), len(failure))
    if actual != expected:
        errors.append(f"split_count={actual},expected={expected}")
    expected_failure_names = sorted(str(item) for item in failure)
    score_flights = sorted(
        pd.read_csv(source.sequence_scores, usecols=["flight"])["flight"].astype(str).unique()
    )
    if score_flights != expected_failure_names:
        errors.append("failure_score_flight_mismatch")
    per_flight = pd.read_csv(source.per_flight_metrics)
    primary_flights = sorted(per_flight.loc[
        (per_flight["threshold_method"].astype(str) == "spot")
        & (per_flight["label_col"].astype(str) == "label_any"),
        "flight",
    ].astype(str).unique())
    if primary_flights != expected_failure_names:
        errors.append(
            f"primary_labeled_flights={len(primary_flights)},expected={len(expected_failure_names)}"
        )
    primary = json.loads(source.primary_metrics.read_text(encoding="utf-8"))
    for key in ("precision", "recall", "f1", "fpr", "auroc", "average_precision"):
        if key not in primary or not np.isfinite(float(primary[key])):
            errors.append(f"non_finite_primary_metric={key}")
    raw_parity = float("nan")
    smooth_parity = float("nan")
    raw_relative_parity = float("nan")
    smooth_relative_parity = float("nan")
    try:
        source_residuals = _canonicalize_sensor_vectors(pd.read_csv(source.failure_residuals))
        rebuilt = aggregate_dataframe(
            source_residuals,
            "mean",
            float(cfg.score_temporal_smooth_alpha),
        )
        sequence = pd.read_csv(source.sequence_scores)
        parity = sequence.merge(
            rebuilt[["flight", "current_index", "raw_total_score", "total_score"]],
            on=["flight", "current_index"],
            suffixes=("_source", "_rebuilt"),
            validate="one_to_one",
        )
        if len(parity) != len(sequence) or len(parity) != len(rebuilt):
            errors.append(
                f"source_score_row_mismatch=sequence:{len(sequence)},residuals:{len(rebuilt)},merged:{len(parity)}"
            )
        else:
            raw_result = compare_scores(
                parity["raw_total_score_source"].to_numpy(),
                parity["raw_total_score_rebuilt"].to_numpy(),
                atol=RAW_SCORE_ATOL,
            )
            smooth_result = compare_scores(
                parity["total_score_source"].to_numpy(),
                parity["total_score_rebuilt"].to_numpy(),
                atol=SMOOTH_SCORE_ATOL,
            )
            raw_parity = raw_result.max_abs_error
            smooth_parity = smooth_result.max_abs_error
            raw_relative_parity = raw_result.max_rel_error
            smooth_relative_parity = smooth_result.max_rel_error
            if not raw_result.passed or not smooth_result.passed:
                errors.append(
                    "source_score_inconsistent="
                    f"raw_abs:{raw_parity:.9g},raw_rel:{raw_relative_parity:.9g},"
                    f"smooth_abs:{smooth_parity:.9g},smooth_rel:{smooth_relative_parity:.9g},"
                    f"rtol:{SCORE_RTOL:.9g}"
                )
    except Exception as exc:
        errors.append(f"source_score_audit_failed={exc!r}")
    return {
        "status": "passed" if not errors else "failed",
        "dataset": source.dataset,
        "seed": source.seed,
        "split_counts": {"train": len(train), "validation": len(validation), "failure": len(failure)},
        "checkpoint_sha256": source.checkpoint_sha256,
        "source_signature": source.source_signature,
        "scored_failure_flights": len(score_flights),
        "primary_labeled_flights": len(primary_flights),
        "source_raw_score_max_abs_error": raw_parity,
        "source_raw_score_max_rel_error": raw_relative_parity,
        "source_smooth_score_max_abs_error": smooth_parity,
        "source_smooth_score_max_rel_error": smooth_relative_parity,
        "primary_metrics_path": str(source.primary_metrics),
        "errors": errors,
    }


def audit_matrix(source_root: Path) -> dict[str, Any]:
    rows = []
    for dataset in DATASETS:
        for seed in MODEL_SEEDS:
            try:
                rows.append(audit_source(resolve_source_run(source_root, dataset, seed)))
            except Exception as exc:
                rows.append({"status": "failed", "dataset": dataset, "seed": seed, "errors": [repr(exc)]})
    passed = sum(row["status"] == "passed" for row in rows)
    return {
        "status": "passed" if passed == len(rows) else "failed",
        "expected_runs": len(rows),
        "passed_runs": passed,
        "rows": rows,
    }
