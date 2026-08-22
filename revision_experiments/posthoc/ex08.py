from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .constants import ROBUSTNESS_CONDITIONS
from .data import corrupt_full_flights, parse_condition
from .evaluation import delay_summary, write_and_evaluate_real
from .inference import LoadedSourceModel, source_primary
from .io import environment_payload, sha256_file, write_json
from .source import SourceRun


DESIGN_VERSION = "ex08_robustness_v1"


def _config_hash(source: SourceRun, condition: str) -> str:
    payload = {
        "design": DESIGN_VERSION,
        "source_signature": source.source_signature,
        "condition": condition,
        "perturbation_seed": source.seed,
        "validation_is_perturbed": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _metric(payload: dict[str, Any], name: str) -> float:
    aliases = {
        "average_precision": ("average_precision", "auprc", "AUPRC"),
        "f1": ("f1", "F1"),
    }
    for key in aliases.get(name, (name,)):
        if key in payload:
            value = float(payload[key])
            if np.isfinite(value):
                return value
    raise RuntimeError(f"Required finite metric {name!r} is missing from {sorted(payload)}")


def _condition_metrics(
    source: SourceRun,
    condition: str,
    primary: dict[str, Any],
    scored: pd.DataFrame,
) -> dict[str, Any]:
    clean = source_primary(source)
    clean_f1 = _metric(clean, "f1")
    clean_ap = _metric(clean, "average_precision")
    current_f1 = _metric(primary, "f1")
    current_ap = _metric(primary, "average_precision")
    delays = delay_summary(scored)
    result = {
        "dataset": source.dataset,
        "seed": source.seed,
        "condition": condition,
        "clean_f1": clean_f1,
        "clean_average_precision": clean_ap,
        "absolute_f1_drop": clean_f1 - current_f1,
        "relative_f1_drop": (clean_f1 - current_f1) / max(abs(clean_f1), 1e-12),
        "auprc_retention": current_ap / max(abs(clean_ap), 1e-12),
        "failure_flights": delays["flights"],
        "detected_failure_flights": delays["detected_flights"],
        "flight_miss_rate": delays["miss_rate"],
        "mean_detection_delay": delays["mean_delay"],
        "median_detection_delay": delays.get("median_delay", float("nan")),
    }
    for key in (
        "precision", "recall", "f1", "fpr", "auroc", "average_precision",
        "accuracy", "tp", "fp", "tn", "fn", "threshold_mean",
    ):
        if key in primary:
            value = primary[key]
            if np.isfinite(float(value)):
                result[key] = value
    return result


def run_ex08(
    source: SourceRun,
    output_root: Path,
    *,
    conditions: Iterable[str] = ROBUSTNESS_CONDITIONS,
    force: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate all requested corruptions while loading the source model only once."""
    selected = list(conditions)
    if smoke:
        selected = selected[:1]
    unknown = sorted(set(selected) - set(ROBUSTNESS_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown EX-08 conditions: {unknown}")

    seed_dir = Path(output_root) / "ex08" / source.dataset / f"seed_{source.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash_before = sha256_file(source.checkpoint)
    pending: list[str] = []
    outcomes: list[dict[str, Any]] = []
    for condition in selected:
        condition_dir = Path(output_root) / "ex08" / source.dataset / condition / f"seed_{source.seed}"
        done_path = condition_dir / "DONE.json"
        expected_hash = _config_hash(source, condition) + ("_smoke" if smoke else "")
        if done_path.is_file() and not force:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                done.get("config_hash") == expected_hash
                and done.get("source_signature") == source.source_signature
            ):
                outcomes.append({"status": "skipped_complete", **done})
                continue
        pending.append(condition)

    loaded = LoadedSourceModel.load(source) if pending else None
    for condition in tqdm(pending, desc=f"EX-08 {source.dataset} seed={source.seed}", unit="condition"):
        condition_dir = Path(output_root) / "ex08" / source.dataset / condition / f"seed_{source.seed}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        config_hash = _config_hash(source, condition) + ("_smoke" if smoke else "")
        done_path = condition_dir / "DONE.json"
        if done_path.exists():
            done_path.unlink()
        try:
            assert loaded is not None
            kind, level = parse_condition(condition)
            validation, validation_manifest = corrupt_full_flights(
                loaded.validation_flights,
                kind,
                level,
                loaded.channel_statistics,
                perturbation_seed=source.seed,
            )
            failure, failure_manifest = corrupt_full_flights(
                loaded.failure_flights,
                kind,
                level,
                loaded.channel_statistics,
                perturbation_seed=source.seed,
            )
            validation_residuals, _ = loaded.score(
                validation,
                description=f"EX-08 {source.dataset} s{source.seed} {condition} validation",
            )
            failure_residuals, _ = loaded.score(
                failure,
                description=f"EX-08 {source.dataset} s{source.seed} {condition} failure",
            )
            primary, scored = write_and_evaluate_real(
                condition_dir,
                loaded,
                validation_residuals,
                failure_residuals,
                "mean",
            )
            metrics = _condition_metrics(source, condition, primary, scored)
            write_json(condition_dir / "condition_metrics.json", metrics)
            write_json(condition_dir / "corruption_manifest.json", {
                "design_version": DESIGN_VERSION,
                "condition": condition,
                "kind": kind,
                "level": level,
                "perturbation_seed": source.seed,
                "statistics_source": "normal training flights only",
                "channel_statistics": loaded.channel_statistics.to_dict(),
                "validation": validation_manifest,
                "failure": failure_manifest,
            })
            write_json(condition_dir / "resolved_analysis_config.json", {
                "design_version": DESIGN_VERSION,
                "condition": condition, "kind": kind, "level": level,
                "model_seed": source.seed, "perturbation_seed": source.seed,
                "validation_is_perturbed": True,
                "fill_policy": "causal forward fill; training median at flight start",
                "statistics_source": "normal training flights only",
                "aggregation_method": "mean", "threshold_method": "flightwise causal SPOT",
                "requires_training": False,
            })
            provenance = {
                "experiment": "ex08",
                "dataset": source.dataset,
                "model_seed": source.seed,
                "perturbation_seed": source.seed,
                "condition": condition,
                "config_hash": config_hash,
                "source": source.to_dict(),
                "environment": environment_payload(),
            }
            write_json(condition_dir / "provenance.json", provenance)
            done = {
                "status": "complete",
                "experiment": "ex08",
                "dataset": source.dataset,
                "seed": source.seed,
                "condition": condition,
                "config_hash": config_hash,
                "source_signature": source.source_signature,
                "source_checkpoint_sha256": source.checkpoint_sha256,
                "metrics_path": str(condition_dir / "condition_metrics.json"),
                "environment": environment_payload(),
            }
            write_json(condition_dir / "DONE.json", done)
            failed_path = condition_dir / "FAILED.json"
            if failed_path.exists():
                failed_path.unlink()
            outcomes.append(done)
        except Exception as exc:
            failed = {
                "status": "failed",
                "experiment": "ex08",
                "dataset": source.dataset,
                "seed": source.seed,
                "condition": condition,
                "config_hash": config_hash,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(condition_dir / "FAILED.json", failed)
            outcomes.append(failed)

    checkpoint_hash_after = sha256_file(source.checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError(f"Source checkpoint changed during EX-08: {source.checkpoint}")
    failed_count = sum(item.get("status") == "failed" for item in outcomes)
    summary = {
        "status": "complete" if failed_count == 0 else "failed",
        "experiment": "ex08",
        "dataset": source.dataset,
        "seed": source.seed,
        "conditions_requested": selected,
        "conditions_complete": sum(item.get("status") in {"complete", "skipped_complete"} for item in outcomes),
        "conditions_failed": failed_count,
        "source_signature": source.source_signature,
        "source_checkpoint_sha256": checkpoint_hash_after,
    }
    final_marker = seed_dir / ("DONE.json" if failed_count == 0 else "FAILED.json")
    stale_marker = seed_dir / ("FAILED.json" if failed_count == 0 else "DONE.json")
    write_json(final_marker, summary)
    if stale_marker.exists():
        stale_marker.unlink()
    if failed_count:
        raise RuntimeError(
            f"EX-08 failed for {failed_count}/{len(selected)} conditions: {source.dataset}/seed_{source.seed}"
        )
    return summary
