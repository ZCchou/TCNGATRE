from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for path in (REPO_ROOT, PACKAGE_ROOT, REPO_ROOT / "TCNGATRE"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from revision_experiments.analysis.graph_interpretability import analyze_graph_run
from revision_experiments.analysis.summarize import summarize_experiment_matrix
from revision_experiments.baselines.export_common_data import ensure_common_data
from revision_experiments.baselines.manager import audit_adapter_runs, fetch_and_audit, load_sources
from revision_experiments.baselines.launcher import execute_isolated_baseline
from revision_experiments.core.config import (
    DATASETS,
    load_protocol,
    make_config,
    resolve_experiment_selection,
)
from revision_experiments.core.doctor import run_doctor
from revision_experiments.core.engine import execute_robustness_inference, execute_training_run
from revision_experiments.core.integrity import create_snapshot, verify_snapshot
from revision_experiments.core.paths import EXTERNAL_ROOT, RESULTS_ROOT
from revision_experiments.core.provenance import write_json
from revision_experiments.scoring.postprocess import run_aggregation_suite


TRAINING_EXPERIMENTS = {"ex01", "ex02"}
BASELINE_EXPERIMENTS = {"ex03", "ex04"}


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _datasets(value: str) -> list[str]:
    values = list(DATASETS) if value.strip().lower() == "all" else _csv_list(value)
    if not values:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(values).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate datasets are not allowed: {values}")
    return values


def _seeds(value: str) -> list[int]:
    values = [int(item) for item in _csv_list(value)]
    if not values:
        raise ValueError("At least one model seed is required")
    if any(seed < 0 for seed in values):
        raise ValueError(f"Model seeds must be non-negative: {values}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate model seeds are not allowed: {values}")
    return values


def _task_rows(
    experiments: list[str],
    datasets: list[str],
    seeds: list[int],
    smoke: bool,
    *,
    variants: list[str] | None = None,
    preset: str | None = None,
) -> list[dict]:
    protocol = load_protocol()
    selection = resolve_experiment_selection(
        protocol,
        experiments=experiments,
        variants=variants,
        preset=preset,
    )
    rows = []
    for experiment, experiment_variants in selection.items():
        for dataset in datasets:
            if experiment == "ex03" and dataset != "alfa":
                continue
            for variant in experiment_variants:
                for seed in seeds:
                    cfg = make_config(experiment, dataset, variant, seed, smoke=smoke, protocol=protocol)
                    rows.append({
                        "experiment": experiment,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "smoke": smoke,
                        "config_hash": cfg.config_hash,
                        "run_dir": str(cfg.run_dir),
                    })
    return rows


def _write_manifest(rows: list[dict], name: str) -> Path:
    output = RESULTS_ROOT / "protocol_v1" / "manifests" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def _baseline_placeholder(experiment: str, dataset: str, variant: str, seed: int, smoke: bool) -> dict:
    cfg = make_config(experiment, dataset, variant, seed, smoke=smoke)
    sources = load_sources()
    source = sources[variant]
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    status = source["code_status"]
    audit_path = EXTERNAL_ROOT / "baseline_audit.json"
    audit = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8")).get("baselines", {}).get(variant, {})
    environment = PACKAGE_ROOT / "envs" / f"{variant}.yml"
    if status == "skipped_by_scope":
        run_status = "skipped_by_scope"
    elif status.startswith("not_reproducible"):
        run_status = "not_reproducible"
    elif audit.get("audit_status") == "incomplete_pending_clean_room_review":
        run_status = "pending_reproducibility_review"
    else:
        run_status = "pending_isolated_environment_validation"
    payload = {
        "status": run_status,
        "numeric_result_emitted": False,
        "reason": status,
        "source": source,
        "source_audit": audit,
        "environment_file": str(environment) if environment.exists() else None,
        "config_hash": cfg.config_hash,
    }
    marker = {
        "not_reproducible": "NOT_REPRODUCIBLE.json",
        "skipped_by_scope": "SKIPPED_SCOPE.json",
    }.get(payload["status"], "PENDING_ADAPTER.json")
    write_json(cfg.run_dir / marker, payload)
    return payload


def execute_tasks(rows: list[dict], force: bool = False) -> list[dict]:
    outcomes = []
    for index, row in enumerate(rows, start=1):
        experiment = row["experiment"]
        cfg = make_config(
            experiment, row["dataset"], row["variant"], int(row["seed"]), smoke=bool(row["smoke"])
        )
        print(f"[{index}/{len(rows)}] {experiment}/{cfg.dataset}/{cfg.variant}/seed_{cfg.model_seed}", flush=True)
        if experiment in TRAINING_EXPERIMENTS:
            outcome = execute_training_run(cfg, force=force)
        elif (
            (experiment == "ex04" and cfg.variant in {"catch", "carots"})
            or (experiment == "ex03" and cfg.variant in {"mstgcnet", "tsae_uav"})
        ):
            outcome = execute_isolated_baseline(cfg, force=force)
        elif experiment in BASELINE_EXPERIMENTS:
            outcome = _baseline_placeholder(
                experiment, cfg.dataset, cfg.variant, cfg.model_seed, cfg.smoke
            )
        elif experiment == "ex08":
            source_cfg = make_config("ex01", cfg.dataset, "full", cfg.model_seed, smoke=cfg.smoke)
            outcome = execute_robustness_inference(cfg, source_cfg.run_dir, force=force)
        else:
            outcome = {"status": "analysis_requires_source_runs"}
        outcomes.append({**row, "outcome": outcome.get("status", "unknown")})
    return outcomes


def run_post_analyses(datasets: list[str], seed: int, smoke: bool) -> list[dict]:
    outcomes = []
    for dataset in datasets:
        cfg = make_config("ex01", dataset, "full", seed, smoke=smoke)
        if not (cfg.run_dir / "DONE.json").exists():
            outcomes.append({"dataset": dataset, "status": "missing_full_source"})
            continue
        legacy_cfg = cfg.to_legacy()
        aggregation = run_aggregation_suite(cfg.run_dir, cfg, legacy_cfg)
        nodes = json.loads((cfg.run_dir / "best.pt.sensor_names.json").read_text(encoding="utf-8")) if (cfg.run_dir / "best.pt.sensor_names.json").exists() else None
        if nodes is None:
            import torch
            checkpoint = torch.load(cfg.run_dir / "best.pt", map_location="cpu")
            nodes = checkpoint["sensor_names"]
        graph = analyze_graph_run(cfg.run_dir, nodes)
        outcomes.append({
            "dataset": dataset,
            "status": "complete",
            "aggregation_methods": len(aggregation),
            "graph_metric_rows": graph["metric_rows"],
        })
    return outcomes


def command_snapshot(_args) -> int:
    payload = create_snapshot()
    print(json.dumps({key: payload[key] for key in ("tracked_file_count", "dirty_status_entry_count", "git_head")}, indent=2))
    return 0


def command_doctor(_args) -> int:
    report = run_doctor()
    print(json.dumps({
        "status": report["status"],
        "python_files_parsed": report["python_files_parsed"],
        "datasets": report["datasets"],
        "variant_checks": len(report["variant_checks"]),
    }, ensure_ascii=False, indent=2))
    return 0


def command_verify(_args) -> int:
    print(json.dumps(verify_snapshot(strict=True), ensure_ascii=False, indent=2))
    return 0


def command_fetch(args) -> int:
    names = None if args.baselines == "all" else _csv_list(args.baselines)
    report = fetch_and_audit(names)
    print(json.dumps({name: row.get("audit_status") for name, row in report["baselines"].items()}, ensure_ascii=False, indent=2))
    return 0


def command_prepare_baseline_data(args) -> int:
    verify_snapshot()
    report = {}
    for dataset in _datasets(args.datasets):
        manifest = ensure_common_data(dataset, force=args.force)
        report[dataset] = {
            "status": "ready",
            "nodes": len(manifest["nodes"]),
            "train": len(manifest["train"]),
            "validation": len(manifest["validation"]),
            "failure": len(manifest["failure"]),
            "labels_exported": manifest["labels_exported"],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_audit_adapters(args) -> int:
    report = audit_adapter_runs(seed=args.seed)
    print(json.dumps({
        "status": report["status"],
        "completed_runs": report["completed_runs"],
        "failed_runs": report["failed_runs"],
        "legacy_integrity_ok": report["legacy_integrity"]["ok"],
    }, ensure_ascii=False, indent=2))
    return 0


def command_run(args) -> int:
    experiments = _csv_list(args.experiments) if args.experiments else []
    variants = _csv_list(args.variants) if args.variants else None
    datasets = _datasets(args.datasets)
    seeds = _seeds(args.seeds)
    rows = _task_rows(
        experiments,
        datasets,
        seeds,
        smoke=bool(args.smoke),
        variants=variants,
        preset=args.preset,
    )
    manifest = _write_manifest(rows, args.manifest_name)
    print(f"manifest={manifest} tasks={len(rows)}")
    if args.dry_run:
        return 0
    outcomes = execute_tasks(rows, force=args.force)
    _write_manifest(outcomes, args.manifest_name.replace(".csv", "_outcomes.csv"))
    return 0


def command_smoke(args) -> int:
    datasets = _datasets(args.datasets)
    core_rows = _task_rows([], datasets, [0], smoke=True, preset=args.preset)
    manifest = _write_manifest(core_rows, "smoke_manifest.csv")
    print(f"smoke_manifest={manifest} tasks={len(core_rows)}")
    outcomes = execute_tasks(core_rows, force=args.force)
    if not args.skip_analyses:
        analysis = run_post_analyses(datasets, seed=0, smoke=True)
        robustness_rows = _task_rows(["ex08"], datasets, [0], smoke=True)
        outcomes.extend(execute_tasks(robustness_rows, force=args.force))
        write_json(RESULTS_ROOT / "protocol_v1" / "smoke_analysis_outcomes.json", analysis)
    _write_manifest(outcomes, "smoke_outcomes.csv")
    print("smoke=complete")
    return 0


def command_summarize(args) -> int:
    protocol = load_protocol()
    experiments = _csv_list(args.experiments) if args.experiments else []
    variants = _csv_list(args.variants) if args.variants else None
    selection = resolve_experiment_selection(
        protocol,
        experiments=experiments,
        variants=variants,
        preset=args.preset,
    )
    preset_spec = protocol.get("experiment_presets", {}).get(args.preset or "", {})
    reference_spec = preset_spec.get(
        "reference", {"experiment": "ex01", "variant": "full"}
    )
    report = summarize_experiment_matrix(
        protocol_name=args.protocol,
        selection=selection,
        datasets=_datasets(args.datasets),
        seeds=_seeds(args.seeds),
        preset_name=args.preset or "custom",
        reference=(str(reference_spec["experiment"]), str(reference_spec["variant"])),
        n_resamples=int(args.bootstrap_resamples),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.require_complete and report["status"] != "complete" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated TCNGATRE revision experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot", help="Create the immutable legacy SHA-256 snapshot.").set_defaults(func=command_snapshot)
    sub.add_parser("doctor", help="Run Gate 0--2 checks.").set_defaults(func=command_doctor)
    sub.add_parser("verify-legacy", help="Verify the legacy snapshot.").set_defaults(func=command_verify)

    fetch = sub.add_parser("fetch-baselines", help="Fetch and audit official baseline repositories.")
    fetch.add_argument("--baselines", default="all")
    fetch.set_defaults(func=command_fetch)

    prepare_data = sub.add_parser(
        "prepare-baseline-data",
        help="Build canonical graphs and export validated label-free baseline data.",
    )
    prepare_data.add_argument("--datasets", default="all")
    prepare_data.add_argument("--force", action="store_true")
    prepare_data.set_defaults(func=command_prepare_baseline_data)

    adapter_audit = sub.add_parser("audit-adapters", help="Validate isolated CATCH/CAROTS result artifacts.")
    adapter_audit.add_argument("--seed", type=int, default=0)
    adapter_audit.set_defaults(func=command_audit_adapters)

    run = sub.add_parser("run", help="Generate and optionally execute a revision task matrix.")
    run.add_argument("--experiments")
    run.add_argument("--variants")
    run.add_argument("--preset")
    run.add_argument("--datasets", default="all")
    run.add_argument("--seeds", default="0,1,2,3,4")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--manifest-name", default="run_manifest.csv")
    run.set_defaults(func=command_run)

    smoke = sub.add_parser("smoke", help="Run one-epoch core experiments and inference analyses.")
    smoke.add_argument("--datasets", default="all")
    smoke.add_argument("--preset", default="core_ablation")
    smoke.add_argument("--force", action="store_true")
    smoke.add_argument("--skip-analyses", action="store_true")
    smoke.set_defaults(func=command_smoke)

    summary = sub.add_parser("summarize", help="Collect result tables without selecting a best seed.")
    summary.add_argument("--protocol", default="protocol_v1")
    summary.add_argument("--preset", default="core_ablation")
    summary.add_argument("--experiments")
    summary.add_argument("--variants")
    summary.add_argument("--datasets", default="all")
    summary.add_argument("--seeds", default="0,1,2,3,4")
    summary.add_argument("--bootstrap-resamples", type=int, default=10000)
    summary.add_argument("--require-complete", action="store_true")
    summary.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
