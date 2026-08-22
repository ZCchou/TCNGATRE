from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for path in (REPO_ROOT, PACKAGE_ROOT, REPO_ROOT / "TCNGATRE"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from revision_experiments.posthoc.constants import (  # noqa: E402
    DATASETS,
    EXPERIMENTS,
    MODEL_SEEDS,
    ROBUSTNESS_CONDITIONS,
    default_output_root,
)
from revision_experiments.posthoc.ex05 import run_ex05  # noqa: E402
from revision_experiments.posthoc.ex07 import run_ex07  # noqa: E402
from revision_experiments.posthoc.ex08 import run_ex08  # noqa: E402
from revision_experiments.posthoc.io import write_json  # noqa: E402
from revision_experiments.posthoc.source import (  # noqa: E402
    audit_source,
    resolve_source_run,
)
from revision_experiments.posthoc.summarize import summarize_posthoc  # noqa: E402


def _csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _selection(value: str, choices: tuple[str, ...], name: str) -> list[str]:
    values = list(choices) if str(value).strip().lower() == "all" else _csv_list(value)
    unknown = sorted(set(values) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown {name}: {unknown}")
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"Invalid or duplicate {name}: {values}")
    return values


def _seeds(value: str) -> list[int]:
    values = [int(item) for item in _csv_list(value)]
    if not values or any(seed < 0 for seed in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"Invalid or duplicate seeds: {values}")
    return values


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _common_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiments", default="all", help="ex05,ex07,ex08 or all")
    parser.add_argument("--datasets", default="all", help="alfa,gpsdata,simulate or all")
    parser.add_argument("--seeds", default=",".join(map(str, MODEL_SEEDS)))
    parser.add_argument(
        "--source-root",
        type=_path,
        default=(REPO_ROOT / "revision_results" / "protocol_v1" / "main_comparison").resolve(),
    )
    parser.add_argument("--output-root", type=_path, default=default_output_root(REPO_ROOT).resolve())


def _resolved(args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    return (
        _selection(args.experiments, EXPERIMENTS, "experiments"),
        _selection(args.datasets, DATASETS, "datasets"),
        _seeds(args.seeds),
    )


def _manifest_rows(
    experiments: list[str], datasets: list[str], seeds: list[int], smoke: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for dataset in datasets:
            for seed in seeds:
                if experiment == "ex08":
                    conditions = ROBUSTNESS_CONDITIONS[:1] if smoke else ROBUSTNESS_CONDITIONS
                    for condition in conditions:
                        rows.append({
                            "run_id": f"ex08/{dataset}/{condition}/seed_{seed}",
                            "experiment": experiment,
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "requires_training": False,
                        })
                else:
                    rows.append({
                        "run_id": f"{experiment}/{dataset}/seed_{seed}",
                        "experiment": experiment,
                        "dataset": dataset,
                        "seed": seed,
                        "condition": "",
                        "requires_training": False,
                    })
    return rows


def command_doctor(args: argparse.Namespace) -> int:
    _, datasets, seeds = _resolved(args)
    rows = []
    for dataset in datasets:
        for seed in seeds:
            try:
                rows.append(audit_source(resolve_source_run(args.source_root, dataset, seed)))
            except Exception as exc:
                rows.append({
                    "status": "failed", "dataset": dataset, "seed": seed,
                    "errors": [repr(exc)], "traceback": traceback.format_exc(),
                })
    passed = sum(row.get("status") == "passed" for row in rows)
    report = {
        "status": "passed" if passed == len(rows) else "failed",
        "source_root": str(args.source_root),
        "expected_runs": len(rows),
        "passed_runs": passed,
        "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "doctor_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def command_run(args: argparse.Namespace) -> int:
    experiments, datasets, seeds = _resolved(args)
    manifest = _manifest_rows(experiments, datasets, seeds, args.smoke)
    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / args.manifest_name
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"manifest={manifest_path} units={len(manifest)} training_units=0")
    if args.dry_run:
        for index, row in enumerate(manifest, 1):
            print(f"[{index}/{len(manifest)}] {row['run_id']}")
        return 0

    failures: list[dict[str, Any]] = []
    execution_count = len(experiments) * len(datasets) * len(seeds)
    current = 0
    for dataset in datasets:
        for seed in seeds:
            try:
                source = resolve_source_run(args.source_root, dataset, seed)
                audit = audit_source(source)
                if audit["status"] != "passed":
                    raise RuntimeError(json.dumps(audit, ensure_ascii=False))
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "experiment": "source", "error": repr(exc)})
                print(f"[FAILED] source/{dataset}/seed_{seed}: {exc}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    write_json(args.output_root / "run_failures.json", failures)
                    return 1
                continue
            for experiment in experiments:
                current += 1
                run_id = f"{experiment}/{dataset}/seed_{seed}"
                print(f"[{current}/{execution_count}] {run_id}", flush=True)
                try:
                    if experiment == "ex05":
                        outcome = run_ex05(source, args.output_root, force=args.force, smoke=args.smoke)
                    elif experiment == "ex07":
                        outcome = run_ex07(source, args.output_root, force=args.force, smoke=args.smoke)
                    else:
                        outcome = run_ex08(source, args.output_root, force=args.force, smoke=args.smoke)
                    print(f"[DONE] {run_id}: {outcome.get('status')}", flush=True)
                except KeyboardInterrupt:
                    print(f"[INTERRUPTED] {run_id}", file=sys.stderr, flush=True)
                    raise
                except Exception as exc:
                    failures.append({"dataset": dataset, "seed": seed, "experiment": experiment, "error": repr(exc)})
                    print(f"[FAILED] {run_id}: {exc}", file=sys.stderr, flush=True)
                    if not args.keep_going:
                        write_json(args.output_root / "run_failures.json", failures)
                        return 1
    write_json(args.output_root / "run_failures.json", failures)
    print(f"completed_execution_groups={execution_count - len(failures)} failures={len(failures)}")
    return 1 if failures else 0


def command_summarize(args: argparse.Namespace) -> int:
    experiments, datasets, seeds = _resolved(args)
    try:
        result = summarize_posthoc(
            args.output_root,
            experiments,
            datasets,
            seeds,
            require_complete=args.require_complete,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated, inference-only TCNGATRE posthoc experiments (EX-05/07/08)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Audit source checkpoints and fixed splits")
    _common_selection(doctor)
    doctor.set_defaults(func=command_doctor)

    run = subparsers.add_parser("run", help="Run inference-only posthoc experiments")
    _common_selection(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--manifest-name", default="posthoc_run_manifest.csv")
    run.set_defaults(func=command_run)

    summarize = subparsers.add_parser("summarize", help="Summarize posthoc results")
    _common_selection(summarize)
    summarize.add_argument("--require-complete", action="store_true")
    summarize.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
