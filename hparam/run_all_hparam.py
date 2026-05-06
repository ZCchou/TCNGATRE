from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


HPARAM_ROOT = Path(__file__).resolve().parent
MODEL_ORDER = [
    "H1_Default",
    "H2_Small",
    "H3_Large",
    "H4_LongWindow",
    "H5_DenseGraph",
    "H6_Conservative",
]
DATASET_ORDER = ["alfa", "alfa4hz", "simulate", "gpsdata"]
STAGE_ORDER = ["train", "infer", "eval"]


@dataclass
class JobSpec:
    index: int
    model: str
    dataset: str
    stage: str
    script_name: str
    workdir: Path
    command: list[str]
    log_path: Path

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["workdir"] = str(self.workdir)
        payload["log_path"] = str(self.log_path)
        return payload


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run all TCNGATRE hyperparameter variants across one or more datasets."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Subset of hparam configs to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Subset of datasets to run.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        help="Subset of stages to run.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later jobs even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved job list without executing.",
    )
    parser.add_argument(
        "--log-root",
        default=str(HPARAM_ROOT / "batch_logs"),
        help="Directory for stdout/stderr logs and batch summary.",
    )
    return parser.parse_args(argv)


def normalize_ordered(unique_values: list[str], canonical_order: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in canonical_order:
        if item in unique_values and item not in seen:
            ordered.append(item)
            seen.add(item)
    for item in unique_values:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def build_job_specs(args) -> tuple[list[JobSpec], Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = Path(args.log_root) / f"batch_{timestamp}"
    jobs: list[JobSpec] = []
    job_index = 0

    models = normalize_ordered(list(args.models), MODEL_ORDER)
    datasets = normalize_ordered(list(args.datasets), DATASET_ORDER)
    stages = normalize_ordered(list(args.stages), STAGE_ORDER)

    for dataset in datasets:
        for model in models:
            workdir = HPARAM_ROOT / model
            for stage in stages:
                script_name = f"{stage}.py"
                job_index += 1
                log_path = batch_root / f"{job_index:02d}__{model}__{dataset}__{stage}.log"
                jobs.append(
                    JobSpec(
                        index=job_index,
                        model=model,
                        dataset=dataset,
                        stage=stage,
                        script_name=script_name,
                        workdir=workdir,
                        command=[str(args.python), script_name, "--dataset", dataset],
                        log_path=log_path,
                    )
                )
    return jobs, batch_root


def stream_process(job: JobSpec) -> tuple[int, float]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    print(
        f"\n[{job.index:02d}] {job.model} | {job.dataset} | {job.stage}\n"
        f"cwd={job.workdir}\n"
        f"cmd={' '.join(job.command)}\n"
        f"log={job.log_path}"
    )
    with job.log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write(f"cwd={job.workdir}\n")
        log_file.write(f"cmd={' '.join(job.command)}\n\n")
        _passthrough = {"UAV_TCNGATRE_BATCH_SIZE"}
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("UAV_TCNGATRE_") or k in _passthrough}
        process = subprocess.Popen(
            job.command,
            cwd=str(job.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=clean_env,
        )
        assert process.stdout is not None
        prefix = f"[{job.model}/{job.dataset}/{job.stage}] "
        for line in process.stdout:
            sys.stdout.write(prefix + line)
            log_file.write(line)
        return_code = process.wait()
    elapsed_sec = time.time() - start_time
    print(
        f"[DONE] {job.model} | {job.dataset} | {job.stage} | "
        f"return_code={return_code} | elapsed_sec={elapsed_sec:.1f}"
    )
    return return_code, elapsed_sec


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs, batch_root = build_job_specs(args)

    if len(jobs) <= 0:
        print("No runnable jobs matched the requested model/dataset/stage selection.")
        return 0

    summary_records: list[dict] = []
    print(f"[BATCH] hparam_root={HPARAM_ROOT}")
    print(f"[BATCH] python={args.python}")
    print(f"[BATCH] log_root={batch_root}")
    print(f"[BATCH] jobs={len(jobs)}")
    for job in jobs:
        print(f"  - [{job.index:02d}] {job.model} | {job.dataset} | {job.stage}")

    if args.dry_run:
        return 0

    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "planned_jobs.json").write_text(
        json.dumps([job.to_dict() for job in jobs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failed = False
    for job in jobs:
        return_code, elapsed_sec = stream_process(job)
        record = job.to_dict()
        record["elapsed_sec"] = float(elapsed_sec)
        record["return_code"] = int(return_code)
        record["status"] = "ok" if return_code == 0 else "failed"
        summary_records.append(record)
        (batch_root / "summary.json").write_text(
            json.dumps(summary_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if return_code != 0:
            failed = True
            print(f"[FAIL] stopping on first failure: {job.model} | {job.dataset} | {job.stage}")
            if not args.keep_going:
                break

    status_payload = {
        "hparam_root": str(HPARAM_ROOT),
        "python": str(args.python),
        "jobs_total": int(len(jobs)),
        "jobs_finished": int(len(summary_records)),
        "jobs_failed": int(sum(1 for row in summary_records if int(row["return_code"]) != 0)),
        "summary_path": str(batch_root / "summary.json"),
        "batch_root": str(batch_root),
    }
    (batch_root / "batch_status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[BATCH] summary={batch_root / 'summary.json'}")
    print(f"[BATCH] status={batch_root / 'batch_status.json'}")
    return 1 if failed and not args.keep_going else 0


if __name__ == "__main__":
    raise SystemExit(main())
