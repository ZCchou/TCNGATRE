from __future__ import annotations

import argparse
import csv
import hashlib
import json
import locale
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REVISION_ROOT = ROOT / "revision_experiments"
DEFAULT_RESULT_ROOT = ROOT / "revision_results" / "protocol_v1" / "main_comparison"
DEFAULT_SMOKE_RESULT_ROOT = ROOT / "revision_results" / "protocol_v1" / "main_comparison_smoke"
DETERMINISTIC_ENTRYPOINT = REVISION_ROOT / "main_comparison" / "deterministic_entrypoint.py"
DEFAULT_TCNGATRE_GPS_BATCH_SIZE = 32
DEFAULT_TCNGATRE_ALFA_SAMPLE_STRIDE = 16

MODEL_ORDER = [
    "USAD",
    "Recurrent_AE",
    "TranAD",
    "OmniAnomaly",
    "BeatGAN",
    "TCNGATRE",
]
DATASET_ORDER = ["alfa", "simulate", "gpsdata"]
STAGE_ORDER = ["train", "infer", "eval"]

MODEL_SCRIPTS: dict[str, dict[str, str]] = {
    "USAD": {"train": "train_usad.py", "infer": "infer_usad.py"},
    "Recurrent_AE": {
        "train": "train_recurrent_ae.py",
        "infer": "infer_recurrent_ae.py",
        "eval": "eval_recurrent_ae.py",
    },
    "TranAD": {
        "train": "train_tranad.py",
        "infer": "infer_tranad.py",
        "eval": "eval_tranad.py",
    },
    "OmniAnomaly": {
        "train": "train_omni_anomaly.py",
        "infer": "infer_omni_anomaly.py",
        "eval": "eval_omni_anomaly.py",
    },
    "BeatGAN": {
        "train": "train_beatgan.py",
        "infer": "infer_beatgan.py",
        "eval": "eval_beatgan.py",
    },
    "TCNGATRE": {
        "train": "train_tcngatre.py",
        "infer": "infer_tcngatre.py",
        "eval": "eval_tcngatre.py",
    },
}

MODEL_ENV: dict[str, dict[str, str]] = {
    "USAD": {
        "seed": "UAV_USAD_SEED",
        "run_root": "UAV_USAD_RUN_ROOT",
        "epochs": "UAV_USAD_NUM_EPOCHS",
        "plot": "UAV_USAD_PLOT_SCORES",
        "plot_compare": "UAV_USAD_PLOT_COMPARE_TIMELINES",
    },
    "Recurrent_AE": {
        "seed": "UAV_RAE_SEED",
        "run_root": "UAV_RAE_RUN_ROOT",
        "epochs": "UAV_RAE_NUM_EPOCHS",
        "plot": "UAV_RAE_PLOT_SCORES",
        "plot_compare": "UAV_RAE_PLOT_COMPARE_TIMELINES",
    },
    "TranAD": {
        "seed": "UAV_TRANAD_SEED",
        "run_root": "UAV_TRANAD_RUN_ROOT",
        "epochs": "UAV_TRANAD_NUM_EPOCHS",
        "plot": "UAV_TRANAD_PLOT_SCORES",
        "plot_compare": "UAV_TRANAD_PLOT_COMPARE_TIMELINES",
    },
    "OmniAnomaly": {
        "seed": "UAV_OA_SEED",
        "run_root": "UAV_OA_RUN_ROOT",
        "epochs": "UAV_OA_NUM_EPOCHS",
        "plot": "UAV_OA_PLOT_SCORES",
        "plot_compare": "UAV_OA_PLOT_COMPARE_TIMELINES",
    },
    "BeatGAN": {
        "seed": "UAV_BEATGAN_SEED",
        "run_root": "UAV_BEATGAN_RUN_ROOT",
        "epochs": "UAV_BEATGAN_NUM_EPOCHS",
        "plot": "UAV_BEATGAN_PLOT_SCORES",
        "plot_compare": "UAV_BEATGAN_PLOT_COMPARE_TIMELINES",
    },
    "TCNGATRE": {
        # The legacy name is retained for compatibility. Dataset manifests fix
        # the train/validation flights, so this value controls model randomness.
        "seed": "UAV_TCNGATRE_SPLIT_SEED",
        "run_root": "UAV_TCNGATRE_RUN_ROOT",
        "epochs": "UAV_TCNGATRE_NUM_EPOCHS",
        "plot": "UAV_TCNGATRE_PLOT_SCORES",
    },
}

INFER_OUTPUT_NAMES = {
    "USAD": "infer_usad_global_threshold",
    "Recurrent_AE": "infer_recurrent_ae_failure",
    "TranAD": "infer_tranad_failure",
    "OmniAnomaly": "infer_future_window_failure",
    "BeatGAN": "infer_beatgan_failure",
    "TCNGATRE": "infer_tcngatre_failure",
}


def _tcngatre_data_protocol_signature(dataset: str) -> str:
    paths = (
        ROOT / "dataset" / str(dataset) / "dataset_manifest.json",
        ROOT / "TCNGATRE" / "data" / "alfa_shared.py",
        ROOT / "TCNGATRE" / "tcngatre_runtime.py",
        ROOT / "TCNGATRE" / "util" / "build_set_a_graph.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"TCNGATRE data protocol input is missing: {path}")
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    seed: int | None
    run_root: Path | None
    env_overrides: dict[str, str]
    stage_marker: Path | None
    smoke: bool
    determinism: str
    plots: bool

    @property
    def run_id(self) -> str:
        if self.seed is None:
            return f"{self.dataset}/{self.model}/legacy"
        return f"{self.dataset}/{self.model}/seed_{self.seed}"

    @property
    def data_protocol_signature(self) -> str | None:
        if self.model != "TCNGATRE":
            return None
        return _tcngatre_data_protocol_signature(self.dataset)

    @property
    def signature(self) -> str:
        semantic_env = {
            key: value for key, value in self.env_overrides.items()
            if key != "PYTHONUNBUFFERED"
        }
        payload = {
            "model": self.model,
            "dataset": self.dataset,
            "stage": self.stage,
            "seed": self.seed,
            "smoke": self.smoke,
            "determinism": self.determinism,
            "plots": self.plots,
            "command": self.command,
            "env_overrides": semantic_env,
        }
        if self.data_protocol_signature is not None:
            payload["data_protocol_signature"] = self.data_protocol_signature
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("workdir", "log_path", "run_root", "stage_marker"):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        payload["run_id"] = self.run_id
        payload["signature"] = self.signature
        payload["data_protocol_signature"] = self.data_protocol_signature
        return payload


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run all bundled models across one or more datasets."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch sub-scripts. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Subset of models to run.",
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
        help="Subset of stages to run. Unsupported stages for a model are skipped automatically.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Model seeds for isolated repeated runs, for example --seeds 0 1 2 3 4. "
            "If omitted, the original single-run behavior is preserved."
        ),
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help=(
            "Output root for seeded runs. Defaults to "
            "revision_results/protocol_v1/main_comparison (or main_comparison_smoke)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one training epoch and disable score plotting. Requires --seeds.",
    )
    parser.add_argument(
        "--determinism",
        choices=["seeded", "strict"],
        default="seeded",
        help=(
            "Randomness policy for seeded runs. 'seeded' fixes all RNG seeds while allowing "
            "optimized CUDA kernels; 'strict' additionally requests deterministic PyTorch kernels."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate native per-flight score plots in seeded formal runs. Disabled by default "
            "because plots do not affect Micro metrics and add substantial I/O."
        ),
    )
    parser.add_argument(
        "--tcngatre-gps-batch-size",
        type=int,
        default=DEFAULT_TCNGATRE_GPS_BATCH_SIZE,
        help=(
            "Physical batch size used only for seeded TCNGATRE runs on GPSData. "
            f"Defaults to {DEFAULT_TCNGATRE_GPS_BATCH_SIZE} to avoid OOM with its larger sensor graph."
        ),
    )
    parser.add_argument(
        "--tcngatre-alfa-sample-stride",
        type=int,
        default=DEFAULT_TCNGATRE_ALFA_SAMPLE_STRIDE,
        help=(
            "Window stride used only by seeded formal TCNGATRE runs on ALFA. "
            f"Defaults to {DEFAULT_TCNGATRE_ALFA_SAMPLE_STRIDE}; other datasets keep their "
            "native stride and smoke runs keep stride 64."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun seeded stages even when matching completion markers already exist.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later jobs even if one job fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved job list, without executing anything.",
    )
    parser.add_argument(
        "--log-root",
        default=str(ROOT / "batch_logs"),
        help=(
            "Directory used for legacy batch logs. In seeded mode, per-stage logs live "
            "inside each run and this directory only stores the batch summary."
        ),
    )
    args = parser.parse_args(argv)
    if args.seeds is None and (
        args.smoke
        or args.force
        or args.result_root is not None
        or args.plots
        or args.determinism != "seeded"
    ):
        parser.error("--smoke, --force, --result-root, --plots and --determinism require --seeds")
    if args.seeds is not None and any(int(seed) < 0 for seed in args.seeds):
        parser.error("--seeds values must be non-negative integers")
    if int(args.tcngatre_gps_batch_size) < 1:
        parser.error("--tcngatre-gps-batch-size must be a positive integer")
    if int(args.tcngatre_alfa_sample_stride) < 1:
        parser.error("--tcngatre-alfa-sample-stride must be a positive integer")
    return args


def normalize_ordered(unique_values: list[Any], canonical_order: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for item in canonical_order:
        if item in unique_values and item not in seen:
            ordered.append(item)
            seen.add(item)
    for item in unique_values:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _unique_seeds(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        seed = int(value)
        if seed not in seen:
            result.append(seed)
            seen.add(seed)
    return result


def resolve_result_root(args) -> Path | None:
    if args.seeds is None:
        return None
    if args.result_root:
        return Path(args.result_root).expanduser().resolve()
    return (DEFAULT_SMOKE_RESULT_ROOT if args.smoke else DEFAULT_RESULT_ROOT).resolve()


def _seeded_env(
    model: str,
    dataset: str,
    seed: int,
    run_root: Path,
    result_root: Path,
    smoke: bool,
    determinism: str,
    plots: bool,
    tcngatre_gps_batch_size: int,
    tcngatre_alfa_sample_stride: int,
) -> dict[str, str]:
    names = MODEL_ENV[model]
    env = {
        names["seed"]: str(seed),
        names["run_root"]: str(run_root),
        "PYTHONHASHSEED": str(seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if determinism == "strict":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if model == "TCNGATRE":
        graph_root = result_root / "_shared" / "tcngatre_graph" / dataset
        env["UAV_TCNGATRE_GRAPH_DIR"] = str(graph_root)
        if smoke:
            env["UAV_TCNGATRE_SAMPLE_STRIDE"] = "64"
        elif dataset == "alfa":
            env["UAV_TCNGATRE_SAMPLE_STRIDE"] = str(int(tcngatre_alfa_sample_stride))
        if dataset == "gpsdata":
            env["UAV_TCNGATRE_BATCH_SIZE"] = str(int(tcngatre_gps_batch_size))
    plot_value = "1" if plots and not smoke else "0"
    env[names["plot"]] = plot_value
    if "plot_compare" in names:
        env[names["plot_compare"]] = plot_value
    if smoke:
        env[names["epochs"]] = "1"
    return env


def build_job_specs(args) -> tuple[list[JobSpec], Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = resolve_result_root(args)
    if result_root is None:
        batch_root = Path(args.log_root) / f"batch_{timestamp}"
    else:
        batch_root = result_root / "_batches" / f"batch_{timestamp}"

    jobs: list[JobSpec] = []
    job_index = 0
    models = normalize_ordered(list(args.models), MODEL_ORDER)
    datasets = normalize_ordered(list(args.datasets), DATASET_ORDER)
    stages = normalize_ordered(list(args.stages), STAGE_ORDER)
    seeds: list[int | None] = [None] if args.seeds is None else _unique_seeds(list(args.seeds))

    for dataset in datasets:
        for model in models:
            workdir = ROOT / model
            script_map = MODEL_SCRIPTS[model]
            for seed in seeds:
                run_root = None if result_root is None else result_root / dataset / model / f"seed_{seed}"
                env_overrides = (
                    {}
                    if run_root is None or seed is None
                    else _seeded_env(
                        model,
                        dataset,
                        seed,
                        run_root,
                        result_root,
                        bool(args.smoke),
                        str(args.determinism),
                        bool(args.plots),
                        int(args.tcngatre_gps_batch_size),
                        int(args.tcngatre_alfa_sample_stride),
                    )
                )
                for stage in stages:
                    script_name = script_map.get(stage)
                    if script_name is None:
                        continue
                    job_index += 1
                    if run_root is None or seed is None:
                        command = [str(args.python), script_name, "--dataset", dataset]
                        log_path = batch_root / f"{job_index:02d}__{model}__{dataset}__{stage}.log"
                        stage_marker = None
                    else:
                        command = [
                            str(args.python),
                            str(DETERMINISTIC_ENTRYPOINT),
                            "--seed",
                            str(seed),
                            "--determinism",
                            str(args.determinism),
                            "--script",
                            str(workdir / script_name),
                            "--",
                            "--dataset",
                            dataset,
                        ]
                        log_path = run_root / "logs" / f"{stage}.log"
                        stage_marker = run_root / "status" / f"{stage}.json"
                    jobs.append(
                        JobSpec(
                            index=job_index,
                            model=model,
                            dataset=dataset,
                            stage=stage,
                            script_name=script_name,
                            workdir=workdir,
                            command=command,
                            log_path=log_path,
                            seed=seed,
                            run_root=run_root,
                            env_overrides=dict(env_overrides),
                            stage_marker=stage_marker,
                            smoke=bool(args.smoke),
                            determinism=str(args.determinism),
                            plots=bool(args.plots),
                        )
                    )
    return jobs, batch_root


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _artifact_paths(job: JobSpec) -> list[Path]:
    if job.run_root is None:
        return []
    if job.stage == "train":
        return [
            job.run_root / "best.pt",
            job.run_root / "last.pt",
            job.run_root / "config.json",
            job.run_root / "val_normal_scores.csv",
        ]
    infer_root = job.run_root / INFER_OUTPUT_NAMES[job.model]
    if job.stage == "infer":
        paths = [infer_root / "sequence_scores.csv"]
        if job.model == "USAD":
            paths.append(infer_root / "score_threshold_analysis" / "summary_metrics.csv")
        return paths
    if job.stage == "eval":
        return [infer_root / "score_threshold_analysis" / "summary_metrics.csv"]
    return []


def _missing_artifacts(job: JobSpec) -> list[str]:
    return [str(path) for path in _artifact_paths(job) if not path.is_file()]


def _stage_is_complete(job: JobSpec) -> bool:
    if job.stage_marker is None or not job.stage_marker.is_file():
        return False
    try:
        payload = json.loads(job.stage_marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        payload.get("status") == "ok"
        and payload.get("signature") == job.signature
        and not _missing_artifacts(job)
    )


def _write_stage_marker(
    job: JobSpec,
    status: str,
    return_code: int,
    elapsed_sec: float,
    missing_artifacts: list[str] | None = None,
) -> None:
    if job.stage_marker is None:
        return
    _write_json(
        job.stage_marker,
        {
            "status": status,
            "return_code": int(return_code),
            "elapsed_sec": float(elapsed_sec),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "signature": job.signature,
            "run_id": job.run_id,
            "stage": job.stage,
            "command": job.command,
            "env_overrides": job.env_overrides,
            "missing_artifacts": list(missing_artifacts or []),
            "log_path": str(job.log_path),
        },
    )


def _write_console(value: str) -> None:
    try:
        sys.stdout.write(value)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or locale.getpreferredencoding(False) or "utf-8"
        safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_value)


def stream_process(job: JobSpec, force: bool = False) -> tuple[int, float, str]:
    if job.seed is not None and not force and _stage_is_complete(job):
        print(f"[SKIP] {job.run_id} | {job.stage} | matching completion marker")
        return 0, 0.0, "skipped"

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if job.run_root is not None:
        job.run_root.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    seed_text = "" if job.seed is None else f" | seed={job.seed}"
    print(
        f"\n[{job.index:03d}] {job.model} | {job.dataset} | {job.stage}{seed_text}\n"
        f"cwd={job.workdir}\ncmd={' '.join(job.command)}\nlog={job.log_path}"
    )
    env = os.environ.copy()
    env.update(job.env_overrides)
    missing: list[str] = []
    with job.log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write(f"cwd={job.workdir}\ncmd={' '.join(job.command)}\n")
        if job.env_overrides:
            log_file.write(
                "env_overrides=" + json.dumps(job.env_overrides, ensure_ascii=False, sort_keys=True) + "\n"
            )
        log_file.write("\n")
        process = subprocess.Popen(
            job.command, cwd=str(job.workdir), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            encoding=locale.getpreferredencoding(False) or "utf-8", errors="replace",
            bufsize=1, env=env,
        )
        assert process.stdout is not None
        prefix = f"[{job.model}/{job.dataset}/{job.stage}] "
        for line in process.stdout:
            _write_console(prefix + line)
            log_file.write(line)
        return_code = int(process.wait())
        missing = _missing_artifacts(job) if return_code == 0 else []
        if missing:
            return_code = 3
            message = "Expected artifacts missing after a successful process:\n" + "\n".join(missing)
            _write_console(prefix + message + "\n")
            log_file.write("\n" + message + "\n")

    elapsed_sec = time.time() - start_time
    status = "ok" if return_code == 0 else "failed"
    _write_stage_marker(job, status, return_code, elapsed_sec, missing_artifacts=missing)
    if return_code != 0 and job.run_root is not None:
        _write_json(
            job.run_root / "FAILED.json",
            {
                "status": "failed", "run_id": job.run_id, "failed_stage": job.stage,
                "return_code": return_code, "log_path": str(job.log_path),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    print(
        f"[DONE] {job.model} | {job.dataset} | {job.stage}{seed_text} | "
        f"return_code={return_code} | elapsed_sec={elapsed_sec:.1f}"
    )
    return return_code, elapsed_sec, status


def _run_rows(jobs: list[JobSpec]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for job in jobs:
        if job.seed is None or job.run_root is None or job.run_id in seen:
            continue
        seen.add(job.run_id)
        rows.append(
            {
                "run_id": job.run_id,
                "dataset": job.dataset,
                "model": job.model,
                "seed": int(job.seed),
                "smoke": bool(job.smoke),
                "determinism": job.determinism,
                "plots": bool(job.plots),
                "data_protocol_signature": job.data_protocol_signature,
                "sample_stride": (
                    int(job.env_overrides.get("UAV_TCNGATRE_SAMPLE_STRIDE", "4"))
                    if job.model == "TCNGATRE" else None
                ),
                "batch_size": (
                    int(job.env_overrides.get("UAV_TCNGATRE_BATCH_SIZE", "128"))
                    if job.model == "TCNGATRE" else None
                ),
                "run_root": str(job.run_root),
                "stages": ",".join(MODEL_SCRIPTS[job.model]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else ["run_id"])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_provenance(run_rows: list[dict], jobs: list[JobSpec], force: bool) -> None:
    grouped: dict[str, list[JobSpec]] = {}
    for job in jobs:
        grouped.setdefault(job.run_id, []).append(job)
    common = {
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "data_split_policy": "fixed dataset manifest; independent of model seed",
        "data_split_seed_protocol": 64,
    }
    for row in run_rows:
        run_jobs = grouped[row["run_id"]]
        run_root = Path(row["run_root"])
        provenance_path = run_root / "provenance.json"
        desired_stage_signatures = {job.stage: job.signature for job in run_jobs}
        if provenance_path.exists() and not force:
            try:
                existing = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                existing = {}
            if (
                existing.get("data_protocol_signature") == row.get("data_protocol_signature")
                and existing.get("stage_signatures") == desired_stage_signatures
            ):
                continue
        first = run_jobs[0]
        payload = {
            **common,
            **row,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "env_overrides": first.env_overrides,
            "commands": {job.stage: job.command for job in run_jobs},
            "stage_signatures": desired_stage_signatures,
            "randomness_policy": (
                "fixed Python/NumPy/PyTorch/CUDA seeds; optimized CUDA kernels allowed"
                if row["determinism"] == "seeded"
                else "fixed seeds; deterministic cuDNN and PyTorch algorithms requested in warn-only mode"
            ),
            "tcngatre_seed_compatibility_note": (
                "UAV_TCNGATRE_SPLIT_SEED controls model randomness; flight splits are fixed by the manifest."
                if row["model"] == "TCNGATRE" else None
            ),
        }
        _write_json(provenance_path, payload)


def _prepare_seeded_outputs(jobs: list[JobSpec], batch_root: Path, force: bool) -> list[dict]:
    run_rows = _run_rows(jobs)
    if not run_rows:
        return []
    result_root = Path(run_rows[0]["run_root"]).parents[2]
    jobs_by_run: dict[str, list[JobSpec]] = {}
    for job in jobs:
        jobs_by_run.setdefault(job.run_id, []).append(job)
    if force:
        for row in run_rows:
            run_root = Path(row["run_root"])
            for name in ("DONE.json", "FAILED.json", "PARTIAL.json"):
                marker = run_root / name
                if marker.is_file():
                    marker.unlink()
    else:
        for row in run_rows:
            selected_jobs = jobs_by_run[row["run_id"]]
            if all(_stage_is_complete(job) for job in selected_jobs):
                continue
            stale_done = Path(row["run_root"]) / "DONE.json"
            if stale_done.is_file():
                stale_done.unlink()
    _write_csv(result_root / "run_manifest.csv", run_rows)
    _write_csv(batch_root / "run_manifest.csv", run_rows)
    _write_provenance(run_rows, jobs, force=force)
    return run_rows


def _finalize_seeded_runs(jobs: list[JobSpec]) -> None:
    grouped: dict[str, list[JobSpec]] = {}
    for job in jobs:
        if job.seed is not None and job.run_root is not None:
            grouped.setdefault(job.run_id, []).append(job)
    for run_id, selected_jobs in grouped.items():
        template = selected_jobs[0]
        run_root = template.run_root
        assert run_root is not None
        status_by_stage: dict[str, str] = {}
        all_complete = True
        for stage in MODEL_SCRIPTS[template.model]:
            matching = [job for job in selected_jobs if job.stage == stage]
            if not matching:
                status_by_stage[stage] = "not_requested"
                all_complete = False
                continue
            complete = _stage_is_complete(matching[0])
            status_by_stage[stage] = "ok" if complete else "missing_or_failed"
            all_complete = all_complete and complete
        payload = {
            "status": "complete" if all_complete else "partial",
            "run_id": run_id, "dataset": template.dataset, "model": template.model,
            "model_seed": template.seed, "data_split_policy": "fixed dataset manifest",
            "data_protocol_signature": template.data_protocol_signature,
            "sample_stride": (
                int(template.env_overrides.get("UAV_TCNGATRE_SAMPLE_STRIDE", "4"))
                if template.model == "TCNGATRE" else None
            ),
            "batch_size": (
                int(template.env_overrides.get("UAV_TCNGATRE_BATCH_SIZE", "128"))
                if template.model == "TCNGATRE" else None
            ),
            "stages": status_by_stage,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if all_complete:
            _write_json(run_root / "DONE.json", payload)
            for name in ("FAILED.json", "PARTIAL.json"):
                path = run_root / name
                if path.is_file():
                    path.unlink()
        elif not (run_root / "FAILED.json").exists():
            _write_json(run_root / "PARTIAL.json", payload)


def _summarize_seeded(result_root: Path, run_rows: list[dict]) -> dict:
    from revision_experiments.analysis.main_comparison import summarize_main_comparison

    return summarize_main_comparison(result_root=result_root, expected_runs=run_rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs, batch_root = build_job_specs(args)
    if not jobs:
        print("No runnable jobs matched the requested model/dataset/stage selection.")
        return 0

    run_rows = _run_rows(jobs)
    print(f"[BATCH] root={ROOT}")
    print(f"[BATCH] python={args.python}")
    print(f"[BATCH] log_root={batch_root}")
    if args.seeds is not None:
        print(f"[BATCH] isolated_runs={len(run_rows)}")
        print(f"[BATCH] seeds={_unique_seeds(list(args.seeds))}")
        print(f"[BATCH] result_root={resolve_result_root(args)}")
        if "TCNGATRE" in args.models:
            resolved_stride = 64 if args.smoke else int(args.tcngatre_alfa_sample_stride)
            print(f"[BATCH] tcngatre_alfa_sample_stride={resolved_stride}")
            if "gpsdata" in args.datasets:
                print(f"[BATCH] tcngatre_gps_batch_size={int(args.tcngatre_gps_batch_size)}")
    print(f"[BATCH] stage_jobs={len(jobs)}")
    for job in jobs:
        seed_text = "" if job.seed is None else f" | seed={job.seed}"
        print(f"  - [{job.index:03d}] {job.model} | {job.dataset} | {job.stage}{seed_text}")
    if args.dry_run:
        return 0

    if args.seeds is not None:
        from revision_experiments.core.integrity import verify_snapshot

        integrity = verify_snapshot()
        print(
            f"[INTEGRITY] ok={integrity['ok']} checked={integrity['checked']} "
            f"approved_changes={len(integrity.get('approved_changes', []))}"
        )

    batch_root.mkdir(parents=True, exist_ok=True)
    _write_json(batch_root / "planned_jobs.json", [job.to_dict() for job in jobs])
    if args.seeds is not None:
        run_rows = _prepare_seeded_outputs(jobs, batch_root, force=bool(args.force))

    summary_records: list[dict] = []
    failed = False
    for job in jobs:
        return_code, elapsed_sec, execution_status = stream_process(job, force=bool(args.force))
        record = job.to_dict()
        record["elapsed_sec"] = float(elapsed_sec)
        record["return_code"] = int(return_code)
        record["status"] = execution_status
        summary_records.append(record)
        _write_json(batch_root / "summary.json", summary_records)
        if return_code != 0:
            failed = True
            print(f"[FAIL] {job.model} | {job.dataset} | {job.stage} | seed={job.seed}")
            if not args.keep_going:
                break

    if args.seeds is not None:
        _finalize_seeded_runs(jobs)
        result_root = resolve_result_root(args)
        assert result_root is not None
        summary_payload = _summarize_seeded(result_root, run_rows)
        print(
            f"[SUMMARY] complete_runs={summary_payload['complete_runs']} "
            f"expected_runs={summary_payload['expected_runs']} "
            f"missing_runs={summary_payload['missing_runs']}"
        )
        integrity = verify_snapshot()
        print(
            f"[INTEGRITY] final_ok={integrity['ok']} "
            f"approved_changes={len(integrity.get('approved_changes', []))}"
        )

    status_payload = {
        "root": str(ROOT), "python": str(args.python),
        "runs_total": int(len(run_rows)) if args.seeds is not None else None,
        "jobs_total": int(len(jobs)), "jobs_finished": int(len(summary_records)),
        "jobs_failed": int(sum(1 for row in summary_records if int(row["return_code"]) != 0)),
        "summary_path": str(batch_root / "summary.json"), "batch_root": str(batch_root),
    }
    _write_json(batch_root / "batch_status.json", status_payload)
    print(f"[BATCH] summary={batch_root / 'summary.json'}")
    print(f"[BATCH] status={batch_root / 'batch_status.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
