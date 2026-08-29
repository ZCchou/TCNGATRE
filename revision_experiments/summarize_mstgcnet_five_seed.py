from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


METRICS = (
    "precision",
    "recall",
    "f1",
    "accuracy",
    "fpr",
    "auroc",
    "average_precision",
)
COUNTS = ("num_samples", "positives", "negatives", "tp", "fp", "tn", "fn")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recomputed_metrics(row: dict) -> dict[str, float]:
    tp, fp, tn, fn = (int(row[name]) for name in ("tp", "fp", "tn", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, float.fromhex("0x1p-52"))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "fpr": fp / max(fp + tn, 1),
    }


def _collect_seed(run_root: Path, dataset: str, seed: int) -> tuple[dict, dict]:
    run_dir = run_root / f"seed_{seed}"
    done_path = run_dir / "DONE.json"
    metric_path = run_dir / "native_evaluation" / "primary_metrics.json"
    config_path = run_dir / "config_resolved.json"
    checkpoint_path = run_dir / "best.pt"
    required = (done_path, metric_path, config_path, checkpoint_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {}, {
            "dataset": dataset,
            "model": "MSTGCNet",
            "seed": int(seed),
            "status": "missing",
            "missing": " | ".join(missing),
            "run_dir": str(run_dir.resolve()),
        }

    done = _read_json(done_path)
    metrics = _read_json(metric_path)
    config = _read_json(config_path)
    parameters = dict(config.get("baseline_parameters", {}))
    profile = str(parameters.get("parameter_profile", ""))
    errors: list[str] = []
    if done.get("status") != "complete":
        errors.append(f"DONE.status={done.get('status')!r}")
    if profile != "paper_faithful":
        errors.append(f"parameter_profile={profile!r}")
    for name in (*COUNTS, *METRICS):
        if name not in metrics:
            errors.append(f"metric_missing={name}")
        elif name in METRICS and not math.isfinite(float(metrics[name])):
            errors.append(f"metric_non_finite={name}")
    if not errors:
        recomputed = _recomputed_metrics(metrics)
        for name, expected in recomputed.items():
            if abs(float(metrics[name]) - expected) > 1e-12:
                errors.append(
                    f"metric_mismatch={name}:{float(metrics[name]):.16g}!={expected:.16g}"
                )

    history_path = run_dir / "history.csv"
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    row = {
        "run_id": f"mstgcnet/{dataset}/seed_{seed}",
        "dataset": dataset,
        "model": "MSTGCNet",
        "seed": int(seed),
        "status": "complete" if not errors else "invalid",
        "parameter_profile": profile,
        "window": parameters.get("window"),
        "train_stride": parameters.get("train_stride"),
        "validation_stride": parameters.get("validation_stride"),
        "score_stride": parameters.get("score_stride"),
        "epochs_configured": parameters.get("epochs"),
        "epochs_completed": int(len(history)),
        "early_stop_patience": parameters.get("early_stop_patience"),
        "effective_batch_size": parameters.get("effective_batch_size"),
        "physical_batch_size": parameters.get("physical_batch_size"),
        "gradient_accumulation_steps": parameters.get("gradient_accumulation_steps"),
        "best_checkpoint_sha256": _sha256(checkpoint_path),
        "evaluation_method": done.get(
            "paper_evaluation_method", "ATSSD + label-dependent point adjustment"
        ),
        **{name: int(metrics[name]) for name in COUNTS if name in metrics},
        **{name: float(metrics[name]) for name in METRICS if name in metrics},
        "run_dir": str(run_dir.resolve()),
    }
    status = {
        "dataset": dataset,
        "model": "MSTGCNet",
        "seed": int(seed),
        "status": row["status"],
        "errors": " | ".join(errors),
        "run_dir": str(run_dir.resolve()),
    }
    return row, status


def summarize(
    *,
    run_root: Path,
    output_dir: Path,
    dataset: str,
    seeds: list[int],
    require_complete: bool,
) -> dict:
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, statuses = [], []
    for seed in seeds:
        row, status = _collect_seed(run_root, dataset, int(seed))
        statuses.append(status)
        if row:
            rows.append(row)

    all_runs = pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()
    status_frame = pd.DataFrame(statuses).sort_values("seed")
    complete = (
        len(all_runs) == len(seeds)
        and not status_frame.empty
        and status_frame["status"].eq("complete").all()
    )

    summary_rows: list[dict] = []
    paper_rows: list[dict] = []
    if not all_runs.empty:
        summary = {
            "dataset": dataset,
            "model": "MSTGCNet",
            "seed_count": int(all_runs["seed"].nunique()),
            "seeds": ",".join(str(int(value)) for value in sorted(all_runs["seed"])),
        }
        paper = {"dataset": dataset, "model": "MSTGCNet"}
        for metric in METRICS:
            values = pd.to_numeric(all_runs[metric], errors="raise")
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sample_sd"] = sample_sd
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
            paper[metric] = f"{mean:.4f} ± {sample_sd:.4f}"
        summary_rows.append(summary)
        paper_rows.append(paper)

    all_runs_path = output_dir / "mstgcnet_five_seed_all_runs.csv"
    summary_path = output_dir / "mstgcnet_five_seed_summary.csv"
    paper_path = output_dir / "mstgcnet_five_seed_paper_table.csv"
    status_path = output_dir / "run_status.csv"
    all_runs.to_csv(all_runs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(paper_rows).to_csv(paper_path, index=False, encoding="utf-8-sig")
    status_frame.to_csv(status_path, index=False, encoding="utf-8-sig")

    tex_path = output_dir / "mstgcnet_five_seed_paper_table.tex"
    if paper_rows:
        ordered = ("precision", "recall", "f1", "accuracy", "fpr", "auroc", "average_precision")
        values = " & ".join(paper_rows[0][name].replace("±", r"$\pm$") for name in ordered)
        tex = (
            r"MSTGCNet & " + values + r" \\" + "\n"
        )
    else:
        tex = ""
    tex_path.write_text(tex, encoding="utf-8")

    payload = {
        "status": "complete" if complete else "incomplete",
        "dataset": dataset,
        "model": "MSTGCNet",
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": (
            [int(value) for value in all_runs.loc[all_runs["status"] == "complete", "seed"]]
            if not all_runs.empty else []
        ),
        "primary_metric_source": "native_evaluation/primary_metrics.json",
        "aggregation": "mean and sample standard deviation over model seeds",
        "metric_recalculation_tolerance": 1e-12,
        "outputs": {
            "all_runs": str(all_runs_path.resolve()),
            "seed_summary": str(summary_path.resolve()),
            "paper_table_csv": str(paper_path.resolve()),
            "paper_table_tex": str(tex_path.resolve()),
            "run_status": str(status_path.resolve()),
        },
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if require_complete and not complete:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize formal MSTGCNet model seeds."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("revision_results/protocol_v1/ex03/alfa/mstgcnet"),
    )
    parser.add_argument("--dataset", default="alfa")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "summary_paper_faithful_5seed"
    )
    try:
        result = summarize(
            run_root=run_root,
            output_dir=output_dir,
            dataset=str(args.dataset),
            seeds=[int(seed) for seed in args.seeds],
            require_complete=bool(args.require_complete),
        )
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
