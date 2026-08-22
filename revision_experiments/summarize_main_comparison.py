from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revision_experiments.analysis.main_comparison import (  # noqa: E402
    INFER_OUTPUT_NAMES,
    summarize_main_comparison,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "revision_results" / "protocol_v1" / "main_comparison"
MODEL_ORDER = list(INFER_OUTPUT_NAMES)
DATASET_ORDER = ["alfa", "simulate", "gpsdata"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
TCNGATRE_FORMAL_PROFILE = {
    "alfa": {"sample_stride": 16, "batch_size": 128},
    "simulate": {"sample_stride": 4, "batch_size": 128},
    "gpsdata": {"sample_stride": 4, "batch_size": 32},
}


def _tcngatre_data_protocol_signature(dataset: str) -> str:
    paths = (
        REPO_ROOT / "dataset" / dataset / "dataset_manifest.json",
        REPO_ROOT / "TCNGATRE" / "data" / "alfa_shared.py",
        REPO_ROOT / "TCNGATRE" / "tcngatre_runtime.py",
        REPO_ROOT / "TCNGATRE" / "util" / "build_set_a_graph.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"TCNGATRE data protocol input is missing: {path}")
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_expected_runs(
    result_root: Path,
    models: list[str],
    datasets: list[str],
    seeds: list[int],
) -> list[dict]:
    root = Path(result_root).expanduser().resolve()
    protocol_signatures = (
        {dataset: _tcngatre_data_protocol_signature(dataset) for dataset in datasets}
        if "TCNGATRE" in models else {}
    )
    rows: list[dict] = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                row = {
                    "run_id": f"{dataset}/{model}/seed_{int(seed)}",
                    "dataset": dataset,
                    "model": model,
                    "seed": int(seed),
                    "smoke": False,
                    "run_root": str(root / dataset / model / f"seed_{int(seed)}"),
                }
                if model == "TCNGATRE":
                    row.update(TCNGATRE_FORMAL_PROFILE[dataset])
                    row["data_protocol_signature"] = protocol_signatures[dataset]
                rows.append(row)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize existing main_comparison runs using only the Micro "
            "SPOT + label_any result."
        )
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return exit code 2 unless every requested model/dataset/seed cell is valid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("All seeds must be non-negative")
    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("Duplicate seeds are not allowed")
    result_root = Path(args.result_root).expanduser().resolve()
    expected = build_expected_runs(
        result_root=result_root,
        models=list(args.models),
        datasets=list(args.datasets),
        seeds=list(args.seeds),
    )
    report = summarize_main_comparison(result_root, expected)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "complete":
        print(f"[OK] Micro five-seed summary: {report['seed_summary_csv']}")
    else:
        print(f"[INCOMPLETE] Missing/stale cells: {report['missing_cells_csv']}")
    return 2 if args.require_complete and report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
