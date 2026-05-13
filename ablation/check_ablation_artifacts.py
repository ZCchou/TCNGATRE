from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ABLATION_ROOT = Path(__file__).resolve().parent

VARIANT_SPECS = {
    "TCNGATRE_NoCrossD": {
        "run_prefix": "tcngatre_nocrossd",
        "expect_graph_corrections": True,
    },
    "TCNGATRE_NoGraph": {
        "run_prefix": "tcngatre_nograph",
        "expect_graph_corrections": False,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run four artifact checks for TCNGATRE ablation variants: "
            "checkpoint hash, checkpoint structure, sequence_scores hash, "
            "and ablation summary regeneration."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["alfa", "alfa4hz", "simulate", "gpsdata"],
        default="alfa",
        help="Dataset suffix used in the ablation run directory names.",
    )
    parser.add_argument(
        "--variants",
        nargs=2,
        default=["TCNGATRE_NoCrossD", "TCNGATRE_NoGraph"],
        choices=sorted(VARIANT_SPECS.keys()),
        help="Exactly two variants to compare.",
    )
    parser.add_argument(
        "--infer-output-name",
        default="infer_tcngatre_failure",
        help="Inference output directory name under each run root.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip rerunning summarize_ablation.py.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_paths(variant: str, dataset: str, infer_output_name: str) -> dict[str, Path]:
    spec = VARIANT_SPECS[variant]
    run_root = ABLATION_ROOT / variant / "runs" / f"{spec['run_prefix']}_{dataset}"
    ckpt_path = run_root / "best.pt"
    infer_root = run_root / infer_output_name
    seq_path = infer_root / "sequence_scores.csv"
    return {
        "run_root": run_root,
        "ckpt_path": ckpt_path,
        "infer_root": infer_root,
        "seq_path": seq_path,
    }


def inspect_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    keys = list(state.keys())
    has_graph = any(k.startswith("graph_corrections") for k in keys)
    has_static = any("A_static" in k for k in keys)
    has_dyn = any("dyn." in k or "A_dyn" in k for k in keys)
    return {
        "num_state_keys": len(keys),
        "has_graph_corrections": bool(has_graph),
        "has_static_graph_params": bool(has_static),
        "has_dynamic_graph_params": bool(has_dyn),
        "state_key_head": keys[:20],
    }


def run_summary() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "summarize_ablation.py"],
        cwd=str(ABLATION_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def print_json(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variant_a, variant_b = args.variants
    all_payloads: dict[str, dict] = {}

    for variant in (variant_a, variant_b):
        spec = VARIANT_SPECS[variant]
        paths = build_paths(variant, args.dataset, args.infer_output_name)
        payload = {
            "variant": variant,
            "dataset": args.dataset,
            "run_root": str(paths["run_root"]),
            "expect_graph_corrections": bool(spec["expect_graph_corrections"]),
            "checkpoint_exists": paths["ckpt_path"].exists(),
            "sequence_scores_exists": paths["seq_path"].exists(),
        }

        if paths["ckpt_path"].exists():
            payload["checkpoint_sha256"] = sha256_file(paths["ckpt_path"])
            payload["checkpoint_inspect"] = inspect_checkpoint(paths["ckpt_path"])
        else:
            payload["checkpoint_sha256"] = None
            payload["checkpoint_inspect"] = None

        if paths["seq_path"].exists():
            payload["sequence_scores_sha256"] = sha256_file(paths["seq_path"])
            payload["sequence_scores_size_bytes"] = int(paths["seq_path"].stat().st_size)
        else:
            payload["sequence_scores_sha256"] = None
            payload["sequence_scores_size_bytes"] = None

        all_payloads[variant] = payload
        print_json(f"Artifact Check: {variant}", payload)

    compare = {
        "dataset": args.dataset,
        "variants": [variant_a, variant_b],
        "checkpoint_sha256_equal": (
            all_payloads[variant_a]["checkpoint_sha256"] is not None
            and all_payloads[variant_a]["checkpoint_sha256"] == all_payloads[variant_b]["checkpoint_sha256"]
        ),
        "sequence_scores_sha256_equal": (
            all_payloads[variant_a]["sequence_scores_sha256"] is not None
            and all_payloads[variant_a]["sequence_scores_sha256"] == all_payloads[variant_b]["sequence_scores_sha256"]
        ),
        "graph_corrections_flags": {
            variant: (
                None if all_payloads[variant]["checkpoint_inspect"] is None
                else all_payloads[variant]["checkpoint_inspect"]["has_graph_corrections"]
            )
            for variant in (variant_a, variant_b)
        },
    }
    print_json("Comparison Summary", compare)

    if not args.skip_summary:
        result = run_summary()
        summary_payload = {
            "return_code": int(result.returncode),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        print_json("summarize_ablation.py", summary_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
