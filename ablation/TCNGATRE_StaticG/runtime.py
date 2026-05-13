from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = MODEL_ROOT.parent
BASE_ROOT = ABLATION_ROOT / "base"
PROJECT_ROOT = ABLATION_ROOT.parent

for _p in [str(PROJECT_ROOT), str(BASE_ROOT), str(MODEL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base_config import TCNGATREConfig

RUN_PREFIX = "tcngatre_staticg"


def _parse_dataset(argv=None, description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["alfa", "alfa4hz", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_config(argv=None, description="Run TCNGATRE_StaticG ablation.") -> TCNGATREConfig:
    args = _parse_dataset(argv, description)
    os.environ.setdefault(
        "UAV_TCNGATRE_RUN_ROOT",
        str(MODEL_ROOT / "runs" / f"{RUN_PREFIX}_{args.dataset}"),
    )
    # Static-only graph: beta=0 means 100% static; eta is irrelevant since dynamic is overridden in model
    os.environ.setdefault("UAV_TCNGATRE_GRAPH_BETA", "0.0")
    os.environ.setdefault("UAV_TCNGATRE_GRAPH_ETA", "3.0")
    return TCNGATREConfig(dataset_name=args.dataset)


def ensure_graph_ready(cfg: TCNGATREConfig) -> None:
    graph_dir = Path(cfg.graph_dir)
    if (graph_dir / "keep_columns.json").exists() and (graph_dir / "adjacency_dense.csv").exists() and not cfg.graph_overwrite:
        return
    graph_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BASE_ROOT / "util" / "build_set_a_graph.py"),
        "--in_dir", str(cfg.graph_input_dir),
        "--out_dir", str(cfg.graph_dir),
        "--dataset_mode", str(cfg.dataset_mode),
        "--grid_sec", str(cfg.graph_grid_sec),
        "--mic_alpha", str(cfg.graph_mic_alpha),
        "--mic_c", str(cfg.graph_mic_c),
        "--min_overlap", str(cfg.graph_min_overlap),
        "--mic_threshold", str(cfg.graph_mic_threshold),
        "--top_k_per_node", "0",
        "--max_points_per_pair", str(cfg.graph_max_points_per_pair),
        "--num_workers", str(cfg.graph_num_workers),
    ]
    if str(cfg.dataset_name).strip().lower() == "alfa4hz":
        manifest = json.loads(Path(cfg.dataset_manifest_path).read_text(encoding="utf-8"))
        legacy = [str(x) for x in manifest.get("legacy_train_flights", [])]
        if legacy:
            cmd.append("--include_flights")
            cmd.extend(legacy)
    if cfg.graph_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=str(MODEL_ROOT), check=True)


def prepare(cfg: TCNGATREConfig) -> TCNGATREConfig:
    Path(cfg.run_root).mkdir(parents=True, exist_ok=True)
    ensure_graph_ready(cfg)
    return cfg
