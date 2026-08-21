from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = MODEL_ROOT.parent
BASE_ROOT = ABLATION_ROOT / "base"
PROJECT_ROOT = ABLATION_ROOT.parent

for _p in [str(PROJECT_ROOT), str(BASE_ROOT), str(MODEL_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from base_config import TCNGATREConfig

RUN_PREFIX = "tcngatre_full"


def _parse_dataset(argv=None, description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_config(argv=None, description="Run TCNGATRE Full ablation.") -> TCNGATREConfig:
    args = _parse_dataset(argv, description)
    os.environ.setdefault(
        "UAV_TCNGATRE_RUN_ROOT",
        str(MODEL_ROOT / "runs" / f"{RUN_PREFIX}_{args.dataset}"),
    )
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
    if cfg.graph_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=str(MODEL_ROOT), check=True)


def prepare(cfg: TCNGATREConfig) -> TCNGATREConfig:
    Path(cfg.run_root).mkdir(parents=True, exist_ok=True)
    ensure_graph_ready(cfg)
    return cfg
