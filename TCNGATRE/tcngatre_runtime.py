from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tcngatreconfig import BUNDLE_ROOT, TCNGATREConfig
from utils.io import ensure_dir


def parse_args(argv: list[str] | None = None, description: str = "Run TCNGATRE.") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_config_from_argv(argv: list[str] | None = None, description: str = "Run TCNGATRE.") -> TCNGATREConfig:
    args = parse_args(argv=argv, description=description)
    return TCNGATREConfig(dataset_name=args.dataset)


def _run_subprocess(args: list[str]) -> None:
    subprocess.run(args, cwd=str(BUNDLE_ROOT), check=True)


def ensure_graph_ready(cfg: TCNGATREConfig) -> None:
    keep_path = Path(cfg.graph_dir) / "keep_columns.json"
    adj_path = Path(cfg.graph_dir) / "adjacency_dense.csv"
    if keep_path.exists() and adj_path.exists() and not bool(cfg.graph_overwrite):
        return
    ensure_dir(Path(cfg.graph_dir))
    # top_k_per_node=0 → keep all edges passing MIC threshold (no topk filtering)
    cmd = [
        sys.executable,
        str(BUNDLE_ROOT / "util" / "build_set_a_graph.py"),
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
    if bool(cfg.graph_overwrite):
        cmd.append("--overwrite")
    _run_subprocess(cmd)


def _set_env(name: str, value) -> None:
    if value is not None:
        os.environ[str(name)] = str(value)


def apply_env(cfg: TCNGATREConfig) -> None:
    _set_env("UAV_TCNGATRE_DATASET", cfg.dataset_name)
    _set_env("UAV_TCNGATRE_DATA_ROOT", cfg.data_root)
    _set_env("UAV_TCNGATRE_LABELS_ROOT", cfg.labels_root)
    _set_env("UAV_TCNGATRE_RUN_ROOT", cfg.run_root)
    _set_env("UAV_TCNGATRE_SPLIT_INFO_PATH", cfg.split_info_path)
    _set_env("UAV_TCNGATRE_GRAPH_DIR", cfg.graph_dir)
    _set_env("UAV_TCNGATRE_INFER_OUTPUT_NAME", cfg.infer_output_name)
    _set_env("UAV_TCNGATRE_INFER_SOURCE_SPLIT", cfg.infer_source_split)


def prepare_and_apply(cfg: TCNGATREConfig) -> TCNGATREConfig:
    ensure_dir(Path(cfg.run_root))
    ensure_graph_ready(cfg)
    apply_env(cfg)
    return cfg
