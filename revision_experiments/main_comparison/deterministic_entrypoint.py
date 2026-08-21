from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Execute a legacy experiment script with deterministic random state."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        boundary = raw.index("--")
        args = parser.parse_args(raw[:boundary])
        script_args = raw[boundary + 1 :]
    else:
        args, script_args = parser.parse_known_args(raw)
    return args, script_args


def configure_determinism(seed: int) -> dict[str, str | int | bool]:
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": int(seed),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def main(argv: list[str] | None = None) -> None:
    args, script_args = parse_args(argv)
    script = args.script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Experiment entrypoint not found: {script}")
    state = configure_determinism(int(args.seed))
    print(f"[DETERMINISM] {state}", flush=True)

    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv = [str(script), *script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
