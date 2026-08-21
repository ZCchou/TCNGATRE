from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from revision_experiments.core.config import make_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one official isolated baseline adapter.")
    parser.add_argument("--baseline", choices=["catch", "carots"], required=True)
    parser.add_argument("--dataset", choices=["alfa", "gpsdata", "simulate"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = make_config("ex04", args.dataset, args.baseline, args.seed, smoke=args.smoke)
    if args.baseline == "catch":
        from revision_experiments.baselines.catch_adapter import run
    else:
        from revision_experiments.baselines.carots_adapter import run
    outcome = run(cfg, force=args.force)
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
