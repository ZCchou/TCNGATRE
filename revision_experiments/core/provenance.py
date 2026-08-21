from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _git_value(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8",
        ).stdout.strip()
    except Exception:
        return "unknown"


def environment_payload(repo_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_head": _git_value(repo_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(repo_root, ["branch", "--show-current"]),
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
    }
    try:
        import pandas as pd
        import scipy
        import torch

        payload.update({
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        payload["python_stack_error"] = repr(exc)
    try:
        import minepy
        payload["minepy"] = getattr(minepy, "__version__", "installed")
    except Exception as exc:
        payload["minepy_error"] = repr(exc)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
