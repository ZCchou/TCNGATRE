from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from revision_experiments.baselines.export_common_data import ensure_common_data
from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import PACKAGE_ROOT, REPO_ROOT


RUNTIME_PATHS = PACKAGE_ROOT / "envs" / "runtime_paths.json"


def runtime_python(baseline: str) -> Path:
    paths = json.loads(RUNTIME_PATHS.read_text(encoding="utf-8"))
    target = Path(paths[baseline])
    if not target.exists():
        raise FileNotFoundError(
            f"The isolated {baseline} runtime does not exist: {target}. "
            f"Create it from revision_experiments/envs/{baseline}.yml first."
        )
    return target


def execute_isolated_baseline(cfg, force: bool = False) -> dict:
    python = runtime_python(cfg.variant)
    verify_snapshot()
    common_manifest = ensure_common_data(cfg.dataset)
    print(
        f"common_data=ready dataset={cfg.dataset} "
        f"train={len(common_manifest['train'])} "
        f"validation={len(common_manifest['validation'])} "
        f"failure={len(common_manifest['failure'])}",
        flush=True,
    )
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(PACKAGE_ROOT / "baselines" / "run_adapter.py"),
        "--baseline", cfg.variant, "--dataset", cfg.dataset, "--seed", str(cfg.model_seed),
    ]
    if cfg.smoke:
        command.append("--smoke")
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(REPO_ROOT)
    log_path = run_dir / "adapter_console.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(console_encoding, errors="replace").decode(console_encoding, errors="replace")
            print(safe_line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{cfg.variant} adapter exited with code {return_code}; see {log_path}")
    done_path = run_dir / "DONE.json"
    if not done_path.exists():
        raise RuntimeError(f"{cfg.variant} adapter did not create DONE.json")
    return json.loads(done_path.read_text(encoding="utf-8"))
