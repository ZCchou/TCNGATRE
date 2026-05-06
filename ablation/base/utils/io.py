from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(Path(path).parent)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
