from __future__ import annotations

import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parent
_ABLATION_ROOT = _MODEL_ROOT.parent
_BASE_ROOT = _ABLATION_ROOT / "base"
_PROJECT_ROOT = _ABLATION_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_BASE_ROOT), str(_MODEL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime import resolve_config


def main(argv=None):
    cfg = resolve_config(argv, "Infer TCNGATRE_NoGraph ablation variant.")
    from infer_impl import main as _infer
    return _infer(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
