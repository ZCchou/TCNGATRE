from __future__ import annotations

import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parent
_ABLATION_ROOT = _MODEL_ROOT.parent
_BASE_ROOT = _ABLATION_ROOT / "base"
_PROJECT_ROOT = _ABLATION_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_BASE_ROOT), str(_MODEL_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runtime import prepare, resolve_config


def main(argv=None):
    cfg = resolve_config(argv, "Train TCNGATRE_NoCrossD ablation variant.")
    prepare(cfg)
    from train_impl import main as _train
    return _train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
