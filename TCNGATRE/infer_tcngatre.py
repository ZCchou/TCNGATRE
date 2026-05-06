from __future__ import annotations

import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = BUNDLE_ROOT.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))
if str(PORTABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PORTABLE_ROOT))

from tcngatre_runtime import apply_env, resolve_config_from_argv


def main(argv=None):
    cfg = resolve_config_from_argv(argv=argv, description="Run TCNGATRE inference on a bundled dataset.")
    apply_env(cfg)
    from tcngatre_infer_impl import main as infer_main
    return infer_main(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
