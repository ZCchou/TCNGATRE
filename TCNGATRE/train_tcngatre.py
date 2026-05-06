from __future__ import annotations

import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = BUNDLE_ROOT.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))
if str(PORTABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PORTABLE_ROOT))

from tcngatre_runtime import prepare_and_apply, resolve_config_from_argv


def main(argv=None):
    cfg = resolve_config_from_argv(argv=argv, description="Train TCNGATRE on a bundled dataset.")
    prepare_and_apply(cfg)
    from tcngatre_train_impl import main as train_main
    return train_main(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
