from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
LEGACY_ROOT = REPO_ROOT / "TCNGATRE"
RESULTS_ROOT = REPO_ROOT / "revision_results"
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "protocol_v1.json"
MANIFEST_DIR = PACKAGE_ROOT / "manifests"
LEGACY_SNAPSHOT_PATH = MANIFEST_DIR / "legacy_snapshot.json"
BASELINE_SOURCE_PATH = PACKAGE_ROOT / "baselines" / "baseline_sources.json"
EXTERNAL_ROOT = PACKAGE_ROOT / "_external"


def ensure_import_paths() -> None:
    import sys

    for path in (LEGACY_ROOT, REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
