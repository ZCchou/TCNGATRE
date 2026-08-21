from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .paths import (
    APPROVED_LEGACY_CHANGES_PATH,
    LEGACY_SNAPSHOT_PATH,
    PACKAGE_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
)


class LegacyIntegrityError(RuntimeError):
    pass


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
    )
    return completed.stdout


def tracked_legacy_files() -> list[Path]:
    paths: list[Path] = []
    for raw in _git(["ls-files", "-z"]).split("\0"):
        if not raw:
            continue
        path = (REPO_ROOT / raw).resolve()
        if PACKAGE_ROOT.resolve() in path.parents or RESULTS_ROOT.resolve() in path.parents:
            continue
        if path.is_file():
            paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(REPO_ROOT).as_posix())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_hashes(paths: Iterable[Path] | None = None) -> dict[str, str]:
    selected = tracked_legacy_files() if paths is None else list(paths)
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in selected
    }


def load_approved_changes(path: Path = APPROVED_LEGACY_CHANGES_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("changes", [])
    approved: dict[str, dict] = {}
    for row in rows:
        rel = str(row.get("path", "")).strip().replace("\\", "/")
        old_hash = str(row.get("old_sha256", "")).strip().lower()
        new_hash = str(row.get("new_sha256", "")).strip().lower()
        accepted_old = {
            str(value).strip().lower()
            for value in row.get("accepted_old_sha256", [old_hash])
        }
        accepted_new = {
            str(value).strip().lower()
            for value in row.get("accepted_new_sha256", [new_hash])
        }
        if old_hash:
            accepted_old.add(old_hash)
        if new_hash:
            accepted_new.add(new_hash)
        if (
            not rel
            or not accepted_old
            or not accepted_new
            or any(len(value) != 64 for value in accepted_old | accepted_new)
        ):
            raise LegacyIntegrityError(f"Invalid approved legacy change entry: {row}")
        if rel in approved:
            raise LegacyIntegrityError(f"Duplicate approved legacy change: {rel}")
        approved[rel] = {
            **row,
            "path": rel,
            "old_sha256": old_hash,
            "new_sha256": new_hash,
            "accepted_old_sha256": sorted(accepted_old),
            "accepted_new_sha256": sorted(accepted_new),
        }
    return approved


def create_snapshot(path: Path = LEGACY_SNAPSHOT_PATH) -> dict:
    hashes = current_hashes()
    status = _git(["status", "--short"]).splitlines()
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_branch": _git(["branch", "--show-current"]).strip(),
        "git_head": _git(["rev-parse", "HEAD"]).strip(),
        "tracked_file_count": len(hashes),
        "dirty_status_entry_count": len(status),
        "dirty_status": status,
        "files": hashes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_snapshot(path: Path = LEGACY_SNAPSHOT_PATH) -> dict:
    if not path.exists():
        raise LegacyIntegrityError(f"Legacy snapshot is missing: {path}")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, str] = snapshot.get("files", {})
    approvals = load_approved_changes()
    missing: list[str] = []
    changed: list[str] = []
    approved_changes: list[dict] = []
    for rel, expected_hash in expected.items():
        file_path = REPO_ROOT / rel
        if not file_path.is_file():
            missing.append(rel)
            continue
        current_hash = sha256_file(file_path)
        if current_hash == expected_hash:
            continue
        approval = approvals.get(rel)
        if (
            approval is not None
            and str(expected_hash).lower() in approval["accepted_old_sha256"]
            and current_hash.lower() in approval["accepted_new_sha256"]
        ):
            approved_changes.append(approval)
        else:
            changed.append(rel)
    unknown_approvals = sorted(set(approvals).difference(expected))
    current_tracked = set(current_hashes())
    unexpected = sorted(current_tracked.difference(expected))
    result = {
        "ok": not missing and not changed and not unexpected and not unknown_approvals,
        "checked": len(expected),
        "missing": missing,
        "changed": changed,
        "approved_changes": approved_changes,
        "unknown_approvals": unknown_approvals,
        "unexpected_tracked_legacy": unexpected,
    }
    if not result["ok"]:
        raise LegacyIntegrityError(json.dumps(result, ensure_ascii=False, indent=2))
    return result
