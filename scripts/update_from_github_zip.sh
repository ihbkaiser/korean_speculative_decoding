#!/usr/bin/env bash
set -euo pipefail

# Self-contained GitHub Download ZIP updater.  The implementation is embedded
# below so this is the only file that must be copied to the company machine.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ZIP_PATH="${ZIP_PATH:-}"
FOLDER_PATH="${FOLDER_PATH:-$REPO_ROOT}"
export REPO_ROOT

PY_ARGS=()
if [[ -n "$ZIP_PATH" ]]; then
  PY_ARGS+=("$ZIP_PATH")
else
  if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "Usage: ZIP_PATH=/path/update.zip FOLDER_PATH=/path/repo bash $0 [options]" >&2
    echo "   or: bash $0 /path/update.zip /path/repo [options]" >&2
    exit 2
  fi
  ZIP_PATH="$1"
  shift
  if [[ $# -gt 0 && "$1" != -* ]]; then
    FOLDER_PATH="$1"
    shift
  fi
  PY_ARGS+=("$ZIP_PATH")
fi
PY_ARGS+=(--repo "$FOLDER_PATH")

exec "$PYTHON_BIN" - "${PY_ARGS[@]}" "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_PRESERVE = (
    ".git", ".venv", ".hf_home", ".cache", ".env", "hf_token", ".hf_token",
    "data", "models", "metadata", "audit", "runs", "logs", "validation",
    "results", "remote_artifacts",
)


def normalise_archive_name(name: str) -> str:
    value = name.replace("\\", "/")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"unsafe archive path: {name!r}")
    parts = PurePosixPath(value).parts
    if ".." in parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    return "/".join(part for part in parts if part not in ("", "."))


def detect_archive_root(names: Iterable[str]) -> str | None:
    files = [normalise_archive_name(name) for name in names if not name.endswith(("/", "\\"))]
    files = [name for name in files if name]
    if not files:
        return None
    parts = [PurePosixPath(name).parts for name in files]
    if any(len(item) < 2 for item in parts):
        return None
    roots = {item[0] for item in parts}
    return next(iter(roots)) if len(roots) == 1 else None


def relative_member(name: str, root: str | None) -> Path:
    normalised = normalise_archive_name(name)
    if root and (normalised == root or normalised.startswith(root + "/")):
        normalised = normalised[len(root):].lstrip("/")
    if not normalised:
        raise ValueError(f"archive entry has no repository path: {name!r}")
    return Path(*PurePosixPath(normalised).parts)


def extract_archive(archive_path: Path, stage: Path) -> Path:
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"staging directory must be empty: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        root = detect_archive_root(info.filename for info in infos)
        seen: set[Path] = set()
        for info in infos:
            relative = relative_member(info.filename, root)
            if relative in seen:
                raise ValueError(f"duplicate archive path after root stripping: {relative}")
            seen.add(relative)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"symbolic links are not allowed: {info.filename!r}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            permissions = (info.external_attr >> 16) & 0o777
            if permissions:
                destination.chmod(permissions)
    return stage


def normalise_preserve(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        clean = value.replace("\\", "/").strip("/")
        if clean and clean not in result:
            result.append(clean)
    return tuple(result)


def is_preserved(relative: Path, preserve: tuple[str, ...]) -> bool:
    value = relative.as_posix()
    return any(value == item or value.startswith(item + "/") for item in preserve)


def files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def same_file(left: Path, right: Path) -> bool:
    if not left.exists() or left.stat().st_size != right.stat().st_size:
        return False
    return left.read_bytes() == right.read_bytes()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backup_path(repo: Path) -> Path:
    parent = repo.parent / f".{repo.name}_update_backups"
    parent.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def apply_update(
    archive_path: Path,
    repo: Path,
    *,
    preserve: Iterable[str],
    delete_missing: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    repo = repo.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    if not repo.is_dir():
        raise NotADirectoryError(repo)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    preserve_values = normalise_preserve(preserve)

    with tempfile.TemporaryDirectory(prefix="repo_zip_stage-") as temporary:
        source = extract_archive(archive_path, Path(temporary))
        source_files = files(source)
        destination_files = files(repo)
        mutable_source = {
            relative: path for relative, path in source_files.items()
            if not is_preserved(relative, preserve_values)
        }
        mutable_destination = {
            relative: path for relative, path in destination_files.items()
            if not is_preserved(relative, preserve_values)
        }
        added = sorted(relative for relative in mutable_source if relative not in mutable_destination)
        updated = sorted(
            relative for relative in mutable_source
            if relative in mutable_destination
            and not same_file(mutable_destination[relative], mutable_source[relative])
        )
        deleted = sorted(
            relative for relative in mutable_destination
            if delete_missing and relative not in mutable_source
        )
        skipped = sorted(relative for relative in source_files if is_preserved(relative, preserve_values))
        changed = added + updated + deleted
        backup_dir: Path | None = None

        if not dry_run and changed:
            backup_dir = backup_path(repo)
            for relative in updated + deleted:
                existing = repo / relative
                if existing.is_file() and not existing.is_symlink():
                    target = backup_dir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(existing, target)
            manifest = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive": str(archive_path),
                "archive_sha256": sha256(archive_path),
                "repo": str(repo),
                "preserve": list(preserve_values),
                "delete_missing": delete_missing,
                "added": [path.as_posix() for path in added],
                "updated": [path.as_posix() for path in updated],
                "deleted": [path.as_posix() for path in deleted],
            }
            (backup_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        if not dry_run:
            for relative in updated + deleted:
                destination = repo / relative
                if destination.is_symlink() or destination.is_file():
                    destination.unlink()
            for relative in added + updated:
                source_file = mutable_source[relative]
                destination = repo / relative
                if destination.is_symlink():
                    raise ValueError(f"refusing to overwrite symlink: {destination}")
                if destination.exists() and destination.is_dir():
                    raise IsADirectoryError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)

    return {
        "status": "DRY_RUN" if dry_run else "UPDATED",
        "repo": str(repo),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "added": [path.as_posix() for path in added],
        "updated": [path.as_posix() for path in updated],
        "deleted": [path.as_posix() for path in deleted],
        "preserved_archive_entries": [path.as_posix() for path in skipped],
        "backup_dir": str(backup_dir) if backup_dir else None,
        "delete_missing": delete_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely update a checkout from a GitHub Download ZIP")
    parser.add_argument("zip_path", type=Path, help="GitHub Download ZIP file")
    parser.add_argument(
        "--repo", type=Path,
        default=Path(os.environ.get("REPO_ROOT", os.getcwd())),
        help="working repository to update",
    )
    parser.add_argument(
        "--preserve", action="append", default=[],
        help="additional repository-relative path/prefix to preserve; repeatable",
    )
    parser.add_argument(
        "--delete-missing", action="store_true",
        help="delete mutable local files absent from ZIP; backups are created first",
    )
    parser.add_argument("--dry-run", action="store_true", help="report changes without modifying the repo")
    args = parser.parse_args()
    result = apply_update(
        args.zip_path,
        args.repo,
        preserve=(*DEFAULT_PRESERVE, *args.preserve),
        delete_missing=args.delete_missing,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
