#!/usr/bin/env bash
set -euo pipefail

# Self-contained GitHub Download ZIP updater.  The implementation is embedded
# below so this is the only file that must be copied to the company machine.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ZIP_PATH="${ZIP_PATH:-}"
FOLDER_PATH="${FOLDER_PATH:-$REPO_ROOT}"
LOG_FILE="${LOG_FILE:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
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
if [[ -n "$LOG_FILE" ]]; then
  PY_ARGS+=(--log-file "$LOG_FILE")
fi
PY_ARGS+=(--progress-every "$PROGRESS_EVERY")
if [[ "${VERBOSE:-0}" == "1" ]]; then
  PY_ARGS+=(--verbose)
fi

exec "$PYTHON_BIN" - "${PY_ARGS[@]}" "$@" <<'PY'
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_PRESERVE = (
    ".git", ".venv", ".hf_home", ".cache", ".env", "hf_token", ".hf_token",
    "data", "models", "metadata", "audit", "runs", "logs", "validation",
    "results", "remote_artifacts",
)


class RunLogger:
    def __init__(self, log_file: Path | None = None, *, verbose: bool = False):
        self.verbose = verbose
        self.handle = None
        if log_file:
            log_file = log_file.expanduser().resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.handle = log_file.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str, level: str = "INFO") -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"{stamp} [{level}] {message}"
        print(line, file=sys.stderr, flush=True)
        if self.handle:
            self.handle.write(line + "\n")

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


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


def extract_archive(
    archive_path: Path,
    stage: Path,
    *,
    preserve: tuple[str, ...],
    logger: RunLogger,
    target_name: str,
) -> tuple[Path, list[Path]]:
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"staging directory must be empty: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        all_infos = archive.infolist()
        infos = [info for info in all_infos if not info.is_dir()]
        root = detect_archive_root(info.filename for info in infos)
        seen: set[Path] = set()
        mutable_infos = []
        skipped: list[Path] = []
        relative_entries = []
        for info in all_infos:
            normalise_archive_name(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"symbolic links are not allowed: {info.filename!r}")
        for info in infos:
            relative = relative_member(info.filename, root)
            relative_entries.append((info, relative))

        # A locally packaged checkout can add a second wrapper, e.g.
        # korean_speculative_decoding-main/repo/src/... . When the inner
        # directory matches the destination basename, remove it so updating
        # FOLDER_PATH=/.../repo never creates /.../repo/repo/....
        inner_root = None
        if relative_entries and all(
            relative.parts
            and relative.parts[0] == target_name
            and len(relative.parts) >= 2
            for _, relative in relative_entries
        ):
            inner_root = target_name

        for info, relative in relative_entries:
            if inner_root:
                relative = Path(*relative.parts[1:])
            if relative in seen:
                raise ValueError(f"duplicate archive path after root stripping: {relative}")
            seen.add(relative)
            if is_preserved(relative, preserve):
                skipped.append(relative)
            else:
                mutable_infos.append(info)
        logger.log(
            f"extract mutable ZIP entries: total={len(infos)}, "
            f"mutable={len(mutable_infos)}, preserved={len(skipped)}, "
            f"root={root or '<none>'}, inner_root={inner_root or '<none>'}"
        )
        # The archive is fully validated above. Extract only mutable entries in
        # one archive operation; data/models/cache files are never copied to tmp.
        archive.extractall(stage, members=mutable_infos)
    source = stage
    if root:
        source /= root
    if inner_root:
        source /= inner_root
    logger.log(f"temporary extraction complete: files={len(mutable_infos)}, elapsed={elapsed(started)}")
    return source, skipped


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
    if not root.exists():
        return {}
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def same_file(left: Path, right: Path) -> bool:
    if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    return filecmp.cmp(left, right, shallow=False)


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
    logger: RunLogger,
    progress_every: int = 100,
) -> dict[str, object]:
    repo = repo.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    if not repo.is_dir():
        raise NotADirectoryError(repo)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    preserve_values = normalise_preserve(preserve)
    logger.log(
        f"start update: archive={archive_path}, repo={repo}, dry_run={dry_run}, "
        f"delete_missing={delete_missing}"
    )

    temporary = tempfile.TemporaryDirectory(prefix="repo_zip_stage-")
    try:
        stage = Path(temporary.name)
        logger.log(f"temporary stage created: {stage}")
        source, skipped = extract_archive(
            archive_path,
            stage,
            preserve=preserve_values,
            logger=logger,
            target_name=repo.name,
        )
        source_files = files(source)
        mutable_source = source_files
        logger.log(f"scan temporary code tree: files={len(mutable_source)}")

        if delete_missing:
            logger.log("scan repository for deletions: full mutable tree")
            destination_files = files(repo)
            mutable_destination = {
                relative: path for relative, path in destination_files.items()
                if not is_preserved(relative, preserve_values)
            }
        else:
            logger.log("scan repository: only ZIP paths (delete scan disabled)")
            mutable_destination = {}
            for relative in mutable_source:
                candidate = repo / relative
                if candidate.is_file() and not candidate.is_symlink():
                    mutable_destination[relative] = candidate

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
        changed = added + updated + deleted
        backup_dir: Path | None = None
        logger.log(
            f"plan ready: added={len(added)}, updated={len(updated)}, "
            f"deleted={len(deleted)}, preserved={len(skipped)}"
        )

        if not dry_run and changed:
            backup_started = time.perf_counter()
            backup_dir = backup_path(repo)
            logger.log(f"create backup: {backup_dir}")
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
            logger.log(f"backup complete: elapsed={elapsed(backup_started)}")

        if not dry_run:
            apply_started = time.perf_counter()
            logger.log(f"apply changes: files={len(changed)}")
            for relative in updated + deleted:
                destination = repo / relative
                if destination.is_symlink() or destination.is_file():
                    destination.unlink()
            for index, relative in enumerate(added + updated, start=1):
                source_file = mutable_source[relative]
                destination = repo / relative
                if destination.is_symlink():
                    raise ValueError(f"refusing to overwrite symlink: {destination}")
                if destination.exists() and destination.is_dir():
                    raise IsADirectoryError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
                if logger.verbose:
                    logger.log(f"copied [{index}/{len(added) + len(updated)}]: {relative}")
                elif index % max(1, progress_every) == 0:
                    logger.log(f"copy progress: {index}/{len(added) + len(updated)}")
            logger.log(f"apply complete: elapsed={elapsed(apply_started)}")
        elif dry_run:
            logger.log("dry-run: no files changed")
        else:
            logger.log("no mutable changes detected")
    finally:
        temporary.cleanup()
        logger.log("temporary stage removed")

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
    parser.add_argument("--log-file", type=Path, help="also write timestamped logs to this file")
    parser.add_argument("--verbose", action="store_true", help="log every copied file")
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="log copy progress every N files (default: 100)",
    )
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    logger = RunLogger(args.log_file, verbose=args.verbose)
    try:
        result = apply_update(
            args.zip_path,
            args.repo,
            preserve=(*DEFAULT_PRESERVE, *args.preserve),
            delete_missing=args.delete_missing,
            dry_run=args.dry_run,
            logger=logger,
            progress_every=args.progress_every,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logger.log(f"update failed: {type(exc).__name__}: {exc}", "ERROR")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
PY
