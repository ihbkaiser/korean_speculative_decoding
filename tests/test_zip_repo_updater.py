from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_from_github_zip.sh"


def _write_zip(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_updater_is_self_contained_shell_file():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import zipfile" in text
    assert "update_repo_from_github_zip.py" not in text
    assert 'ZIP_PATH="${ZIP_PATH:-' in text
    assert 'FOLDER_PATH="${FOLDER_PATH:-' in text


def test_updater_has_fast_staging_and_detailed_logging():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "archive.extractall" in text
    assert "TemporaryDirectory" in text
    assert "--log-file" in text
    assert "temporary stage removed" in text
    assert "scan repository" in text


def _run_shell_updater(archive: Path, repo: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to execute the Linux launcher")
    probe = subprocess.run([bash, "-c", "exit 0"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("a working Bash runtime is required to execute the Linux launcher")
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    return subprocess.run(
        [bash, str(SCRIPT), str(archive), "--repo", str(repo), *extra],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_shell_updater_updates_code_but_preserves_data(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "src").mkdir(parents=True)
    (repo / "data" / "frozen.parquet").write_bytes(b"frozen")
    (repo / "src" / "old.py").write_text("old", encoding="utf-8")
    archive = tmp_path / "download.zip"
    _write_zip(archive, {
        "repo-main/src/old.py": "new",
        "repo-main/src/new.py": "added",
        "repo-main/data/frozen.parquet": "must-not-overwrite",
    })

    result = _run_shell_updater(archive, repo)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (repo / "src" / "old.py").read_text(encoding="utf-8") == "new"
    assert (repo / "src" / "new.py").read_text(encoding="utf-8") == "added"
    assert (repo / "data" / "frozen.parquet").read_bytes() == b"frozen"
    assert Path(report["backup_dir"]).exists()


def test_shell_updater_rejects_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = tmp_path / "bad.zip"
    _write_zip(archive, {"repo-main/../../escape.txt": "bad"})

    result = _run_shell_updater(archive, repo)
    assert result.returncode != 0
    assert "unsafe archive path" in result.stderr


def test_shell_updater_deletes_missing_only_when_requested(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "stale.py").write_text("stale", encoding="utf-8")
    archive = tmp_path / "download.zip"
    _write_zip(archive, {"repo-main/src/current.py": "current"})

    result = _run_shell_updater(archive, repo)
    assert result.returncode == 0, result.stderr
    assert (repo / "src" / "stale.py").exists()

    result = _run_shell_updater(archive, repo, "--delete-missing")
    assert result.returncode == 0, result.stderr
    assert not (repo / "src" / "stale.py").exists()
