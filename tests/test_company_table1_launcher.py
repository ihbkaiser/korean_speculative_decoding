"""Exercise shell orchestration with a tiny CLI stand-in, never model inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def company_launcher(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.exists() else None
    if not bash:
        pytest.skip("Bash is required for launcher integration checks")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "scripts/run_company_table1.sh", scripts)
    shutil.copy2(root / "scripts/run_company_table1_fast.sh", scripts)
    shutil.copy2(root / "run_table1.sh", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "data/prompts_40k.parquet").touch()
    (tmp_path / "metadata/dataset_revision.json").write_text("{}")
    (scripts / "table1_pipeline.py").write_text('''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"args": args, "unbuffered": os.environ.get("PYTHONUNBUFFERED")}) + "\\n")
print("stub stdout " + " ".join(args), flush=True)
print("stub stderr", file=sys.stderr, flush=True)
if "--skip-align" in args and os.environ.get("FAIL_PAIR") in args:
    sys.exit(7)
''', encoding="utf-8")
    env = {**os.environ, "PYTHON_BIN": sys.executable.replace("\\", "/"),
           "NUM_SHARDS": "3", "CALL_LOG": str(tmp_path / "calls.jsonl"),
           "TABLE1_LOG_TAG": "test-full"}

    def invoke(pair="all", extra_args=(), **overrides):
        result = subprocess.run([bash, str(tmp_path / "run_table1.sh"), pair, *extra_args],
                                env={**env, **overrides}, text=True,
                                capture_output=True, timeout=60)
        call_log = tmp_path / "calls.jsonl"
        calls = [json.loads(line) for line in call_log.read_text().splitlines()] if call_log.exists() else []
        return result, calls

    return invoke, tmp_path


def test_full_launcher_aligns_every_shard_and_captures_both_streams(company_launcher):
    invoke, root = company_launcher
    result, calls = invoke()
    assert result.returncode == 0, result.stderr
    align = [row["args"] for row in calls if "align-morphology" in row["args"]]
    assert len(align) == 15, "all shards of all five pairs must be aligned"
    assert {args[args.index("--shard-index") + 1] for args in align} == {"0", "1", "2"}
    assert all(row["unbuffered"] == "1" for row in calls)
    assert sum("build-table1" in row["args"] for row in calls) == 1
    for pair in ("Q1", "Q2", "Q3", "M1", "G1"):
        for stage in ("sd", "align"):
            log = (root / f"logs/test-full_{pair}_{stage}.log").read_text()
            assert "stub stdout" in log and "stub stderr" in log


def test_launcher_appends_logs_when_same_pair_resumes(company_launcher):
    invoke, root = company_launcher
    assert invoke("Q1")[0].returncode == 0
    assert invoke("Q1")[0].returncode == 0
    log = (root / "logs/test-full_Q1.log").read_text()
    assert log.count("stub stdout") == 2 and log.count("stub stderr") == 2


def test_gpu_failure_does_not_start_later_pairs_or_build_table(company_launcher):
    invoke, _ = company_launcher
    result, calls = invoke(FAIL_PAIR="Q3")
    assert result.returncode != 0
    assert not any("build-table1" in row["args"] for row in calls)
    assert not any("M1" in row["args"] or "G1" in row["args"] for row in calls)


def test_custom_output_directory_is_forwarded_to_every_stage(company_launcher):
    invoke, root = company_launcher
    output = root / "external output"
    result, calls = invoke(extra_args=("--output-dir", output.as_posix()))
    assert result.returncode == 0, result.stderr
    assert len(calls) == 21
    for row in calls:
        args = row["args"]
        assert "--output-dir" in args
        assert Path(args[args.index("--output-dir") + 1]) == output
    assert not (root / "logs").exists()
    assert len(list((output / "logs").glob("*.log"))) == 11
    assert "stub stderr" in (output / "logs/test-full_build.log").read_text()


@pytest.mark.parametrize("extra_args", [("--output-dir",), ("--output-dir", ""), ("--unknown",)])
def test_invalid_launcher_options_fail_before_running(company_launcher, extra_args):
    invoke, _ = company_launcher
    result, calls = invoke("Q1", extra_args=extra_args)
    assert result.returncode != 0
    assert not calls
