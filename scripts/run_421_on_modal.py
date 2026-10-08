#!/usr/bin/env python3
"""Run the matched-population 4.2.1 reanalysis on a Modal T4."""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "runs" / "proposal_side_boundary_validation"
INPUT_NAMES = [
    "proposal_boundary_rows_wikipedia_p1.parquet",
    "proposal_boundary_rows_wikipedia_p2.parquet",
    "proposal_boundary_rows_flores_p1.parquet",
    "proposal_boundary_rows_flores_p2.parquet",
]
REMOTE_INPUT = "/root/inputs"
REMOTE_OUTPUT = "/root/results"
REMOTE_REPO = "/root/repo"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pyarrow==24.0.0",
        "scipy==1.14.1",
        "patsy==1.0.1",
        "statsmodels==0.14.1",
        "transformers==5.17.0",
        "tokenizers==0.23.1",
        "huggingface-hub==1.5.0",
        "kiwipiepy==0.24.0",
        "kiwipiepy_model==0.24.0",
    )
    .add_local_dir(str(ROOT / "src"), remote_path=f"{REMOTE_REPO}/src")
    .add_local_dir(str(ROOT / "scripts"), remote_path=f"{REMOTE_REPO}/scripts")
)
for input_name in INPUT_NAMES:
    image = image.add_local_file(
        str(INPUTS / input_name),
        remote_path=f"{REMOTE_INPUT}/{input_name}",
    )

app = modal.App("naacl-421-matched-population")


def _records_without_nan(frame: Any) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(frame.notna(), None)
    return clean.to_dict(orient="records")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
)
def run_reanalysis() -> dict[str, Any]:
    import importlib.metadata
    import pandas as pd

    started = time.monotonic()
    remote_output = Path(REMOTE_OUTPUT)
    remote_output.mkdir(parents=True, exist_ok=True)
    script = Path(REMOTE_REPO) / "scripts" / "analyze_421_aggregate_vs_directional.py"
    command = [
        sys.executable,
        str(script),
        "--allow-hub",
        "--source-dir",
        str(REMOTE_INPUT),
        "--output-dir",
        str(REMOTE_OUTPUT),
    ]
    env = os.environ.copy()
    env["HF_HOME"] = "/root/cache/huggingface"
    env["HF_HUB_CACHE"] = "/root/cache/huggingface/hub"
    print(
        "MY_ENV_JSON="
        + json.dumps(
            {
                "python": platform.python_version(),
                "requested_gpu": "T4",
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy", "pandas", "pyarrow", "scipy", "patsy",
                        "statsmodels", "transformers", "tokenizers",
                        "huggingface-hub", "kiwipiepy", "kiwipiepy_model",
                    ]
                },
                "input_files": INPUT_NAMES,
                "command": command,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    gpu_probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"MY_GPU_NAME={gpu_probe.stdout.strip() or 'unavailable'}", flush=True)

    completed = subprocess.run(command, cwd=str(REMOTE_REPO), env=env, check=False)
    output_files = sorted(path for path in remote_output.rglob("*") if path.is_file())
    bundle_buffer = io.BytesIO()
    with zipfile.ZipFile(bundle_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output_files:
            bundle.write(path, arcname=path.relative_to(remote_output).as_posix())

    summary: dict[str, Any] = {
        "return_code": int(completed.returncode),
        "requested_gpu": "T4",
        "gpu_probe": gpu_probe.stdout.strip() or None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_files": [path.relative_to(remote_output).as_posix() for path in output_files],
    }
    audit_path = remote_output / "population_audit.csv"
    comparison_path = remote_output / "model_comparison.csv"
    if audit_path.is_file():
        summary["population_audit"] = _records_without_nan(pd.read_csv(audit_path))
    if comparison_path.is_file():
        summary["model_comparison"] = _records_without_nan(pd.read_csv(comparison_path))
    print("MY_SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return {"summary": summary, "archive": bundle_buffer.getvalue()}


@app.local_entrypoint()
def main() -> None:
    run_stamp = os.environ.get("MODAL_421_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "remote_artifacts" / f"modal_421_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_label": "4.2.1_matched_population_aggregate_vs_directional",
        "modal_app": "naacl-421-matched-population",
        "gpu": "T4",
        "cpu": 4,
        "memory_mib": 16384,
        "timeout_seconds": 3600,
        "input_files": INPUT_NAMES,
        "local_data_analysis": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("MY_RUN_DIR=" + str(run_dir), flush=True)
    result = run_reanalysis.remote()
    archive_path = run_dir / "results.zip"
    archive_path.write_bytes(result["archive"])
    summary_path = run_dir / "remote_summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("MY_ARTIFACT_PATH=" + str(archive_path), flush=True)
    print("MY_SUMMARY_JSON=" + json.dumps(result["summary"], ensure_ascii=False, default=str), flush=True)
    if result["summary"]["return_code"] != 0:
        raise RuntimeError(f"Remote analysis exited with code {result['summary']['return_code']}")
