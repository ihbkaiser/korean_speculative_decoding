"""Runtime metadata helpers for GPU rebuttal jobs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_gpu_metadata(
    destination: str | Path,
    *,
    job: str,
    details: dict[str, Any] | None = None,
    require_idle: bool = False,
    merge_existing: bool = False,
) -> dict[str, Any]:
    """Capture package/GPU state and the literal nvidia-smi output before work starts."""
    import torch
    import transformers

    full = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True).stdout
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if require_idle and compute:
        raise RuntimeError(f"Timing job requires an idle GPU; compute processes are listed: {compute}")
    metadata = {
        "job": job,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": full,
        "nvidia_smi_compute_apps": compute,
        "details": details or {},
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = metadata
    if merge_existing and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot merge GPU metadata into non-object JSON: {path}")
        existing["gpu_runtime"] = metadata
        payload = existing
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return metadata
