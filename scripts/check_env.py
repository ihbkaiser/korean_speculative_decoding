#!/usr/bin/env python3
"""Check core dependencies, CUDA hardware, and Qwen tokenizer compatibility."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ["torch", "transformers", "datasets", "kiwipiepy", "pyarrow", "pandas", "statsmodels", "matplotlib", "yaml"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.yaml"))
    parser.add_argument("--skip-tokenizer-check", action="store_true", help="Only inspect local packages and CUDA")
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["experiment"]
    packages = {}
    for name in REQUIRED:
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available = False
        distribution = "PyYAML" if name == "yaml" else name
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"available": available, "version": version}

    result = {"python": sys.version, "packages": packages, "cuda": {"available": False}, "tokenizer_compatibility": None}
    missing = [name for name, item in packages.items() if not item["available"]]
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {"name": torch.cuda.get_device_name(index), "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2)}
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        result["cuda"]["error"] = f"{type(exc).__name__}: {exc}"

    if not args.skip_tokenizer_check and not missing:
        try:
            from src.models import load_compatible_tokenizers

            _, report = load_compatible_tokenizers(config["draft_model"], config["target_model"])
            result["tokenizer_compatibility"] = report
        except Exception as exc:
            result["tokenizer_compatibility"] = {"compatible": False, "error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        print("Missing packages: " + ", ".join(missing), file=sys.stderr)
        return 2
    if not result["cuda"].get("available"):
        print("CUDA is unavailable; this experiment requires the requested GPU.", file=sys.stderr)
        return 3
    compatibility = result.get("tokenizer_compatibility")
    if compatibility is not None and not compatibility.get("compatible", False):
        print("Draft and target tokenizers are not compatible.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
