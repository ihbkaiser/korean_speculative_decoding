#!/usr/bin/env python3
"""Run the pinned model downloader with a token read from a local secret file."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "/workspace/hf_token"))


def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Hugging Face token file is empty: {TOKEN_FILE}")
    os.environ["HF_TOKEN"] = token
    downloader = ROOT / "scripts/download_models.py"
    sys.argv = [str(downloader), *sys.argv[1:]]
    runpy.run_path(str(downloader), run_name="__main__")


if __name__ == "__main__":
    main()
