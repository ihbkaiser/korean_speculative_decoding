#!/usr/bin/env python3
"""Analyze saved pilot parquets and write statistical tables/figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import analyze_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path such as runs/<run_id>")
    args = parser.parse_args()
    result = analyze_run(args.run_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
