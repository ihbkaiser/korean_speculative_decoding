#!/usr/bin/env python3
"""Run E1 token-frequency analysis from an existing H2 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e1_frequency import analyze_e1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory containing the existing H2 token table")
    parser.add_argument(
        "--prompt-cache",
        default=str(ROOT / "data/korean_wikipedia_20231101_ko.jsonl"),
        help="Local Korean Wikipedia prompt cache used for token frequencies",
    )
    args = parser.parse_args()
    result = analyze_e1(args.run_dir, args.prompt_cache)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
