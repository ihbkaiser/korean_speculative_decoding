#!/usr/bin/env python3
"""Build the token-level H2 analysis from an existing completed run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.h2_analysis import analyze_h2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Completed run directory containing reference_outputs.jsonl")
    parser.add_argument(
        "--prompt-cache",
        default=str(ROOT / "data/korean_wikipedia_20231101_ko.jsonl"),
        help="Prompt cache used to reconstruct tokenizer context",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    result = analyze_h2(args.run_dir, args.prompt_cache, bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
