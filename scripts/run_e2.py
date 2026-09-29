#!/usr/bin/env python3
"""Run E2 robustness checks across Qwen3 model pairs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/e2_model_pair_replication")
    parser.add_argument("--prompt-cache", default="data/korean_wikipedia_20231101_ko.jsonl")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--speculative-k", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16"], default="float16")
    parser.add_argument("--seed", type=int, default=3090)
    parser.add_argument("--checkpoint-prompts", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check tokenizer compatibility and prompt identity without loading model weights or using the GPU")
    args = parser.parse_args()
    if args.num_prompts < 1 or args.checkpoint_prompts < 1:
        parser.error("--num-prompts and --checkpoint-prompts must be positive")
    if not args.preflight_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        parser.error("GPU work must be launched with CUDA_VISIBLE_DEVICES=7; the process uses cuda:0")

    from src.e2_model_pairs import run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
