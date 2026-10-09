#!/usr/bin/env python3
"""Measure SD throughput and scalar differences without rerunning mismatches.

Use --synthetic for a download-free kernel/cache smoke benchmark. Real B200
measurements use the pinned local model paths in the selected config.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.table1_pipeline import load_config
from src.batched_speculative import speculative_greedy_microbatch
from src.data import encode_prompt
from src.models import load_models
from src.speculative_decoding import greedy_generate, speculative_greedy_cached
from src.table1_runner import load_pair_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/table1_fast_b200.yaml"))
    parser.add_argument("--pair", default="Q1", choices=["Q1", "Q2", "Q3", "M1", "G1"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--require-scalar-parity", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prompts < 1 or args.max_new_tokens < 1 or any(b < 1 for b in args.batch_sizes):
        parser.error("prompts, max-new-tokens and batch sizes must be positive")
    config = load_config(args.config)
    pair = config["pairs"][args.pair]
    inference = config["inference"]
    if args.synthetic:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(20261009)
        tiny = LlamaConfig(vocab_size=4096, hidden_size=128, intermediate_size=256,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=1024, eos_token_id=None, pad_token_id=0)
        draft = LlamaForCausalLM(tiny).to(args.device).eval()
        target = LlamaForCausalLM(tiny).to(args.device).eval()
        prompts = [torch.randint(1, 4096, (32 + i % 3,)).tolist() for i in range(args.prompts)]
        eos = None
    else:
        from transformers import AutoTokenizer

        paths = config["model_paths"][args.pair]
        tokenizer = AutoTokenizer.from_pretrained(paths["target"], use_fast=True, local_files_only=True)
        draft, target = load_models(
            paths["draft"], paths["target"], device=args.device,
            dtype=pair.get("dtype", inference["dtype"]),
            attention_backend=pair.get("attention_backend", inference["attention_backend"]),
        )
        records = load_pair_records(args.root / config["paths"]["prompts"], pair)[:args.prompts]
        prompts = [encode_prompt(tokenizer, row["text"], inference["max_prompt_tokens"]) for row in records]
        eos = tokenizer.eos_token_id

    cuda = str(args.device).startswith("cuda")

    def synchronize():
        if cuda:
            torch.cuda.synchronize(args.device)

    # The independent scalar baseline is generated ONCE for diagnostics only.
    references = [greedy_generate(target, p, args.max_new_tokens, eos) for p in prompts]
    speculative_greedy_cached(draft, target, prompts[0], 4, eos, 4, batch_target_verification=True)
    speculative_greedy_microbatch(draft, target, prompts[:min(4, len(prompts))], 4, eos, 4)
    synchronize()
    measurements = []
    variants = [("cached_block_single_prompt", 1)] + [("microbatched", b) for b in args.batch_sizes]
    for mode, size in variants:
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)
        synchronize()
        started = time.perf_counter()
        outputs, all_events = [], []
        try:
            for offset in range(0, len(prompts), size):
                chunk = prompts[offset:offset + size]
                if mode == "cached_block_single_prompt":
                    out, events = speculative_greedy_cached(
                        draft, target, chunk[0], args.max_new_tokens, eos,
                        inference["speculative_k"], batch_target_verification=True,
                    )
                    outputs.append(out)
                    all_events.append(events)
                else:
                    out, events = speculative_greedy_microbatch(
                        draft, target, chunk, args.max_new_tokens, eos, inference["speculative_k"],
                    )
                    outputs.extend(out)
                    all_events.extend(events)
            synchronize()
        except torch.cuda.OutOfMemoryError:
            row = {"mode": mode, "batch_size": size, "status": "oom"}
            measurements.append(row)
            print(json.dumps(row), flush=True)
            continue
        elapsed = time.perf_counter() - started
        differences = []
        for index, (actual, expected) in enumerate(zip(outputs, references)):
            if actual != expected:
                first = next((j for j, (a, b) in enumerate(zip(actual, expected)) if a != b),
                             min(len(actual), len(expected)))
                differences.append({"prompt_index": index, "first_difference": first})
        row = {
            "mode": mode, "batch_size": size, "status": "complete",
            "prompts": len(prompts), "seconds": elapsed,
            "prompts_per_second": len(prompts) / elapsed,
            "tokens_per_second": sum(map(len, outputs)) / elapsed,
            "scalar_exact_prompts": len(prompts) - len(differences),
            "scalar_mismatches": differences,
            "proposal_events": sum(map(len, all_events)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3 if cuda else None,
        }
        measurements.append(row)
        print(json.dumps(row), flush=True)
    successful = [row for row in measurements if row["status"] == "complete"]
    best = max((row for row in successful if row["mode"] == "microbatched"),
               key=lambda row: row["prompts_per_second"], default=None)
    report = {
        "pair": args.pair, "synthetic": args.synthetic,
        "gpu": torch.cuda.get_device_name(args.device) if cuda else "cpu",
        "torch": torch.__version__, "config": str(args.config),
        "max_new_tokens": args.max_new_tokens,
        "numeric_path": "random_tiny_fp32" if args.synthetic else {
            "dtype": pair.get("dtype", inference["dtype"]),
            "attention_backend": pair.get("attention_backend", inference["attention_backend"]),
        },
        "measurements": measurements,
        "recommended_batch_size": best["batch_size"] if best else None,
        "note": "Scalar differences are reported, never silently repaired. "
                "Timing includes audit collection but excludes model loading and checkpoint serialization.",
    }
    output = args.output or args.root / "profile_output" / f"table1_fast_{args.pair.lower()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}", flush=True)
    return int(not best or (args.require_scalar_parity and any(row["scalar_mismatches"] for row in successful)))


if __name__ == "__main__":
    raise SystemExit(main())
