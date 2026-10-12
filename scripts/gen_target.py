#!/usr/bin/env python3
"""Batched greedy target continuations, checkpointed in 2,000-prompt Parquet shards."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import encode_prompt
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    load_tokenizer,
    model_path,
    output_root,
    prompt_shards,
    prompt_ids_sha256,
    select_prompts,
    sha256_file,
)


START_BATCH = {
    "qwen06": 192,
    "qwen17": 192,
    "qwen4": 128,
    "qwen8": 96,
    "qwen14": 48,
    "gemma4": 64,
    "gemma12": 48,
    "ministral3": 64,
    "ministral8": 64,
}


def _padded_inputs(rows: list[dict], pad_id: int, device: str):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        values = row["prompt_ids"]
        ids[i, width - len(values) :] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, width - len(values) :] = 1
    return ids, mask, width


def _decode_batch(model, tokenizer, rows, max_new_tokens, device):
    input_ids, attention_mask, prompt_width = _padded_inputs(
        rows, int(tokenizer.pad_token_id), device
    )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": False,
        "logits_to_keep": 1,
    }
    with torch.inference_mode():
        sequences = model.generate(**kwargs)
    outputs = []
    for i in range(len(rows)):
        generated = sequences[i, prompt_width:].tolist()
        eos_id = tokenizer.eos_token_id
        eos_hit = eos_id is not None and eos_id in generated
        if eos_hit:
            generated = generated[: generated.index(eos_id) + 1]
        outputs.append((list(map(int, generated)), bool(eos_hit)))
    return outputs


def _run_batch_with_backoff(model, tokenizer, rows, max_new_tokens, device, used_sizes):
    if not rows:
        return []
    try:
        result = _decode_batch(model, tokenizer, rows, max_new_tokens, device)
        used_sizes.add(len(rows))
        return result
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda(torch)
        if len(rows) == 1:
            raise RuntimeError("Target generation ran out of GPU memory at batch size 1")
        midpoint = len(rows) // 2
        return (
            _run_batch_with_backoff(model, tokenizer, rows[:midpoint], max_new_tokens, device, used_sizes)
            + _run_batch_with_backoff(model, tokenizer, rows[midpoint:], max_new_tokens, device, used_sizes)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--target", required=True, help="Model key, e.g. qwen4 or gemma12")
    parser.add_argument("--split", choices=("common", "all", "extra"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, help="Local model snapshot override")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-prompts", type=int, help="Bounded pilot; omitted means the full frozen split")
    parser.add_argument("--force", action="store_true", help="Replace matching completed Parquet shards")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target not in config["models"]:
        parser.error(f"unknown target model key {args.target!r}")
    spec = config["models"][args.target]
    prompts = select_prompts(config, args.split)
    if args.max_prompts is not None:
        if args.max_prompts < 1:
            parser.error("--max-prompts must be positive")
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise RuntimeError("The selected frozen prompt split is empty")

    expected_path = model_path(config, args.target)
    if args.model_path:
        path = args.model_path.expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config at {path}")
    else:
        path = expected_path
    batch_size = args.batch_size or START_BATCH[args.target]
    if batch_size < 1:
        parser.error("--batch-size must be positive")
    shard_size = int(config["data"]["shard_prompts"])
    max_new_tokens = int(config["data"]["max_new_tokens"])
    out = output_root(config, args.output_dir)
    destination = out / "continuations" / args.target / args.split
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "run_metadata.json"
    prompt_sha = prompt_ids_sha256(prompts)
    base_manifest = {
        "stage": "target_greedy_generation",
        "model_key": args.target,
        "model_id": spec["id"],
        "revision": spec["revision"],
        "split": args.split,
        "prompt_count": len(prompts),
        "prompt_ids_sha256": prompt_sha,
        "max_prompt_tokens": int(config["data"]["max_prompt_tokens"]),
        "max_new_tokens": max_new_tokens,
        "dtype": config["inference"]["dtype"],
        "attention_backend": config["inference"]["attention_backend"],
        "requested_batch_size": batch_size,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_pool_sha256": sha256_file(ROOT / config["data"]["prompts"]),
        "model_path": str(path),
    }
    if manifest_path.exists() and not args.force:
        old = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("model_key", "revision", "split", "prompt_count", "prompt_ids_sha256", "max_new_tokens"):
            if old.get(field) != base_manifest.get(field):
                raise RuntimeError(f"Existing output metadata differs at {field}; use --force or another output directory")

    if args.model_path:
        from src.models import load_table1_tokenizer
        from transformers import AutoTokenizer

        if spec.get("tokenizer_backend") == "tokenizers":
            tokenizer = load_table1_tokenizer(
                path,
                backend="tokenizers",
                fix_mistral_regex=bool(spec.get("fix_mistral_regex", False)),
                local_files_only=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = load_tokenizer(config, args.target)
    tokenizer.padding_side = "left"
    encoded = []
    for record in prompts:
        prompt_ids = encode_prompt(tokenizer, str(record["text"]), int(config["data"]["max_prompt_tokens"]))
        encoded.append({**record, "prompt_ids": prompt_ids})

    print(
        f"GEN_START target={args.target} split={args.split} prompts={len(prompts)} "
        f"requested_batch={batch_size} max_new_tokens={max_new_tokens}",
        flush=True,
    )
    started = time.monotonic()
    model = load_model(config, args.target, snapshot=path)
    actual_batch_sizes: set[int] = set()
    for shard_index, shard_rows in prompt_shards(encoded, shard_size):
        shard_path = destination / f"shard-{shard_index:05d}.parquet"
        if shard_path.exists() and not args.force:
            print(f"GEN_RESUME_SKIP {shard_path}", flush=True)
            continue
        by_length: dict[int, list[dict]] = defaultdict(list)
        for row in shard_rows:
            by_length[len(row["prompt_ids"])].append(row)
        outputs_by_id = {}
        for length in sorted(by_length, reverse=True):
            bucket = by_length[length]
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start : start + batch_size]
                outputs = _run_batch_with_backoff(
                    model, tokenizer, chunk, max_new_tokens, "cuda", actual_batch_sizes
                )
                for source, (continuation_ids, eos_hit) in zip(chunk, outputs):
                    outputs_by_id[str(source["doc_id"])] = (continuation_ids, eos_hit)
        result_rows = []
        for row in shard_rows:
            continuation_ids, eos_hit = outputs_by_id[str(row["doc_id"])]
            result_rows.append({
                "doc_id": str(row["doc_id"]),
                "source_index": int(row["source_index"]),
                "pool_index": int(row["pool_index"]),
                "pool_split": str(row["pool_split"]),
                "raw_text_hash": str(row["raw_text_hash"]),
                "prompt_ids": row["prompt_ids"],
                "continuation_ids": continuation_ids,
                "prompt_length": len(row["prompt_ids"]),
                "continuation_length": len(continuation_ids),
                "eos_hit": eos_hit,
                "target_model_key": args.target,
                "target_model_id": spec["id"],
                "target_revision": spec["revision"],
                "split": args.split,
            })
        atomic_parquet(result_rows, shard_path)
        print(f"GEN_SHARD_COMPLETE target={args.target} shard={shard_index:05d} prompts={len(result_rows)}", flush=True)

    manifest = {
        **base_manifest,
        "actual_batch_sizes": sorted(actual_batch_sizes),
        "wall_seconds": time.monotonic() - started,
        "status": "COMPLETE",
    }
    atomic_json(manifest_path, manifest)
    del model
    cleanup_cuda(torch)
    print(f"GEN_COMPLETE target={args.target} output={destination} wall_seconds={manifest['wall_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
