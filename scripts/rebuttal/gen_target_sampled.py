#!/usr/bin/env python3
"""Generate frozen target-sampled continuations without truncating the distribution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.score_cross_family_judge import _ensure_snapshot, _load_streamed_model
from src.data import encode_prompt
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_tokenizer,
    prompt_ids_sha256,
    select_prompts,
)
from src.rebuttal.runtime import write_gpu_metadata
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

SEED = 20261125
SHARD_DOCS = 100


def _padded_inputs(rows: list[dict], pad_id: int, device: str):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = row["prompt_ids"]
        ids[index, width - len(values):] = torch.tensor(values, dtype=torch.long, device=device)
        mask[index, width - len(values):] = 1
    return ids, mask, width


@torch.inference_mode()
def _sample_batch(model, tokenizer, rows, max_new_tokens, temperature, actual_batch_sizes):
    input_ids, attention_mask, prompt_width = _padded_inputs(
        rows, int(tokenizer.pad_token_id), "cuda"
    )
    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        typical_p=1.0,
        min_p=0.0,
        epsilon_cutoff=0.0,
        eta_cutoff=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
        logits_to_keep=1,
    )
    outputs = []
    for index in range(len(rows)):
        generated = list(map(int, sequences[index, prompt_width:].tolist()))
        eos_hit = tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated
        if eos_hit:
            generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
        outputs.append((generated, bool(eos_hit)))
    actual_batch_sizes.add(len(rows))
    return outputs


def _sample_with_backoff(model, tokenizer, rows, max_new_tokens, temperature, batch_sizes):
    try:
        return _sample_batch(model, tokenizer, rows, max_new_tokens, temperature, batch_sizes)
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda(torch)
        if len(rows) == 1:
            raise RuntimeError("Sampled target generation ran out of GPU memory at batch size 1")
        midpoint = len(rows) // 2
        return (
            _sample_with_backoff(model, tokenizer, rows[:midpoint], max_new_tokens, temperature, batch_sizes)
            + _sample_with_backoff(model, tokenizer, rows[midpoint:], max_new_tokens, temperature, batch_sizes)
        )


def _write_manifest(path: Path, identity: dict, completed: list[dict], status: str, extra: dict | None = None):
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {**identity, **(extra or {}), "completed_shards": completed, "status": status}
    if "gpu_runtime" in old:
        payload["gpu_runtime"] = old["gpu_runtime"]
    atomic_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("qwen4", "qwen8"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rebuttal/sampled_T1")
    args = parser.parse_args()
    if args.temperature <= 0 or args.max_prompts < 1 or args.batch_size < 1:
        parser.error("temperature, max-prompts, and batch-size must be positive")

    config = load_config(args.config)
    if args.target not in config["models"]:
        parser.error(f"unknown target model key {args.target!r}")
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face token is unavailable; set HF_TOKEN from /workspace/hf_token")
    api = HfApi(token=token)
    snapshot, _, _ = _ensure_snapshot(config, args.target, api, token)
    tokenizer = load_tokenizer(config, args.target, snapshot=snapshot)
    tokenizer.padding_side = "left"
    prompts = select_prompts(config, "common")[:args.max_prompts]
    if len(prompts) != args.max_prompts:
        raise RuntimeError(f"Requested {args.max_prompts} common prompts; found {len(prompts)}")
    rows = []
    for row in prompts:
        ids = encode_prompt(tokenizer, str(row["text"]), int(config["data"]["max_prompt_tokens"]))
        rows.append({**row, "prompt_ids": ids})

    spec = config["models"][args.target]
    out_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    destination = out_root / "continuations" / args.target / "common"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "run_metadata.json"
    identity = {
        "stage": "target_sampled_generation",
        "model_key": args.target,
        "model_id": spec["id"],
        "revision": spec["revision"],
        "split": "common",
        "prompt_count": len(rows),
        "prompt_ids_sha256": prompt_ids_sha256(rows),
        "prompt_pool_sha256": __import__("hashlib").sha256(
            (ROOT / config["data"]["prompts"]).read_bytes()
        ).hexdigest(),
        "max_prompt_tokens": int(config["data"]["max_prompt_tokens"]),
        "max_new_tokens": int(config["data"]["max_new_tokens"]),
        "dtype": config["inference"]["dtype"],
        "attention_backend": config["inference"]["attention_backend"],
        "requested_batch_size": args.batch_size,
        "seed": SEED,
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "typical_p": 1.0,
            "min_p": 0.0,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "renormalize_logits": False,
        },
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    completed = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(old.get(key) != identity.get(key) for key in ("model_key", "revision", "prompt_count", "prompt_ids_sha256", "seed", "sampling")):
            raise RuntimeError(f"Existing sampled output identity differs at {manifest_path}")
        completed = {int(item["shard"]): item for item in old.get("completed_shards", [])}
    write_gpu_metadata(
        manifest_path,
        job="target_sampled_generation",
        details={"target": args.target, "temperature": args.temperature, "seed": SEED},
        merge_existing=manifest_path.exists(),
    )
    _write_manifest(manifest_path, identity, [completed[i] for i in sorted(completed)], "RUNNING")

    model, model_path, storage_mode = _load_streamed_model(config, args.target, api, token)
    model.config.use_cache = True
    max_new_tokens = int(config["data"]["max_new_tokens"])
    shard_count = (len(rows) + SHARD_DOCS - 1) // SHARD_DOCS
    actual_batch_sizes: set[int] = set()
    started = time.monotonic()
    for shard_index in range(shard_count):
        shard_rows = rows[shard_index * SHARD_DOCS:(shard_index + 1) * SHARD_DOCS]
        shard_path = destination / f"shard-{shard_index:05d}.parquet"
        if shard_index in completed and shard_path.is_file():
            print(f"SAMPLE_RESUME_SKIP {shard_path}", flush=True)
            continue
        # A stable per-shard seed makes completed shards independent of resume order.
        torch.manual_seed(SEED + shard_index)
        torch.cuda.manual_seed_all(SEED + shard_index)
        by_length: dict[int, list[dict]] = defaultdict(list)
        for row in shard_rows:
            by_length[len(row["prompt_ids"])].append(row)
        outputs_by_id = {}
        for length in sorted(by_length, reverse=True):
            bucket = by_length[length]
            for start in range(0, len(bucket), args.batch_size):
                chunk = bucket[start:start + args.batch_size]
                sampled = _sample_with_backoff(
                    model, tokenizer, chunk, max_new_tokens, args.temperature, actual_batch_sizes
                )
                for source, result in zip(chunk, sampled):
                    outputs_by_id[str(source["doc_id"])] = result
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
                "split": "common",
                "sampling_seed": SEED + shard_index,
                "sampling_temperature": args.temperature,
            })
        atomic_parquet(result_rows, shard_path)
        completed[shard_index] = {
            "shard": shard_index,
            "prompt_count": len(shard_rows),
            "continuation_token_count": int(sum(len(row["continuation_ids"]) for row in result_rows)),
            "actual_batch_sizes": sorted(actual_batch_sizes),
        }
        _write_manifest(manifest_path, identity, [completed[i] for i in sorted(completed)], "RUNNING",
                         {"actual_batch_sizes": sorted(actual_batch_sizes), "model_storage_mode": storage_mode,
                          "model_path": str(model_path), "wall_seconds_so_far": time.monotonic() - started})
        print(f"SAMPLE_SHARD_COMPLETE target={args.target} shard={shard_index:05d} docs={len(result_rows)}", flush=True)
    _write_manifest(manifest_path, identity, [completed[i] for i in sorted(completed)], "COMPLETE",
                     {"actual_batch_sizes": sorted(actual_batch_sizes), "model_storage_mode": storage_mode,
                      "model_path": str(model_path), "wall_seconds": time.monotonic() - started})
    del model
    cleanup_cuda(torch)
    print(f"SAMPLE_COMPLETE target={args.target} temperature={args.temperature} docs={len(rows)} output={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
