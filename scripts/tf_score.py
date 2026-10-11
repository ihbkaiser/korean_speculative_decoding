#!/usr/bin/env python3
"""Teacher-force a model over target continuations and store reduced logits stats.

The transformer body runs once per microbatch. Its final hidden states are
projected through the LM head in 32-position chunks, so the full vocabulary
logit tensor is never materialized for the full sequence.
"""

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
    read_parquet_records,
    select_prompts,
    sha256_file,
)


def _continuation_files(root: Path, target: str, split: str) -> list[Path]:
    directory = root / "continuations" / target / split
    paths = sorted(directory.glob("shard-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No target continuation Parquet shards found under {directory}")
    return paths


def _load_generated_refs(root, target, split, selection):
    wanted = {str(row["doc_id"]): row for row in selection}
    refs = {}
    source_split = split
    try:
        files = _continuation_files(root, target, source_split)
    except FileNotFoundError:
        if split != "common":
            raise
        source_split = "all"
        files = _continuation_files(root, target, source_split)
    for path in files:
        for row in read_parquet_records(path):
            doc_id = str(row["doc_id"])
            if doc_id in wanted:
                refs[doc_id] = row
    missing = set(wanted) - set(refs)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target continuations for target={target}, split={split}")
    return [refs[str(row["doc_id"])] for row in selection]


def _build_human_refs(prompts, tokenizer, max_prompt, max_new):
    refs = []
    skipped = []
    for row in prompts:
        full = list(tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"])
        prompt = full[:max_prompt]
        continuation = full[len(prompt) : len(prompt) + max_new]
        if not prompt or not continuation:
            skipped.append(str(row["doc_id"]))
            continue
        refs.append({
            "doc_id": str(row["doc_id"]),
            "source_index": int(row["source_index"]),
            "pool_index": int(row["pool_index"]),
            "pool_split": str(row["pool_split"]),
            "raw_text_hash": str(row["raw_text_hash"]),
            "prompt_ids": prompt,
            "continuation_ids": continuation,
            "prompt_length": len(prompt),
            "continuation_length": len(continuation),
            "eos_hit": False,
            "split": "human",
        })
    return refs, skipped


def _body_and_head(model):
    if getattr(model.config, "model_type", None) == "mistral3":
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Mistral3 checkpoint has no text-only language_model")
        body = getattr(language_model, "model", None)
        head = getattr(language_model, "lm_head", None)
    else:
        body = getattr(model, "model", None)
        head = model.get_output_embeddings()
    if body is None or head is None:
        raise RuntimeError(
            f"Cannot access a text transformer body and LM head for {type(model).__name__}"
        )
    return body, head


def _hidden_states(body, input_ids, attention_mask, position_ids):
    output = body(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state


def _native_logits(model, input_ids, attention_mask, position_ids, positions):
    # Only one document and one position are used for the calibration check.
    kwargs = {
        "input_ids": input_ids[:1],
        "attention_mask": attention_mask[:1],
        "position_ids": position_ids[:1],
        "use_cache": False,
    }
    position_tensor = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
    try:
        output = model(**kwargs, logits_to_keep=position_tensor)
        logits = output.logits
        if logits.shape[1] != len(positions):
            raise TypeError("model ignored indexed logits_to_keep")
        return logits
    except (TypeError, ValueError, AttributeError):
        output = model(**kwargs)
        return output.logits[:, position_tensor, :]


def _calibrate_scale(model, body, head, input_ids, attention_mask, position_ids, prompt_len):
    positions = [prompt_len - 1]
    native = _native_logits(model, input_ids, attention_mask, position_ids, positions).float()
    hidden = _hidden_states(body, input_ids[:1], attention_mask[:1], position_ids[:1])
    selected = hidden[:, positions, :]
    manual = head(selected).float()
    denominator = (manual * manual).sum()
    scale = float(((native * manual).sum() / denominator.clamp_min(1e-20)).item())
    candidate = manual * scale
    max_error = float((native - candidate).abs().max().item())
    if not torch.allclose(native, candidate, atol=0.08, rtol=0.025):
        raise RuntimeError(
            "Chunked LM-head logits do not match the model forward: "
            f"max_abs_error={max_error:.5g}, fitted_scale={scale:.8g}. "
            "Do not use this model's TF output until its text path is inspected."
        )
    return scale, max_error


def _collate(rows, pad_id, device):
    sequences = [
        [int(x) for x in row["prompt_ids"]] + [int(x) for x in row["continuation_ids"][:-1]]
        for row in rows
    ]
    width = max(map(len, sequences))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def _stats_for_chunk(logits, labels, role, scale):
    logits = logits.float() * scale
    normalizer = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = normalizer - (probabilities * logits).sum(dim=-1)
    top_count = 20 if role == "target" else 2
    top_values, top_ids = torch.topk(logits, k=top_count, dim=-1)
    top_logp = top_values - normalizer.unsqueeze(-1)
    selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return {
        "argmax_id": top_ids[..., 0],
        "logp_argmax": top_logp[..., 0],
        "logp_target_token": selected - normalizer,
        "entropy": entropy,
        "top2_margin": top_values[..., 0] - top_values[..., 1],
        "top_ids": top_ids,
        "top_logp": top_logp,
    }


@torch.inference_mode()
def _score_batch(model, body, head, rows, role, chunk_positions, scale_state):
    device = next(model.parameters()).device
    pad_id = int(getattr(model.config, "pad_token_id", None) or 0)
    input_ids, attention_mask, position_ids = _collate(rows, pad_id, device)
    prompt_len = len(rows[0]["prompt_ids"])
    max_targets = max(len(row["continuation_ids"]) for row in rows)
    hidden = _hidden_states(body, input_ids, attention_mask, position_ids)
    if scale_state["scale"] is None:
        scale, error = _calibrate_scale(
            model, body, head, input_ids, attention_mask, position_ids, prompt_len
        )
        scale_state.update(scale=scale, max_abs_error=error)
        print(f"TF_HEAD_CHECK_OK scale={scale:.9g} max_abs_error={error:.4g}", flush=True)

    labels = [list(map(int, row["continuation_ids"])) for row in rows]
    out = [[] for _ in rows]
    batch_positions = torch.arange(len(rows), device=device)
    for start in range(0, max_targets, chunk_positions):
        end = min(max_targets, start + chunk_positions)
        sequence_positions = torch.arange(
            prompt_len - 1 + start,
            prompt_len - 1 + end,
            dtype=torch.long,
            device=device,
        )
        hidden_chunk = hidden[batch_positions[:, None], sequence_positions[None, :]]
        logits = head(hidden_chunk)
        valid_counts = [max(0, min(end, len(targets)) - start) for targets in labels]
        label_chunk = torch.full(
            (len(rows), end - start),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for i, target_ids in enumerate(labels):
            count = valid_counts[i]
            if count:
                label_chunk[i, :count] = torch.tensor(
                    target_ids[start : start + count], dtype=torch.long, device=device
                )
        stats = _stats_for_chunk(logits, label_chunk, role, scale_state["scale"])
        for i, row in enumerate(rows):
            for offset in range(valid_counts[i]):
                record = {
                    "doc_id": str(row["doc_id"]),
                    "source_index": int(row["source_index"]),
                    "pool_index": int(row["pool_index"]),
                    "position": start + offset,
                    "target_token_id": labels[i][start + offset],
                    "argmax_id": int(stats["argmax_id"][i, offset].item()),
                    "logp_argmax": float(stats["logp_argmax"][i, offset].item()),
                    "logp_target_token": float(stats["logp_target_token"][i, offset].item()),
                    "entropy": float(stats["entropy"][i, offset].item()),
                    "top2_margin": float(stats["top2_margin"][i, offset].item()),
                }
                if role == "target":
                    record["top20_ids"] = [int(x) for x in stats["top_ids"][i, offset].tolist()]
                    record["top20_logp"] = [float(x) for x in stats["top_logp"][i, offset].tolist()]
                out[i].append(record)
        del hidden_chunk, logits, stats, label_chunk
    del hidden
    return out


def _score_with_backoff(model, body, head, rows, role, chunk_positions, scale_state, batch_sizes):
    try:
        values = _score_batch(model, body, head, rows, role, chunk_positions, scale_state)
        batch_sizes.add(len(rows))
        return values
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda(torch)
        if len(rows) == 1:
            raise RuntimeError("Teacher-forcing ran out of GPU memory at batch size 1")
        middle = len(rows) // 2
        return (
            _score_with_backoff(model, body, head, rows[:middle], role, chunk_positions, scale_state, batch_sizes)
            + _score_with_backoff(model, body, head, rows[middle:], role, chunk_positions, scale_state, batch_sizes)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--model", required=True, help="Model key to score")
    parser.add_argument("--target", required=True, help="Target whose continuation is scored")
    parser.add_argument("--split", choices=("common", "all", "extra"), required=True)
    parser.add_argument("--role", choices=("target", "draft"), required=True)
    parser.add_argument("--continuation", choices=("generated", "human"), default="generated")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-positions", type=int, default=32)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model not in config["models"] or args.target not in config["models"]:
        parser.error("--model and --target must be model keys from the config")
    if args.role == "target" and args.model != args.target:
        parser.error("--role target requires --model and --target to be the same model key")
    if args.batch_size < 1 or args.chunk_positions < 1:
        parser.error("--batch-size and --chunk-positions must be positive")

    prompts = select_prompts(config, args.split)
    if args.max_prompts is not None:
        if args.max_prompts < 1:
            parser.error("--max-prompts must be positive")
        prompts = prompts[: args.max_prompts]
    out = output_root(config, args.output_dir)
    tokenizer = load_tokenizer(config, args.model, args.model_path)
    if args.continuation == "generated":
        refs = _load_generated_refs(out, args.target, args.split, prompts)
    else:
        refs, skipped = _build_human_refs(
            prompts,
            tokenizer,
            int(config["data"]["max_prompt_tokens"]),
            int(config["data"]["max_new_tokens"]),
        )
        if skipped:
            print(f"TF_HUMAN_SKIPPED short_documents={len(skipped)}", flush=True)
        prompt_by_id = {str(row["doc_id"]): row for row in prompts}
        prompts = [prompt_by_id[ref["doc_id"]] for ref in refs]
    if not refs:
        raise RuntimeError("No continuations are available for scoring")

    target_spec = config["models"][args.target]
    model_spec = config["models"][args.model]
    output_split = args.split
    destination = out / "teacher_forced" / args.continuation / args.target / args.model / output_split
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "run_metadata.json"
    prompt_sha = prompt_ids_sha256(prompts)
    model_path_used = args.model_path.expanduser().resolve() if args.model_path else model_path(config, args.model)
    metadata = {
        "stage": "teacher_forced_scoring",
        "continuation": args.continuation,
        "target_key": args.target,
        "target_id": target_spec["id"],
        "target_revision": target_spec["revision"],
        "scored_model_key": args.model,
        "scored_model_id": model_spec["id"],
        "scored_model_revision": model_spec["revision"],
        "role": args.role,
        "split": args.split,
        "prompt_count": len(prompts),
        "prompt_ids_sha256": prompt_sha,
        "chunk_positions": args.chunk_positions,
        "requested_batch_size": args.batch_size,
        "dtype": config["inference"]["dtype"],
        "attention_backend": config["inference"]["attention_backend"],
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": str(model_path_used),
        "prompt_pool_sha256": sha256_file(ROOT / config["data"]["prompts"]),
    }
    if metadata_path.exists() and not args.force:
        old = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
        for field in ("continuation", "target_key", "scored_model_key", "split", "prompt_count", "prompt_ids_sha256"):
            if old.get(field) != metadata.get(field):
                raise RuntimeError(f"Existing TF metadata differs at {field}; use --force or another output directory")

    shard_size = int(config["data"]["shard_prompts"])
    source_by_id = {str(row["doc_id"]): row for row in prompts}
    shard_inputs = list(prompt_shards(prompts, shard_size))
    pending = []
    previous_shards = {}
    old_meta = {}
    if metadata_path.exists():
        old_meta = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
        previous_shards = {int(row["shard"]): row for row in old_meta.get("completed_shards", [])}
    for shard_index, shard_prompts in shard_inputs:
        shard_path = destination / f"shard-{shard_index:05d}.parquet"
        if shard_path.exists() and not args.force:
            print(f"TF_RESUME_SKIP {shard_path}", flush=True)
            if shard_index not in previous_shards:
                previous_shards[shard_index] = {
                    "shard": shard_index,
                    "prompt_count": len(shard_prompts),
                    "scored_token_count": None,
                    "resumed_from_parquet": True,
                }
            continue
        pending.append((shard_index, shard_prompts, shard_path))
    if not pending:
        atomic_json(metadata_path, {**metadata, "completed_shards": [previous_shards[i] for i, _ in shard_inputs], "status": "COMPLETE"})
        print(f"TF_COMPLETE model={args.model} target={args.target} shards={len(shard_inputs)} (resumed)", flush=True)
        return 0

    print(f"TF_LOAD model={args.model} target={args.target} role={args.role}", flush=True)
    model = load_model(config, args.model, snapshot=args.model_path.expanduser().resolve() if args.model_path else None)
    body, head = _body_and_head(model)
    scale_state = {"scale": None, "max_abs_error": None}
    for shard_index, shard_prompts, shard_path in pending:
        shard_ids = {str(row["doc_id"]) for row in shard_prompts}
        ref_rows = [row for row in refs if str(row["doc_id"]) in shard_ids]
        ref_rows.sort(key=lambda row: int(source_by_id[str(row["doc_id"])]["pool_index"]))
        chunks = []
        by_length: dict[int, list[dict]] = defaultdict(list)
        for ref in ref_rows:
            by_length[len(ref["prompt_ids"])].append(ref)
        for length in sorted(by_length, reverse=True):
            values = by_length[length]
            for start in range(0, len(values), args.batch_size):
                chunks.append(values[start : start + args.batch_size])
        actual_batch_sizes: set[int] = set()
        result_rows = []
        started = time.monotonic()
        for chunk in tqdm(chunks, desc=f"TF {args.model} on {args.target} shard {shard_index}", unit="batch", dynamic_ncols=True, mininterval=1):
            scored = _score_with_backoff(
                model, body, head, chunk, args.role, args.chunk_positions, scale_state, actual_batch_sizes
            )
            for per_prompt in scored:
                result_rows.extend(per_prompt)
        atomic_parquet(result_rows, shard_path)
        previous_shards[shard_index] = {
            "shard": shard_index,
            "prompt_count": len(shard_prompts),
            "scored_token_count": len(result_rows),
            "actual_batch_sizes": sorted(actual_batch_sizes),
            "head_scale": scale_state["scale"],
            "head_check_max_abs_error": scale_state["max_abs_error"],
            "wall_seconds": time.monotonic() - started,
        }
        print(f"TF_SHARD_COMPLETE model={args.model} shard={shard_index:05d} prompts={len(shard_prompts)} tokens={len(result_rows)}", flush=True)

    del body, head, model
    cleanup_cuda(torch)
    atomic_json(metadata_path, {**metadata, "completed_shards": [previous_shards[i] for i, _ in shard_inputs], "status": "COMPLETE"})
    print(f"TF_COMPLETE model={args.model} target={args.target} files={len(list(destination.glob('shard-*.parquet')))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
