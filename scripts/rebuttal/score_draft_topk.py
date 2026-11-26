#!/usr/bin/env python3
"""Store draft top-eight distributions on frozen target continuations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tf_score import (
    _body_and_head,
    _calibrate_scale,
    _collate,
    _hidden_states,
    _load_generated_refs,
)
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    load_tokenizer,
    output_root,
    prompt_shards,
    select_prompts,
)
from src.rebuttal.runtime import write_gpu_metadata


TOP_K = 8
CHUNK_POSITIONS = 32
SHARD_PROMPTS = 100
SEED = 20261125


@torch.inference_mode()
def _score_batch(model, body, head, rows, scale_state):
    device = next(model.parameters()).device
    pad_id = int(getattr(model.config, "pad_token_id", None) or 0)
    input_ids, attention_mask, position_ids = _collate(rows, pad_id, device)
    prompt_len = len(rows[0]["prompt_ids"])
    hidden = _hidden_states(body, input_ids, attention_mask, position_ids)
    if scale_state["scale"] is None:
        scale, error = _calibrate_scale(model, body, head, input_ids, attention_mask, position_ids, prompt_len)
        scale_state.update(scale=scale, max_abs_error=error)
        print(f"DRAFT_TOPK_HEAD_CHECK_OK scale={scale:.9g} max_abs_error={error:.4g}", flush=True)

    labels = [list(map(int, row["continuation_ids"])) for row in rows]
    outputs: list[list[dict[str, Any]]] = [[] for _ in rows]
    batch_indices = torch.arange(len(rows), device=device)
    max_targets = max(map(len, labels))
    for start in range(0, max_targets, CHUNK_POSITIONS):
        end = min(max_targets, start + CHUNK_POSITIONS)
        positions = torch.arange(prompt_len - 1 + start, prompt_len - 1 + end, device=device)
        selected = hidden[batch_indices[:, None], positions[None, :]]
        logits = head(selected).float() * scale_state["scale"]
        normalizer = torch.logsumexp(logits, dim=-1)
        top2_values, top2_ids = torch.topk(logits, k=2, dim=-1)
        top8_values, top8_ids = torch.topk(logits, k=TOP_K, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = normalizer - (probabilities * logits).sum(dim=-1)
        for row_index, row in enumerate(rows):
            count = max(0, min(end, len(labels[row_index])) - start)
            for offset in range(count):
                position = start + offset
                top1_id = int(top2_ids[row_index, offset, 0].item())
                ids8 = [int(value) for value in top8_ids[row_index, offset].tolist()]
                values8 = [float(value) for value in top8_values[row_index, offset].tolist()]
                if top1_id not in ids8:
                    ids8[-1] = top1_id
                    values8[-1] = float(logits[row_index, offset, top1_id].item())
                if ids8[0] != top1_id:
                    top1_index = ids8.index(top1_id)
                    ids8 = [ids8[top1_index], *ids8[:top1_index], *ids8[top1_index + 1:]]
                    values8 = [values8[top1_index], *values8[:top1_index], *values8[top1_index + 1:]]
                outputs[row_index].append({
                    "doc_id": str(row["doc_id"]),
                    "pool_index": int(row["pool_index"]),
                    "position": position,
                    "target_token_id": labels[row_index][position],
                    "argmax_id": top1_id,
                    "logp_argmax": float((top2_values[row_index, offset, 0] - normalizer[row_index, offset]).item()),
                    "entropy": float(entropy[row_index, offset].item()),
                    "top2_margin": float((top2_values[row_index, offset, 0] - top2_values[row_index, offset, 1]).item()),
                    "top8_ids": ids8,
                    "top8_logp": [value - float(normalizer[row_index, offset].item()) for value in values8],
                })
        del selected, logits, normalizer, top2_values, top2_ids, top8_values, top8_ids, probabilities, entropy
    del hidden
    return outputs


def _score_with_backoff(model, body, head, rows, scale_state, batch_sizes):
    try:
        output = _score_batch(model, body, head, rows, scale_state)
        batch_sizes.add(len(rows))
        return output
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda(torch)
        if len(rows) == 1:
            raise RuntimeError("Draft top-k scoring ran out of GPU memory at batch size 1")
        midpoint = len(rows) // 2
        return (
            _score_with_backoff(model, body, head, rows[:midpoint], scale_state, batch_sizes)
            + _score_with_backoff(model, body, head, rows[midpoint:], scale_state, batch_sizes)
        )


def _read_stored(root: Path, continuation: str, target: str, draft: str, split: str, wanted: set[str]):
    directory = root / "teacher_forced" / continuation / target / draft / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Stored draft teacher-forced data missing under {directory}")
    parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=["doc_id", "pool_index", "position", "argmax_id", "top2_margin"])
        frame = frame[frame.doc_id.astype(str).isin(wanted)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rebuttal/draft_topk")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    if args.max_prompts < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("max-prompts and batch-size must be positive")
    pair = config["pairs"][args.pair]
    target_key, draft_key, split = pair["target"], pair["draft"], pair["split"]
    tokenizer = load_tokenizer(config, target_key)
    prompts = select_prompts(config, "common")[:args.max_prompts]
    refs = _load_generated_refs(output_root(config), target_key, split, prompts)
    if len(refs) != len(prompts):
        raise RuntimeError(f"Frozen target references cover {len(refs)} of {len(prompts)} selected prompts")
    reference_shard_size = int(config["data"].get("shard_prompts", 2000))
    reference_prompt_count = (
        (args.max_prompts + reference_shard_size - 1) // reference_shard_size
    ) * reference_shard_size
    reference_prompts = select_prompts(config, split)[:reference_prompt_count]
    reference_refs = _load_generated_refs(output_root(config), target_key, split, reference_prompts)
    docs = [str(row["doc_id"]) for row in refs]
    selected_docs = set(docs)
    reference_docs = {str(row["doc_id"]) for row in reference_refs}
    if not selected_docs.issubset(reference_docs):
        raise RuntimeError("Original teacher-forcing batches do not cover selected target references")

    frozen_tf_dir = (
        output_root(config) / "teacher_forced" / "generated" / target_key / draft_key / split
    )
    frozen_metadata = json.loads((frozen_tf_dir / "run_metadata.json").read_text(encoding="utf-8"))
    batch_size = args.batch_size or int(frozen_metadata.get("requested_batch_size", 32))

    output_root_path = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_dir = output_root_path / "teacher_forced/generated" / target_key / draft_key / "common"
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "pair": args.pair,
        "target_model": target_key,
        "draft_model": draft_key,
        "target_revision": config["models"][target_key]["revision"],
        "draft_revision": config["models"][draft_key]["revision"],
        "frozen_source_root": str(output_root(config)),
        "source_split": split,
        "doc_ids": docs,
        "top_k": TOP_K,
        "batch_size": batch_size,
        "reference_batching": {
            "rows": "frozen_generated_refs",
            "reference_doc_count": len(reference_refs),
            "shard_prompts": reference_shard_size,
            "sort_prompt_lengths": "descending",
        },
        "seed": SEED,
    }
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("details", {}).get("run_identity") != identity:
            raise RuntimeError("Existing draft top-k output identity differs; use another output root")
    write_gpu_metadata(metadata_path, job="draft_topk_scoring", details={"run_identity": identity})

    model = load_model(config, draft_key)
    body, head = _body_and_head(model)
    scale_state: dict[str, Any] = {"scale": None, "max_abs_error": None}
    batch_sizes: set[int] = set()
    saved_paths = []
    work_dir = output_dir / "_reference_shards"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_parts = []
    for shard_index, shard_refs in tqdm(
        list(prompt_shards(reference_refs, reference_shard_size)),
        desc=f"draft-topk {args.pair} reference batches",
    ):
        shard_docs = {
            str(row["doc_id"]) for row in shard_refs
            if str(row["doc_id"]) in selected_docs
        }
        if not shard_docs:
            continue
        work_path = work_dir / f"shard-{shard_index:05d}.parquet"
        expected_rows = sum(
            len(row["continuation_ids"]) for row in shard_refs
            if str(row["doc_id"]) in shard_docs
        )
        if work_path.exists():
            try:
                saved = pd.read_parquet(work_path, columns=["doc_id", "position", "top8_ids"])
                if set(saved.doc_id.astype(str)) == shard_docs and len(saved) == expected_rows:
                    print(f"DRAFT_TOPK_REFERENCE_RESUME_SKIP {work_path}", flush=True)
                    reference_parts.append(pd.read_parquet(work_path))
                    continue
            except Exception:
                pass
            work_path.unlink(missing_ok=True)
        rows = []
        length_groups: dict[int, list[dict[str, Any]]] = {}
        for ref in shard_refs:
            length_groups.setdefault(len(ref["prompt_ids"]), []).append(ref)
        for prompt_length in sorted(length_groups, reverse=True):
            group = length_groups[prompt_length]
            for batch_start in range(0, len(group), batch_size):
                batch = group[batch_start:batch_start + batch_size]
                outputs = _score_with_backoff(model, body, head, batch, scale_state, batch_sizes)
                rows.extend(
                    record
                    for document_rows in outputs
                    for record in document_rows
                    if str(record["doc_id"]) in shard_docs
                )
        atomic_parquet(rows, work_path)
        reference_parts.append(pd.DataFrame(rows))
        print(
            f"DRAFT_TOPK_REFERENCE_SHARD_COMPLETE pair={args.pair} shard={shard_index:05d} "
            f"docs={len(shard_docs)} positions={len(rows)}",
            flush=True,
        )

    selected_output = pd.concat(reference_parts, ignore_index=True)
    for output_index, start in enumerate(range(0, len(refs), SHARD_PROMPTS)):
        output_refs = refs[start:start + SHARD_PROMPTS]
        output_docs = {str(row["doc_id"]) for row in output_refs}
        output_records = selected_output[
            selected_output.doc_id.astype(str).isin(output_docs)
        ]
        shard_path = output_dir / f"shard-{output_index:05d}.parquet"
        if len(output_records) != sum(len(row["continuation_ids"]) for row in output_refs):
            raise RuntimeError(f"Draft top-k output shard {output_index} is incomplete")
        atomic_parquet(output_records.to_dict("records"), shard_path)
        saved_paths.append(shard_path)
        print(
            f"DRAFT_TOPK_SHARD_COMPLETE pair={args.pair} shard={output_index:05d} "
            f"positions={len(output_records)}",
            flush=True,
        )

    output = pd.concat([pd.read_parquet(path) for path in saved_paths], ignore_index=True)
    stored_split = split
    stored = _read_stored(output_root(config), "generated", target_key, draft_key, stored_split, set(docs))
    joined = output[["doc_id", "pool_index", "position", "argmax_id", "top2_margin"]].merge(
        stored, on=["doc_id", "pool_index", "position"], suffixes=("_new", "_stored"),
        how="inner", validate="one_to_one",
    )
    if len(joined) != len(output):
        raise RuntimeError(f"Draft log consistency join covered {len(joined)} of {len(output)} positions")
    consistency = {
        "positions": int(len(joined)),
        "argmax_agreement": float((joined.argmax_id_new == joined.argmax_id_stored).mean()),
        "median_abs_top2_margin_delta": float((joined.top2_margin_new - joined.top2_margin_stored).abs().median()),
    }
    atomic_json(output_dir / "consistency.json", consistency)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["details"].update({"actual_batch_sizes": sorted(batch_sizes), "scale": scale_state, "consistency": consistency})
    atomic_json(metadata_path, metadata)
    if consistency["argmax_agreement"] < 0.985 or consistency["median_abs_top2_margin_delta"] != 0.0:
        raise RuntimeError(f"Draft top-k consistency gate failed; inspect {output_dir / 'consistency.json'}")
    del model
    cleanup_cuda(torch)
    print(f"DRAFT_TOPK_COMPLETE pair={args.pair} docs={len(docs)} positions={len(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
