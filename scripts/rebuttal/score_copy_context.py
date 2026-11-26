#!/usr/bin/env python3
"""Rescore fixed human continuations after controlled context-copy interventions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tf_score import (
    _body_and_head,
    _build_human_refs,
    _score_with_backoff,
    _score_batch,
)
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    load_tokenizer,
    prompt_shards,
    select_prompts,
)
from src.rebuttal.runtime import write_gpu_metadata


SEED = 20261125
SHARD_PROMPTS = 100
REFERENCE_BATCH_SIZE = 32
CONDITIONS = ("A", "B", "C", "D")


def _contexts(refs, tokenizer, successor_refs):
    separator = list(map(int, tokenizer("\n\n", add_special_tokens=False)["input_ids"]))
    sorted_refs = sorted(successor_refs, key=lambda row: int(row["pool_index"]))
    next_by_id = {
        str(row["doc_id"]): sorted_refs[index + 1]
        for index, row in enumerate(sorted_refs[:-1])
    }
    contexts: dict[str, dict[str, list[int]]] = {condition: {} for condition in CONDITIONS}
    selected_ids = {str(row["doc_id"]) for row in refs}
    for ref in refs:
        doc_id = str(ref["doc_id"])
        if doc_id not in selected_ids:
            continue
        prompt = list(map(int, ref["prompt_ids"]))
        continuation = list(map(int, ref["continuation_ids"]))
        decoded = tokenizer.decode(continuation, skip_special_tokens=False)
        eojeols = decoded.split()
        random.Random(SEED + int(ref["pool_index"])).shuffle(eojeols)
        shuffled_text = " ".join(eojeols)
        shuffled_ids = list(map(int, tokenizer(shuffled_text, add_special_tokens=False)["input_ids"]))
        next_ref = next_by_id.get(doc_id)
        if next_ref is None:
            raise RuntimeError(f"No next-document continuation available for {doc_id}")
        next_continuation = list(map(int, next_ref["continuation_ids"]))
        contexts["A"][doc_id] = prompt
        contexts["B"][doc_id] = prompt + continuation + separator + prompt
        contexts["C"][doc_id] = prompt + shuffled_ids + separator + prompt
        contexts["D"][doc_id] = prompt + next_continuation + separator + prompt
    return contexts, separator


def _read_stored(root: Path, target: str, model: str, wanted: set[str]) -> pd.DataFrame:
    directory = root / "teacher_forced/human" / target / model / "common"
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Stored human teacher-forced logs missing under {directory}")
    parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=["doc_id", "pool_index", "position", "argmax_id", "top2_margin"])
        frame = frame[frame.doc_id.astype(str).isin(wanted)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _shard_is_complete(path: Path, expected_docs: set[str], expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        saved = pd.read_parquet(path, columns=["doc_id", "position"])
        return (
            set(saved.doc_id.astype(str)) == expected_docs
            and len(saved) == expected_rows
        )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=("Q2", "M1"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rebuttal/copy_context")
    args = parser.parse_args()
    if args.max_prompts < 1 or args.batch_size < 1:
        parser.error("max-prompts and batch-size must be positive")
    config = load_config(args.config)
    pair = config["pairs"][args.pair]
    target_key, draft_key = pair["target"], pair["draft"]
    tokenizer = load_tokenizer(config, target_key)
    prompts = select_prompts(config, "common")
    all_refs, skipped = _build_human_refs(
        prompts,
        tokenizer,
        int(config["data"]["max_prompt_tokens"]),
        int(config["data"]["max_new_tokens"]),
    )
    all_refs.sort(key=lambda row: int(row["pool_index"]))
    refs = [row for row in all_refs if len(row["continuation_ids"]) == 128][:args.max_prompts]
    if len(refs) < args.max_prompts:
        raise RuntimeError(f"Only {len(refs)} complete human continuations found; requested {args.max_prompts}")
    context_map, separator = _contexts(refs, tokenizer, all_refs)
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_dir = output_root / args.pair
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "pair": args.pair,
        "target_model": target_key,
        "draft_model": draft_key,
        "target_revision": config["models"][target_key]["revision"],
        "draft_revision": config["models"][draft_key]["revision"],
        "doc_ids": [str(row["doc_id"]) for row in refs],
        "pool_indices": [int(row["pool_index"]) for row in refs],
        "conditions": list(CONDITIONS),
        "separator_token_ids": separator,
        "batch_size": args.batch_size,
        "seed": SEED,
    }
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("details", {}).get("run_identity") != identity:
            raise RuntimeError("Existing copy-context output identity differs; use a fresh output root")
    write_gpu_metadata(metadata_path, job="copy_context_scoring", details={
        "run_identity": identity,
        "human_refs_skipped": len(skipped),
    })

    draft_model = load_model(config, draft_key)
    target_model = load_model(config, target_key)
    draft_body, draft_head = _body_and_head(draft_model)
    target_body, target_head = _body_and_head(target_model)
    scale_states: dict[tuple[str, str], dict[str, Any]] = {}
    actual_batch_sizes: dict[str, set[int]] = defaultdict(set)
    selected_ids = {str(row["doc_id"]) for row in refs}
    original_shard_size = int(config["data"].get("shard_prompts", 2000))

    for condition in CONDITIONS:
        for model_key in (target_key, draft_key):
            model = target_model if model_key == target_key else draft_model
            body, head = _body_and_head(model)
            role = "target" if model_key == target_key else "draft"
            destination = output_dir / condition / model_key
            destination.mkdir(parents=True, exist_ok=True)
            scale_state = {"scale": None, "max_abs_error": None}
            scale_states[(condition, model_key)] = scale_state
            if condition == "A":
                # Recreate tf_score.py's original batch composition (all common
                # documents in each 2,000-prompt shard, grouped by prompt length)
                # before retaining only the first 2,000 complete continuations.
                model_tokenizer = load_tokenizer(config, model_key)
                model_refs, _ = _build_human_refs(
                    prompts,
                    model_tokenizer,
                    int(config["data"]["max_prompt_tokens"]),
                    int(config["data"]["max_new_tokens"]),
                )
                model_refs.sort(key=lambda row: int(row["pool_index"]))
                model_selected = [
                    row for row in model_refs
                    if len(row["continuation_ids"]) == 128
                ][:args.max_prompts]
                if [str(row["doc_id"]) for row in model_selected] != [str(row["doc_id"]) for row in refs]:
                    raise RuntimeError(f"{model_key} tokenizer selected a different complete-document set")
                work_dir = destination / "_reference_shards"
                work_dir.mkdir(parents=True, exist_ok=True)
                reference_parts = []
                for ref_shard_index, shard_refs in tqdm(
                    list(prompt_shards(model_refs, original_shard_size)),
                    desc=f"copy {args.pair}/A/{model_key} reference batches",
                ):
                    shard_selected_ids = {
                        str(row["doc_id"]) for row in shard_refs
                        if str(row["doc_id"]) in selected_ids
                    }
                    if not shard_selected_ids:
                        continue
                    shard_path = work_dir / f"shard-{ref_shard_index:05d}.parquet"
                    expected_rows = sum(
                        len(row["continuation_ids"]) for row in shard_refs
                        if str(row["doc_id"]) in shard_selected_ids
                    )
                    if _shard_is_complete(shard_path, shard_selected_ids, expected_rows):
                        print(f"COPY_CONTEXT_RESUME_SKIP {shard_path}", flush=True)
                        reference_parts.append(pd.read_parquet(shard_path))
                        continue

                    length_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for ref in shard_refs:
                        length_groups[len(ref["prompt_ids"])].append(ref)
                    records = []
                    for prompt_len in sorted(length_groups, reverse=True):
                        group = length_groups[prompt_len]
                        for batch_start in range(0, len(group), args.batch_size):
                            rows = group[batch_start:batch_start + args.batch_size]
                            scored = _score_with_backoff(
                                model, body, head, rows, role, 32, scale_state,
                                actual_batch_sizes[model_key],
                            )
                            records.extend(
                                record
                                for document_rows in scored
                                for record in document_rows
                                if str(record["doc_id"]) in shard_selected_ids
                            )
                    atomic_parquet(records, shard_path)
                    reference_parts.append(pd.DataFrame(records))
                    print(
                        f"COPY_CONTEXT_REFERENCE_SHARD_COMPLETE pair={args.pair} model={model_key} "
                        f"shard={ref_shard_index:05d} docs={len(shard_selected_ids)} positions={len(records)}",
                        flush=True,
                    )

                selected_records = pd.concat(reference_parts, ignore_index=True)
                for output_index, start in enumerate(range(0, len(refs), SHARD_PROMPTS)):
                    output_refs = refs[start:start + SHARD_PROMPTS]
                    output_docs = {str(row["doc_id"]) for row in output_refs}
                    output_records = selected_records[
                        selected_records.doc_id.astype(str).isin(output_docs)
                    ]
                    output_path = destination / f"shard-{output_index:05d}.parquet"
                    if len(output_records) != sum(len(row["continuation_ids"]) for row in output_refs):
                        raise RuntimeError(f"Reference-batched A output is incomplete for {model_key}/{output_index}")
                    atomic_parquet(output_records.to_dict("records"), output_path)
                    print(
                        f"COPY_CONTEXT_SHARD_COMPLETE pair={args.pair} condition=A model={model_key} "
                        f"shard={output_index:05d} positions={len(output_records)}",
                        flush=True,
                    )
            else:
                for shard_start in tqdm(
                    range(0, len(refs), SHARD_PROMPTS),
                    desc=f"copy {args.pair}/{condition}/{model_key}",
                ):
                    shard_index = shard_start // SHARD_PROMPTS
                    shard_path = destination / f"shard-{shard_index:05d}.parquet"
                    shard_refs = refs[shard_start:shard_start + SHARD_PROMPTS]
                    expected_docs = {str(row["doc_id"]) for row in shard_refs}
                    expected_rows = sum(len(row["continuation_ids"]) for row in shard_refs)
                    if _shard_is_complete(shard_path, expected_docs, expected_rows):
                        print(f"COPY_CONTEXT_RESUME_SKIP {shard_path}", flush=True)
                        continue

                    length_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for ref in shard_refs:
                        doc_id = str(ref["doc_id"])
                        length_groups[len(context_map[condition][doc_id])].append({
                            **ref,
                            "prompt_ids": context_map[condition][doc_id],
                        })
                    records = []
                    for context_len in sorted(length_groups):
                        group = length_groups[context_len]
                        for batch_start in range(0, len(group), args.batch_size):
                            rows = group[batch_start:batch_start + args.batch_size]
                            scored = _score_with_backoff(
                                model, body, head, rows, role, 32, scale_state,
                                actual_batch_sizes[model_key],
                            )
                            records.extend(record for document_rows in scored for record in document_rows)
                    atomic_parquet(records, shard_path)
                    print(
                        f"COPY_CONTEXT_SHARD_COMPLETE pair={args.pair} condition={condition} "
                        f"model={model_key} shard={shard_index:05d} positions={len(records)}",
                        flush=True,
                    )

    # Condition A must reproduce the published human teacher-forced scores.
    consistency = {}
    for model_key, prefix in ((target_key, "target"), (draft_key, "draft")):
        current_parts = []
        for path in sorted((output_dir / "A" / model_key).glob("shard-*.parquet")):
            current_parts.append(pd.read_parquet(path))
        current = pd.concat(current_parts, ignore_index=True)
        stored = _read_stored(ROOT / "results/naacl_experiments", target_key, model_key, set(identity["doc_ids"]))
        joined = current[["doc_id", "pool_index", "position", "argmax_id", "top2_margin"]].merge(
            stored, on=["doc_id", "pool_index", "position"], suffixes=("_new", "_stored"),
            how="inner", validate="one_to_one",
        )
        if len(joined) != len(current):
            raise RuntimeError(f"Condition A join for {model_key} missed {len(current)-len(joined)} positions")
        consistency[model_key] = {
            "positions": int(len(joined)),
            "argmax_agreement": float((joined.argmax_id_new == joined.argmax_id_stored).mean()),
            "median_abs_top2_margin_delta": float((joined.top2_margin_new - joined.top2_margin_stored).abs().median()),
        }
    atomic_json(output_dir / "consistency_A.json", consistency)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["details"].update({
        "actual_batch_sizes": {key: sorted(value) for key, value in actual_batch_sizes.items()},
        "scales": {f"{condition}/{model}": value for (condition, model), value in scale_states.items()},
        "condition_A_consistency": consistency,
    })
    atomic_json(metadata_path, metadata)
    if any(row["argmax_agreement"] < 0.985 or row["median_abs_top2_margin_delta"] != 0.0 for row in consistency.values()):
        raise RuntimeError(f"Condition A consistency gate failed; inspect {output_dir / 'consistency_A.json'}")
    del draft_model, target_model
    cleanup_cuda(torch)
    print(f"COPY_CONTEXT_COMPLETE pair={args.pair} documents={len(refs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
