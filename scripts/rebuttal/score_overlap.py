#!/usr/bin/env python3
"""Score exact speculative-sampling overlap on human or generated continuations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tf_score import (
    _body_and_head,
    _build_human_refs,
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


SEED = 20261125
CHUNK_POSITIONS = 32
SHARD_PROMPTS = 100
REFERENCE_BATCH_SIZE = 32


def _logit_statistics(
    logits: torch.Tensor, scale: float, top_count: int = 2
) -> dict[str, torch.Tensor]:
    values = logits.float() * scale
    normalizer = torch.logsumexp(values, dim=-1)
    top_values, top_ids = torch.topk(values, k=top_count, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    entropy = normalizer - (probabilities * values).sum(dim=-1)
    return {
        "argmax_id": top_ids[..., 0],
        "top2_margin": top_values[..., 0] - top_values[..., 1],
        "entropy": entropy,
        "logp_argmax": top_values[..., 0] - normalizer,
    }


@torch.inference_mode()
def _score_batch(draft, target, draft_body, draft_head, target_body, target_head,
                 rows, scales, chunk_positions=CHUNK_POSITIONS):
    device = next(target.parameters()).device
    target_pad = int(getattr(target.config, "pad_token_id", None) or 0)
    draft_pad = int(getattr(draft.config, "pad_token_id", None) or 0)
    if target_pad != draft_pad:
        raise RuntimeError("Draft and target padding IDs differ; T2a requires their stored tokenizer pair")
    input_ids, attention_mask, position_ids = _collate(rows, target_pad, device)
    prompt_len = len(rows[0]["prompt_ids"])
    target_hidden = _hidden_states(target_body, input_ids, attention_mask, position_ids)
    draft_hidden = _hidden_states(draft_body, input_ids, attention_mask, position_ids)
    if scales["target"] is None:
        scales["target"], scales["target_calibration_error"] = _calibrate_scale(
            target, target_body, target_head, input_ids, attention_mask, position_ids, prompt_len
        )
    if scales["draft"] is None:
        scales["draft"], scales["draft_calibration_error"] = _calibrate_scale(
            draft, draft_body, draft_head, input_ids, attention_mask, position_ids, prompt_len
        )

    labels = [list(map(int, row["continuation_ids"])) for row in rows]
    valid_counts = [len(sequence) for sequence in labels]
    outputs: list[list[dict[str, Any]]] = [[] for _ in rows]
    batch_indices = torch.arange(len(rows), device=device)
    max_targets = max(valid_counts)
    for start in range(0, max_targets, chunk_positions):
        end = min(max_targets, start + chunk_positions)
        sequence_positions = torch.arange(
            prompt_len - 1 + start,
            prompt_len - 1 + end,
            dtype=torch.long,
            device=device,
        )
        positions = batch_indices[:, None]
        selected = target_hidden[positions, sequence_positions[None, :]]
        target_logits = target_head(selected).float() * scales["target"]
        selected_draft = draft_hidden[positions, sequence_positions[None, :]]
        draft_logits = draft_head(selected_draft).float() * scales["draft"]

        # tf_score.py persists the target argmax as topk(20)[..., 0] and the
        # draft argmax as topk(2)[..., 0]. Match both tie-breaking paths.
        t_stats = _logit_statistics(target_logits, 1.0, top_count=20)
        d_stats = _logit_statistics(draft_logits, 1.0, top_count=2)
        alpha_by_temp = {}
        for name, temperature in (("alpha_T1", 1.0), ("alpha_T07", 0.7)):
            t_logp = torch.log_softmax(target_logits / temperature, dim=-1)
            d_logp = torch.log_softmax(draft_logits / temperature, dim=-1)
            alpha_by_temp[name] = torch.minimum(t_logp, d_logp).exp().sum(dim=-1)
            del t_logp, d_logp

        for row_index, row in enumerate(rows):
            count = max(0, min(end, valid_counts[row_index]) - start)
            for offset in range(count):
                position = start + offset
                outputs[row_index].append({
                    "doc_id": str(row["doc_id"]),
                    "pool_index": int(row["pool_index"]),
                    "position": position,
                    "target_token_id": labels[row_index][position],
                    "alpha_T1": float(alpha_by_temp["alpha_T1"][row_index, offset].item()),
                    "alpha_T07": float(alpha_by_temp["alpha_T07"][row_index, offset].item()),
                    "target_argmax_id": int(t_stats["argmax_id"][row_index, offset].item()),
                    "target_top2_margin": float(t_stats["top2_margin"][row_index, offset].item()),
                    "target_entropy": float(t_stats["entropy"][row_index, offset].item()),
                    "target_logp_argmax": float(t_stats["logp_argmax"][row_index, offset].item()),
                    "draft_argmax_id": int(d_stats["argmax_id"][row_index, offset].item()),
                    "draft_top2_margin": float(d_stats["top2_margin"][row_index, offset].item()),
                    "draft_entropy": float(d_stats["entropy"][row_index, offset].item()),
                    "draft_logp_argmax": float(d_stats["logp_argmax"][row_index, offset].item()),
                })
        del selected, selected_draft, target_logits, draft_logits, t_stats, d_stats, alpha_by_temp
    del target_hidden, draft_hidden
    return outputs


def _score_with_backoff(models, rows, scales, batch_sizes):
    try:
        scored = _score_batch(*models, rows, scales)
        batch_sizes.add(len(rows))
        return scored
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda(torch)
        if len(rows) == 1:
            raise RuntimeError("Overlap scoring ran out of GPU memory at batch size 1")
        midpoint = len(rows) // 2
        return (
            _score_with_backoff(models, rows[:midpoint], scales, batch_sizes)
            + _score_with_backoff(models, rows[midpoint:], scales, batch_sizes)
        )


def _read_stored(root: Path, continuation: str, target_key: str, model_key: str,
                 split: str, wanted: set[str]) -> pd.DataFrame:
    directory = root / "teacher_forced" / continuation / target_key / model_key / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Stored teacher-forced shards missing under {directory}")
    parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=["doc_id", "pool_index", "position", "argmax_id", "top2_margin"])
        frame = frame[frame.doc_id.astype(str).isin(wanted)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _consistency(output: pd.DataFrame, stored: pd.DataFrame, model_prefix: str) -> dict[str, Any]:
    actual = output[["doc_id", "pool_index", "position", f"{model_prefix}_argmax_id", f"{model_prefix}_top2_margin"]]
    joined = actual.merge(stored, on=["doc_id", "pool_index", "position"], how="inner", validate="one_to_one")
    if len(joined) != len(output):
        raise RuntimeError(f"Consistency join for {model_prefix} covered {len(joined)} of {len(output)} positions")
    agreement = float((joined[f"{model_prefix}_argmax_id"] == joined.argmax_id).mean())
    median_delta = float((joined[f"{model_prefix}_top2_margin"] - joined.top2_margin).abs().median())
    return {"positions": int(len(joined)), "argmax_agreement": agreement, "median_abs_top2_margin_delta": median_delta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--continuation", required=True, choices=("human", "generated", "sampled"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-positions", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rebuttal/overlap")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    pair = config["pairs"][args.pair]
    draft_key, target_key = pair["draft"], pair["target"]
    if args.continuation == "human" and config["models"][target_key].get("family") == "Gemma 3":
        parser.error("Human text is excluded for Gemma; use generated continuations")
    if args.max_prompts < 1 or args.chunk_positions < 1:
        parser.error("max-prompts and chunk-positions must be positive")
    batch_size = args.batch_size or (
        2 if config["models"][target_key].get("family") == "Gemma 3" else REFERENCE_BATCH_SIZE
    )
    if batch_size < 1:
        parser.error("batch-size must be positive")

    tokenizer = load_tokenizer(config, target_key)
    common_prompts = select_prompts(config, "common")
    reference_refs = None
    if args.continuation == "human":
        prompts = common_prompts
        reference_refs, _ = _build_human_refs(prompts, tokenizer, 128, 128)
        reference_refs.sort(key=lambda row: int(row["pool_index"]))
        refs = [row for row in reference_refs if len(row["continuation_ids"]) == 128]
        refs = refs[:args.max_prompts]
        split = "common"
    elif args.continuation == "generated":
        split = pair["split"]
        prompts = common_prompts[:args.max_prompts]
        refs = _load_generated_refs(output_root(config), target_key, split, prompts)
        if len(refs) != len(prompts):
            raise RuntimeError(f"Generated references cover {len(refs)} of {len(prompts)} requested documents")
        reference_shard_size = int(config["data"].get("shard_prompts", 2000))
        reference_prompt_count = (
            (args.max_prompts + reference_shard_size - 1) // reference_shard_size
        ) * reference_shard_size
        reference_prompts = select_prompts(config, split)[:reference_prompt_count]
        reference_refs = _load_generated_refs(
            output_root(config), target_key, split, reference_prompts
        )
        selected_reference_ids = {str(row["doc_id"]) for row in refs}
        if not selected_reference_ids.issubset({str(row["doc_id"]) for row in reference_refs}):
            raise RuntimeError("Original generated reference batches do not cover the selected documents")
    else:
        split = "common"
        prompts = common_prompts[:args.max_prompts]
        sampled_root = ROOT / "results/rebuttal/sampled_T1"
        refs = _load_generated_refs(sampled_root, target_key, "common", prompts)
        if len(refs) != len(prompts):
            raise RuntimeError(f"Sampled references cover {len(refs)} of {len(prompts)} requested documents")
    if not refs:
        raise RuntimeError(f"No {args.continuation} references found for {args.pair}")
    selected_ids = {str(row["doc_id"]) for row in refs}

    output_root_path = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_dir = output_root_path / args.pair / args.continuation
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = [str(row["doc_id"]) for row in refs]
    identity = {
        "pair": args.pair,
        "draft_model": draft_key,
        "target_model": target_key,
        "draft_revision": config["models"][draft_key]["revision"],
        "target_revision": config["models"][target_key]["revision"],
        "continuation": args.continuation,
        "source_split": split,
        "doc_ids": docs,
        "batch_size": batch_size,
        "chunk_positions": args.chunk_positions,
        "seed": SEED,
        "reference_batching": (
            {
                "rows": "all_human_refs" if args.continuation == "human" else "frozen_generated_refs",
                "reference_doc_count": len(reference_refs),
                "shard_prompts": int(config["data"].get("shard_prompts", 2000)),
                "sort_prompt_lengths": "descending",
            }
            if reference_refs is not None else None
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("details", {}).get("run_identity") != identity:
            raise RuntimeError("Existing overlap output identity differs; use a fresh output root")
    write_gpu_metadata(
        metadata_path,
        job="sampling_overlap",
        details={"run_identity": identity},
    )

    draft_model = load_model(config, draft_key)
    target_model = load_model(config, target_key)
    draft_body, draft_head = _body_and_head(draft_model)
    target_body, target_head = _body_and_head(target_model)
    models = (draft_model, target_model, draft_body, draft_head, target_body, target_head)
    scales: dict[str, Any] = {"target": None, "draft": None}
    used_batch_sizes: set[int] = set()
    saved_paths = []
    selected_docs = selected_ids
    if reference_refs is not None:
        work_dir = output_dir / "_reference_shards"
        work_dir.mkdir(parents=True, exist_ok=True)
        reference_parts = []
        reference_shard_size = int(config["data"].get("shard_prompts", 2000))
        for shard_index, shard_refs in tqdm(
            list(prompt_shards(reference_refs, reference_shard_size)),
            desc=f"overlap {args.pair}/{args.continuation} reference batches",
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
                    saved = pd.read_parquet(work_path, columns=["doc_id", "position"])
                    if set(saved.doc_id.astype(str)) == shard_docs and len(saved) == expected_rows:
                        print(f"OVERLAP_REFERENCE_RESUME_SKIP {work_path}", flush=True)
                        reference_parts.append(pd.read_parquet(work_path))
                        continue
                except Exception:
                    pass
                work_path.unlink(missing_ok=True)

            length_groups: dict[int, list[dict[str, Any]]] = {}
            for row in shard_refs:
                length_groups.setdefault(len(row["prompt_ids"]), []).append(row)
            records = []
            for prompt_length in sorted(length_groups, reverse=True):
                group = length_groups[prompt_length]
                for batch_start in range(0, len(group), batch_size):
                    rows = group[batch_start:batch_start + batch_size]
                    scored = _score_with_backoff(models, rows, scales, used_batch_sizes)
                    records.extend(
                        record
                        for document_rows in scored
                        for record in document_rows
                        if str(record["doc_id"]) in shard_docs
                    )
            atomic_parquet(records, work_path)
            reference_parts.append(pd.DataFrame(records))
            print(
                f"OVERLAP_REFERENCE_SHARD_COMPLETE pair={args.pair} continuation={args.continuation} "
                f"shard={shard_index:05d} docs={len(shard_docs)} positions={len(records)}",
                flush=True,
            )

        selected_outputs = pd.concat(reference_parts, ignore_index=True)
        for output_index, start in enumerate(range(0, len(refs), SHARD_PROMPTS)):
            output_refs = refs[start:start + SHARD_PROMPTS]
            output_docs = {str(row["doc_id"]) for row in output_refs}
            output_records = selected_outputs[
                selected_outputs.doc_id.astype(str).isin(output_docs)
            ]
            output_path = output_dir / f"shard-{output_index:05d}.parquet"
            if len(output_records) != sum(len(row["continuation_ids"]) for row in output_refs):
                raise RuntimeError(f"Human overlap output is incomplete for selected shard {output_index}")
            atomic_parquet(output_records.to_dict("records"), output_path)
            saved_paths.append(output_path)
            print(
                f"OVERLAP_SHARD_COMPLETE pair={args.pair} continuation=human "
                f"shard={output_index:05d} positions={len(output_records)}",
                flush=True,
            )
    else:
        for shard_start in tqdm(range(0, len(refs), SHARD_PROMPTS), desc=f"overlap {args.pair}/{args.continuation}"):
            shard_index = shard_start // SHARD_PROMPTS
            shard_path = output_dir / f"shard-{shard_index:05d}.parquet"
            shard_refs = refs[shard_start:shard_start + SHARD_PROMPTS]
            if shard_path.exists():
                try:
                    saved = pd.read_parquet(shard_path, columns=["doc_id", "position"])
                    expected_docs = {str(row["doc_id"]) for row in shard_refs}
                    if set(saved.doc_id.astype(str)) == expected_docs and len(saved) == sum(len(row["continuation_ids"]) for row in shard_refs):
                        print(f"OVERLAP_RESUME_SKIP {shard_path}", flush=True)
                        saved_paths.append(shard_path)
                        continue
                except Exception:
                    pass
                shard_path.unlink(missing_ok=True)
            records = []
            length_groups: dict[int, list[dict[str, Any]]] = {}
            for row in shard_refs:
                length_groups.setdefault(len(row["prompt_ids"]), []).append(row)
            for prompt_length in sorted(length_groups, reverse=True):
                group = length_groups[prompt_length]
                for batch_start in range(0, len(group), batch_size):
                    rows = group[batch_start:batch_start + batch_size]
                    scored = _score_with_backoff(models, rows, scales, used_batch_sizes)
                    records.extend(record for document_rows in scored for record in document_rows)
            atomic_parquet(records, shard_path)
            saved_paths.append(shard_path)
            print(f"OVERLAP_SHARD_COMPLETE pair={args.pair} continuation={args.continuation} shard={shard_index:05d} positions={len(records)}", flush=True)

    outputs = pd.concat([pd.read_parquet(path) for path in saved_paths], ignore_index=True)
    output_root_for_tf = output_root(config)
    stored_split = split
    if args.continuation == "human":
        stored_split = "common"
    elif args.pair == "Q2" and args.continuation == "generated":
        stored_split = "all"
    wanted = set(docs)
    target_stored = _read_stored(output_root_for_tf, args.continuation, target_key, target_key, stored_split, wanted)
    draft_stored = _read_stored(output_root_for_tf, args.continuation, target_key, draft_key, stored_split, wanted)
    checks = {
        "target": _consistency(outputs, target_stored, "target"),
        "draft": _consistency(outputs, draft_stored, "draft"),
    }
    atomic_json(output_dir / "consistency.json", checks)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["details"].update({
        "actual_batch_sizes": sorted(used_batch_sizes),
        "scales": scales,
        "consistency": checks,
    })
    atomic_json(metadata_path, metadata)
    if any(value["argmax_agreement"] < 0.985 or value["median_abs_top2_margin_delta"] != 0.0 for value in checks.values()):
        raise RuntimeError(f"T2a consistency gate failed; inspect {output_dir / 'consistency.json'} before analysis")
    del draft_model, target_model
    cleanup_cuda(torch)
    print(f"OVERLAP_COMPLETE pair={args.pair} continuation={args.continuation} docs={len(docs)} positions={len(outputs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
