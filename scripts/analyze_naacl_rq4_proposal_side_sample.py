#!/usr/bin/env python3
"""Project Q2 draft candidates onto Kiwi morphology on a selected prompt pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from transformers import AutoTokenizer

from analyze_naacl_cpu import fit_cross_ame, load_pair_frame
from src.morphology_guard import project_candidate_block
from src.naacl_experiment import atomic_json, load_config, output_root


SEED = 20261012
PAIR = "Q2"


def boundary_environment(record: dict[str, Any]) -> str:
    morph_class = str(record.get("h2_morph_class", ""))
    if morph_class != "CROSS_MORPHEME":
        return ""
    values = str(record.get("crossed_boundary_sequence") or "").split("|")
    return values[0] if int(record.get("num_boundaries_crossed", 0)) == 1 and values else "MULTI_BOUNDARY"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--split", choices=("common", "all"), default="common")
    parser.add_argument("--output-stem", default="rq4_proposal_side_sample")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    split_limit = 20_000 if args.split == "common" else 40_000
    if args.max_docs <= 0 or args.max_docs > split_limit:
        raise ValueError(f"--max-docs must be in 1..{split_limit:,} {args.split}-pool prompts")
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        raise ValueError("--output-stem must be a non-empty filename stem")
    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    prompt_pool = pd.read_parquet(ROOT / config["data"]["prompts"], columns=["doc_id", "pool_index"])
    split_pool = prompt_pool.loc[
        prompt_pool.pool_index.astype(int).lt(20_000) if args.split == "common"
        else prompt_pool.pool_index.astype(int).lt(40_000)
    ].sort_values("pool_index")
    if len(split_pool) < args.max_docs:
        raise AssertionError(f"Prompt file has {len(split_pool)} rows in {args.split}, expected at least {args.max_docs}")
    selected = split_pool.iloc[np.sort(rng.choice(len(split_pool), size=args.max_docs, replace=False))]
    selected_ids = set(selected.doc_id.astype(str))
    selected_pool_indices = sorted(selected.pool_index.astype(int).tolist())
    print(f"PROPOSAL_PROJECTION_DOCS split={args.split} n={len(selected_ids):,} seed={args.seed}", flush=True)

    model_spec = config["models"]["qwen06"]
    print("LOAD_PINNED_QWEN06_TOKENIZER", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], use_fast=True, local_files_only=False,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Proposal-side projection requires the pinned fast Qwen tokenizer")
    backend_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    frequency_metadata = json.loads((ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache_metadata.json").read_text(encoding="utf-8"))
    if backend_sha256 != frequency_metadata["tokenizer_backend_sha256"]:
        raise RuntimeError("Downloaded Qwen06 tokenizer backend hash differs from the frozen Qwen tokenizer")
    kiwi = Kiwi()

    continuations = []
    continuation_dir = root / "continuations" / "qwen4" / "all"
    for path in sorted(continuation_dir.glob("shard-*.parquet")):
        part = pd.read_parquet(path, columns=["doc_id", "pool_index", "prompt_ids", "continuation_ids"])
        part = part.loc[part.doc_id.astype(str).isin(selected_ids)]
        if len(part):
            continuations.append(part)
    continuation_frame = pd.concat(continuations, ignore_index=True)
    continuation_frame["doc_id"] = continuation_frame.doc_id.astype(str)
    if set(continuation_frame.doc_id) != selected_ids:
        raise AssertionError("Selected prompts do not have complete saved Q2 target continuations")
    references = {
        str(row.doc_id): {
            "prompt_ids": [int(x) for x in row.prompt_ids],
            "continuation_ids": [int(x) for x in row.continuation_ids],
            "pool_index": int(row.pool_index),
        }
        for row in continuation_frame.itertuples(index=False)
    }

    print(f"LOAD_Q2_{args.split.upper()}_EVENTS_ALIGNMENT_AND_CONTROLS", flush=True)
    full, join_meta = load_pair_frame(root, PAIR, config, split_override=args.split)
    sample = full.loc[full.doc_id.astype(str).isin(selected_ids)].copy()
    # The shared RQ1 loader omits the simulator's numeric block_id; the
    # contiguous eligible positions and zero-based slots recover its start.
    sample["block_id"] = sample.position.astype("int64") - sample.slot.astype("int64")
    sample = sample.sort_values(["doc_id", "block_id", "slot"], kind="stable")
    print(f"PROPOSAL_EVENTS split={args.split} rows={len(sample):,} docs={sample.doc_id.nunique():,}", flush=True)

    rows: list[dict[str, Any]] = []
    block_count = sample.groupby(["doc_id", "block_id"], sort=False).ngroups
    invalid_projection_blocks = 0
    for block_number, ((doc_id, block_id), block) in enumerate(sample.groupby(["doc_id", "block_id"], sort=False), start=1):
        block = block.sort_values("slot", kind="stable")
        base = int(block.position.min())
        reference = references[str(doc_id)]
        proposal_ids = block.draft_argmax_id.astype("int64").tolist()
        expected_slots = block.slot.astype("int64").tolist()
        if expected_slots != list(range(len(expected_slots))) or [int(x) for x in block.position] != [base + i for i in expected_slots]:
            raise AssertionError(f"Non-contiguous eligible proposals in {doc_id}/{block_id}")
        try:
            projected, projection_meta = project_candidate_block(
                tokenizer,
                kiwi,
                reference["prompt_ids"],
                reference["continuation_ids"][:base],
                proposal_ids,
                int(reference["pool_index"]),
                int(block_id),
                PAIR,
                allow_local_exact_spans=True,
            )
        except Exception as exc:
            invalid_projection_blocks += 1
            print(f"PROJECTION_BLOCK_ERROR doc={doc_id} block={block_id} error={type(exc).__name__}:{exc}", flush=True)
            continue
        projection_by_slot = {int(record["proposal_slot"]) - 1: record for record in projected}
        for event in block.itertuples(index=False):
            rec = projection_by_slot[int(event.slot)]
            proposal_class = (
                "CROSS" if rec["h2_morph_class"] == "CROSS_MORPHEME"
                else "WITHIN_SPLIT" if rec["h2_morph_class"] == "WITHIN_SPLIT"
                else str(rec["h2_morph_class"])
            )
            rows.append({
                "doc_id": str(doc_id), "pool_index": reference["pool_index"],
                "position": int(event.position), "block_id": int(block_id), "slot": int(event.slot),
                "target_token_id": int(event.target_token_id), "draft_argmax_id": int(event.draft_argmax_id),
                "rejected": bool(event.rejected), "proposal_side_class": proposal_class,
                "proposal_detector_status": str(rec["detector_status"]),
                "proposal_boundary_count": int(rec["num_boundaries_crossed"]),
                "proposal_boundary_environment": boundary_environment(rec),
                "proposal_token_surface": str(rec["token_surface"]),
                "proposal_eojeol": rec.get("eojeol"),
                "target_side_class": str(event.alignment_class),
                "target_side_status": str(event.alignment_status),
                "proposal_class_matches_target": proposal_class == ("CROSS" if event.alignment_class == "CROSS" else "WITHIN_SPLIT" if event.alignment_class == "WITHIN_SPLIT" else str(event.alignment_class)),
                "roundtrip_exact": bool(rec["alignment_roundtrip_exact"]),
                "proposal_span_exact": bool(rec["span_exact"]),
                "proposal_context_roundtrip_exact": bool(projection_meta["roundtrip_exact"]),
                "eojeol_token_count": int(event.eojeol_token_count),
                "eojeol_char_len": int(event.eojeol_char_len),
                "draft_entropy": float(event.draft_entropy), "target_entropy": float(event.target_entropy),
                "draft_margin": float(event.draft_margin), "target_margin": float(event.target_margin),
            })
        if block_number % 1000 == 0:
            print(f"PROJECTION_PROGRESS blocks={block_number:,}/{block_count:,} rows={len(rows):,}", flush=True)

    data = pd.DataFrame(rows)
    valid = data.loc[
        data.proposal_detector_status.eq("VALID")
        & data.proposal_side_class.isin(["CROSS", "WITHIN_SPLIT"])
    ].copy()
    valid["alignment_class"] = valid.proposal_side_class
    fit: dict[str, Any] | None = None
    if valid.alignment_class.nunique() == 2 and valid.rejected.nunique() == 2:
        fit = fit_cross_ame(valid, label=f"RQ4:proposal_side_{args.split}_{args.max_docs}_prompts")

    target_comparable = valid.loc[valid.target_side_class.isin(["CROSS", "WITHIN_SPLIT"])]
    if args.max_docs <= 2_000:
        selected_pool_provenance: dict[str, Any] = {"pool_indices": selected_pool_indices}
    else:
        selected_pool_provenance = {
            "pool_indices_sha256": hashlib.sha256(
                ",".join(str(x) for x in selected_pool_indices).encode("ascii")
            ).hexdigest(),
            "pool_index_min": min(selected_pool_indices),
            "pool_index_max": max(selected_pool_indices),
        }
    is_full_40k = args.split == "all" and args.max_docs == 40_000
    sample_label = (
        "all 40,000 prompts from the frozen all pool" if is_full_40k
        else f"random {args.max_docs:,} prompts from the {args.split} pool"
    )
    rows_path = f"{args.output_stem}_rows.parquet"
    summary_path = f"{args.output_stem}_summary.json"
    summary = {
        "pair": PAIR,
        "target": "Qwen3-4B-Base",
        "draft": "Qwen3-0.6B-Base",
        "sample": sample_label,
        "split": args.split,
        "selected_documents": int(len(selected_ids)),
        "seed": int(args.seed),
        **selected_pool_provenance,
        "tokenizer_revision": model_spec["revision"],
        "tokenizer_backend_sha256": backend_sha256,
        "eligible_proposal_rows": int(len(data)),
        "join": join_meta,
        "projection_error_blocks": int(invalid_projection_blocks),
        "valid_proposal_class_rows": int(len(valid)),
        "proposal_cross_rows": int(valid.proposal_side_class.eq("CROSS").sum()),
        "proposal_within_rows": int(valid.proposal_side_class.eq("WITHIN_SPLIT").sum()),
        "valid_class_match_rows": int(target_comparable.proposal_class_matches_target.sum()),
        "valid_class_comparable_rows": int(len(target_comparable)),
        "class_agreement_with_target_side": float(target_comparable.proposal_class_matches_target.mean()) if len(target_comparable) else None,
        "valid_proposal_roundtrip_exact_fraction": float(valid.roundtrip_exact.mean()) if len(valid) else None,
        "ame_fit": fit,
        "scope": (
            "Full frozen Q2 all-pool proposal-side projection and refit across 40,000 prompts. Each eligible teacher-forced draft argmax is projected on its saved target-prefix context."
            if is_full_40k else
            f"Exploratory random {args.max_docs:,}-prompt subset of the frozen {args.split} pool. Proposal labels project each eligible teacher-forced draft argmax on its saved target-prefix context; this is not a full 40k proposal-side refit."
        ),
        "outputs": [
            rows_path,
            summary_path,
        ],
    }
    data.to_parquet(out / rows_path, index=False)
    atomic_json(out / summary_path, summary)
    print(
        f"PROPOSAL_PROJECTION_COMPLETE split={args.split} rows={len(data):,} valid={len(valid):,} "
        f"class_agreement={summary['class_agreement_with_target_side']} fit={fit and fit['AME_cross_minus_within']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
