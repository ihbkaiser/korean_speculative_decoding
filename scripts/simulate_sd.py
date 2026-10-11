#!/usr/bin/env python3
"""CPU simulation of greedy speculative-decoding labels from TF argmax logs."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import atomic_json, atomic_parquet, load_config, output_root, read_parquet_records, select_prompts


def _tf_files(root: Path, target: str, model: str, split: str):
    path = root / "teacher_forced" / "generated" / target / model / split
    files = sorted(path.glob("shard-*.parquet"))
    if files:
        return files
    if split == "common":
        fallback = root / "teacher_forced" / "generated" / target / model / "all"
        files = sorted(fallback.glob("shard-*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No TF score shards found under {path}")


def _read_tf(root, target, model, split, pool_indices):
    frames = [pd.read_parquet(path) for path in _tf_files(root, target, model, split)]
    frame = pd.concat(frames, ignore_index=True)
    if split == "common":
        frame = frame[frame["pool_index"].astype(int) < 20_000]
    elif split == "extra":
        frame = frame[frame["pool_index"].astype(int) >= 20_000]
    if pool_indices is not None:
        frame = frame[frame["pool_index"].astype(int).isin(pool_indices)]
    frame["doc_id"] = frame["doc_id"].astype(str)
    return frame


def _simulate_prompt(group, block_size, pair_id, split):
    group = group.sort_values("position")
    positions = {int(row.position): row for row in group.itertuples(index=False)}
    if not positions:
        return [], {"calls": 0, "eligible_proposals": 0, "rejections": 0, "invalidated_suffix": 0, "bonus_tokens": 0, "target_tokens": 0}
    doc_id = str(group.iloc[0]["doc_id"])
    ordered = sorted(positions)
    if ordered != list(range(len(ordered))):
        raise RuntimeError(f"Non-contiguous continuation positions for {doc_id}")
    total = len(ordered)
    cursor = 0
    block_id = 0
    events = []
    accepted = rejected = invalidated = bonus = 0
    while cursor < total:
        block_end = min(total, cursor + block_size)
        slots = []
        mismatch_slot = None
        for position in range(cursor, block_end):
            row = positions[position]
            proposal_id = int(row.draft_argmax_id)
            target_id = int(row.target_token_id)
            match = proposal_id == target_id
            slot = position - cursor
            event = {
                "pair_id": pair_id,
                "split": split,
                "doc_id": doc_id,
                "position": position,
                "block_id": block_id,
                "slot": slot,
                "block_size": block_size,
                "target_token_id": target_id,
                "draft_argmax_id": proposal_id,
                "target_argmax_id": int(row.target_argmax_id),
                "target_matches_continuation": bool(int(row.target_argmax_id) == target_id),
                "eligible": True,
                "accepted": bool(match),
                "rejected": bool(not match),
                "invalidated_after": False,
                "proposal_context_observed": True,
                "event_type": "verified_proposal",
            }
            events.append(event)
            slots.append(position)
            if not match:
                mismatch_slot = slot
                rejected += 1
                break
            accepted += 1

        if mismatch_slot is not None:
            # The draft computed the remainder of its block before verification.
            # Those suffix tokens are work, but their target-forced predictions
            # are not the draft's actual proposals and therefore get no token ID.
            for slot in range(mismatch_slot + 1, block_end - cursor):
                position = cursor + slot
                row = positions[position]
                events.append({
                    "pair_id": pair_id,
                    "split": split,
                    "doc_id": doc_id,
                    "position": position,
                    "block_id": block_id,
                    "slot": slot,
                    "block_size": block_size,
                    "target_token_id": int(row.target_token_id),
                    "draft_argmax_id": None,
                    "target_argmax_id": int(row.target_argmax_id),
                    "target_matches_continuation": bool(int(row.target_argmax_id) == int(row.target_token_id)),
                    "eligible": False,
                    "accepted": False,
                    "rejected": None,
                    "invalidated_after": True,
                    "proposal_context_observed": False,
                    "event_type": "unobserved_invalidated_suffix",
                })
                invalidated += 1
            cursor += mismatch_slot + 1  # target correction token is output at the rejection position
        else:
            # A fully accepted block earns the target's bonus token, which is
            # deliberately not represented as a draft proposal event.
            cursor = block_end
            if cursor < total:
                bonus += 1
                cursor += 1
        block_id += 1

    summary = {
        "calls": block_id,
        "eligible_proposals": accepted + rejected,
        "accepted_proposals": accepted,
        "rejections": rejected,
        "invalidated_suffix": invalidated,
        "bonus_tokens": bonus,
        "target_tokens": total,
    }
    return events, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--split", choices=("common", "all", "extra"), required=True)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    if any(k < 1 for k in args.block_sizes):
        parser.error("block sizes must be positive")
    pair = config["pairs"][args.pair]
    prompts = select_prompts(config, args.split)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    pool_indices = {int(row["pool_index"]) for row in prompts}
    out = output_root(config, args.output_dir)
    target_frame = _read_tf(out, pair["target"], pair["target"], args.split, pool_indices)
    draft_frame = _read_tf(out, pair["target"], pair["draft"], args.split, pool_indices)
    draft_frame = draft_frame.rename(columns={"argmax_id": "draft_argmax_id"})
    merged = draft_frame.merge(
        target_frame[["doc_id", "pool_index", "position", "target_token_id", "argmax_id"]].rename(
            columns={"argmax_id": "target_argmax_id"}
        ),
        on=["doc_id", "pool_index", "position"],
        suffixes=("_draft", "_target"),
        validate="one_to_one",
    )
    if len(merged) != len(draft_frame) or len(merged) != len(target_frame):
        raise RuntimeError(
            f"Draft/target TF position counts do not reconcile: draft={len(draft_frame)}, "
            f"target={len(target_frame)}, joined={len(merged)}"
        )
    merged = merged.rename(columns={"target_token_id_draft": "target_token_id"})
    merged["target_matches_continuation"] = (
        merged["target_argmax_id"].astype("int64") == merged["target_token_id"].astype("int64")
    )
    result_dir = out / "simulated" / args.pair / args.split
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for block_size in args.block_sizes:
        events = []
        prompt_summaries = []
        for _, group in merged.groupby("doc_id", sort=False):
            prompt_events, summary = _simulate_prompt(group, block_size, args.pair, args.split)
            events.extend(prompt_events)
            prompt_summaries.append(summary)
        event_path = result_dir / f"events_k{block_size}.parquet"
        if event_path.exists() and not args.force:
            raise FileExistsError(f"{event_path} exists; pass --force to replace it")
        atomic_parquet(events, event_path)
        total_eligible = sum(row["eligible_proposals"] for row in prompt_summaries)
        total_rejections = sum(row["rejections"] for row in prompt_summaries)
        total_calls = sum(row["calls"] for row in prompt_summaries)
        summary_rows.append({
            "Pair": args.pair,
            "Split": args.split,
            "Block size": block_size,
            "Prompts": len(prompt_summaries),
            "Target tokens": sum(row["target_tokens"] for row in prompt_summaries),
            "Target calls": total_calls,
            "Eligible proposals": total_eligible,
            "Accepted proposals": sum(row["accepted_proposals"] for row in prompt_summaries),
            "Rejections": total_rejections,
            "Rejection rate": total_rejections / total_eligible if total_eligible else None,
            "Invalidated suffix work": sum(row["invalidated_suffix"] for row in prompt_summaries),
            "Bonus tokens": sum(row["bonus_tokens"] for row in prompt_summaries),
            "Target TF label disagreement": int((~merged["target_matches_continuation"].astype(bool)).sum()),
        })
        print(
            f"SIM pair={args.pair} k={block_size} prompts={len(prompt_summaries)} "
            f"eligible={total_eligible} reject={total_rejections} "
            f"rate={summary_rows[-1]['Rejection rate']:.6f}",
            flush=True,
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(result_dir / "summary_by_k.csv", index=False)
    (result_dir / "summary_by_k.md").write_text(summary_frame.to_markdown(index=False) + "\n", encoding="utf-8")
    atomic_json(result_dir / "run_metadata.json", {
        "pair": args.pair,
        "draft": pair["draft"],
        "target": pair["target"],
        "split": args.split,
        "block_sizes": args.block_sizes,
        "prompt_count": len(prompts),
        "position_count": len(merged),
        "event_files": [f"events_k{k}.parquet" for k in args.block_sizes],
        "status": "COMPLETE",
        "semantics": "Only target-verified proposals count toward rejection rate; invalidated suffix slots have no inferred draft token ID; fully accepted blocks skip the target bonus position.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
