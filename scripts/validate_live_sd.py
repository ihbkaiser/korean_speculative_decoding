#!/usr/bin/env python3
"""Compare teacher-forced k=4 labels with live cached SD on the V0 sample."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batched_speculative import speculative_greedy_microbatch
from src.data import encode_prompt
from src.models import load_table1_tokenizer
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    model_path,
    output_root,
    read_parquet_records,
    select_prompts,
)


def _load_refs(out, target, prompts):
    wanted = {str(row["doc_id"]): row for row in prompts}
    paths = sorted((out / "continuations" / target / "common").glob("shard-*.parquet"))
    if not paths:
        paths = sorted((out / "continuations" / target / "all").glob("shard-*.parquet"))
    refs = {}
    for path in paths:
        for row in read_parquet_records(path):
            doc_id = str(row["doc_id"])
            if doc_id in wanted:
                refs[doc_id] = row
    missing = set(wanted) - set(refs)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} V0 reference continuations for target={target}")
    return [refs[str(row["doc_id"])] for row in prompts]


def _label_agreement(
    live: pd.DataFrame,
    simulated: pd.DataFrame,
    aligned: pd.DataFrame | None,
    common_prefix_lengths: dict[str, int],
):
    live_eligible = live[~live["invalidated_after_first_rejection"].fillna(False).astype(bool)].copy()
    simulated_eligible = simulated[simulated["eligible"].astype(bool)].copy()
    live_eligible = live_eligible.rename(columns={
        "prompt_id": "doc_id",
        "output_token_position": "position",
        "rejected": "live_rejected",
        "draft_token_id": "live_draft_id",
        "target_greedy_token_id": "live_target_id",
    })
    simulated_eligible = simulated_eligible.rename(columns={
        "rejected": "sim_rejected",
        "draft_argmax_id": "sim_draft_id",
        "target_token_id": "sim_target_id",
    })
    join = live_eligible.merge(
        simulated_eligible[["doc_id", "position", "sim_rejected", "sim_draft_id", "sim_target_id"]],
        on=["doc_id", "position"],
        how="inner",
        validate="one_to_one",
    )
    join["common_prefix_length"] = join["doc_id"].map(common_prefix_lengths).fillna(0).astype(int)
    # A proposal at position t is comparable when the committed prefix before
    # t matches. If the output at t is the first divergence, its verification
    # still used the same context; later proposals did not.
    shared_context = join[join["position"] <= join["common_prefix_length"]]
    cross_join = None
    shared_cross_join = None
    if aligned is not None and not join.empty:
        labels = aligned[["doc_id", "generated_token_position", "alignment_class", "alignment_status"]].copy()
        labels["position"] = labels["generated_token_position"].astype(int)
        cross_join = join.merge(labels[["doc_id", "position", "alignment_class", "alignment_status"]], on=["doc_id", "position"], how="left", validate="many_to_one")
        cross_join = cross_join[cross_join["alignment_class"].eq("CROSS") & cross_join["alignment_status"].eq("ALIGNED")]
        shared_cross_join = cross_join[
            cross_join["position"] <= cross_join["common_prefix_length"]
        ]

    def measure(frame):
        if frame is None or frame.empty:
            return {"compared": 0, "rejection_label_agreement": None, "draft_id_agreement": None, "target_id_agreement": None}
        return {
            "compared": len(frame),
            "rejection_label_agreement": float((frame["live_rejected"].astype(bool) == frame["sim_rejected"].fillna(False).astype(bool)).mean()),
            "draft_id_agreement": float((frame["live_draft_id"].astype("int64") == frame["sim_draft_id"].astype("int64")).mean()),
            "target_id_agreement": float((frame["live_target_id"].astype("int64") == frame["sim_target_id"].astype("int64")).mean()),
        }

    return {
        "all_positions": measure(join),
        "shared_context": measure(shared_context),
        "cross_all_positions": measure(cross_join),
        "cross_shared_context": measure(shared_cross_join),
        "all_joined_positions": len(join),
        "shared_context_joined_positions": len(shared_context),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--pair", required=True, help="V0 pair; plan specifies Q2 and G1")
    parser.add_argument("--prompts", type=int, help="V0 sample size; default 1,000 for Q2 and 500 for G1")
    parser.add_argument("--batch-size", type=int, help="Default 64 for Q2 and 32 for G1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    if args.pair not in {"Q2", "G1"}:
        parser.error("V0 plan specifies only Q2 and G1")
    pair = config["pairs"][args.pair]
    default_count = 1_000 if args.pair == "Q2" else 500
    default_batch = 64 if args.pair == "Q2" else 32
    count = args.prompts or default_count
    batch_size = args.batch_size or default_batch
    if count < 1 or batch_size < 1:
        parser.error("--prompts and --batch-size must be positive")
    selected = select_prompts(config, "common")[:count]
    out = output_root(config, args.output_dir)
    target_refs = _load_refs(out, pair["target"], selected)
    tokenizer_spec = config["models"][pair["target"]]
    tokenizer = load_table1_tokenizer(
        model_path(config, pair["target"]),
        backend=tokenizer_spec.get("tokenizer_backend", "auto"),
        fix_mistral_regex=bool(tokenizer_spec.get("fix_mistral_regex", False)),
        local_files_only=True,
    )
    prompt_rows = []
    ref_by_id = {}
    for prompt, ref in zip(selected, target_refs):
        prompt_ids = encode_prompt(tokenizer, str(prompt["text"]), int(config["data"]["max_prompt_tokens"]))
        if prompt_ids != [int(x) for x in ref["prompt_ids"]]:
            raise RuntimeError(f"Frozen prompt encoding differs for V0 doc_id={prompt['doc_id']}")
        prompt_rows.append(prompt_ids)
        ref_by_id[str(prompt["doc_id"])] = ref

    destination = out / "validation" / "v0" / args.pair
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "live_events.parquet"
    parity_path = destination / "target_sequence_parity.parquet"
    if (events_path.exists() or parity_path.exists()) and not args.force:
        raise FileExistsError(f"V0 output already exists under {destination}; pass --force to replace")

    draft = load_model(config, pair["draft"])
    target = load_model(config, pair["target"])
    all_events = []
    parity_rows = []
    common_prefix_lengths = {}
    started = time.monotonic()
    max_new = int(config["data"]["max_new_tokens"])
    eos = tokenizer.eos_token_id
    for start in tqdm(range(0, count, batch_size), desc=f"V0 live SD {args.pair}", unit="batch", dynamic_ncols=True, mininterval=1):
        stop = min(count, start + batch_size)
        outputs, event_batches = speculative_greedy_microbatch(
            draft,
            target,
            prompt_rows[start:stop],
            max_new,
            eos,
            k=4,
            prompt_ids=[str(row["doc_id"]) for row in selected[start:stop]],
        )
        for local, (output_ids, events) in enumerate(zip(outputs, event_batches)):
            doc_id = str(selected[start + local]["doc_id"])
            reference_ids = [int(x) for x in ref_by_id[doc_id]["continuation_ids"]]
            common = 0
            for generated_id, reference_id in zip(output_ids, reference_ids):
                if int(generated_id) != reference_id:
                    break
                common += 1
            common_prefix_lengths[doc_id] = common
            parity_rows.append({
                "pair_id": args.pair,
                "doc_id": doc_id,
                "pool_index": int(selected[start + local]["pool_index"]),
                "target_generated_length": len(reference_ids),
                "live_sd_length": len(output_ids),
                "common_prefix_length": common,
                "compared_tokens": max(len(output_ids), len(reference_ids)),
                "exact_sequence_match": output_ids == reference_ids,
            })
            for event in events:
                all_events.append({
                    **event,
                    "pair_id": args.pair,
                    "prompt_id": doc_id,
                    "output_token_position": int(event["output_token_position"]),
                    "invalidated_after_first_rejection": bool(event["invalidated_after_first_rejection"]),
                })

    atomic_parquet(all_events, events_path)
    atomic_parquet(parity_rows, parity_path)
    del draft, target
    cleanup_cuda(torch)

    live = pd.DataFrame(all_events)
    simulation_path = out / "simulated" / args.pair / "common" / "events_k4.parquet"
    if not simulation_path.is_file():
        raise FileNotFoundError(f"Run scripts/simulate_sd.py --pair {args.pair} --split common before V0")
    simulation = pd.read_parquet(simulation_path)
    selected_ids = {str(row["doc_id"]) for row in selected}
    simulation = simulation[simulation["doc_id"].astype(str).isin(selected_ids)]
    aligned_path = out / "aligned" / "kiwi" / pair["target"] / "common"
    if not list(aligned_path.glob("shard-*.parquet")):
        aligned_path = out / "aligned" / "kiwi" / pair["target"] / "all"
    aligned_files = sorted(aligned_path.glob("shard-*.parquet"))
    aligned = pd.concat((pd.read_parquet(path) for path in aligned_files), ignore_index=True) if aligned_files else None
    if aligned is not None:
        aligned = aligned[aligned["doc_id"].astype(str).isin(selected_ids)]
    agreement = _label_agreement(live, simulation, aligned, common_prefix_lengths)
    parity = pd.DataFrame(parity_rows)
    total_ref_tokens = int(parity["target_generated_length"].sum())
    exact_sequences = int(parity["exact_sequence_match"].sum())
    sequence_position_agreement = int(parity["common_prefix_length"].sum()) / max(1, int(parity["compared_tokens"].sum()))
    shared_agreement = agreement["shared_context"]
    gate_status = (
        "PASS"
        if shared_agreement["rejection_label_agreement"] is not None
        and shared_agreement["rejection_label_agreement"] >= 0.99
        else "FAIL"
    )
    metadata = {
        "pair": args.pair,
        "draft": pair["draft"],
        "target": pair["target"],
        "sample_prompts": count,
        "batch_size": batch_size,
        "block_size": 4,
        "target_sequence_exact_prompts": exact_sequences,
        "target_sequence_exact_fraction": exact_sequences / count,
        "target_sequence_common_prefix_fraction": sequence_position_agreement,
        "target_tokens_in_generation_reference": total_ref_tokens,
        "live_vs_simulation_eligible_label_agreement_all_positions": agreement["all_positions"],
        "live_vs_simulation_cross_label_agreement_all_positions": agreement["cross_all_positions"],
        "live_vs_simulation_eligible_label_agreement_shared_context": agreement["shared_context"],
        "live_vs_simulation_cross_label_agreement_shared_context": agreement["cross_shared_context"],
        "live_vs_simulation_joined_proposal_positions_all": agreement["all_joined_positions"],
        "live_vs_simulation_joined_proposal_positions_shared_context": agreement["shared_context_joined_positions"],
        "g0_gate_status": gate_status,
        "wall_seconds": time.monotonic() - started,
        "status": "COMPLETE",
    }
    atomic_json(destination / "agreement_report.json", metadata)
    print(
        f"V0_COMPLETE pair={args.pair} prompts={count} exact_sequences={exact_sequences}/{count} "
        f"shared_context_label_agreement={shared_agreement['rejection_label_agreement']} "
        f"all_position_label_agreement={agreement['all_positions']['rejection_label_agreement']} "
        f"g0_gate={gate_status}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
