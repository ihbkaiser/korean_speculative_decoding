#!/usr/bin/env python3
"""Compare teacher-forced k=4 labels with live cached SD on the V0 sample."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
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
    prompt_ids_sha256,
    read_parquet_records,
    select_prompts,
    sha256_file,
)


G0_REJECTION_AGREEMENT_THRESHOLD = 0.95
PROMPT_BOOTSTRAP_REPLICATES = 2_000
MARGIN_THRESHOLDS = (0.0, 0.001, 0.01, 0.1, 0.5)


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
        "draft_top2_margin": "live_draft_top2_margin",
        "target_top2_margin": "live_target_top2_margin",
    })
    simulated_eligible = simulated_eligible.rename(columns={
        "rejected": "sim_rejected",
        "draft_argmax_id": "sim_draft_id",
        "target_token_id": "sim_target_id",
        "draft_top2_margin": "sim_draft_top2_margin",
        "target_top2_margin": "sim_target_top2_margin",
    })
    margin_columns = [
        "live_draft_top2_margin",
        "live_target_top2_margin",
        "sim_draft_top2_margin",
        "sim_target_top2_margin",
    ]
    missing_margins = [
        column for column in margin_columns if column.startswith("live_")
        and column not in live_eligible.columns
    ] + [
        column for column in margin_columns if column.startswith("sim_")
        and column not in simulated_eligible.columns
    ]
    if missing_margins:
        raise RuntimeError(
            "V0 margin audit inputs are missing: " + ", ".join(missing_margins)
            + ". Regenerate the simulation and live validation outputs with the current code."
        )
    comparison = live_eligible.merge(
        simulated_eligible[[
            "doc_id", "position", "sim_rejected", "sim_draft_id", "sim_target_id",
            "sim_draft_top2_margin", "sim_target_top2_margin",
        ]],
        on=["doc_id", "position"],
        how="outer",
        validate="one_to_one",
        indicator="position_join_status",
    )
    comparison["position_join_status"] = comparison["position_join_status"].astype(str)
    comparison["common_prefix_length"] = comparison["doc_id"].map(common_prefix_lengths).fillna(0).astype(int)
    # A proposal at position t is comparable when the committed prefix before
    # t matches. If the output at t is the first divergence, its verification
    # still used the same context; later proposals did not.
    comparison["shared_context"] = comparison["position"] <= comparison["common_prefix_length"]
    if aligned is not None and not comparison.empty:
        labels = aligned[["doc_id", "generated_token_position", "alignment_class", "alignment_status"]].copy()
        labels["position"] = labels["generated_token_position"].astype(int)
        comparison = comparison.merge(
            labels[["doc_id", "position", "alignment_class", "alignment_status"]],
            on=["doc_id", "position"],
            how="left",
            validate="many_to_one",
        )
    else:
        comparison["alignment_class"] = None
        comparison["alignment_status"] = None

    joined = comparison[comparison["position_join_status"].eq("both")]
    shared_context = joined[joined["shared_context"]]
    cross_join = joined[
        joined["alignment_class"].eq("CROSS")
        & joined["alignment_status"].eq("ALIGNED")
    ]
    shared_cross_join = cross_join[cross_join["shared_context"]]

    def prompt_cluster_ci(frame: pd.DataFrame):
        if frame.empty:
            return None
        matches = (
            frame["live_rejected"].fillna(False).astype(bool)
            == frame["sim_rejected"].fillna(False).astype(bool)
        )
        per_prompt = pd.DataFrame({"doc_id": frame["doc_id"].astype(str), "match": matches})
        grouped = per_prompt.groupby("doc_id", sort=True)["match"].agg(["sum", "count"])
        successes = grouped["sum"].to_numpy(dtype=np.float64)
        totals = grouped["count"].to_numpy(dtype=np.float64)
        if len(successes) == 1:
            point = float(successes.sum() / totals.sum())
            return {
                "method": "prompt_cluster_bootstrap_percentile",
                "confidence_level": 0.95,
                "replicates": PROMPT_BOOTSTRAP_REPLICATES,
                "lower": point,
                "upper": point,
                "prompt_clusters": 1,
            }
        rng = np.random.default_rng(20261011)
        draws = rng.integers(
            0,
            len(successes),
            size=(PROMPT_BOOTSTRAP_REPLICATES, len(successes)),
        )
        sampled_successes = successes[draws].sum(axis=1)
        sampled_totals = totals[draws].sum(axis=1)
        values = sampled_successes / sampled_totals
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "method": "prompt_cluster_bootstrap_percentile",
            "confidence_level": 0.95,
            "replicates": PROMPT_BOOTSTRAP_REPLICATES,
            "lower": float(lower),
            "upper": float(upper),
            "prompt_clusters": int(len(successes)),
        }

    def measure(frame: pd.DataFrame, include_ci: bool = True):
        if frame is None or frame.empty:
            return {
                "compared": 0,
                "prompt_count": 0,
                "rejection_label_agreement": None,
                "rejection_label_agreement_ci95_prompt_cluster_bootstrap": None,
                "draft_id_agreement": None,
                "target_id_agreement": None,
                "rejection_confusion": {
                    "both_reject": 0,
                    "both_accept": 0,
                    "live_reject_sim_accept": 0,
                    "live_accept_sim_reject": 0,
                },
            }
        live_rejected = frame["live_rejected"].fillna(False).astype(bool)
        sim_rejected = frame["sim_rejected"].fillna(False).astype(bool)
        confusion = {
            "both_reject": int((live_rejected & sim_rejected).sum()),
            "both_accept": int((~live_rejected & ~sim_rejected).sum()),
            "live_reject_sim_accept": int((live_rejected & ~sim_rejected).sum()),
            "live_accept_sim_reject": int((~live_rejected & sim_rejected).sum()),
        }
        return {
            "compared": len(frame),
            "prompt_count": int(frame["doc_id"].nunique()),
            "rejection_label_agreement": float((live_rejected == sim_rejected).mean()),
            "rejection_label_agreement_ci95_prompt_cluster_bootstrap": (
                prompt_cluster_ci(frame) if include_ci else None
            ),
            "draft_id_agreement": float((frame["live_draft_id"].astype("int64") == frame["sim_draft_id"].astype("int64")).mean()),
            "target_id_agreement": float((frame["live_target_id"].astype("int64") == frame["sim_target_id"].astype("int64")).mean()),
            "rejection_confusion": confusion,
        }

    def margin_sensitivity(frame: pd.DataFrame):
        usable = frame[margin_columns].apply(pd.to_numeric, errors="coerce")
        complete = usable.notna().all(axis=1)
        minimum_margin = usable.min(axis=1)
        rows = []
        for threshold in MARGIN_THRESHOLDS:
            selected = frame[complete & minimum_margin.ge(threshold)]
            measured = measure(selected, include_ci=False)
            rows.append({
                "minimum_margin_threshold": threshold,
                "compared": measured["compared"],
                "coverage_of_compared_positions": (
                    float(measured["compared"] / len(frame)) if len(frame) else None
                ),
                "rejection_label_agreement": measured["rejection_label_agreement"],
            })
        return {
            "definition": "Minimum of live and teacher-forced draft and target top-two logit margins.",
            "positions_with_all_four_margins": int(complete.sum()),
            "positions_missing_a_margin": int((~complete).sum()),
            "by_minimum_margin_threshold": rows,
        }

    return {
        "all_positions": measure(joined),
        "shared_context": measure(shared_context),
        "cross_all_positions": measure(cross_join),
        "cross_shared_context": measure(shared_cross_join),
        "all_joined_positions": len(joined),
        "shared_context_joined_positions": len(shared_context),
        "shared_context_unmatched_live_positions": int((
            comparison["shared_context"] & comparison["position_join_status"].eq("left_only")
        ).sum()),
        "shared_context_unmatched_simulation_positions": int((
            comparison["shared_context"] & comparison["position_join_status"].eq("right_only")
        ).sum()),
        "unmatched_live_positions_all_contexts": int(comparison["position_join_status"].eq("left_only").sum()),
        "unmatched_simulation_positions_all_contexts": int(comparison["position_join_status"].eq("right_only").sum()),
        "margin_sensitivity_shared_context": margin_sensitivity(shared_context),
        "margin_sensitivity_cross_shared_context": margin_sensitivity(shared_cross_join),
        "comparison": comparison,
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
    count = args.prompts if args.prompts is not None else default_count
    batch_size = args.batch_size if args.batch_size is not None else default_batch
    if count < 1 or batch_size < 1:
        parser.error("--prompts and --batch-size must be positive")
    available_prompts = select_prompts(config, "common")
    if count > len(available_prompts):
        parser.error(f"requested {count} prompts but common split contains only {len(available_prompts)}")
    selected = available_prompts[:count]
    out = output_root(config, args.output_dir)
    simulation_path = out / "simulated" / args.pair / "common" / "events_k4.parquet"
    if not simulation_path.is_file():
        raise FileNotFoundError(f"Run scripts/simulate_sd.py --pair {args.pair} --split common before V0")
    simulation = pd.read_parquet(simulation_path)
    required_simulation_margins = {"draft_top2_margin", "target_top2_margin"}
    missing_simulation_margins = required_simulation_margins - set(simulation.columns)
    if missing_simulation_margins:
        raise RuntimeError(
            "Simulation events lack margin audit columns: "
            + ", ".join(sorted(missing_simulation_margins))
            + ". Rerun scripts/simulate_sd.py with the current code before V0."
        )
    selected_ids = {str(row["doc_id"]) for row in selected}
    simulation["doc_id"] = simulation["doc_id"].astype(str)
    simulation = simulation[simulation["doc_id"].isin(selected_ids)]
    missing_simulated_prompts = selected_ids - set(simulation["doc_id"])
    if missing_simulated_prompts:
        raise RuntimeError(
            f"Simulation events are missing {len(missing_simulated_prompts)} selected prompts; "
            "regenerate the simulation for the requested sample."
        )
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
    comparison_path = destination / "position_comparison.parquet"
    report_path = destination / "agreement_report.json"
    existing_outputs = [
        path for path in (events_path, parity_path, comparison_path, report_path)
        if path.exists()
    ]
    if existing_outputs and not args.force:
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
    aligned_path = out / "aligned" / "kiwi" / pair["target"] / "common"
    if not list(aligned_path.glob("shard-*.parquet")):
        aligned_path = out / "aligned" / "kiwi" / pair["target"] / "all"
    aligned_files = sorted(aligned_path.glob("shard-*.parquet"))
    aligned = pd.concat((pd.read_parquet(path) for path in aligned_files), ignore_index=True) if aligned_files else None
    if aligned is not None:
        aligned = aligned[aligned["doc_id"].astype(str).isin(selected_ids)]
    agreement = _label_agreement(live, simulation, aligned, common_prefix_lengths)
    comparison = agreement.pop("comparison")
    atomic_parquet(comparison.to_dict(orient="records"), comparison_path)
    parity = pd.DataFrame(parity_rows)
    total_ref_tokens = int(parity["target_generated_length"].sum())
    exact_sequences = int(parity["exact_sequence_match"].sum())
    sequence_position_agreement = int(parity["common_prefix_length"].sum()) / max(1, int(parity["compared_tokens"].sum()))
    shared_agreement = agreement["shared_context"]
    shared_context_unmatched = (
        agreement["shared_context_unmatched_live_positions"]
        + agreement["shared_context_unmatched_simulation_positions"]
    )
    gate_status = (
        "PASS"
        if shared_agreement["rejection_label_agreement"] is not None
        and shared_agreement["rejection_label_agreement"] >= G0_REJECTION_AGREEMENT_THRESHOLD
        and shared_context_unmatched == 0
        else "FAIL"
    )
    if gate_status == "PASS":
        gate_reason = None
    elif shared_context_unmatched:
        gate_reason = "One or more shared-context positions are missing from one side of the comparison."
    else:
        gate_reason = "Shared-context rejection-label agreement is below the configured threshold."
    validation_scope = (
        "g0_pilot"
        if args.pair == "Q2" and count == 100
        else "full_v0"
        if (args.pair == "Q2" and count == 1_000) or (args.pair == "G1" and count == 500)
        else "custom"
    )
    metadata = {
        "pair": args.pair,
        "draft": pair["draft"],
        "target": pair["target"],
        "draft_revision": config["models"][pair["draft"]]["revision"],
        "target_revision": config["models"][pair["target"]]["revision"],
        "sample_prompts": count,
        "validation_scope": validation_scope,
        "prompt_ids_sha256": prompt_ids_sha256(selected),
        "config_sha256": sha256_file(args.config),
        "experiment_plan_sha256": sha256_file(ROOT / "NAACL_experiments.md"),
        "simulation_events_sha256": sha256_file(simulation_path),
        "batch_size": batch_size,
        "block_size": 4,
        "g0_gate_metric": "shared-context eligible rejection-label agreement",
        "g0_gate_threshold": G0_REJECTION_AGREEMENT_THRESHOLD,
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
        "shared_context_unmatched_live_positions": agreement["shared_context_unmatched_live_positions"],
        "shared_context_unmatched_simulation_positions": agreement["shared_context_unmatched_simulation_positions"],
        "unmatched_live_positions_all_contexts": agreement["unmatched_live_positions_all_contexts"],
        "unmatched_simulation_positions_all_contexts": agreement["unmatched_simulation_positions_all_contexts"],
        "margin_sensitivity_shared_context": agreement["margin_sensitivity_shared_context"],
        "margin_sensitivity_cross_shared_context": agreement["margin_sensitivity_cross_shared_context"],
        "g0_gate_status": gate_status,
        "g0_gate_reason": gate_reason,
        "g0_eta_status": "NOT_ASSESSED_BY_VALIDATION_SCRIPT",
        "position_comparison_artifact": str(comparison_path),
        "position_comparison_rows": len(comparison),
        "confidence_interval_method": "2,000-replicate percentile bootstrap sampling prompts as clusters; seed 20261011",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "wall_seconds": time.monotonic() - started,
        "status": "COMPLETE",
    }
    atomic_json(report_path, metadata)
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
