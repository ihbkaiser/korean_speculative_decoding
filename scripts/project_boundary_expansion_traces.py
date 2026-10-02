#!/usr/bin/env python3
"""Project newly collected speculative proposals into morphology/outcome rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from kiwipiepy import Kiwi

from scripts.analyze_proposal_side import FREQUENCY, load_tokenizer
from scripts.analyze_proposal_side_boundary import proposal_environment
from src.morphology_guard import project_candidate_block

OUT = ROOT / "runs/proposal_boundary_data_expansion"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def project_file(workload: str, pair: str, tokenizer: Any, kiwi: Any,
                 prompts: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    slug = f"{workload.lower()}_{pair.lower()}"
    trace_path = OUT / "traces" / f"{slug}.jsonl"
    metadata_path = OUT / "traces" / f"{slug}.metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Trace run has no completion metadata yet: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    traces = read_jsonl(trace_path)
    if len(traces) != int(metadata["prompt_count"]):
        raise AssertionError(f"Trace count differs from completion metadata for {slug}")

    rows: list[dict[str, Any]] = []
    status_counts: defaultdict[str, int] = defaultdict(int)
    for prompt_i, trace in enumerate(traces, start=1):
        if prompt_i % 100 == 0:
            print(f"[{slug}] projected {prompt_i}/{len(traces)} prompts", flush=True)
        pid = int(trace["prompt_id"])
        prompt = prompts[pid]
        prompt_ids = [int(x) for x in prompt["input_ids"]]
        if prompt_ids != [int(x) for x in trace["input_ids"]]:
            raise AssertionError(f"Prompt-token IDs changed for {slug}/prompt={pid}")
        output_ids = [int(x) for x in trace["sd_token_ids"]]

        blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in trace["events"]:
            blocks[int(event["round_index"])].append(event)
        for rnd, block in blocks.items():
            block.sort(key=lambda event: int(event["proposal_slot"]))
            actual = [event for event in block if bool(event["sd_valid"])]
            if not actual:
                continue
            base = min(int(event["output_token_position"]) for event in block)
            proposal_ids = [int(event["draft_proposed_token_id"]) for event in block]
            if base > len(output_ids):
                raise AssertionError(f"Proposal block begins past output end: {slug}/{pid}/{rnd}")
            projected, projection_meta = project_candidate_block(
                tokenizer, kiwi, prompt_ids, output_ids[:base], proposal_ids,
                pid, rnd, pair, allow_local_exact_spans=True,
            )
            projected_by_slot = {int(record["proposal_slot"]): record for record in projected}
            for event in actual:
                slot = int(event["proposal_slot"])
                pos = int(event["output_token_position"])
                rec = projected_by_slot[slot]
                draft_id = int(event["draft_proposed_token_id"])
                target_id = int(event["target_greedy_token_id"])
                rejected = event["rejected"]
                if rejected is None:
                    raise AssertionError(f"Valid event lacks a reject label: {slug}/{pid}/{rnd}/{slot}")
                if int(rec["token_id"]) != draft_id:
                    raise AssertionError(f"Projected proposal ID mismatch: {slug}/{pid}/{rnd}/{slot}")
                if pos >= len(output_ids) or output_ids[pos] != target_id:
                    raise AssertionError(f"Target decision differs from speculative output: {slug}/{pid}/{rnd}/{slot}")
                if bool(rejected) == (draft_id == target_id):
                    raise AssertionError(f"Accept/reject label contradicts token IDs: {slug}/{pid}/{rnd}/{slot}")
                env, env_source = proposal_environment(rec)
                status = str(rec["detector_status"])
                status_counts[status] += 1
                rows.append({
                    "trace_source": "expansion",
                    "workload": workload, "pair": pair, "prompt_id": pid,
                    "round_index": rnd, "proposal_slot": slot,
                    "output_token_position": pos, "draft_token_id": draft_id,
                    "target_token_id": target_id, "sd_rejected": int(bool(rejected)),
                    "proposal_morph_class": str(rec["h2_morph_class"]),
                    "projection_status": status, "proposal_morph_environment": env,
                    "environment_source": env_source,
                    "num_boundaries_crossed": int(rec["num_boundaries_crossed"]),
                    "candidate_surface": rec["token_surface"], "candidate_eojeol": rec["eojeol"],
                    "candidate_token_char_length": int(rec["token_char_end"]) - int(rec["token_char_start"]),
                    "candidate_eojeol_char_length": len(str(rec.get("eojeol") or "")),
                    "fragmentation_bin": None, "relative_position": None,
                    "first_token": None, "last_token": None,
                    "generation_position": int(event["generation_position"]),
                    "proposal_slot_ctrl": slot,
                    "draft_entropy": event["draft_entropy"],
                    "target_entropy": event["target_entropy"],
                    "fullblock_roundtrip_exact": bool(projection_meta["roundtrip_exact"]),
                })

    frame = pd.DataFrame(rows)
    frequency = pd.read_parquet(FREQUENCY)[["token_id", "log_token_count"]].rename(
        columns={"token_id": "draft_token_id", "log_token_count": "log_candidate_token_count"}
    )
    frame = frame.merge(frequency, on="draft_token_id", how="left", validate="many_to_one")
    if frame.log_candidate_token_count.isna().any():
        raise AssertionError(f"Some candidate IDs lack frequency values for {slug}")
    result = {
        "workload": workload, "pair": pair, "trace_prompts": len(traces),
        "valid_decision_rows": len(frame),
        "reject_rows": int(frame.sd_rejected.sum()),
        "projection_status_counts": dict(status_counts),
        "valid_primary_classes": int((frame.projection_status.eq("VALID") & frame.proposal_morph_class.isin(["WITHIN_SPLIT", "CROSS_MORPHEME"])).sum()),
        "valid_single_boundary_cross": int((frame.projection_status.eq("VALID") & frame.proposal_morph_class.eq("CROSS_MORPHEME") & frame.num_boundaries_crossed.eq(1)).sum()),
        "valid_n2p_cross": int((frame.projection_status.eq("VALID") & frame.proposal_morph_class.eq("CROSS_MORPHEME") & frame.num_boundaries_crossed.eq(1) & frame.proposal_morph_environment.eq("NOMINAL_TO_PARTICLE")).sum()),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    return frame, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["P1", "P2", "all"], default="all")
    parser.add_argument("--workload", choices=["WIKIPEDIA", "FLORES", "all"], default="all")
    parser.add_argument("--output", type=Path, default=OUT / "analysis/proposal_rows_new.parquet")
    args = parser.parse_args()

    prompt_rows = read_jsonl(OUT / "prompts.jsonl")
    prompts = {str(workload): {int(row["prompt_id"]): row for row in prompt_rows if row["workload"] == workload}
               for workload in ("WIKIPEDIA", "FLORES")}
    tokenizer, tok_meta = load_tokenizer()
    kiwi = Kiwi()
    pairs = ("P1", "P2") if args.pair == "all" else (args.pair,)
    workloads = ("WIKIPEDIA", "FLORES") if args.workload == "all" else (args.workload,)
    frames, reports = [], []
    for pair in pairs:
        for workload in workloads:
            frame, report = project_file(workload, pair, tokenizer, kiwi, prompts[workload])
            frames.append(frame)
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    combined = pd.concat(frames, ignore_index=True)
    rows_path = args.output
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(rows_path, index=False)
    (rows_path.parent / f"{rows_path.stem}.projection_manifest.json").write_text(json.dumps({
        "tokenizer": tok_meta, "rows_file": rows_path.name,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "records": reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} valid decision rows to {rows_path}", flush=True)


if __name__ == "__main__":
    main()
