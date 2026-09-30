#!/usr/bin/env python3
"""Prepare exact prompts/references and CPU-only guard feasibility audits."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.e2_model_pairs import MODELS, PAIRS, sha256_file
from src.morphology_guard import project_candidate_block

OUT = ROOT / "runs/morphology_aware_boundary_guard"
WIKI = ROOT / "runs/e2_model_pair_replication"
LEGACY = ROOT / "runs/20260926T184145Z_pilot1000"
FLORES = ROOT / "runs/flores200_en_ko_replication"
SEED = 3090
SLOT_SEED = 20260930
FEASIBILITY_VERSION = "all_proposals_v2"
MAX_NEW = 128
K = 4
EOS = 151643


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha(path: Path) -> str:
    return sha256_file(path)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    headers = [str(x) for x in frame.columns]
    body = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    rows = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in body]
    return "\n".join([head, sep, *rows])


def _load_sources(tokenizer: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    prompt_hashes: dict[str, str] = {}
    target_reference_resolution: dict[str, Any] = {}
    with (WIKI / "prompt_ids.jsonl").open(encoding="utf-8") as f:
        wiki_prompts = [json.loads(line) for line in f if line.strip()]
    if len(wiki_prompts) != 1000:
        raise AssertionError(f"Expected 1,000 Wikipedia prompts, found {len(wiki_prompts)}")
    for row in wiki_prompts:
        pid = int(row["prompt_id"])
        prompt_ids = [int(x) for x in row["input_ids"]]
        records.append({
            "workload": "WIKIPEDIA", "prompt_id": pid, "prompt_text": "",
            "prompt_ids_json": json.dumps(prompt_ids, separators=(",", ":")),
            "prompt_hash": row["prompt_hash"], "source_index": int(row["source_index"]),
        })
        prompt_hashes[f"WIKIPEDIA:{pid}"] = str(row["prompt_hash"])

    flores_manifest = pd.read_csv(FLORES / "data_manifest.csv")
    if len(flores_manifest) != 1012 or not flores_manifest["included"].astype(bool).all():
        raise AssertionError("FLORES manifest does not contain the full included 1,012-row devtest split")
    for row in flores_manifest.itertuples(index=False):
        pid = int(row.prompt_id)
        rendered = str(row.rendered_prompt)
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
        prompt_ids = [int(x) for x in encoded]
        token_hash = hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")).hexdigest()
        if token_hash != str(row.tokenized_prompt_sha256):
            raise AssertionError(f"FLORES prompt token hash mismatch at row={pid}")
        records.append({
            "workload": "FLORES", "prompt_id": pid, "prompt_text": rendered,
            "prompt_ids_json": json.dumps(prompt_ids, separators=(",", ":")),
            "prompt_hash": str(row.rendered_prompt_sha256), "source_index": pid,
        })
        prompt_hashes[f"FLORES:{pid}"] = str(row.rendered_prompt_sha256)

    for pair, spec in PAIRS.items():
        slug = spec["slug"]
        for workload in ["WIKIPEDIA", "FLORES"]:
            if workload == "WIKIPEDIA" and pair == "P3":
                # P2 and P3 share the same target ID/revision (the pinned 4B).
                # The old P3 reference cache came from the historical 3090 run
                # and is not token-identical to the current pinned target on
                # 160/1,000 prompts. Canonicalize P3 to the current P2 target
                # continuations after checking prompt identity.
                historical_path = LEGACY / "reference_outputs.jsonl"
                with historical_path.open(encoding="utf-8") as f:
                    historical = [json.loads(line) for line in f if line.strip()]
                historical_by_prompt = {int(r["prompt_id"]): r for r in historical}
                wiki_prompt_by_id = {int(r["prompt_id"]): r for r in wiki_prompts}
                prompt_hash_by_id = {int(r["prompt_id"]): str(r["prompt_hash"]) for r in wiki_prompts}
                p2_base = WIKI / PAIRS["P2"]["slug"]
                p2_continuations = pd.read_parquet(p2_base / "continuations.parquet")
                p2_by_prompt = {int(r.prompt_id): r for r in p2_continuations.itertuples(index=False)}
                expected_ids = set(prompt_hash_by_id)
                if set(p2_by_prompt) != expected_ids or set(historical_by_prompt) != expected_ids:
                    raise AssertionError("P2 and historical P3 references must cover the same exact Wikipedia prompt IDs")
                if any(str(p2_by_prompt[pid].prompt_hash) != prompt_hash_by_id[pid] for pid in expected_ids):
                    raise AssertionError("P2 target continuation hashes do not match the frozen Wikipedia prompt manifest")

                exact_historical_matches = 0
                historical_prompt_metadata_matches = 0
                first_mismatch: dict[str, Any] | None = None
                for pid in sorted(expected_ids):
                    current_ids = [int(x) for x in p2_by_prompt[pid].reference_token_ids]
                    historical_row = historical_by_prompt[pid]
                    historical_ids = [int(x) for x in historical_row["reference_token_ids"]]
                    current_prompt = wiki_prompt_by_id[pid]
                    if (
                        int(historical_row.get("source_index", -1)) == int(current_prompt["source_index"])
                        and str(historical_row.get("title", "")) == str(current_prompt.get("title", ""))
                    ):
                        historical_prompt_metadata_matches += 1
                    if current_ids == historical_ids:
                        exact_historical_matches += 1
                    elif first_mismatch is None:
                        mismatch_pos = next(
                            (i for i, (a, b) in enumerate(zip(current_ids, historical_ids)) if a != b),
                            min(len(current_ids), len(historical_ids)),
                        )
                        first_mismatch = {
                            "prompt_id": pid,
                            "first_differing_output_position_zero_based": mismatch_pos,
                            "p2_pinned_target_token_id": current_ids[mismatch_pos] if mismatch_pos < len(current_ids) else None,
                            "historical_p3_token_id": historical_ids[mismatch_pos] if mismatch_pos < len(historical_ids) else None,
                        }
                target_reference_resolution["WIKIPEDIA_P3"] = {
                    "canonical_reference_pair": "P2",
                    "target_model_id": MODELS["4B"]["id"],
                    "target_revision": MODELS["4B"]["revision"],
                    "canonical_source": str(p2_base / "continuations.parquet"),
                    "rejected_legacy_source": str(historical_path),
                    "shared_prompt_ids": len(expected_ids),
                    "p2_prompt_hash_matches_frozen_manifest": len(expected_ids),
                    "legacy_source_indices_and_titles_match_current_manifest": historical_prompt_metadata_matches,
                    "historical_token_continuations_equal_to_pinned_p2": exact_historical_matches,
                    "historical_token_continuations_differ_from_pinned_p2": len(expected_ids) - exact_historical_matches,
                    "first_historical_mismatch": first_mismatch,
                    "resolution": "P2 references are reused for P3 because both pairs use the same pinned 4B target and identical prompt IDs/hashes; the historical P3 cache is not compatible with the current target outputs.",
                }
                for pid in sorted(expected_ids):
                    row = p2_by_prompt[pid]
                    references.append({
                        "workload": workload, "pair": pair, "prompt_id": pid,
                        "target_token_ids": [int(x) for x in row.reference_token_ids],
                        "prompt_hash": str(row.prompt_hash),
                        "target_reference_source": f"{p2_base / 'continuations.parquet'} (canonical current 4B target reference reused for P3)",
                    })
            else:
                base = WIKI / slug if workload == "WIKIPEDIA" else FLORES / "traces" / slug
                if workload == "FLORES":
                    base = FLORES / "token_tables" / slug
                cont = pd.read_parquet(base / "continuations.parquet")
                for row in cont.itertuples(index=False):
                    references.append({
                        "workload": workload, "pair": pair, "prompt_id": int(row.prompt_id),
                        "target_token_ids": [int(x) for x in row.reference_token_ids],
                        "prompt_hash": str(row.prompt_hash), "target_reference_source": str(base / "continuations.parquet"),
                    })
    prompt_frame = pd.DataFrame(records)
    ref_frame = pd.DataFrame(references)
    if prompt_frame.duplicated(["workload", "prompt_id"]).any():
        raise AssertionError("Duplicate workload/prompt manifest key")
    if ref_frame.duplicated(["workload", "pair", "prompt_id"]).any():
        raise AssertionError("Duplicate target reference key")
    for (workload, pair), group in ref_frame.groupby(["workload", "pair"]):
        expected = 1000 if workload == "WIKIPEDIA" else 1012
        if len(group) != expected:
            raise AssertionError(f"{workload}/{pair}: expected {expected} saved target references, found {len(group)}")
    return prompt_frame, ref_frame, {
        "prompt_count_wikipedia": len(wiki_prompts),
        "prompt_count_flores": len(flores_manifest),
        "prompt_hashes": prompt_hashes,
        "target_reference_resolution": target_reference_resolution,
    }


def _event_groups(path: Path, pair: str, workload: str):
    events = pd.read_parquet(path)
    if "proposal_slot" not in events:
        events = events.copy()
        # The legacy P3 trace stores proposal_position as a one-based slot,
        # matching the later E2/FLORES proposal_slot field.
        events["proposal_slot"] = pd.to_numeric(events["proposal_position"], errors="coerce").astype(int)
    if "draft_proposed_token_id" not in events:
        events = events.copy()
        events["draft_proposed_token_id"] = events["draft_token_id"]
    if "sd_valid" not in events:
        events = events.copy()
        events["sd_valid"] = events["rejected"].notna()
    events["proposal_slot"] = pd.to_numeric(events["proposal_slot"], errors="coerce").astype(int)
    events["output_token_position"] = pd.to_numeric(events["output_token_position"], errors="coerce").astype(int)
    for (pid, rnd), group in events.groupby(["prompt_id", "round_index"], sort=False):
        ordered = group.sort_values("proposal_slot")
        slots = ordered["proposal_slot"].astype(int).tolist()
        if slots != list(range(1, len(slots) + 1)):
            raise AssertionError(f"Non-contiguous proposals in {workload}/{pair}/{pid}/{rnd}: {slots}")
        base = int(ordered.iloc[0]["output_token_position"]) - (int(ordered.iloc[0]["proposal_slot"]) - 1)
        rejected = ordered.loc[ordered["sd_valid"].astype(bool) & ordered["rejected"].fillna(False).astype(bool)]
        reject_slot = int(rejected["proposal_slot"].min()) if len(rejected) else None
        yield {
            "workload": workload, "pair": pair, "prompt_id": int(pid), "round_index": int(rnd),
            "base_position": base, "reject_slot": reject_slot,
            "proposal_ids": [int(x) for x in ordered["draft_proposed_token_id"]],
        }


def _artifact_paths(pair: str, workload: str) -> tuple[Path, Path]:
    slug = PAIRS[pair]["slug"]
    if workload == "WIKIPEDIA":
        events = WIKI / slug / "sd_events.parquet" if pair != "P3" else LEGACY / "sd_events.parquet"
        morph = WIKI / "h3" / "h3_token_environment.parquet"
    else:
        events = FLORES / "traces" / slug / "sd_events.parquet"
        morph = FLORES / "token_tables" / slug / "h3_token_environment.parquet"
    return events, morph


def _source_artifacts() -> dict[str, str]:
    paths = [
        WIKI / "config.json", WIKI / "prompt_ids.jsonl", WIKI / "h3/h3_token_environment.parquet",
        LEGACY / "sd_events.parquet", LEGACY / "reference_outputs.jsonl", LEGACY / "h2_token_table.parquet",
        FLORES / "flores_config.json", FLORES / "data_manifest.csv", FLORES / "data_manifest.sha256",
    ]
    for pair in PAIRS:
        slug = PAIRS[pair]["slug"]
        if pair != "P3":
            paths.extend([WIKI / slug / "sd_events.parquet", WIKI / slug / "continuations.parquet"])
        paths.extend([
            FLORES / "traces" / slug / "sd_events.parquet",
            FLORES / "token_tables" / slug / "continuations.parquet",
            FLORES / "token_tables" / slug / "h3_token_environment.parquet",
        ])
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def _load_h3(workload: str, pair: str) -> pd.DataFrame:
    _, path = _artifact_paths(pair, workload)
    frame = pd.read_parquet(path)
    if "pair" in frame:
        frame = frame.loc[frame["pair"].astype(str).eq(pair)]
    return frame.set_index(["prompt_id", "generation_pos"], drop=False)


def _offline_pair(
    pair: str,
    workload: str,
    prompt_by_key: dict[tuple[str, int], list[int]],
    reference_by_key: dict[tuple[str, str, int], list[int]],
    tokenizer: Any,
    kiwi: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_path, _ = _artifact_paths(pair, workload)
    h3 = _load_h3(workload, pair)
    out: list[dict[str, Any]] = []
    entropy_rounds: list[dict[str, Any]] = []
    teacher_path = (
        (WIKI / PAIRS[pair]["slug"] / "teacher_forced_tokens.parquet" if pair != "P3" else LEGACY / "teacher_forced_tokens.parquet")
        if workload == "WIKIPEDIA" else FLORES / "traces" / PAIRS[pair]["slug"] / "teacher_forced_tokens.parquet"
    )
    teacher = pd.read_parquet(teacher_path)[["prompt_id", "output_token_position", "draft_entropy"]]
    entropy_lookup = {
        (int(r.prompt_id), int(r.output_token_position)): float(r.draft_entropy)
        for r in teacher.itertuples(index=False) if pd.notna(r.draft_entropy)
    }
    for group in _event_groups(event_path, pair, workload):
        pid, base, reject_slot = group["prompt_id"], group["base_position"], group["reject_slot"]
        proposals = group["proposal_ids"]
        pids = prompt_by_key[(workload, pid)]
        internal_max = min(K - 1, len(proposals) - 1)
        accessible_max = internal_max
        h3_rows: list[dict[str, Any] | None] = []
        for slot in range(1, accessible_max + 1):
            key = (pid, base + slot - 1)
            try:
                row = h3.loc[key]
                if isinstance(row, pd.DataFrame):
                    raise AssertionError(f"duplicate H3 row key {key}")
                h3_rows.append(row.to_dict())
            except KeyError:
                h3_rows.append(None)
        candidate_rows: list[dict[str, Any]] = []
        if reject_slot is None:
            # Every proposal in a fully accepted block is the saved target
            # continuation token; H3 already contains the exact H2/H3 spans.
            for slot, row in enumerate(h3_rows, start=1):
                if row is None:
                    continue
                if int(row.get("token_id", -1)) != int(proposals[slot - 1]):
                    raise AssertionError(f"Accepted proposal does not equal target continuation token: {workload}/{pair}/{pid}/{base}/{slot}")
                n_cross = row.get("num_boundaries_crossed")
                n_cross = int(n_cross) if pd.notna(n_cross) else 0
                if bool(row.get("sd_valid", True)) and str(row.get("morph_class", "")) == "CROSS_MORPHEME" and n_cross == 1:
                    candidate_rows.append({
                        "slot": slot, "morph_class": "CROSS_MORPHEME", "num_boundaries_crossed": n_cross,
                        "environment": str(row.get("morph_environment", "")),
                        "crossed_fine_pos_sequence": str(row.get("crossed_fine_pos_sequence", "")),
                        "source": "saved_H3_exact_target_token",
                    })
        else:
            # Once a block rejects, the hypothetical draft suffix differs from
            # the target continuation. Reproject the full proposal block so
            # exact token spans for even the accepted prefix reflect the actual
            # proposed UTF-8/token sequence rather than later target tokens.
            committed = reference_by_key[(workload, pair, pid)][:base]
            projected, _ = project_candidate_block(
                tokenizer, kiwi, pids, committed, proposals, pid, group["round_index"], pair,
            )
            for projected_record in projected:
                slot = int(projected_record["proposal_slot"])
                if slot > internal_max:
                    continue
                if (
                    projected_record["detector_status"] == "VALID"
                    and projected_record["h2_morph_class"] == "CROSS_MORPHEME"
                    and int(projected_record["num_boundaries_crossed"]) == 1
                ):
                    candidate_rows.append({
                        "slot": slot, "morph_class": "CROSS_MORPHEME", "num_boundaries_crossed": 1,
                        "environment": str(projected_record["crossed_boundary_sequence"]),
                        "crossed_fine_pos_sequence": str(projected_record["crossed_fine_pos_sequence"]),
                        "source": "projected_saved_draft_candidate_at_or_after_rejection",
                    })
        candidate_rows.sort(key=lambda x: int(x["slot"]))
        # Entropy threshold calibration uses only proposal contexts whose
        # preceding speculative prefix was accepted, hence teacher-forced
        # draft entropy has the same committed context as the online draft.
        entropy_slots = []
        for slot in range(1, min(K - 1, len(proposals) - 1) + 1):
            if reject_slot is not None and slot > reject_slot:
                break
            entropy = entropy_lookup.get((pid, base + slot - 1))
            if entropy is not None:
                entropy_slots.append({"slot": slot, "entropy": entropy})
        entropy_rounds.append({
            "prompt_id": pid, "round_index": group["round_index"], "reject_slot": reject_slot,
            "entropy_slots": entropy_slots,
        })
        np_hit = next((x for x in candidate_rows if x["environment"] == "NOMINAL_TO_PARTICLE"), None)
        any_hit = candidate_rows[0] if candidate_rows else None
        for policy, hit in [("NP_BOUNDARY_GUARD", np_hit), ("ALL_MORPH_BOUNDARY_GUARD", any_hit)]:
            if hit is None:
                continue
            slot = int(hit["slot"])
            out.append({
                "workload": workload, "pair": pair, "prompt_id": pid,
                "round_index": group["round_index"], "base_generation_position": base,
                "policy": policy, "guard_slot": slot, "guard_source": hit["source"],
                "baseline_reject_slot": reject_slot,
                "baseline_rejection_relation": "AT_GUARD" if reject_slot == slot else ("BEFORE_GUARD" if reject_slot is not None and reject_slot < slot else ("AFTER_GUARD" if reject_slot is not None else "NO_REJECTION")),
                "reachable_before_or_at_first_rejection": reject_slot is None or reject_slot >= slot,
                "proposals_in_block": len(proposals), "verifier_positions_from_cut_including_candidate": len(proposals) - slot + 1,
                "verifier_positions_after_candidate": max(0, len(proposals) - slot),
                "crossed_fine_pos_sequence": hit["crossed_fine_pos_sequence"],
            })
    return out, entropy_rounds


def calibrate_entropy_threshold(np_rows: pd.DataFrame, rounds: list[dict[str, Any]], pair: str) -> dict[str, Any]:
    target = np_rows.loc[np_rows["pair"].eq(pair) & np_rows["workload"].eq("WIKIPEDIA") & np_rows["policy"].eq("NP_BOUNDARY_GUARD")]
    target_hist = Counter(target["guard_slot"].astype(int))
    target_n = int(len(target))
    values = np.asarray([float(item["entropy"]) for group in rounds for item in group["entropy_slots"]], dtype=float)
    if len(values) == 0:
        return {"threshold": None, "target_np_activation_blocks": target_n, "matched_activation_blocks": 0, "reason": "no_entropy_observations"}
    max_internal_slot = K - 1
    entropy_matrix = np.full((len(rounds), max_internal_slot), -np.inf, dtype=float)
    for row_idx, group in enumerate(rounds):
        for item in group["entropy_slots"]:
            slot = int(item["slot"])
            if 1 <= slot <= max_internal_slot:
                entropy_matrix[row_idx, slot - 1] = float(item["entropy"])
    # A fixed 201-point empirical quantile grid makes threshold selection
    # deterministic and avoids quadratic Python loops over every unique FP16
    # entropy value while still matching the activation/slot distribution.
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    best = None
    for threshold in candidates:
        crossings = entropy_matrix >= float(threshold)
        active = crossings.any(axis=1)
        first_slot = crossings.argmax(axis=1) + 1
        hist = Counter(int(x) for x in first_slot[active])
        matched_hist = {slot: min(int(target_hist.get(slot, 0)), int(hist.get(slot, 0))) for slot in target_hist}
        matched_n = int(sum(matched_hist.values()))
        eligible_n = int(sum(hist.values()))
        # A scalar entropy cutoff can have a discontinuous activation rate.
        # Freeze a deterministic, slot-specific thinning probability on
        # Wikipedia so the entropy control approximately matches the NP guard
        # rate and cut-position distribution without consulting FLORES.
        probabilities = {
            str(slot): float(matched_hist[slot] / max(1, int(hist.get(slot, 0))))
            for slot in target_hist
        }
        extra_eligible = eligible_n - matched_n
        row = {
            "threshold": float(threshold), "matched_blocks": matched_n,
            "eligible_blocks_before_thinning": eligible_n,
            "eligible_slot_histogram": {str(k): int(v) for k, v in hist.items()},
            "target_np_slot_histogram": {str(k): int(v) for k, v in target_hist.items()},
            "slot_thinning_probabilities": probabilities,
            "extra_eligible_blocks": extra_eligible,
        }
        # Maximize the number of target slots supported, then minimize
        # thinning; use the higher cutoff as a deterministic tie break.
        score = (-matched_n, extra_eligible, -float(threshold))
        if best is None or score < best["score"]:
            row["score"] = score
            best = row
    return {
        **best, "score": list(best["score"]), "target_np_activation_blocks": target_n,
        "calibration_source": "WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix",
        "method": "201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities",
    }


def main() -> None:
    from transformers import AutoTokenizer
    from kiwipiepy import Kiwi

    for directory in ["feasibility", "correctness", "pilot", "full_benchmark", "figures"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    config_path = WIKI / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["4B"]["id"], revision=MODELS["4B"]["revision"], use_fast=True, local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Exact offset mapping requires the previously validated fast tokenizer")
    prompt_frame, ref_frame, prompt_summary = _load_sources(tokenizer)
    rng = np.random.default_rng(SEED)
    pilot_ids: dict[str, set[int]] = {}
    correct_ids: dict[str, set[int]] = {}
    for workload in ["WIKIPEDIA", "FLORES"]:
        rows = prompt_frame.loc[prompt_frame.workload.eq(workload)].sort_values("source_index")
        selected = set(int(x) for x in rng.choice(rows.prompt_id.to_numpy(), size=200, replace=False))
        ordered_selected = [int(x) for x in rows.prompt_id if int(x) in selected]
        pilot_ids[workload] = set(ordered_selected)
        correct_ids[workload] = set(ordered_selected[:100])
    prompt_frame["pilot_selected"] = [int(r.prompt_id) in pilot_ids[str(r.workload)] for r in prompt_frame.itertuples(index=False)]
    prompt_frame["correctness_selected"] = [int(r.prompt_id) in correct_ids[str(r.workload)] for r in prompt_frame.itertuples(index=False)]
    atomic_csv(prompt_frame, OUT / "prompt_manifest.csv")
    ref_path = OUT / "source_target_references.parquet"
    tmp_ref = ref_path.with_suffix(".parquet.tmp")
    ref_frame.to_parquet(tmp_ref, index=False)
    tmp_ref.replace(ref_path)
    pilot_frame = prompt_frame.loc[prompt_frame.pilot_selected].copy()
    pilot_frame["pilot_order"] = pilot_frame.groupby("workload", sort=False).cumcount()
    atomic_csv(pilot_frame, OUT / "pilot/prompt_manifest.csv")

    prompt_by_key = {
        (str(row.workload), int(row.prompt_id)): [int(x) for x in json.loads(row.prompt_ids_json)]
        for row in prompt_frame.itertuples(index=False)
    }
    reference_by_key = {
        (str(row.workload), str(row.pair), int(row.prompt_id)): [int(x) for x in row.target_token_ids]
        for row in ref_frame.itertuples(index=False)
    }
    kiwi = Kiwi()
    opportunity_rows: list[dict[str, Any]] = []
    wiki_entropy_rounds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_dir = OUT / "feasibility/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for workload in ["WIKIPEDIA", "FLORES"]:
        for pair in PAIRS:
            key = f"{FEASIBILITY_VERSION}_{workload.lower()}_{pair.lower()}"
            rows_path = checkpoint_dir / f"{key}_opportunities.csv"
            entropy_path = checkpoint_dir / f"{key}_entropy_rounds.json"
            if rows_path.exists() and entropy_path.exists():
                rows = pd.read_csv(rows_path).to_dict("records")
                entropy_rounds = json.loads(entropy_path.read_text(encoding="utf-8"))
                opportunity_rows.extend(rows)
                if workload == "WIKIPEDIA":
                    wiki_entropy_rounds[pair] = entropy_rounds
                print(f"CPU feasibility {workload}/{pair}: loaded checkpoint ({len(rows):,} guard opportunity rows)", flush=True)
                continue
            started = time.perf_counter()
            rows, entropy_rounds = _offline_pair(pair, workload, prompt_by_key, reference_by_key, tokenizer, kiwi)
            opportunity_rows.extend(rows)
            if workload == "WIKIPEDIA":
                wiki_entropy_rounds[pair] = entropy_rounds
            atomic_csv(pd.DataFrame(rows), rows_path)
            atomic_json(entropy_path, entropy_rounds)
            print(f"CPU feasibility {workload}/{pair}: {len(rows):,} guard opportunity rows; {time.perf_counter()-started:.1f}s", flush=True)
    opportunities = pd.DataFrame(opportunity_rows)
    if opportunities.empty:
        opportunities = pd.DataFrame(columns=["workload", "pair", "prompt_id", "round_index", "policy", "guard_slot"])
    atomic_csv(opportunities, OUT / "feasibility/offline_guard_opportunity.csv")

    thresholds = {
        pair: calibrate_entropy_threshold(opportunities, wiki_entropy_rounds[pair], pair)
        for pair in PAIRS
    }
    thresholds_json = {pair: {**row, "threshold": (None if row.get("threshold") is None else float(row["threshold"]))} for pair, row in thresholds.items()}
    slot_distributions: dict[str, dict[str, float]] = {}
    for pair in PAIRS:
        selected_slots = opportunities.loc[
            (opportunities["workload"] == "WIKIPEDIA")
            & (opportunities["pair"] == pair)
            & (opportunities["policy"] == "NP_BOUNDARY_GUARD"),
            "guard_slot",
        ].astype(int)
        denominator = max(1, len(selected_slots))
        slot_distributions[pair] = {
            str(int(slot)): float(count / denominator)
            for slot, count in Counter(selected_slots).items()
        }
    artifact_hashes = _source_artifacts()
    config_out = {
        "experiment": "Training-free exact morphology-aware speculative-decoding scheduling intervention",
        "method": "NP_BOUNDARY_GUARD",
        "seed": SEED,
        "random_guard_seed": SLOT_SEED,
        "models": MODELS,
        "pairs": PAIRS,
        "dtype": "float16",
        "attention_backend": "sdpa",
        "max_prompt_tokens": 128,
        "max_new_tokens": MAX_NEW,
        "speculative_k": K,
        "eos_token_id": EOS,
        "generation": "greedy, no sampling, use_cache=True",
        "batch_target_verification": False,
        "verification_optimization": "Enable causal block target verification only after exact token/event parity against the original sequential verifier is established on 8 frozen prompts per workload/model-pair/SD-variant.",
        "prompt_counts": {"WIKIPEDIA": 1000, "FLORES": 1012},
        "prompt_order": "source order; pilot uses seed 3090 choice without replacement then returns to source order; 100 selected pilot IDs per workload are correctness suite",
        "target_references_reused": True,
        "reference_artifact": str(ref_path.relative_to(ROOT)),
        "policy_variants": ["TARGET_ONLY", "FIXED_K_SD", "NP_BOUNDARY_GUARD", "ALL_MORPH_BOUNDARY_GUARD", "RANDOM_MATCHED_GUARD", "ENTROPY_MATCHED_GUARD"],
        "entropy_thresholds_wikipedia_only": thresholds_json,
        "entropy_policy": "earliest internal slot with online draft entropy >= frozen pair-specific threshold; a deterministic slotwise thinning probability calibrated on WIKIPEDIA matches the approximate NP activation and slot distribution; both remain frozen on FLORES",
        "wikipedia_pilot_slot_distributions": slot_distributions,
        "gpu_requirement": {
            "physical_gpu_index": 3, "gpu_name": "NVIDIA A100 80GB", "CUDA_VISIBLE_DEVICES": "3",
            "framework_local_device": "cuda:0", "inventory_and_versions_recorded_before_first_GPU_job": False,
        },
        "gpu_metadata": None,
        "source_artifact_sha256": artifact_hashes,
        "prompt_summary": {k: v for k, v in prompt_summary.items() if k != "prompt_hashes"},
        "target_reference_resolution": prompt_summary.get("target_reference_resolution", {}),
        "prompt_hashes_sha256": hashlib.sha256("\n".join(prompt_summary["prompt_hashes"].values()).encode()).hexdigest(),
        "software": {
            name: (importlib.metadata.version(name) if _has_dist(name) else None)
            for name in ["torch", "transformers", "tokenizers", "kiwipiepy", "numpy", "pandas", "pyarrow"]
        },
        "stage": "CPU preprocessing and offline opportunity analysis complete; GPU stages not started",
    }
    atomic_json(OUT / "method_config.json", config_out)
    if prompt_summary.get("target_reference_resolution"):
        atomic_json(
            OUT / "correctness/target_reference_compatibility.json",
            prompt_summary["target_reference_resolution"],
        )
    summary = opportunities.groupby(["workload", "pair", "policy"], dropna=False).agg(
        candidate_blocks=("prompt_id", "size"), prompts=("prompt_id", "nunique"),
        mean_guard_slot=("guard_slot", "mean"), mean_positions_after=("verifier_positions_after_candidate", "mean"),
        reachable_share=("reachable_before_or_at_first_rejection", "mean"),
    ).reset_index() if len(opportunities) else pd.DataFrame()
    lines = [
        "# Offline detector validation", "",
        "This is a CPU-only feasibility audit of saved baseline traces; it is not a runtime or speed claim.",
        "For proposals before the first baseline rejection (and all proposals in fully accepted rounds), exact H3 target-token labels are reused because the proposed token ID equals the saved target continuation ID. At and after a rejection, the entire saved draft proposal block is reprojected from the committed target prefix through the H2 tokenizer-span routine and H3 POS transition function. This records both reachable opportunities and candidates following a rejection; the latter are marked unreachable for that baseline block.",
        "", "## Opportunity counts", "",
        markdown_table(summary) if len(summary) else "No valid opportunities found.",
        "", "## Frozen entropy thresholds", "",
        markdown_table(pd.DataFrame([{"pair": pair, **row} for pair, row in thresholds_json.items()])),
        "", "## Alignment caveat", "",
        "The offline opportunity table conservatively excludes any candidate whose H3 token record is missing or whose exact rejected-token projection is ambiguous. The online controller uses the full saved H2/H3 projection and disables guarding for a block whenever the full candidate sequence does not round-trip exactly.", "",
    ]
    (OUT / "feasibility/detector_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote prompt manifest, target refs, threshold config, and {len(opportunities):,} opportunity rows.", flush=True)


def _has_dist(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
