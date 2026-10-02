#!/usr/bin/env python3
"""Exploratory boundary-type analysis on actual saved draft proposals.

Projects every target-observed proposal in its complete draft block, then
assigns CROSS boundaries or the nearest adjacent boundary for WITHIN_SPLIT
tokens using the established H3 taxonomy. Invalidated suffixes never become
outcome rows. Primary reporting is limited to P1/P2 on Wiki and FLORES; Wiki
P3 is excluded because its historical draft revision is unresolved.
"""

from __future__ import annotations

import json
import math
import sys
import argparse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from kiwipiepy import Kiwi

from scripts import analyze_h3 as h3
from scripts.analyze_proposal_side import (
    FLORES,
    FREQUENCY,
    WIKI,
    load_prompts,
    load_references,
    load_tokenizer,
    normalize_events,
    paths,
    target_lookup,
)
from src.e2_model_pairs import PAIRS
from src.morphology_guard import project_candidate_block

OUT = ROOT / "runs/proposal_side_boundary_validation"
JOIN_KEYS = ["workload", "pair", "prompt_id", "round_index", "proposal_slot", "output_token_position"]


def proposal_environment(record: dict[str, Any]) -> tuple[str, str]:
    """Assign H3 boundary environment from this draft token's own full-block span."""
    morph_class = str(record.get("h2_morph_class", ""))
    n_boundaries = int(record.get("num_boundaries_crossed", 0) or 0)
    if morph_class == "CROSS_MORPHEME":
        seq = str(record.get("crossed_boundary_sequence") or "")
        if n_boundaries != 1 or not seq or "|" in seq:
            return "MULTI_BOUNDARY", "crossed_boundary"
        return seq, "crossed_boundary"
    if morph_class != "WITHIN_SPLIT":
        return "OTHER", "not_primary_class"

    start, end = int(record["token_char_start"]), int(record["token_char_end"])
    morphs = record.get("kiwi_sequence") or []
    overlap = [i for i, morph in enumerate(morphs)
               if max(0, min(end, int(morph["end"])) - max(start, int(morph["start"]))) > 0]
    if len(overlap) != 1:
        return "OTHER", "within_split_not_single_local_morpheme"
    idx = overlap[0]
    candidates: list[tuple[int, int, str]] = []
    for edge_idx in (idx - 1, idx):
        if not 0 <= edge_idx < len(morphs) - 1:
            continue
        left, right = morphs[edge_idx], morphs[edge_idx + 1]
        if int(left["end"]) != int(right["start"]):
            continue
        boundary = int(left["end"])
        distance = boundary - end if end <= boundary else start - boundary if start >= boundary else 0
        env = h3.boundary_type(str(left["fine_pos"]), str(right["fine_pos"]))
        # Match H3: on equal distances, the right/following edge wins.
        candidates.append((distance, 1 if edge_idx == idx else 0, env))
    if not candidates:
        return "OTHER", "no_valid_adjacent_boundary"
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][2], "nearest_adjacent_boundary"


def build_pair(workload: str, pair: str, tokenizer: Any, kiwi: Any, prompt_map: dict[int, list[int]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_path, target_path, reference_path = paths(workload, pair)
    events = normalize_events(pd.read_parquet(event_path))
    targets = pd.read_parquet(target_path)
    if "pair" in targets.columns:
        targets = targets.loc[targets.pair.astype(str).eq(pair)]
    target_by_position = target_lookup(targets)
    references = load_references(workload, pair, reference_path)
    saved = pd.read_parquet(OUT.parent / "proposal_side_validation" / "pairs" / f"{workload.lower()}_{pair.lower()}" / "proposal_side_rows.parquet")
    saved["workload"] = workload
    saved["pair"] = pair
    rows: list[dict[str, Any]] = []
    event_mismatches = accepted_id_mismatches = 0
    projection_valid = projection_invalid = exact_roundtrip = 0
    blocks = events.groupby(["prompt_id", "round_index"], sort=False, observed=True)
    for block_i, ((raw_pid, raw_round), block) in enumerate(blocks, start=1):
        if block_i % 3000 == 0:
            print(f"[{workload}/{pair}] projected {block_i}/{blocks.ngroups} blocks", flush=True)
        block = block.sort_values("proposal_slot")
        actual = block.loc[block.rejected.notna()].copy()
        if actual.empty:
            continue
        pid, rnd = int(raw_pid), int(raw_round)
        base = int(block.output_token_position.min())
        proposals = [int(x) for x in block.draft_proposed_token_id]
        committed = references[pid][:base]
        projected, meta = project_candidate_block(
            tokenizer, kiwi, prompt_map[pid], committed, proposals, pid, rnd, pair,
            allow_local_exact_spans=True,
        )
        by_slot = {int(r["proposal_slot"]): r for r in projected}
        for event in actual.itertuples(index=False):
            pos, slot = int(event.output_token_position), int(event.proposal_slot)
            rec = by_slot.get(slot)
            target = target_by_position.get((pid, pos))
            if rec is None or target is None:
                raise AssertionError(f"Missing projection/target {workload}/{pair}/{pid}/{rnd}/{slot}")
            if int(rec["token_id"]) != int(event.draft_proposed_token_id):
                raise AssertionError("Projected draft token ID differs from saved proposal event")
            target_id = int(target["token_id"])
            if pd.notna(getattr(event, "target_greedy_token_id", np.nan)) and int(event.target_greedy_token_id) != target_id:
                event_mismatches += 1
                raise AssertionError("Saved event target token differs from target continuation")
            if bool(event.rejected):
                if int(event.draft_proposed_token_id) == target_id:
                    raise AssertionError("Rejected draft proposal equals target token")
            elif int(event.draft_proposed_token_id) != target_id:
                accepted_id_mismatches += 1
                raise AssertionError("Accepted draft token differs from target token")

            status = str(rec["detector_status"])
            projection_valid += int(status == "VALID")
            projection_invalid += int(status != "VALID")
            exact_roundtrip += int(bool(rec["alignment_roundtrip_exact"]))
            env, env_source = proposal_environment(rec)
            rows.append({
                "workload": workload, "pair": pair, "prompt_id": pid, "round_index": rnd,
                "proposal_slot": slot, "output_token_position": pos,
                "draft_token_id": int(event.draft_proposed_token_id), "target_token_id": target_id,
                "sd_rejected": int(bool(event.rejected)), "proposal_morph_class": str(rec["h2_morph_class"]),
                "projection_status": status, "proposal_morph_environment": env,
                "environment_source": env_source, "num_boundaries_crossed": int(rec["num_boundaries_crossed"]),
                "candidate_surface": rec["token_surface"], "candidate_eojeol": rec["eojeol"],
                "candidate_token_char_length": int(rec["token_char_end"]) - int(rec["token_char_start"]),
                "candidate_eojeol_char_length": len(str(rec.get("eojeol") or "")),
                "fragmentation_bin": target.get("fragmentation_bin"),
                "relative_position": target.get("relative_position"),
                "first_token": target.get("first_token"), "last_token": target.get("last_token"),
                "generation_position": int(getattr(event, "generation_position", target.get("generation_position", pos))),
                "proposal_slot_ctrl": slot, "draft_entropy": target.get("draft_entropy"),
                "target_entropy": target.get("target_entropy"),
                "fullblock_roundtrip_exact": bool(meta["roundtrip_exact"]),
            })

    fresh = pd.DataFrame(rows)
    saved_keys = saved[JOIN_KEYS + ["draft_token_id", "target_token_id"]].copy()
    fresh_keys = fresh[JOIN_KEYS + ["draft_token_id", "target_token_id"]].copy()
    if len(saved_keys) != len(fresh_keys):
        raise AssertionError(f"Saved/fresh population mismatch: {len(saved_keys)} vs {len(fresh_keys)}")
    check = saved_keys.merge(fresh_keys, on=JOIN_KEYS + ["draft_token_id", "target_token_id"], how="outer", indicator=True, validate="one_to_one")
    if not check._merge.eq("both").all():
        raise AssertionError("Fresh projection population differs from validated proposal-side rows")
    frequency = pd.read_parquet(FREQUENCY)[["token_id", "log_token_count"]].rename(columns={"token_id": "draft_token_id", "log_token_count": "log_candidate_token_count"})
    fresh = fresh.merge(frequency, on="draft_token_id", how="left", validate="many_to_one", suffixes=("", "_freq"))
    if fresh.log_candidate_token_count.isna().any():
        raise AssertionError("Some draft candidates lack pinned frequency counts")
    return fresh, {
        "workload": workload, "pair": pair, "saved_valid_proposals": int(len(saved)),
        "fullblock_projection_valid": int(projection_valid), "fullblock_projection_invalid": int(projection_invalid),
        "fullblock_roundtrip_exact": int(exact_roundtrip), "event_target_mismatches": int(event_mismatches),
        "accepted_id_mismatches": int(accepted_id_mismatches),
        "valid_primary_class": int((fresh.projection_status.eq("VALID") & fresh.proposal_morph_class.isin(["CROSS_MORPHEME", "WITHIN_SPLIT"])).sum()),
        "valid_single_boundary_cross": int((fresh.projection_status.eq("VALID") & fresh.proposal_morph_class.eq("CROSS_MORPHEME") & fresh.num_boundaries_crossed.eq(1)).sum()),
        "valid_n2p_cross": int((fresh.projection_status.eq("VALID") & fresh.proposal_morph_class.eq("CROSS_MORPHEME") & fresh.num_boundaries_crossed.eq(1) & fresh.proposal_morph_environment.eq("NOMINAL_TO_PARTICLE")).sum()),
    }


def fit_h3(frame: pd.DataFrame, pair_effects: bool = False) -> list[dict[str, Any]]:
    data = frame.loc[
        frame.projection_status.eq("VALID")
        & (frame.proposal_morph_class.eq("WITHIN_SPLIT")
           | (frame.proposal_morph_class.eq("CROSS_MORPHEME") & frame.num_boundaries_crossed.eq(1)))
        & frame.fragmentation_bin.astype(str).isin(h3.FRAGMENTATION_ORDER)
    ].copy()
    data = data.rename(columns={
        "proposal_morph_class": "morph_class", "proposal_morph_environment": "morph_environment",
        "log_candidate_token_count": "log_token_count",
        "candidate_token_char_length": "token_char_length",
        "candidate_eojeol_char_length": "eojeol_char_length",
    })
    data["morph_environment"] = data.morph_environment.where(data.morph_environment.isin(h3.ENVIRONMENTS), "OTHER")
    data, env_ref = h3._categories(data, "morph_environment", "LEXICAL_TO_LEXICAL")
    data, frag_ref = h3._categories(data, "fragmentation_bin", "2")
    formula = h3.formula_for("", interaction=True)
    if env_ref != "LEXICAL_TO_LEXICAL":
        formula = formula.replace("reference='LEXICAL_TO_LEXICAL'", f"reference='{env_ref}'")
    if frag_ref != "2":
        formula = formula.replace("reference='2'", f"reference='{frag_ref}'")
    if pair_effects:
        pair_formula = "C(pair, Treatment(reference='P1')) + C(pair, Treatment(reference='P1')):C(morph_class, Treatment(reference='WITHIN_SPLIT')) + "
        formula = formula.replace("sd_rejected ~ ", "sd_rejected ~ " + pair_formula)
    required = ["sd_rejected", "prompt_id", "log_token_count", "fragmentation_bin"] + h3.STRUCTURAL + h3.ENTROPY
    data = data.dropna(subset=required).copy()
    fitted, diag = h3.fit_clustered(data, formula, {"morph_environment": env_ref, "fragmentation_bin": frag_ref})
    res = h3.adjusted_contrast(fitted, data, "NOMINAL_TO_PARTICLE")
    res.update({
        "pair": "P1+P2" if pair_effects else str(frame.pair.iloc[0]),
        "workload": str(frame.workload.iloc[0]), "formula": formula,
        "model_converged": bool(diag.get("converged", False)), "model_n_rows": diag.get("n_tokens"),
        "model_n_prompts": diag.get("n_prompts"), "model_rank": diag.get("rank"),
        "model_n_columns": diag.get("n_columns"), "model_fit_error": diag.get("fit_error"),
    })
    return [res]


def refit_existing() -> None:
    """Refit saved full-block projections without repeating tokenizer/kiwi work."""
    OUT.mkdir(parents=True, exist_ok=True)
    all_frames, audits = [], []
    for workload in ("WIKIPEDIA", "FLORES"):
        for pair in ("P1", "P2"):
            path = OUT / f"proposal_boundary_rows_{workload.lower()}_{pair.lower()}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            all_frames.append(pd.read_parquet(path))
    audit_df = pd.read_csv(OUT / "projection_population_audit.csv")
    results = []
    for frame in all_frames:
        results.extend(fit_h3(frame))
    table = pd.concat(all_frames, ignore_index=True)
    for workload in ("WIKIPEDIA", "FLORES"):
        results.extend(fit_h3(table.loc[table.workload.eq(workload)].copy(), pair_effects=True))
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "nominal_to_particle_adjusted_results.csv", index=False)
    lines = [
        "# Proposal-side Nominal→Particle analysis", "",
        "Exploratory offline reanalysis of the saved P1/P2 Wiki and English→Korean FLORES traces. Every observed proposal is reprojected from the complete draft block, including accepted proposals; suffix proposals after the first reject are excluded. Accepted proposals are not assigned target-side labels. This tests actual draft proposal segmentation.", "",
        "The H3 taxonomy is applied to the candidate span. A CROSS candidate enters the primary boundary analysis only if it crosses exactly one boundary. A WITHIN_SPLIT candidate is assigned to the nearest adjacent morpheme boundary, with right/following-edge tie breaks as in H3. The model estimates adjusted CROSS−SPLIT probability difference within NOMINAL_TO_PARTICLE, controlling for the H3 structural variables, draft/target entropy, and a four-df spline of the draft candidate token's Wikipedia frequency; SEs are clustered by prompt.", "",
        "## Projection audit", "", audit_df.to_string(index=False), "",
        "## Adjusted proposal-side effects", "", results_df.to_string(index=False), "",
        "## Interpretation", "",
        "This is a follow-up exploratory result on actual proposal-side spans; it is distinct from the prior target-side H3 result. Wiki P1/P2 and FLORES P1/P2 are the primary evidence. The result does not establish causality or speedup. Projection-invalid and multi-boundary CROSS proposals are not in the primary contrast; see saved population audit.", "",
    ]
    (OUT / "proposal_side_boundary_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Refit proposal-side boundary models from saved projections", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-existing", action="store_true", help="refit models using saved projection parquet files")
    if parser.parse_args().fit_existing:
        refit_existing()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer, tok_meta = load_tokenizer()
    kiwi = Kiwi()
    all_frames, audits = [], []
    for workload in ("WIKIPEDIA", "FLORES"):
        prompts = load_prompts(workload, tokenizer)
        for pair in ("P1", "P2"):
            print(f"Starting actual-proposal boundary projection {workload}/{pair}", flush=True)
            frame, audit = build_pair(workload, pair, tokenizer, kiwi, prompts)
            frame.to_parquet(OUT / f"proposal_boundary_rows_{workload.lower()}_{pair.lower()}.parquet", index=False)
            all_frames.append(frame)
            audits.append(audit)
            print(f"Finished {workload}/{pair}: {audit['valid_n2p_cross']:,} valid single-boundary N→P draft proposals", flush=True)
    table = pd.concat(all_frames, ignore_index=True)
    results = []
    for frame in all_frames:
        results.extend(fit_h3(frame))
    for workload in ("WIKIPEDIA", "FLORES"):
        pooled = table.loc[table.workload.eq(workload)].copy()
        results.extend(fit_h3(pooled, pair_effects=True))
    audit_df = pd.DataFrame(audits)
    results_df = pd.DataFrame(results)
    audit_df.to_csv(OUT / "projection_population_audit.csv", index=False)
    results_df.to_csv(OUT / "nominal_to_particle_adjusted_results.csv", index=False)
    table.to_parquet(OUT / "proposal_boundary_rows_all.parquet", index=False)
    lines = [
        "# Proposal-side Nominal→Particle analysis", "",
        "Exploratory offline reanalysis of the saved P1/P2 Wiki and English→Korean FLORES traces. Every observed proposal is reprojected from the complete draft block, including accepted proposals; suffix proposals after the first reject are excluded. Accepted proposals are not assigned target-side labels. This tests actual draft proposal segmentation.", "",
        "The H3 taxonomy is applied to the candidate span. A CROSS candidate enters the primary boundary analysis only if it crosses exactly one boundary. A WITHIN_SPLIT candidate is assigned to the nearest adjacent morpheme boundary, with right/following-edge tie breaks as in H3. The model estimates adjusted CROSS−SPLIT probability difference within NOMINAL_TO_PARTICLE, controlling for the H3 structural variables, draft/target entropy, and a four-df spline of the draft candidate token's Wikipedia frequency; SEs are clustered by prompt.", "",
        "## Projection audit", "", audit_df.to_string(index=False), "",
        "## Adjusted proposal-side effects", "", results_df.to_string(index=False), "",
        "## Interpretation", "",
        "This is a follow-up exploratory result on actual proposal-side spans; it is distinct from the prior target-side H3 result. Wiki P1/P2 and FLORES P1/P2 are the primary evidence. The result does not establish causality or speedup. Projection-invalid and multi-boundary CROSS proposals are not in the primary contrast; see saved population audit.", "",
    ]
    (OUT / "proposal_side_boundary_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps({
        "analysis": "actual draft proposal boundary-specific N2P CROSS vs nearest-boundary WITHIN_SPLIT",
        "workloads": ["WIKIPEDIA", "FLORES"], "pairs": ["P1", "P2"],
        "candidate_context": "complete saved draft proposal block for both accepted and first-rejected observed proposals",
        "invalidated_suffixes": "excluded as outcome rows",
        "tokenizer": tok_meta, "kiwi_version": "0.24.0",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote proposal-side boundary analysis to", OUT, flush=True)


if __name__ == "__main__":
    main()
