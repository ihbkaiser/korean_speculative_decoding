#!/usr/bin/env python3
"""Reanalyze speculative-decoding agreement with morphology on draft proposals.

This is an offline analysis of saved greedy-SD traces. It does not load model
weights or rerun decoding. Accepted proposal labels reuse the exact target-side
span label because the proposed token ID equals the target token ID. At the
first rejection in each block, the draft candidate is reprojected separately
from (a) the valid prefix through the rejected candidate and (b) the full
draft block. Proposals invalidated after the first rejection are never outcome
rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from kiwipiepy import Kiwi
from transformers import AutoTokenizer

from src.e2_model_pairs import MODELS, PAIRS, sha256_file
from src.morphology_guard import project_candidate_block

WIKI = ROOT / "runs/e2_model_pair_replication"
LEGACY = ROOT / "runs/20260926T184145Z_pilot1000"
FLORES = ROOT / "runs/flores200_en_ko_replication"
FREQUENCY = LEGACY / "e1_token_frequency_cache.parquet"
FREQ_META_JSON = LEGACY / "e1_token_frequency_cache_metadata.json"
PRIMARY_CLASSES = {"WITHIN_SPLIT", "CROSS_MORPHEME"}
FRAGMENTATION_ORDER = ["2", "3", "4", "5", "6", "7", "8+"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_tokenizer() -> tuple[Any, dict[str, Any]]:
    model = MODELS["0.6B"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"], use_fast=True,
        local_files_only=os.environ.get("PROPOSAL_SIDE_ALLOW_HUB", "0") != "1",
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Proposal-side morphology requires the pinned fast tokenizer")
    backend_hash = sha256_bytes(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    expected = json.loads(FREQ_META_JSON.read_text(encoding="utf-8"))
    if backend_hash != expected["tokenizer_backend_sha256"] or len(tokenizer) != int(expected["vocab_size"]):
        raise RuntimeError("Tokenizer does not match the saved E1 token-frequency cache")
    return tokenizer, {
        "model_id": model["id"], "revision": model["revision"],
        "tokenizer_class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer),
        "backend_sha256": backend_hash,
    }


def load_prompts(workload: str, tokenizer: Any) -> dict[int, list[int]]:
    if workload == "WIKIPEDIA":
        path = WIKI / "prompt_ids.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        prompts = {int(row["prompt_id"]): [int(x) for x in row["input_ids"]] for row in rows}
        if len(prompts) != 1000:
            raise AssertionError(f"Expected 1,000 frozen Wiki prompts, found {len(prompts)}")
        return prompts

    manifest = pd.read_csv(FLORES / "data_manifest.csv")
    if len(manifest) != 1012 or not manifest["included"].astype(bool).all():
        raise AssertionError("Expected all 1,012 FLORES devtest rows")
    prompts: dict[int, list[int]] = {}
    for row in manifest.itertuples(index=False):
        ids = tokenizer(str(row.rendered_prompt), add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
        digest = sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("ascii"))
        if digest != str(row.tokenized_prompt_sha256):
            raise AssertionError(f"FLORES prompt-token hash mismatch at prompt_id={row.prompt_id}")
        prompts[int(row.prompt_id)] = [int(x) for x in ids]
    return prompts


def paths(workload: str, pair: str) -> tuple[Path, Path, Path]:
    slug = PAIRS[pair]["slug"]
    if workload == "WIKIPEDIA":
        if pair == "P3":
            return LEGACY / "sd_events.parquet", LEGACY / "h2_token_table.parquet", LEGACY / "reference_outputs.jsonl"
        return WIKI / slug / "sd_events.parquet", WIKI / slug / "token_table.parquet", WIKI / slug / "continuations.parquet"
    return (
        FLORES / "traces" / slug / "sd_events.parquet",
        FLORES / "token_tables" / slug / "token_table.parquet",
        FLORES / "token_tables" / slug / "continuations.parquet",
    )


def load_references(workload: str, pair: str, reference_path: Path) -> dict[int, list[int]]:
    if workload == "WIKIPEDIA" and pair == "P3":
        with reference_path.open(encoding="utf-8") as handle:
            return {
                int(row["prompt_id"]): [int(x) for x in row["reference_token_ids"]]
                for row in (json.loads(line) for line in handle if line.strip())
            }
    frame = pd.read_parquet(reference_path)
    return {int(row.prompt_id): [int(x) for x in row.reference_token_ids] for row in frame.itertuples(index=False)}


def normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "proposal_slot" not in frame:
        frame["proposal_slot"] = frame["proposal_position"]
    if "draft_proposed_token_id" not in frame:
        frame["draft_proposed_token_id"] = frame["draft_token_id"]
    if "sd_valid" not in frame:
        frame["sd_valid"] = frame["rejected"].notna()
    frame["proposal_slot"] = pd.to_numeric(frame["proposal_slot"], errors="raise").astype(int)
    frame["output_token_position"] = pd.to_numeric(frame["output_token_position"], errors="raise").astype(int)
    return frame


def target_lookup(frame: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    frame = frame.copy()
    if "generation_pos" not in frame:
        raise AssertionError("Target morphology table has no generation_pos")
    if frame.duplicated(["prompt_id", "generation_pos"]).any():
        raise AssertionError("Duplicate target morphology rows")
    return {
        (int(row.prompt_id), int(row.generation_pos)): row._asdict()
        for row in frame.itertuples(index=False)
    }


def project_rejection(
    tokenizer: Any,
    kiwi: Any,
    prompt_ids: list[int],
    committed: list[int],
    proposals: list[int],
    reject_slot: int,
    prompt_id: int,
    round_index: int,
    pair: str,
    do_prefix_sensitivity: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prefix_records = None
    if do_prefix_sensitivity:
        prefix_records, _ = project_candidate_block(
            tokenizer, kiwi, prompt_ids, committed, proposals[:reject_slot], prompt_id, round_index, pair,
            allow_local_exact_spans=True,
        )
    full_records, _ = project_candidate_block(
        tokenizer, kiwi, prompt_ids, committed, proposals, prompt_id, round_index, pair,
        allow_local_exact_spans=True,
    )
    prefix = next((r for r in prefix_records if int(r["proposal_slot"]) == reject_slot), None) if prefix_records is not None else None
    full = next((r for r in full_records if int(r["proposal_slot"]) == reject_slot), None)
    return prefix, full


def build_pair_rows(
    workload: str,
    pair: str,
    tokenizer: Any,
    kiwi: Any,
    prompt_ids_by_id: dict[int, list[int]],
    max_prompts: int | None = None,
    prefix_audit_fraction: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    event_path, target_path, reference_path = paths(workload, pair)
    events = normalize_events(pd.read_parquet(event_path))
    targets = pd.read_parquet(target_path)
    if "pair" in targets.columns:
        targets = targets.loc[targets["pair"].astype(str).eq(pair)]
    target_by_position = target_lookup(targets)
    references = load_references(workload, pair, reference_path)
    prompt_ids = prompt_ids_by_id
    if max_prompts is not None:
        keep = set(sorted(prompt_ids)[:max_prompts])
        events = events.loc[events.prompt_id.astype(int).isin(keep)].copy()

    rows: list[dict[str, Any]] = []
    projection_audit: list[dict[str, Any]] = []
    groups = events.groupby(["prompt_id", "round_index"], sort=False, observed=True)
    n_invalidated = n_projection_rounds = n_projection_prefix_valid = n_projection_full_valid = 0
    n_projection_both_valid = 0
    n_event_target_mismatch = n_projection_class_disagreement = 0
    total_blocks = int(groups.ngroups)
    for block_number, ((raw_pid, raw_round), block) in enumerate(groups, start=1):
        if block_number % 2000 == 0:
            print(f"[{workload}/{pair}] processed {block_number}/{total_blocks} proposal blocks", flush=True)
        pid, rnd = int(raw_pid), int(raw_round)
        block = block.sort_values("proposal_slot")
        actual = block.loc[block["rejected"].notna()].copy()
        n_invalidated += len(block) - len(actual)
        if actual.empty:
            continue
        base = int(block.output_token_position.min())
        if any(int(r.output_token_position) != base + int(r.proposal_slot) - 1 for r in block.itertuples(index=False)):
            raise AssertionError(f"Non-contiguous event positions for {workload}/{pair}/{pid}/{rnd}")
        proposal_ids = [int(x) for x in block.draft_proposed_token_id]
        accepted_rows = actual.loc[~actual["rejected"].astype(bool)]
        rejected_rows = actual.loc[actual["rejected"].astype(bool)]
        if len(rejected_rows) > 1:
            raise AssertionError(f"Multiple first rejects in {workload}/{pair}/{pid}/{rnd}")
        reject_slot = int(rejected_rows.proposal_slot.iloc[0]) if len(rejected_rows) else None
        reject_projection_prefix = reject_projection_full = None
        if reject_slot is not None:
            n_projection_rounds += 1
            reject_event = rejected_rows.iloc[0]
            reject_position = int(reject_event.output_token_position)
            ref = references[pid]
            committed = ref[:base]
            sample_key = f"{workload}:{pair}:{pid}:{rnd}".encode("utf-8")
            sample_value = int(hashlib.sha256(sample_key).hexdigest()[:8], 16) / 0xFFFFFFFF
            do_prefix_sensitivity = sample_value < prefix_audit_fraction
            reject_projection_prefix, reject_projection_full = project_rejection(
                tokenizer, kiwi, prompt_ids[pid], committed, proposal_ids, reject_slot, pid, rnd, pair,
                do_prefix_sensitivity,
            )
            for record in [reject_projection_prefix, reject_projection_full]:
                if record and record.get("detector_status") == "VALID":
                    if record is reject_projection_prefix:
                        n_projection_prefix_valid += 1
                    else:
                        n_projection_full_valid += 1
            if reject_projection_prefix and reject_projection_full:
                if reject_projection_prefix.get("detector_status") == "VALID" and reject_projection_full.get("detector_status") == "VALID":
                    n_projection_both_valid += 1
                    if reject_projection_prefix.get("h2_morph_class") != reject_projection_full.get("h2_morph_class"):
                        n_projection_class_disagreement += 1
            projection_audit.append({
                "workload": workload, "pair": pair, "prompt_id": pid, "round_index": rnd,
                "output_token_position": reject_position, "proposal_slot": reject_slot,
                "draft_token_id": int(reject_event.draft_proposed_token_id),
                "target_token_id": int(reject_event.target_greedy_token_id),
                "prefix_candidate_surface": reject_projection_prefix.get("token_surface") if reject_projection_prefix else None,
                "prefix_class": reject_projection_prefix.get("h2_morph_class") if reject_projection_prefix else None,
                "prefix_status": reject_projection_prefix.get("detector_status") if reject_projection_prefix else "MISSING",
                "prefix_span_exact": reject_projection_prefix.get("span_exact") if reject_projection_prefix else None,
                "prefix_roundtrip_exact": reject_projection_prefix.get("alignment_roundtrip_exact") if reject_projection_prefix else None,
                "fullblock_candidate_surface": reject_projection_full.get("token_surface") if reject_projection_full else None,
                "fullblock_eojeol_surface": reject_projection_full.get("eojeol") if reject_projection_full else None,
                "fullblock_class": reject_projection_full.get("h2_morph_class") if reject_projection_full else None,
                "fullblock_status": reject_projection_full.get("detector_status") if reject_projection_full else "MISSING",
                "fullblock_span_exact": reject_projection_full.get("span_exact") if reject_projection_full else None,
                "fullblock_roundtrip_exact": reject_projection_full.get("alignment_roundtrip_exact") if reject_projection_full else None,
                "crossed_boundary_sequence": reject_projection_full.get("crossed_boundary_sequence") if reject_projection_full else None,
                "crossed_fine_pos_sequence": reject_projection_full.get("crossed_fine_pos_sequence") if reject_projection_full else None,
                "kiwi_sequence": json.dumps(reject_projection_full.get("kiwi_sequence"), ensure_ascii=False) if reject_projection_full else None,
                "projection_class_agrees": bool(
                    reject_projection_prefix and reject_projection_full
                    and reject_projection_prefix.get("detector_status") == "VALID"
                    and reject_projection_full.get("detector_status") == "VALID"
                    and reject_projection_prefix.get("h2_morph_class") == reject_projection_full.get("h2_morph_class")
                ),
            })

        for event in actual.itertuples(index=False):
            pos, slot = int(event.output_token_position), int(event.proposal_slot)
            target = target_by_position.get((pid, pos))
            if target is None:
                raise AssertionError(f"Missing target-side token row: {workload}/{pair}/{pid}/{pos}")
            draft_id = int(event.draft_proposed_token_id)
            target_id = int(target["token_id"])
            if pd.notna(getattr(event, "target_greedy_token_id", np.nan)) and int(event.target_greedy_token_id) != target_id:
                n_event_target_mismatch += 1
                raise AssertionError(f"Event target ID differs from saved target continuation: {workload}/{pair}/{pid}/{pos}")
            if bool(event.rejected):
                if draft_id == target_id:
                    raise AssertionError("Rejected proposal unexpectedly equals target token")
                # The drafter generated the whole block before target
                # verification. Use that proposal sequence to resolve local
                # morphology; suffix proposals after this slot are context
                # only and are never included as outcome rows.
                projected = reject_projection_full if slot == reject_slot else None
                candidate_class = projected.get("h2_morph_class") if projected else None
                candidate_status = projected.get("detector_status", "MISSING") if projected else "MISSING"
                candidate_surface = projected.get("token_surface") if projected else None
                candidate_char_len = (
                    int(projected["token_char_end"]) - int(projected["token_char_start"])
                    if projected and projected.get("detector_status") == "VALID" else np.nan
                )
                candidate_eojeol_len = len(str(projected.get("eojeol") or "")) if projected else np.nan
                projected_prefix = reject_projection_prefix if slot == reject_slot else None
                candidate_class_prefix = projected_prefix.get("h2_morph_class") if projected_prefix else None
                candidate_status_prefix = projected_prefix.get("detector_status", "MISSING") if projected_prefix else "MISSING"
                candidate_char_len_prefix = (
                    int(projected_prefix["token_char_end"]) - int(projected_prefix["token_char_start"])
                    if projected_prefix and projected_prefix.get("detector_status") == "VALID" else np.nan
                )
                candidate_eojeol_len_prefix = len(str(projected_prefix.get("eojeol") or "")) if projected_prefix else np.nan
            else:
                if draft_id != target_id:
                    raise AssertionError("Accepted proposal ID differs from target token")
                candidate_class = target.get("morph_class")
                candidate_status = "REUSED_EXACT_TARGET_LABEL"
                candidate_surface = target.get("token_text")
                candidate_char_len = target.get("token_char_length", np.nan)
                candidate_eojeol_len = target.get("eojeol_char_length", np.nan)
                candidate_class_prefix = candidate_class
                candidate_status_prefix = "REUSED_EXACT_TARGET_LABEL"
                candidate_char_len_prefix = candidate_char_len
                candidate_eojeol_len_prefix = candidate_eojeol_len

            if target.get("sd_valid") is not None and not bool(target.get("sd_valid")):
                raise AssertionError("Target-side table marks an observed proposal event invalid")
            rows.append({
                "workload": workload, "pair": pair, "prompt_id": pid, "round_index": rnd,
                "proposal_slot": slot, "generation_position": int(getattr(event, "generation_position", target.get("generation_position", pos))),
                "output_token_position": pos, "draft_token_id": draft_id, "target_token_id": target_id,
                "sd_rejected": int(bool(event.rejected)), "accepted": int(bool(event.accepted)),
                "candidate_morph_class": candidate_class, "candidate_projection_status": candidate_status,
                "candidate_morph_class_prefix": candidate_class_prefix,
                "candidate_projection_status_prefix": candidate_status_prefix,
                "candidate_token_surface": candidate_surface,
                "candidate_token_char_length": candidate_char_len,
                "candidate_eojeol_char_length": candidate_eojeol_len,
                "candidate_token_char_length_prefix": candidate_char_len_prefix,
                "candidate_eojeol_char_length_prefix": candidate_eojeol_len_prefix,
                "target_morph_class": target.get("morph_class"),
                "target_span_exact": target.get("span_exact"),
                "fragmentation_bin": target.get("fragmentation_bin"),
                "relative_position": target.get("relative_position"),
                "first_token": target.get("first_token"), "last_token": target.get("last_token"),
                "target_token_char_length": target.get("token_char_length"),
                "target_eojeol_char_length": target.get("eojeol_char_length"),
                "draft_entropy": target.get("draft_entropy"), "target_entropy": target.get("target_entropy"),
            })

    frame = pd.DataFrame(rows)
    frequencies = pd.read_parquet(FREQUENCY)[["token_id", "token_count", "log_token_count"]]
    frequencies = frequencies.rename(columns={"token_id": "draft_token_id", "token_count": "candidate_token_count", "log_token_count": "log_candidate_token_count"})
    frame = frame.merge(frequencies, on="draft_token_id", how="left", validate="many_to_one")
    target_frequencies = pd.read_parquet(FREQUENCY)[["token_id", "token_count", "log_token_count"]].rename(
        columns={"token_id": "target_token_id", "token_count": "target_token_count", "log_token_count": "log_target_token_count"}
    )
    frame = frame.merge(target_frequencies, on="target_token_id", how="left", validate="many_to_one")
    if frame[["log_candidate_token_count", "log_target_token_count"]].isna().any().any():
        raise AssertionError("A proposal or target token is absent from the pinned tokenizer frequency cache")
    audit = {
        "workload": workload, "pair": pair, "prompts": int(frame.prompt_id.nunique()),
        "trace_events_all": int(len(events)), "valid_proposals": int(len(frame)),
        "accepted_proposals": int(frame.accepted.sum()), "first_rejections": int(frame.sd_rejected.sum()),
        "invalidated_proposals_excluded": int(n_invalidated),
        "rejected_blocks_reprojected": int(n_projection_rounds),
        "prefix_projection_valid": int(n_projection_prefix_valid),
        "fullblock_projection_valid": int(n_projection_full_valid),
        "fullblock_local_span_valid_but_roundtrip_not_exact": int(sum(
            bool(r.get("fullblock_status") == "VALID" and r.get("fullblock_span_exact") and not r.get("fullblock_roundtrip_exact"))
            for r in projection_audit
        )),
        "prefix_and_fullblock_both_valid": int(n_projection_both_valid),
        "prefix_fullblock_class_disagreement": int(n_projection_class_disagreement),
        "prefix_fullblock_class_agreement_rate": (
            float(1 - n_projection_class_disagreement / n_projection_both_valid) if n_projection_both_valid else None
        ),
        "event_target_mismatches": int(n_event_target_mismatch),
        "candidate_class_counts": {str(k): int(v) for k, v in frame.candidate_morph_class.value_counts(dropna=False).items()},
        "candidate_projection_status_counts": {str(k): int(v) for k, v in frame.candidate_projection_status.value_counts(dropna=False).items()},
    }
    return frame, audit, projection_audit


def fit_models(frame: pd.DataFrame, include_pair_fixed_effect: bool = False, result_suffix: str = "") -> list[dict[str, Any]]:
    controls = [
        "C(fragmentation_bin, Treatment(reference='2'))", "relative_position_ctrl",
        "C(first_token_ctrl)", "C(last_token_ctrl)", "C(proposal_slot)", "generation_position",
        "candidate_token_char_length", "candidate_eojeol_char_length", "draft_entropy_ctrl", "target_entropy_ctrl",
        "bs(log_candidate_token_count, df=4, degree=3, include_intercept=False)",
    ]
    formulas = [
        ("M5_PROPOSAL_SIDE", "candidate_morph_class", "candidate_token_char_length", "candidate_eojeol_char_length", False),
        ("M5_PREFIX_CONTEXT_SENSITIVITY", "candidate_morph_class_prefix", "candidate_token_char_length_prefix", "candidate_eojeol_char_length_prefix", False),
        ("M6_TARGET_SIDE_ADJUSTED", "candidate_morph_class", "candidate_token_char_length", "candidate_eojeol_char_length", True),
    ]
    rows: list[dict[str, Any]] = []
    for base_name, class_column, token_length_column, eojeol_length_column, add_target_side in formulas:
        name = base_name + result_suffix
        controls_for_model = [
            control.replace("candidate_token_char_length", token_length_column)
            .replace("candidate_eojeol_char_length", eojeol_length_column)
            for control in controls
        ]
        formula = (
            f"sd_rejected ~ C({class_column}, Treatment(reference='WITHIN_SPLIT')) + "
            + ("C(pair) + " if include_pair_fixed_effect else "")
            + ("C(target_morph_class) + " if add_target_side else "")
            + " + ".join(controls_for_model)
            + (" + bs(log_target_token_count, df=4, degree=3, include_intercept=False)" if add_target_side else "")
        )
        clean = frame.copy()
        missing_terms: list[str] = []
        clean["relative_position_ctrl"] = pd.to_numeric(clean.relative_position, errors="coerce").fillna(-1.0)
        clean["first_token_ctrl"] = clean.first_token.fillna("UNKNOWN").astype(str)
        clean["last_token_ctrl"] = clean.last_token.fillna("UNKNOWN").astype(str)
        for col in ["draft_entropy", "target_entropy"]:
            clean[f"{col}_ctrl"] = pd.to_numeric(clean[col], errors="coerce")
            clean[f"{col}_ctrl_missing"] = clean[f"{col}_ctrl"].isna().astype(int)
            if clean[f"{col}_ctrl_missing"].nunique() > 1:
                missing_terms.append(f"{col}_ctrl_missing")
            median = clean[f"{col}_ctrl"].median()
            clean[f"{col}_ctrl"] = clean[f"{col}_ctrl"].fillna(median)
        # Include missingness indicators only when they vary; constant columns
        # make small-sample pilot designs singular.
        formula_with_missing = formula + ((" + " + " + ".join(missing_terms)) if missing_terms else "")
        eligible = clean.loc[
            clean[class_column].isin(PRIMARY_CLASSES)
            & clean.fragmentation_bin.astype(str).isin(FRAGMENTATION_ORDER)
        ].copy()
        eligible[class_column] = pd.Categorical(
            eligible[class_column], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"],
        )
        eligible["fragmentation_bin"] = pd.Categorical(
            eligible.fragmentation_bin.astype(str), categories=FRAGMENTATION_ORDER, ordered=True,
        )
        if include_pair_fixed_effect:
            eligible["pair"] = eligible.pair.astype(str)
        if add_target_side:
            eligible["target_morph_class"] = eligible.target_morph_class.fillna("UNKNOWN").astype(str)
        result = {
            "model": name, "n_rows": int(len(eligible)), "n_prompts": int(eligible.prompt_id.nunique()),
            "n_model_rows": None, "n_model_prompts": None,
            "cross_or": None, "or_ci_low": None, "or_ci_high": None, "p_value": None, "status": "not_estimable",
            "formula": formula_with_missing,
        }
        if len(eligible) < 100 or eligible.sd_rejected.nunique() < 2 or eligible[class_column].nunique() < 2:
            result["status"] = "not_estimable:insufficient_rows_or_classes"
            rows.append(result)
            continue
        try:
            model = smf.logit(formula_with_missing, data=eligible, missing="drop")
            retained = eligible.loc[model.data.row_labels]
            model_row_count = int(len(model.endog))
            if len(retained) != model_row_count:
                raise AssertionError(f"Formula rows ({model_row_count}) differ from prompt-cluster rows ({len(retained)})")
            result["n_model_rows"] = int(len(retained))
            result["n_model_prompts"] = int(retained.prompt_id.nunique())
            fitted = model.fit(
                disp=False, maxiter=200, cov_type="cluster",
                cov_kwds={"groups": retained.prompt_id, "use_correction": True},
            )
            converged = bool(fitted.mle_retvals.get("converged", True))
            result["converged"] = converged
            if not converged:
                # Statsmodels returns provisional coefficients and covariance
                # even when the optimizer has not converged. Keep the row
                # counts and flag, but do not publish those estimates as
                # inferential results.
                result["status"] = "fit:nonconverged_estimates_withheld"
                rows.append(result)
                continue
            term = next(
                key for key in fitted.params.index
                if class_column in key and "CROSS_MORPHEME" in key
            )
            low, high = fitted.conf_int().loc[term].tolist()
            result.update({
                "beta_cross": float(fitted.params[term]), "std_error": float(fitted.bse[term]),
                "cross_or": float(math.exp(fitted.params[term])),
                "or_ci_low": float(math.exp(low)), "or_ci_high": float(math.exp(high)),
                "p_value": float(fitted.pvalues[term]),
                "status": "fit",
            })
        except Exception as exc:
            result["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
        rows.append(result)
    return rows


def run(args: argparse.Namespace) -> None:
    if args.fit_existing:
        refit_existing(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_metadata = load_tokenizer()
    kiwi = Kiwi()
    combined: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    source_files: set[Path] = {FREQUENCY, FREQ_META_JSON}

    for workload in args.workloads:
        prompt_map = load_prompts(workload, tokenizer)
        for pair in args.pairs:
            print(f"Starting proposal-side analysis: {workload}/{pair}", flush=True)
            pair_dir = args.output_dir / "pairs" / f"{workload.lower()}_{pair.lower()}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            cached_rows = pair_dir / "proposal_side_rows.parquet"
            cached_audit = pair_dir / "population_and_alignment_audit.csv"
            cached_projection = pair_dir / "rejected_candidate_projection_audit.csv"
            if args.skip_existing_checkpoints and cached_rows.exists() and cached_audit.exists() and cached_projection.exists():
                frame = pd.read_parquet(cached_rows)
                audit = pd.read_csv(cached_audit).iloc[0].to_dict()
                projection_audit = pd.read_csv(cached_projection).to_dict(orient="records")
                print(f"Reusing saved morphology labels: {workload}/{pair}", flush=True)
            else:
                frame, audit, projection_audit = build_pair_rows(
                    workload, pair, tokenizer, kiwi, prompt_map, args.max_prompts, args.prefix_audit_fraction,
                )
            frame.to_parquet(pair_dir / "proposal_side_rows.parquet", index=False)
            pd.DataFrame([audit]).to_csv(pair_dir / "population_and_alignment_audit.csv", index=False)
            pd.DataFrame(projection_audit).to_csv(pair_dir / "rejected_candidate_projection_audit.csv", index=False)
            pair_models = pd.DataFrame([{**audit, **result} for result in fit_models(frame)])
            pair_models.to_csv(pair_dir / "adjusted_model_results.csv", index=False)
            combined.append(frame)
            audits.append(audit)
            projections.extend(projection_audit)
            model_rows.extend(pair_models.to_dict(orient="records"))
            source_files.update(paths(workload, pair))
            print(f"Finished {workload}/{pair}: {len(frame):,} valid proposals; {audit['first_rejections']:,} first rejects", flush=True)
    table = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    for workload in args.workloads:
        pooled = table.loc[table.workload.eq(workload) & table.pair.isin(["P1", "P2"])].copy()
        if pooled.pair.nunique() >= 2:
            pooled_audit = {
                "workload": workload, "pair": "P1+P2", "prompts": int(pooled.prompt_id.nunique()),
                "valid_proposals": int(len(pooled)), "accepted_proposals": int(pooled.accepted.sum()),
                "first_rejections": int(pooled.sd_rejected.sum()),
            }
            for result in fit_models(pooled, include_pair_fixed_effect=True, result_suffix="_POOLED_P1P2"):
                model_rows.append({**pooled_audit, **result})
    projections_frame = pd.DataFrame(projections)
    audit_frame = pd.DataFrame(audits)
    models_frame = pd.DataFrame(model_rows)
    table.to_parquet(args.output_dir / "proposal_side_rows.parquet", index=False)
    projections_frame.to_csv(args.output_dir / "rejected_candidate_projection_audit.csv", index=False)
    audit_frame.to_csv(args.output_dir / "population_and_alignment_audit.csv", index=False)
    models_frame.to_csv(args.output_dir / "adjusted_model_results.csv", index=False)
    metadata = {
        "purpose": "Proposal-side morphology validation for greedy speculative-decoding agreement",
        "gpu_used": False, "max_prompts_per_workload": args.max_prompts,
        "prefix_audit_fraction": args.prefix_audit_fraction,
        "workloads": args.workloads, "pairs": args.pairs,
        "primary_candidate_label_source": "accepted tokens reuse exact target label; first rejected candidate is reprojected in the complete draft proposal block",
        "invalidated_suffix_rule": "proposal rows after first rejection are excluded from all regression outcomes",
        "sensitivity_label_source": "strict projection of a deterministic sample of valid proposal prefixes through the first rejected candidate",
        "tokenizer": tokenizer_metadata,
        "kiwi_version": "0.24.0",
        "frequency_cache_metadata": json.loads(FREQ_META_JSON.read_text(encoding="utf-8")),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files if p.exists() and ROOT in p.parents},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Proposal-side morphology validation", "",
        "This offline reanalysis tests whether the morphology of the actual draft proposal predicts greedy draft–target disagreement. It does not rerun generation or use GPU compute.", "",
        "## Measurement", "",
        "Accepted proposals reuse target morphology only after exact draft/target token-ID equality is asserted. The first rejected proposal in each block is independently projected from the prompt, committed target prefix, and the complete draft proposal block. The candidate token must itself have an exact retokenized character span and exact Kiwi/H2-H3 alignment; whole-block round-trip failure elsewhere does not invalidate that candidate span. Proposals after the first reject are used only as local right context for morphology and are excluded as observations. A deterministic sample of strict-prefix projections through the rejected candidate is saved as a context sensitivity.", "",
        "## Statistical test", "",
        "The outcome is `sd_rejected` (1 for the first target-verified mismatch, 0 for accepted proposals). Pairwise `M5_PROPOSAL_SIDE` models estimate the CROSS_MORPHEME versus WITHIN_SPLIT odds ratio while controlling for target-continuation fragmentation bin, relative position, first/last token flags, proposal slot, generation position, candidate token and eojeol character lengths, draft/target entropy, and a four-df spline of the candidate token's Wikipedia frequency. Standard errors are clustered by prompt. `M5_PREFIX_CONTEXT_SENSITIVITY` repeats the model using strict-prefix labels. The pooled `M5_PROPOSAL_SIDE_POOLED_P1P2` adds a pair fixed effect and uses prompt-clustered errors. `M6_TARGET_SIDE_ADJUSTED` additionally controls for the target token's morphology class and frequency; when optimization does not converge, its coefficient estimates are withheld.", "",
        "## Results", "",
        markdown_table(audit_frame) if not audit_frame.empty else "No data.", "",
        markdown_table(models_frame[[c for c in ["workload", "pair", "model", "n_rows", "n_prompts", "cross_or", "or_ci_low", "or_ci_high", "p_value", "converged", "status"] if c in models_frame.columns]]) if not models_frame.empty else "No fitted model.", "",
        "## Interpretation limit", "",
        "The main estimand is a conditional association for the saved deterministic greedy runs. A positive adjusted odds ratio does not establish that morphology causes rejection. Wiki P3 uses the historical saved model run with an unresolved draft checkpoint revision and is treated as a sensitivity result; P1/P2 are the primary Wiki replications. In primary P1/P2, the proposal-side projection is valid for about 87–89% of first-rejected candidates; among blocks with both full-block and strict-prefix labels valid, those labels agree about 74–78%. The projection audit is computational; no Korean expert has yet adjudicated the labels. Thus context sensitivity and exclusion of ambiguous projections matter, and the operational labeling context should be validated with the prepared blinded annotation sheet before describing the measure as human-validated.", "",
    ]
    (args.output_dir / "proposal_side_validation.md").write_text("\n".join(lines), encoding="utf-8")


def refit_existing(args: argparse.Namespace) -> None:
    """Refit models from pair checkpoints without reprojecting morphology."""
    pair_frames: dict[tuple[str, str], pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for workload in args.workloads:
        for pair in args.pairs:
            pair_dir = args.output_dir / "pairs" / f"{workload.lower()}_{pair.lower()}"
            frame_path = pair_dir / "proposal_side_rows.parquet"
            audit_path = pair_dir / "population_and_alignment_audit.csv"
            if not frame_path.exists() or not audit_path.exists():
                raise FileNotFoundError(f"Missing completed pair checkpoint: {frame_path}")
            frame = pd.read_parquet(frame_path)
            audit = pd.read_csv(audit_path).iloc[0].to_dict()
            pair_frames[(workload, pair)] = frame
            audits.append(audit)
            pair_results = pd.DataFrame([{**audit, **result} for result in fit_models(frame)])
            pair_results.to_csv(pair_dir / "adjusted_model_results.csv", index=False)
            model_rows.extend(pair_results.to_dict(orient="records"))
            print(f"Refit {workload}/{pair} from saved proposal labels", flush=True)

    all_frames = list(pair_frames.values())
    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_parquet(args.output_dir / "proposal_side_rows.parquet", index=False)
    audit_frame = pd.DataFrame(audits)
    for workload in args.workloads:
        pooled_parts = [pair_frames[(workload, pair)] for pair in ["P1", "P2"] if (workload, pair) in pair_frames]
        pooled = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()
        if not pooled.empty and pooled.pair.nunique() >= 2:
            pooled_audit = {
                "workload": workload, "pair": "P1+P2", "prompts": int(pooled.prompt_id.nunique()),
                "valid_proposals": int(len(pooled)), "accepted_proposals": int(pooled.accepted.sum()),
                "first_rejections": int(pooled.sd_rejected.sum()),
            }
            model_rows.extend({**pooled_audit, **result} for result in fit_models(
                pooled, include_pair_fixed_effect=True, result_suffix="_POOLED_P1P2",
            ))
    models_frame = pd.DataFrame(model_rows)
    models_frame.to_csv(args.output_dir / "adjusted_model_results.csv", index=False)
    projection_paths = [
        args.output_dir / "pairs" / f"{w.lower()}_{p.lower()}" / "rejected_candidate_projection_audit.csv"
        for w in args.workloads for p in args.pairs
    ]
    projection_frames = [pd.read_csv(p) for p in projection_paths if p.exists()]
    if projection_frames:
        pd.concat(projection_frames, ignore_index=True).to_csv(
            args.output_dir / "rejected_candidate_projection_audit.csv", index=False,
        )
    audit_frame.to_csv(args.output_dir / "population_and_alignment_audit.csv", index=False)
    lines = [
        "# Proposal-side morphology validation", "",
        "This offline reanalysis tests whether the morphology of the actual draft proposal predicts greedy draft–target disagreement. It does not rerun generation or use GPU compute.", "",
        "## Measurement", "",
        "Accepted proposals reuse target morphology only after exact draft/target token-ID equality is asserted. The first rejected proposal in each block is independently projected from the prompt, committed target prefix, and the complete draft proposal block. The candidate token must itself have an exact local tokenizer span and exact Kiwi/H2-H3 alignment. Invalidated suffix proposals are never outcome rows. A deterministic strict-prefix projection is saved as a context sensitivity.", "",
        "## Statistical test", "",
        "Pairwise M5 estimates the proposal CROSS_MORPHEME versus WITHIN_SPLIT odds ratio, controlling for target-continuation fragmentation, position, token lengths, draft/target entropy, and a four-df spline of Wikipedia candidate-token frequency. Standard errors are clustered by prompt. The pooled P1+P2 model adds pair fixed effects. The prefix-context model and target-side-adjusted model are sensitivity analyses; target-side-adjusted estimates are withheld if optimization does not converge.", "",
        "## Population and alignment", "", markdown_table(audit_frame), "",
        "## Adjusted results", "",
        markdown_table(models_frame[[c for c in ["workload", "pair", "model", "n_rows", "n_model_rows", "n_model_prompts", "cross_or", "or_ci_low", "or_ci_high", "p_value", "converged", "status"] if c in models_frame.columns]]), "",
        "## Decision rule", "",
        "Confirm only if the pooled P1+P2 proposal-side OR is greater than 1 with its prompt-clustered 95% CI excluding 1 in both Wiki and FLORES, and pair-specific P1/P2 point estimates are positive in both datasets. P3 Wikipedia remains a sensitivity result because its historical draft revision is unresolved.", "",
        "## Interpretation limit", "",
        "These are conditional associations in deterministic greedy decoding. They do not establish a causal mechanism, cross-lingual generality, or a speedup policy. In primary P1/P2, the proposal-side projection is valid for about 87–89% of first-rejected candidates; among blocks with both full-block and strict-prefix labels valid, those labels agree about 74–78%. The projection audit is computational; no Korean expert has yet adjudicated the labels. Thus context sensitivity and exclusion of ambiguous projections matter, and the operational labeling context should be stated explicitly and validated with the prepared blinded annotation sheet before calling the measure human-validated.", "",
    ]
    (args.output_dir / "proposal_side_validation.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=["WIKIPEDIA", "FLORES"], default=["WIKIPEDIA", "FLORES"])
    parser.add_argument("--pairs", nargs="+", choices=["P1", "P2", "P3"], default=["P1", "P2", "P3"])
    parser.add_argument("--max-prompts", type=int, default=None, help="Use the first N prompt IDs per workload for pipeline validation")
    parser.add_argument("--prefix-audit-fraction", type=float, default=1.0, help="Deterministic share of rejected blocks with strict-prefix sensitivity projection")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/proposal_side_validation")
    parser.add_argument("--fit-existing", action="store_true", help="Refit from completed pair checkpoints without reprojecting morphology")
    parser.add_argument("--skip-existing-checkpoints", action="store_true", help="Reuse completed pair label checkpoints while processing missing pairs")
    return parser.parse_args()


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


if __name__ == "__main__":
    run(parse_args())
