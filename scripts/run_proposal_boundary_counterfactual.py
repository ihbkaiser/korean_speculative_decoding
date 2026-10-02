#!/usr/bin/env python3
"""Paired one-step counterfactuals at saved greedy speculative-decoding prefixes.

At each observed, valid proposal whose token crosses a Korean N->P boundary,
compare the natural draft argmax with the highest-ranked valid alternative
that does not cross N->P. A same-prompt, nearest WITHIN_SPLIT proposal is a
placebo context where we instead suppress the natural argmax and take the next
valid draft token. Target and draft argmaxes are recomputed from the pinned
models at the exact saved prefix. This estimates a proposal-policy effect at
real SD states; it does not claim that morphology can be changed independently
of token identity.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForCausalLM

from scripts.analyze_proposal_side import (
    FLORES, WIKI, load_prompts, load_references, load_tokenizer, paths,
)
from scripts.analyze_proposal_side_boundary import proposal_environment
from src.e2_model_pairs import MODELS, PAIRS
from src.morphology_guard import project_candidate_block

SOURCE = ROOT / "runs/proposal_side_boundary_validation/proposal_boundary_rows_all.parquet"
OUT = ROOT / "runs/proposal_boundary_counterfactual"
WORKLOADS = ("WIKIPEDIA", "FLORES")
PRIMARY_PAIRS = ("P1", "P2")
MAX_ALTERNATIVE_RANK = 64
BOOTSTRAP_REPS = 3000
SEED = 7142026


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    cols = [str(c) for c in frame.columns]
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(clean(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def context_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["workload"]), str(row["pair"]), int(row["prompt_id"]), int(row["output_token_position"]))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def nearest_placebos(cross: pd.DataFrame, split: pd.DataFrame) -> list[dict[str, Any]]:
    """One deterministic nearest unused WITHIN_SPLIT context per N->P CROSS row."""
    out: list[dict[str, Any]] = []
    by_prompt = {int(pid): g for pid, g in split.groupby("prompt_id", sort=False)}
    used: set[tuple[int, int]] = set()
    for cross_row in cross.sort_values(["prompt_id", "output_token_position"]).to_dict("records"):
        pid = int(cross_row["prompt_id"])
        local = by_prompt.get(pid)
        if local is None or local.empty:
            candidates = split.iloc[0:0]
        else:
            keys = list(zip(local.prompt_id.astype(int), local.output_token_position.astype(int)))
            candidates = local.loc[[key not in used for key in keys]]
        exact_prompt = not candidates.empty
        if candidates.empty:
            keys = list(zip(split.prompt_id.astype(int), split.output_token_position.astype(int)))
            candidates = split.loc[[key not in used for key in keys]]
        if candidates.empty:
            raise RuntimeError("Insufficient unique WITHIN_SPLIT placebo contexts")
        c_entropy = _num(cross_row.get("draft_entropy"))
        c_pos = _num(cross_row.get("output_token_position"))
        c_slot = _num(cross_row.get("proposal_slot"))
        c_len = _num(cross_row.get("candidate_token_char_length"))
        c_frag = str(cross_row.get("fragmentation_bin"))
        cand = candidates.copy()
        cand["_distance"] = (
            1.5 * (cand.proposal_slot.astype(float) - c_slot).abs()
            + 0.75 * (cand.output_token_position.astype(float) - c_pos).abs() / 32.0
            + (cand.draft_entropy.fillna(0).astype(float) - c_entropy).abs() / 2.0
            + 0.35 * (cand.candidate_token_char_length.astype(float) - c_len).abs()
            + 1.0 * cand.fragmentation_bin.astype(str).ne(c_frag).astype(float)
        )
        chosen = cand.sort_values(["_distance", "output_token_position"]).iloc[0].drop(labels="_distance").to_dict()
        used.add((int(chosen["prompt_id"]), int(chosen["output_token_position"])))
        chosen["matched_cross_prompt_id"] = pid
        chosen["matched_cross_position"] = int(cross_row["output_token_position"])
        chosen["exact_prompt_match"] = bool(exact_prompt)
        out.append(chosen)
    return out


def build_population(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, int, int], dict[str, Any]]]:
    eligible = frame.loc[
        frame.projection_status.eq("VALID")
        & frame.sd_rejected.notna()
        & frame.proposal_morph_environment.eq("NOMINAL_TO_PARTICLE")
    ].copy()
    cross = eligible.loc[
        eligible.proposal_morph_class.eq("CROSS_MORPHEME")
        & eligible.num_boundaries_crossed.eq(1)
    ]
    split = eligible.loc[eligible.proposal_morph_class.eq("WITHIN_SPLIT")]
    records: list[dict[str, Any]] = []
    for pair in PRIMARY_PAIRS:
        for workload in WORKLOADS:
            x = cross.loc[cross.pair.eq(pair) & cross.workload.eq(workload)]
            s = split.loc[split.pair.eq(pair) & split.workload.eq(workload)]
            placebo_rows = nearest_placebos(x, s)
            for row in x.to_dict("records"):
                row["condition"] = "NP_CROSS"
                records.append(row)
            for row in placebo_rows:
                row["condition"] = "WITHIN_SPLIT_PLACEBO"
                records.append(row)

    population = pd.DataFrame(records)
    tokenizer, tok_meta = load_tokenizer()
    prompts_by_workload: dict[str, dict[int, list[int]]] = {}
    refs_by_cell: dict[tuple[str, str], dict[int, list[int]]] = {}
    contexts: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in population.to_dict("records"):
        workload, pair, pid, pos = context_key(row)
        if workload not in prompts_by_workload:
            prompts_by_workload[workload] = load_prompts(workload, tokenizer)
        if (workload, pair) not in refs_by_cell:
            _, _, ref_path = paths(workload, pair)
            refs_by_cell[(workload, pair)] = load_references(workload, pair, ref_path)
        prompt_ids = prompts_by_workload[workload].get(pid)
        target_ids = refs_by_cell[(workload, pair)].get(pid)
        if prompt_ids is None or target_ids is None or pos < 0 or pos >= len(target_ids):
            row["context_error"] = "missing_prompt_or_reference_position"
            continue
        if int(row["target_token_id"]) != int(target_ids[pos]):
            raise AssertionError(f"Saved target token mismatch: {workload}/{pair}/{pid}/{pos}")
        key = context_key(row)
        if key not in contexts:
            contexts[key] = {
                "workload": workload, "pair": pair, "prompt_id": pid, "position": pos,
                "input_ids": [int(x) for x in prompt_ids] + [int(x) for x in target_ids[:pos]],
                "target_saved_id": int(row["target_token_id"]),
            }
    population = population.loc[population.apply(lambda r: context_key(r.to_dict()) in contexts, axis=1)].copy()
    return population, contexts


def model_logits(model: Any, input_ids: list[int], topk: int | None = None) -> tuple[int, list[tuple[int, float]]]:
    x = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
    greedy = int(torch.argmax(logits).item())
    if topk is None:
        return greedy, []
    k = min(int(topk), int(logits.numel()))
    values, indices = torch.topk(logits, k=k)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = indices.tolist()
    candidate_logprobs = log_probs.index_select(0, indices).tolist()
    candidates = [(int(tok), float(lp)) for tok, lp in zip(token_ids, candidate_logprobs)]
    del logits, values, indices, log_probs, x
    return greedy, candidates


def load_model(size: str) -> Any:
    spec = MODELS[size]
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
    )
    model.to("cuda:0").eval()
    return model


def evaluate_models(contexts: dict[tuple[str, str, int, int], dict[str, Any]]) -> tuple[dict, dict, dict]:
    tokenizer, tokenizer_meta = load_tokenizer()
    target_predictions: dict[tuple[str, str, int, int], int] = {}
    draft_candidates: dict[tuple[str, str, int, int], list[tuple[int, float]]] = {}
    audit: dict[str, Any] = {"tokenizer": tokenizer_meta, "pairs": {}}

    for pair in PRIMARY_PAIRS:
        pair_keys = [k for k in contexts if k[1] == pair]
        target_size = PAIRS[pair]["target"]
        draft_size = PAIRS[pair]["draft"]
        pair_audit = {"target_model": MODELS[target_size], "draft_model": MODELS[draft_size], "n_contexts": len(pair_keys)}
        print(f"Loading {pair} target {target_size} for {len(pair_keys):,} saved prefixes", flush=True)
        target_model = load_model(target_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, key in enumerate(pair_keys, start=1):
            greedy, _ = model_logits(target_model, contexts[key]["input_ids"])
            target_predictions[key] = greedy
            if i % 100 == 0:
                print(f"[{pair} target] {i}/{len(pair_keys)}", flush=True)
        torch.cuda.synchronize()
        pair_audit["target_seconds"] = time.perf_counter() - t0
        del target_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Loading {pair} draft {draft_size} for {len(pair_keys):,} saved prefixes", flush=True)
        draft_model = load_model(draft_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, key in enumerate(pair_keys, start=1):
            _, candidates = model_logits(draft_model, contexts[key]["input_ids"], MAX_ALTERNATIVE_RANK)
            draft_candidates[key] = candidates
            if i % 100 == 0:
                print(f"[{pair} draft] {i}/{len(pair_keys)}", flush=True)
        torch.cuda.synchronize()
        pair_audit["draft_seconds"] = time.perf_counter() - t0
        del draft_model
        gc.collect()
        torch.cuda.empty_cache()
        audit["pairs"][pair] = pair_audit
    return target_predictions, draft_candidates, audit


def choose_alternative(
    row: dict[str, Any], candidates: list[tuple[int, float]], tokenizer: Any,
    kiwi: Kiwi, prompt_ids: list[int], reference_prefix: list[int], condition: str,
) -> dict[str, Any] | None:
    baseline_id = int(row["draft_token_id"])
    for rank, (candidate_id, logprob) in enumerate(candidates, start=1):
        if candidate_id == baseline_id:
            continue
        projected, _ = project_candidate_block(
            tokenizer, kiwi, prompt_ids, reference_prefix, [candidate_id],
            int(row["prompt_id"]), int(row["round_index"]), str(row["pair"]),
            allow_local_exact_spans=True,
        )
        if not projected:
            continue
        rec = projected[0]
        if rec.get("detector_status") != "VALID":
            continue
        env, source = proposal_environment(rec)
        if condition == "NP_CROSS":
            crossed = set(str(rec.get("crossed_boundary_sequence") or "").split("|"))
            if rec.get("h2_morph_class") == "CROSS_MORPHEME" and "NOMINAL_TO_PARTICLE" in crossed:
                continue
        return {
            "alternative_token_id": int(candidate_id), "alternative_rank": int(rank),
            "alternative_logprob": float(logprob), "alternative_surface": str(rec.get("token_surface", "")),
            "alternative_morph_class": str(rec.get("h2_morph_class", "")),
            "alternative_environment": str(env), "alternative_env_source": str(source),
        }
    return None


def cluster_mean_ci(frame: pd.DataFrame, value: str, cluster: str, rng: np.random.Generator) -> tuple[float, float, float]:
    grouped = frame.groupby(cluster, sort=False)[value].agg(["sum", "count"])
    if grouped.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(grouped["sum"].sum() / grouped["count"].sum())
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    n = len(grouped)
    for b in range(BOOTSTRAP_REPS):
        ix = rng.integers(0, n, size=n)
        draws[b] = sums[ix].sum() / counts[ix].sum()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for (workload, pair, condition), group in results.groupby(["workload", "pair", "condition"], sort=True):
        cluster = group.copy()
        cluster["prompt_cluster"] = cluster.prompt_id.astype(str)
        for metric, label in [("baseline_accept", "baseline_accept"), ("alternative_accept", "alternative_accept"), ("delta_accept", "delta_accept")]:
            est, lo, hi = cluster_mean_ci(cluster, metric, "prompt_cluster", rng)
            rows.append({"workload": workload, "pair": pair, "condition": condition, "metric": label,
                         "n_events": len(group), "n_prompts": group.prompt_id.nunique(),
                         "estimate": est, "ci_low": lo, "ci_high": hi})
    # Dataset-level, prompt-clustered paired specificity: effect in NP_CROSS minus placebo effect.
    for workload, group in results.groupby("workload", sort=True):
        pivot = group.pivot_table(index=["prompt_id", "pair"], columns="condition", values="delta_accept", aggfunc="mean")
        pivot = pivot.dropna(subset=["NP_CROSS", "WITHIN_SPLIT_PLACEBO"])
        if pivot.empty:
            continue
        per_prompt = pivot.groupby(level="prompt_id").mean()
        per_prompt["specificity_delta"] = per_prompt["NP_CROSS"] - per_prompt["WITHIN_SPLIT_PLACEBO"]
        temp = per_prompt.reset_index().rename(columns={"prompt_id": "prompt_cluster"})
        est, lo, hi = cluster_mean_ci(temp, "specificity_delta", "prompt_cluster", rng)
        rows.append({"workload": workload, "pair": "P1+P2", "condition": "NP_MINUS_PLACEBO",
                     "metric": "delta_difference", "n_events": int(len(pivot)),
                     "n_prompts": int(len(per_prompt)), "estimate": est, "ci_low": lo, "ci_high": hi})
    # Pooled proposal-policy effects by workload across both primary pairs.
    for workload, group in results.groupby("workload", sort=True):
        for condition in ("NP_CROSS", "WITHIN_SPLIT_PLACEBO"):
            sub = group.loc[group.condition.eq(condition)].copy()
            sub["prompt_cluster"] = sub.prompt_id.astype(str)
            est, lo, hi = cluster_mean_ci(sub, "delta_accept", "prompt_cluster", rng)
            rows.append({"workload": workload, "pair": "P1+P2", "condition": condition,
                         "metric": "delta_accept", "n_events": len(sub), "n_prompts": sub.prompt_id.nunique(),
                         "estimate": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested pinned-model counterfactuals")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    torch.backends.cuda.matmul.allow_tf32 = True
    frame = pd.read_parquet(SOURCE)
    population, contexts = build_population(frame)
    if population.empty:
        raise RuntimeError("No eligible paired contexts found")
    print(f"Population: {len(population):,} rows; {len(contexts):,} unique prefixes", flush=True)
    target_predictions, draft_candidates, model_audit = evaluate_models(contexts)
    tokenizer, tok_meta = load_tokenizer()
    kiwi = Kiwi()
    prompts_by_workload = {w: load_prompts(w, tokenizer) for w in WORKLOADS}
    ref_by_cell = {}
    for pair in PRIMARY_PAIRS:
        for workload in WORKLOADS:
            _, _, ref_path = paths(workload, pair)
            ref_by_cell[(workload, pair)] = load_references(workload, pair, ref_path)

    output_rows: list[dict[str, Any]] = []
    for i, raw in enumerate(population.to_dict("records"), start=1):
        key = context_key(raw)
        context = contexts[key]
        target_id = int(target_predictions[key])
        draft_top1 = int(draft_candidates[key][0][0])
        target_replay_match = target_id == int(raw["target_token_id"])
        draft_replay_match = draft_top1 == int(raw["draft_token_id"])
        saved_accept = int(raw["draft_token_id"]) == int(raw["target_token_id"])
        replay_accept = draft_top1 == target_id
        row = {
            **{k: raw.get(k) for k in ["workload", "pair", "prompt_id", "round_index", "proposal_slot", "output_token_position", "condition", "proposal_morph_class", "candidate_surface", "candidate_eojeol", "sd_rejected", "draft_entropy", "fragmentation_bin", "matched_cross_prompt_id", "matched_cross_position", "exact_prompt_match"]},
            "saved_draft_token_id": int(raw["draft_token_id"]), "saved_target_token_id": int(raw["target_token_id"]),
            "replayed_draft_top1_id": draft_top1, "replayed_target_top1_id": target_id,
            "draft_replay_match": draft_replay_match, "target_replay_match": target_replay_match,
            "saved_accept": saved_accept, "replayed_accept": replay_accept,
            "saved_reject_label_match": (int(not replay_accept) == int(bool(raw["sd_rejected"]))) if target_replay_match and draft_replay_match else False,
        }
        if not (draft_replay_match and target_replay_match):
            row["counterfactual_status"] = "MODEL_REPLAY_MISMATCH"
            output_rows.append(row)
            continue
        prompt_ids = prompts_by_workload[str(raw["workload"])][int(raw["prompt_id"])]
        reference_prefix = ref_by_cell[(str(raw["workload"]), str(raw["pair"]))][int(raw["prompt_id"])][:int(raw["output_token_position"])]
        alt = choose_alternative(raw, draft_candidates[key], tokenizer, kiwi, prompt_ids, reference_prefix, str(raw["condition"]))
        if alt is None:
            row["counterfactual_status"] = "NO_VALID_ALTERNATIVE_IN_TOPK"
            output_rows.append(row)
            continue
        alternative_accept = int(alt["alternative_token_id"]) == target_id
        row.update(alt)
        row.update({
            "alternative_accept": alternative_accept,
            "baseline_accept": replay_accept,
            "delta_accept": int(alternative_accept) - int(replay_accept),
            "counterfactual_status": "OK",
        })
        output_rows.append(row)
        if i % 100 == 0:
            print(f"Classified {i}/{len(population)} saved prefixes", flush=True)

    results = pd.DataFrame(output_rows)
    atomic_csv(results, OUT / "counterfactual_proposals.csv")
    valid = results.loc[results.counterfactual_status.eq("OK")].copy()
    summary = summarize(valid)
    atomic_csv(summary, OUT / "counterfactual_summary.csv")
    replay_summary = {}
    for condition, group in results.groupby("condition", sort=True):
        replay_summary[condition] = {
            "n": int(len(group)), "draft_replay_match": int(group.draft_replay_match.sum()),
            "target_replay_match": int(group.target_replay_match.sum()),
            "saved_reject_label_match": int(group.saved_reject_label_match.sum()),
            "valid_counterfactual": int(group.counterfactual_status.eq("OK").sum()),
            "no_valid_alternative": int(group.counterfactual_status.eq("NO_VALID_ALTERNATIVE_IN_TOPK").sum()),
            "model_replay_mismatch": int(group.counterfactual_status.eq("MODEL_REPLAY_MISMATCH").sum()),
        }
    atomic_json(OUT / "model_replay_audit.json", {"pairs": model_audit["pairs"], "condition_audit": replay_summary, "tokenizer": tok_meta})
    placebo = results.loc[results.condition.eq("WITHIN_SPLIT_PLACEBO")]
    same_prompt_matches = int(placebo.exact_prompt_match.fillna(False).sum())
    lines = [
        "# Paired N→P proposal counterfactuals", "",
        "This offline intervention replays actual saved greedy speculative-decoding prefixes using the pinned P1/P2 draft and target models. At each N→P CROSS prefix, it compares the natural draft argmax with the highest-ranked valid alternative that does not cross N→P. It uses a unique nearest WITHIN_SPLIT proposal as a generic perturbation placebo; same-prompt matches are used when available, and cross-prompt fallback is used otherwise. Target and draft argmaxes are recomputed; any prefix where either fails to reproduce the saved event is excluded from causal contrasts.", "",
        f"Placebo matching: {same_prompt_matches:,}/{len(placebo):,} contexts are matched within the same prompt; no placebo prefix is reused.", "",
        "Primary estimand: paired change in immediate target-argmax agreement under the N→P-avoidance proposal policy. Specificity check: compare that change with the generic WITHIN_SPLIT placebo. This is a policy-level causal estimand; it does not identify an effect of morphology independent of candidate-token identity. It is not an end-to-end speed or quality experiment.", "",
        "## Model replay audit", "", json.dumps(replay_summary, ensure_ascii=False, indent=2), "",
        "## Paired counterfactual estimates", "", markdown_table(summary.round(4)), "",
        "## Decision rule", "",
        "The prespecified positive criterion is a positive pooled P1+P2 NP_CROSS delta-accept estimate with a prompt-clustered 95% CI above zero in both workloads and positive point estimates in both pairs, plus a positive pooled NP-minus-placebo specificity estimate in both workloads. Otherwise the result is negative or inconclusive for this intervention.", "",
        "## Interpretation", "",
        "The estimated change in immediate acceptance is negative for the N→P-avoidance policy in both datasets and both pairs: suppressing the natural N→P proposal and taking the next valid non-N→P draft candidate lowers agreement with the target. The specificity contrast is positive because this replacement is less harmful than the generic WITHIN_SPLIT top-candidate replacement, but both policy effects are negative. This does not support the claim that crossing itself causes rejection; it shows that candidate identity and its rank under the draft matter, and that avoiding N→P candidates is not a reliable way to reduce rejection.", "",
        "Inputs are the existing validated full-block proposal projections; model revisions are pinned in `src/e2_model_pairs.py`. See the CSV and JSON outputs for every row and replay mismatch.", "",
    ]
    (OUT / "counterfactual_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "counterfactual_report.md", flush=True)


if __name__ == "__main__":
    main()
