#!/usr/bin/env python3
"""Measure exact-greedy adaptive SD wall-clock performance on frozen test docs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from ksdlib import load_pair
from policysim import doc_starts, simulate
from src.naacl_experiment import atomic_parquet, load_config, load_model, load_tokenizer
from src.rebuttal.adaptive_sd import (
    EntropyStop,
    FixedK,
    MarginStop,
    Top1Stop,
    greedy_target_only,
    speculative_greedy_adaptive,
)
from src.rebuttal.static_compiled_sd import (
    clear_compiled_models,
    greedy_target_only_static,
    speculative_greedy_static,
)
from src.rebuttal.runtime import write_gpu_metadata


SEED = 20261125
MAX_NEW_TOKENS = 128


def _continuations(target: str, split: str) -> pd.DataFrame:
    directory = ROOT / "results/naacl_experiments/continuations" / target / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No continuations found under {directory}")
    return pd.concat(
        [pd.read_parquet(path, columns=["doc_id", "pool_index", "prompt_ids", "continuation_ids", "eos_hit"])
         for path in files],
        ignore_index=True,
    )


def _select_valid_docs(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[
        frame.pool_index.map(int).lt(20000)
        & frame.pool_index.map(int).mod(2).eq(1)
        & frame.continuation_ids.map(len).eq(MAX_NEW_TOKENS)
        & ~frame.eos_hit.astype(bool)
        & frame.prompt_ids.map(len).ge(1)
    ]
    return eligible.sort_values("pool_index", kind="stable").reset_index(drop=True)


def _selection_dev(frame: pd.DataFrame) -> set[int]:
    eligible = frame[
        frame.pool_index.map(int).lt(20000)
        & frame.pool_index.map(int).mod(2).eq(0)
        & frame.continuation_ids.map(len).eq(MAX_NEW_TOKENS)
        & ~frame.eos_hit.astype(bool)
    ]
    return set(eligible.pool_index.astype(int).tolist())


def _score_replay(
    data: pd.DataFrame,
    doc_ids: set[int],
    stop: np.ndarray,
    kmax: int,
    cost: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    part = data[data.doc.isin(doc_ids)].copy()
    if part.empty:
        raise RuntimeError("Replay selected no teacher-forced positions")
    starts = doc_starts(part.doc.to_numpy())
    match = (part.d_arg.to_numpy() == part.y.to_numpy())
    calls, drafted, _ = simulate(match, stop[part.index.to_numpy()], starts, kmax)
    token_count = len(part)
    call_count = int(calls.sum())
    l_value = token_count / call_count
    d_value = float(drafted.sum()) / call_count
    return l_value / (1.0 + cost * d_value), l_value, d_value, calls, drafted


def _choose_threshold(
    data: pd.DataFrame,
    dev_ids: set[int],
    cost: float,
    field: str,
    signal: np.ndarray,
    stop_low: bool,
    kmax: int = 16,
) -> tuple[float, float]:
    dev = data[data.doc.isin(dev_ids)]
    values = signal[dev.index.to_numpy()]
    quantiles = np.unique(np.quantile(values, np.linspace(0.005, 0.9, 80)))
    best_score = -np.inf
    best_threshold = float(quantiles[0])
    for threshold in quantiles:
        stop = signal < threshold if stop_low else signal > threshold
        score, _, _, _, _ = _score_replay(data, dev_ids, stop, kmax, cost)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    print(f"TUNED {field} threshold={best_threshold:.9g} replay_score={best_score:.6f}", flush=True)
    return best_threshold, best_score


def _fixed_optimum(data: pd.DataFrame, dev_ids: set[int], cost: float) -> tuple[int, float]:
    all_stop = np.zeros(len(data), dtype=bool)
    scores = []
    for k in range(1, 17):
        score, _, _, _, _ = _score_replay(data, dev_ids, all_stop, k, cost)
        scores.append((score, k))
    score, k = max(scores)
    print(f"TUNED fixed_k={k} replay_score={score:.6f}", flush=True)
    return k, float(score)


def _make_policies(data: pd.DataFrame, dev_ids: set[int], cost: float, skipped: set[int],
                   include_k1: bool = False):
    margin, _ = _choose_threshold(data, dev_ids, cost, "margin", data.mD.to_numpy(), True)
    entropy, _ = _choose_threshold(data, dev_ids, cost, "entropy", data.HD.to_numpy(), False)
    top1_signal = np.exp(data.d_lp1.to_numpy())
    top1, _ = _choose_threshold(data, dev_ids, cost, "top1_probability", top1_signal, True)
    k_opt, _ = _fixed_optimum(data, dev_ids, cost)
    policies: dict[str, tuple[Any, float | int | None]] = {"target_only": (None, None)}
    fixed_ks = ([1, 2, 4, 6, 8] if include_k1 else [2, 4, 6, 8])
    if k_opt not in fixed_ks:
        fixed_ks.append(k_opt)
    for k in fixed_ks:
        if k not in skipped:
            policies[f"fixed_k_{k}"] = (FixedK(k), k)
    policies["entropy_tuned"] = (EntropyStop(entropy), entropy)
    policies["top1_tuned"] = (Top1Stop(top1), top1)
    policies["margin_tuned"] = (MarginStop(margin), margin)
    policies["margin_1nat"] = (MarginStop(1.0), 1.0)
    threshold_specs = {
        name: (float(threshold), "margin" if name.startswith("margin") else
               "entropy" if name.startswith("entropy") else
               "top1" if name.startswith("top1") else "fixed_k")
        for name, (_, threshold) in policies.items() if threshold is not None
    }
    return policies, threshold_specs, k_opt


def _replay_by_doc(data: pd.DataFrame, docs: list[int], policies: dict[str, tuple[Any, float | int | None]]):
    result: dict[str, dict[int, tuple[int, int]]] = {}
    doc_set = set(docs)
    for name, (policy, threshold) in policies.items():
        part = data[data.doc.isin(doc_set)].copy()
        starts = doc_starts(part.doc.to_numpy())
        if name == "target_only":
            calls = np.diff(starts).astype(np.int64)
            drafted = np.zeros(len(calls), dtype=np.int64)
        else:
            if isinstance(policy, FixedK):
                stop = np.zeros(len(data), dtype=bool)
                kmax = policy.k
            elif isinstance(policy, MarginStop):
                stop = data.mD.to_numpy() < policy.tau
                kmax = 16
            elif isinstance(policy, EntropyStop):
                stop = data.HD.to_numpy() > policy.h
                kmax = 16
            elif isinstance(policy, Top1Stop):
                stop = np.exp(data.d_lp1.to_numpy()) < policy.p
                kmax = 16
            else:
                raise TypeError(type(policy))
            match = (part.d_arg.to_numpy() == part.y.to_numpy())
            calls, drafted, _ = simulate(match, stop[part.index.to_numpy()], starts, kmax)
        result[name] = {
            int(doc): (int(call), int(draft))
            for doc, call, draft in zip(part.doc.to_numpy()[starts[:-1]], calls, drafted)
        }
    return result


def _run_policy(draft_model, target_model, doc: pd.Series, policy, eos_token_id: int | None,
                backend: str = "dynamic", compile_mode: str = "reduce-overhead"):
    if backend == "static-compiled":
        if policy is None:
            return greedy_target_only_static(
                target_model, list(map(int, doc.prompt_ids)), MAX_NEW_TOKENS, eos_token_id,
                compile_steps=True, compile_mode=compile_mode)
        return speculative_greedy_static(
            draft_model, target_model, list(map(int, doc.prompt_ids)), MAX_NEW_TOKENS,
            eos_token_id, policy, max_block=16, compile_steps=True, compile_mode=compile_mode)
    if policy is None:
        return greedy_target_only(target_model, list(map(int, doc.prompt_ids)), MAX_NEW_TOKENS, eos_token_id)
    return speculative_greedy_adaptive(
        draft_model,
        target_model,
        list(map(int, doc.prompt_ids)),
        MAX_NEW_TOKENS,
        eos_token_id,
        policy,
        max_block=16,
    )


def _seconds_for_call(draft_model, target_model, doc, policy, eos_token_id,
                      backend: str = "dynamic", compile_mode: str = "reduce-overhead"):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = _run_policy(draft_model, target_model, doc, policy, eos_token_id, backend, compile_mode)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated, counters = result
    return generated, counters, seconds


def _resumable_shard(path: Path, doc_ids: list[str], policy_names: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
        if (set(frame.doc_id.astype(str)) == set(doc_ids)
                and set(frame.policy) == set(policy_names)
                and len(frame) == len(doc_ids) * len(policy_names)):
            return frame
    except Exception:
        pass
    path.unlink(missing_ok=True)
    return None


def _bootstrap_time_ratio(
    frame: pd.DataFrame, baseline: str, policy: str, docs: list[str], rng, reps: int,
    exact_only: bool = False,
):
    if not docs:
        return [float("nan"), float("nan")]
    selected = frame[frame.doc_id.astype(str).isin(docs)]
    if exact_only:
        exact_rows = selected[(selected.policy == policy) & selected.exact_match_live][["doc_id", "repeat"]]
        selected = selected.merge(exact_rows.drop_duplicates(), on=["doc_id", "repeat"], how="inner")
    per_doc = selected.groupby(["doc_id", "policy"]).seconds.sum().unstack("policy")
    per_doc = per_doc.reindex(docs).dropna(subset=[baseline, policy])
    values = per_doc[[baseline, policy]].to_numpy(dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    sums = values[draws].sum(axis=1)
    ratios = sums[:, 0] / np.maximum(sums[:, 1], 1e-15)
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def _summarize(frame: pd.DataFrame, replay, threshold_specs, cost: float, bootstrap_reps: int):
    rng = np.random.default_rng(SEED)
    baseline = frame[frame.policy == "target_only"]
    docs_all = sorted(frame.doc_id.astype(str).unique())
    fixed_names = [name for name in frame.policy.unique() if name.startswith("fixed_k_")]
    fixed_sec = {name: frame[frame.policy == name].seconds.sum() for name in fixed_names}
    fastest_fixed = min(fixed_sec, key=fixed_sec.get) if fixed_sec else None
    rows = []

    def metrics_for(name, doc_names, exact_only=False):
        sub = frame[(frame.policy == name) & frame.doc_id.isin(doc_names)]
        base = frame[(frame.policy == "target_only") & frame.doc_id.isin(doc_names)]
        matching = None
        if exact_only:
            sub = sub[sub.exact_match_live]
            matching = sub[["doc_id", "repeat"]].drop_duplicates()
            base = base.merge(matching, on=["doc_id", "repeat"], how="inner")
        total_seconds = float(sub.seconds.sum())
        total_tokens = int(sub.tokens.sum())
        base_seconds = float(base.seconds.sum())
        base_tokens = int(base.tokens.sum())
        total_calls = int(sub.target_forwards.sum())
        drafted = int(sub.drafted_tokens.sum())
        speedup = (total_tokens / total_seconds) / (base_tokens / base_seconds) if total_seconds and base_seconds and base_tokens else float("nan")
        fastest = frame[(frame.policy == fastest_fixed) & frame.doc_id.isin(doc_names)] if fastest_fixed else None
        if exact_only and fastest is not None:
            fastest = fastest.merge(matching, on=["doc_id", "repeat"], how="inner")
        speedup_fixed = (
            (total_tokens / total_seconds) /
            (float(fastest.tokens.sum()) / float(fastest.seconds.sum()))
            if fastest is not None and total_seconds and fastest.seconds.sum() and fastest.tokens.sum()
            else float("nan")
        )
        speedup_fixed_ci = (
            _bootstrap_time_ratio(frame, fastest_fixed, name, list(map(str, doc_names)), rng,
                                  bootstrap_reps, exact_only=exact_only)
            if fastest_fixed and name != fastest_fixed else [1.0, 1.0]
        )
        selected_pools = sub.pool_index.astype(int).tolist()
        replay_calls = sum(replay[name][pool][0] for pool in selected_pools if pool in replay[name])
        replay_drafted = sum(replay[name][pool][1] for pool in selected_pools if pool in replay[name])
        # Replay is evaluated against the stored, eligible 128-token
        # continuation. Live outputs can terminate early after an EOS.
        replay_tokens = MAX_NEW_TOKENS * len(sub)
        replay_l = replay_tokens / replay_calls if replay_calls else float("nan")
        replay_d = replay_drafted / replay_calls if replay_calls else float("nan")
        return {
            "documents": len(doc_names),
            "tokens": total_tokens,
            "seconds": total_seconds,
            "tokens_per_second": total_tokens / total_seconds if total_seconds else float("nan"),
            "speedup_target_only": speedup,
            "speedup_fastest_fixed": speedup_fixed,
            "tokens_per_target_call": total_tokens / total_calls if total_calls else float("nan"),
            "drafted_tokens_per_target_call": drafted / total_calls if total_calls else float("nan"),
            "replay_tokens_per_target_call": replay_l,
            "replay_drafted_tokens_per_call": replay_d,
            "cost_model_speedup": replay_l / (1 + cost * replay_d) if replay_calls else float("nan"),
            "speedup_target_only_ci": _bootstrap_time_ratio(
                frame, "target_only", name, list(map(str, doc_names)), rng, bootstrap_reps,
                exact_only=exact_only,
            ),
            "speedup_fastest_fixed_ci": speedup_fixed_ci,
        }

    for name in frame.policy.drop_duplicates().tolist():
        sub = frame[frame.policy == name]
        live_docs = sorted(sub[sub.exact_match_live].doc_id.astype(str).unique())
        values_all = metrics_for(name, docs_all)
        values_live = metrics_for(name, live_docs, exact_only=True)
        thresholds = threshold_specs.get(name, (None, None))
        rows.append({
            "policy": name,
            "threshold": thresholds[0],
            "threshold_kind": thresholds[1],
            "exact_match_stored_rate": float(sub.exact_match_stored.mean()),
            "exact_match_live_rate": float(sub.exact_match_live.mean()),
            **{f"{key}_all": value for key, value in values_all.items() if not key.startswith("_")},
            **{f"{key}_live_exact": value for key, value in values_live.items() if not key.startswith("_")},
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--pilot", action="store_true", help="Run a 50-document, one-repeat scheduling pilot")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-fixed-k", nargs="*", type=int, default=[])
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/rebuttal/wallclock")
    parser.add_argument("--backend", choices=("dynamic", "static-compiled"), default="dynamic")
    parser.add_argument("--compile-mode", choices=("reduce-overhead", "default"), default="reduce-overhead")
    parser.add_argument("--streamed-models", action="store_true",
                        help="Load pinned weights by shard to support pairs larger than /dev/shm")
    parser.add_argument("--cost-ratio-file", type=Path,
                        help="Override the latency cost-ratio CSV used for replay threshold selection")
    parser.add_argument("--only-policies", nargs="+",
                        help="Restrict the run to these named policies (for a same-GPU dynamic comparator)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config.get("pairs", {}):
        parser.error(f"unknown pair {args.pair!r} in {args.config}")
    if args.pilot:
        args.max_docs, args.repeats = 50, 1
    if args.max_docs < 1 or args.repeats < 1 or args.max_hours <= 0:
        parser.error("max-docs, repeats, and max-hours must be positive")

    pair = config["pairs"][args.pair]
    draft_key, target_key = pair["draft"], pair["target"]
    tokenizer = load_tokenizer(config, target_key)
    eos_token_id = int(tokenizer.eos_token_id)
    continuations = _continuations(target_key, pair["split"])
    valid = _select_valid_docs(continuations)
    docs = valid.head(args.max_docs).copy()
    if not len(docs):
        raise RuntimeError("No test documents with 128 continuation tokens and eos_hit=False")
    dev_ids = _selection_dev(continuations)
    if args.cost_ratio_file is not None:
        pair_latency_path = args.cost_ratio_file if args.cost_ratio_file.is_absolute() else ROOT / args.cost_ratio_file
    elif args.backend == "static-compiled":
        pair_latency_path = ROOT / "results/rebuttal/latency_opt/cost_ratio_opt.csv"
    else:
        pair_latency_path = ROOT / "results/rebuttal/latency/cost_ratio.csv"
    if not pair_latency_path.is_file():
        raise FileNotFoundError(f"Run T1a latency probes first: {pair_latency_path}")
    costs = pd.read_csv(pair_latency_path)
    cost_rows = costs[costs.pair == args.pair]
    if "target_verify_tokens" in cost_rows.columns:
        cost_rows = cost_rows[cost_rows.target_verify_tokens == 17]
    if cost_rows.empty:
        raise RuntimeError(f"No measured cost ratio for pair {args.pair}; run its T1a family probe")
    measured_cost = float(cost_rows.iloc[0].c_measured)
    pair_data = load_pair("generated", target_key, draft_key, common_only=True)
    replay_docs = set(docs.pool_index.astype(int))
    replay_pools = set(valid.pool_index.astype(int)) | dev_ids
    pair_data = pair_data[pair_data.doc.isin(replay_pools)].reset_index(drop=True)
    policies, threshold_specs, fixed_opt = _make_policies(
        pair_data, dev_ids, measured_cost, set(args.skip_fixed_k),
        include_k1=(args.backend == "static-compiled"
                    or bool(args.only_policies and "fixed_k_1" in args.only_policies)))
    if args.only_policies:
        missing = set(args.only_policies) - set(policies)
        if missing:
            parser.error(f"unknown or unavailable policies: {sorted(missing)}")
        policies = {name: policies[name] for name in args.only_policies}
        threshold_specs = {name: value for name, value in threshold_specs.items() if name in policies}

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    # Use the completed Q5 pilot to project its main-run budget. If the
    # measured projection exceeds the four-hour cap, remove k=2 and k=6 first,
    # then reduce the document count to 150 and, only if needed, 100.
    if not args.pilot:
        pilot_path = output_root / args.pair / "pilot/per_doc.parquet"
        # Q2 has no separate pilot by specification. Its draft is the same as
        # Q5 and its measured target verify latency is similar, so use Q5's
        # pilot as a conservative scheduling estimate before starting Q2.
        if not pilot_path.is_file() and args.pair == "Q2":
            pilot_path = output_root / "Q5/pilot/per_doc.parquet"
        if pilot_path.is_file():
            pilot_frame = pd.read_parquet(pilot_path)
            pilot_docs = max(1, pilot_frame.doc_id.nunique())

            def projected_hours(frame):
                per_doc = float(frame.seconds.sum()) / pilot_docs
                return per_doc * args.max_docs * args.repeats / 3600.0

            projection = projected_hours(pilot_frame)
            if projection > args.max_hours:
                removable = {"fixed_k_2", "fixed_k_6"} & set(policies)
                if removable:
                    policies = {name: value for name, value in policies.items() if name not in removable}
                    threshold_specs = {name: value for name, value in threshold_specs.items() if name not in removable}
                    args.skip_fixed_k = sorted(set(args.skip_fixed_k) | {2, 6})
                    pilot_frame = pilot_frame[~pilot_frame.policy.isin(removable)]
                    projection = projected_hours(pilot_frame)
                    print(f"CAP_PLAN drop_fixed_k={sorted(removable)} projected_hours={projection:.2f}", flush=True)
                if projection > args.max_hours:
                    args.max_docs = 150
                    projection = projected_hours(pilot_frame)
                    print(f"CAP_PLAN max_docs=150 projected_hours={projection:.2f}", flush=True)
                if projection > args.max_hours:
                    args.max_docs = 100
                    projection = projected_hours(pilot_frame)
                    print(f"CAP_PLAN max_docs=100 projected_hours={projection:.2f}", flush=True)
            docs = docs.head(args.max_docs).copy()

    test_pool_ids = docs.pool_index.astype(int).tolist()
    replay_map = _replay_by_doc(pair_data, test_pool_ids, policies)

    output_dir = output_root / args.pair / ("pilot" if args.pilot else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "shards"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "pair": args.pair,
        "draft_model": draft_key,
        "target_model": target_key,
        "draft_revision": config["models"][draft_key]["revision"],
        "target_revision": config["models"][target_key]["revision"],
        "pool_indices": test_pool_ids,
        "repeats": args.repeats,
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": eos_token_id,
        "seed": SEED,
        "measured_cost_ratio": measured_cost,
        "thresholds": {key: list(value) for key, value in threshold_specs.items()},
        "fixed_k_optimal_dev": fixed_opt,
        "skipped_fixed_k": sorted(args.skip_fixed_k),
        "policy_names": list(policies),
    }
    if args.backend != "dynamic" or args.streamed_models:
        identity.update({"backend": args.backend,
                         "compile_mode": args.compile_mode if args.backend == "static-compiled" else None,
                         "streamed_models": args.streamed_models or args.backend == "static-compiled"})
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("details", {}).get("run_identity") != identity:
            raise RuntimeError("Existing wall-clock run identity differs; use a fresh output directory")
    metadata_details = {"run_identity": identity, "max_hours": args.max_hours,
                       "bootstrap_reps": args.bootstrap_reps}
    if args.backend != "dynamic" or args.streamed_models:
        metadata_details.update({"backend": args.backend, "compile_mode": args.compile_mode})
    write_gpu_metadata(
        metadata_path,
        job="wallclock_pilot" if args.pilot else "wallclock_main",
        require_idle=True,
        details=metadata_details,
    )

    if args.streamed_models or args.backend == "static-compiled":
        from huggingface_hub import HfApi
        from huggingface_hub.utils import get_token
        from scripts.rebuttal.score_cross_family_judge import _load_streamed_model

        token = get_token()
        if not token:
            raise RuntimeError("Hugging Face token is unavailable; set HF_TOKEN from /workspace/hf_token")
        api = HfApi(token=token)
        draft_model, _, _ = _load_streamed_model(config, draft_key, api, token)
        target_model, _, _ = _load_streamed_model(config, target_key, api, token)
    else:
        draft_model = load_model(config, draft_key)
        target_model = load_model(config, target_key)
    for model in (draft_model, target_model):
        model.eval()
    torch.cuda.synchronize()

    policy_names = list(policies)
    warmup_docs = docs.head(min(10, len(docs)))
    with torch.inference_mode():
        for policy_name in policy_names:
            policy = policies[policy_name][0]
            for _, doc in warmup_docs.iterrows():
                _run_policy(draft_model, target_model, doc, policy, eos_token_id,
                            args.backend, args.compile_mode)
            print(f"WALLCLOCK_WARMUP_COMPLETE pair={args.pair} policy={policy_name} docs={len(warmup_docs)}", flush=True)
        torch.cuda.synchronize()

        job_started = time.perf_counter()
        completed_frames = []
        for repeat in range(args.repeats):
            for block_start in range(0, len(docs), 20):
                block = docs.iloc[block_start:block_start + 20]
                shard_path = checkpoint_dir / f"repeat-{repeat:02d}-block-{block_start // 20:03d}.parquet"
                expected_doc_ids = block.doc_id.astype(str).tolist()
                cached = _resumable_shard(shard_path, expected_doc_ids, policy_names)
                if cached is not None:
                    print(f"RESUME_SKIP {shard_path}", flush=True)
                    completed_frames.append(cached)
                    continue
                block_rows = []
                for _, doc in block.iterrows():
                    order = policy_names.copy()
                    random.Random(SEED + repeat * 1000003 + int(doc.pool_index)).shuffle(order)
                    for policy_name in order:
                        policy = policies[policy_name][0]
                        generated, counters, elapsed = _seconds_for_call(
                            draft_model, target_model, doc, policy, eos_token_id,
                            args.backend, args.compile_mode
                        )
                        stored = list(map(int, doc.continuation_ids))
                        baseline_output = None
                        # The target-only row is available in this document's
                        # block after its randomized execution; compare later.
                        block_rows.append({
                            "pair": args.pair,
                            "policy": policy_name,
                            "threshold": threshold_specs.get(policy_name, (None, None))[0],
                            "repeat": repeat,
                            "doc_id": str(doc.doc_id),
                            "pool_index": int(doc.pool_index),
                            "tokens": len(generated),
                            "seconds": elapsed,
                            "target_forwards": int(counters.get("target_forwards", 0)),
                            "draft_forwards": int(counters.get("draft_forwards", 0)),
                            "drafted_tokens": int(counters.get("drafted_tokens", 0)),
                            "accepted_tokens": int(counters.get("accepted_tokens", 0)),
                            "exact_match_stored": generated == stored,
                            "generated_ids": generated,
                            "_baseline_output": baseline_output,
                        })
                baseline_by_doc = {
                    row["doc_id"]: row["generated_ids"]
                    for row in block_rows if row["policy"] == "target_only"
                }
                for row in block_rows:
                    row["exact_match_live"] = row["generated_ids"] == baseline_by_doc[row["doc_id"]]
                    row.pop("generated_ids", None)
                    row.pop("_baseline_output", None)
                shard = pd.DataFrame(block_rows)
                atomic_parquet(block_rows, shard_path)
                completed_frames.append(shard)
                elapsed_total = time.perf_counter() - job_started
                print(
                    f"WALLCLOCK_BLOCK pair={args.pair} repeat={repeat} block={block_start // 20} "
                    f"docs={len(block)} seconds={elapsed_total:.1f}",
                    flush=True,
                )
                if elapsed_total >= args.max_hours * 3600:
                    print(f"WALLCLOCK_CAP_REACHED hours={args.max_hours}; stopping at completed block", flush=True)
                    break
            if time.perf_counter() - job_started >= args.max_hours * 3600:
                break

    per_doc = pd.concat(completed_frames, ignore_index=True) if completed_frames else pd.DataFrame()
    if not per_doc.empty:
        per_doc.to_parquet(output_dir / "per_doc.parquet", index=False, compression="zstd")
        summary = _summarize(per_doc, replay_map, threshold_specs, measured_cost, args.bootstrap_reps)
        summary.to_csv(output_dir / "summary.csv", index=False)
        summary.to_markdown(output_dir / "summary.md", index=False)
        print(f"WALLCLOCK_COMPLETE pair={args.pair} rows={len(per_doc)} output={output_dir}", flush=True)
    else:
        raise RuntimeError("No completed wall-clock document blocks")
    del draft_model, target_model
    clear_compiled_models()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
