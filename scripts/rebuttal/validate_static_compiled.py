#!/usr/bin/env python3
"""Gate a fixed-shape StaticCache decoder against the existing dynamic path."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.score_cross_family_judge import _load_streamed_model
from scripts.rebuttal.bench_wallclock import _continuations, _select_valid_docs
from src.rebuttal.adaptive_sd import FixedK, MarginStop, greedy_target_only, speculative_greedy_adaptive
from src.rebuttal.static_compiled_sd import (
    clear_compiled_models,
    greedy_target_only_static,
    speculative_greedy_static,
)
from src.naacl_experiment import atomic_json, load_config
from src.rebuttal.runtime import write_gpu_metadata


def _margin_from_dynamic_summary(pair: str) -> float:
    path = ROOT / "results/rebuttal/wallclock" / pair / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No previous dynamic margin threshold at {path}")
    rows = pd.read_csv(path)
    row = rows[rows.policy == "margin_tuned"]
    if row.empty:
        raise RuntimeError(f"No margin_tuned row in {path}")
    return float(row.iloc[0].threshold)


def _decode(model, doc, policy, eos, mode, static, compile_steps=True):
    if policy == "target_only":
        if static:
            return greedy_target_only_static(model["target"], list(map(int, doc.prompt_ids)), 128,
                                              eos, compile_steps=compile_steps, compile_mode=mode)
        return greedy_target_only(model["target"], list(map(int, doc.prompt_ids)), 128, eos)
    spec = FixedK(4) if policy == "fixed_k4" else MarginStop(float(policy.split(":", 1)[1]))
    if static:
        return speculative_greedy_static(model["draft"], model["target"],
                                          list(map(int, doc.prompt_ids)), 128, eos, spec,
                                          compile_steps=compile_steps, compile_mode=mode)
    return speculative_greedy_adaptive(model["draft"], model["target"],
                                       list(map(int, doc.prompt_ids)), 128, eos, spec)


@torch.inference_mode()
def _run_pair(pair: str, mode: str, args, config, api, token) -> dict:
    if pair == "Q2":
        target, draft, split = "qwen4", "qwen06", "all"
    else:
        target, draft, split = "qwen8", "qwen06", "common"
    frame = _select_valid_docs(_continuations(target, split)).head(args.documents)
    if len(frame) != args.documents:
        raise RuntimeError(f"{pair}: expected {args.documents} eligible test docs, got {len(frame)}")
    margin = args.margin_q2 if pair == "Q2" else args.margin_q5
    if margin is None:
        margin = _margin_from_dynamic_summary(pair)
    out = args.output_root / pair / "equivalence"
    out.mkdir(parents=True, exist_ok=True)
    model_rows = {"draft": None, "target": None}
    for key in (draft, target):
        print(f"EQUIV_LOAD model={key}", flush=True)
        model, path, storage_mode = _load_streamed_model(config, key, api, token)
        model.config.use_cache = True
        model_rows["draft" if key == draft else "target"] = model
    eos = getattr(model_rows["target"].generation_config, "eos_token_id", None)
    if isinstance(eos, list):
        eos = eos[0]
    policies = (tuple(args.policies) if args.policies else
                ("target_only", "fixed_k4", f"margin:{margin}"))
    dynamic_runs: dict[tuple[int, str, int], list[int]] = {}
    counters_ok = True
    for repeat in range(2):
        for doc in frame.itertuples(index=False):
            for policy in policies:
                result = _decode(model_rows, doc, policy, eos, mode, static=False)
                generated, counters = result[:2]
                dynamic_runs[(int(doc.pool_index), policy, repeat)] = generated
                if policy != "target_only":
                    counters_ok &= counters["target_forwards"] == counters["blocks"]
    repeat_rates = {}
    for policy in policies:
        repeat_rates[policy] = sum(
            dynamic_runs[(int(doc.pool_index), policy, 0)] == dynamic_runs[(int(doc.pool_index), policy, 1)]
            for doc in frame.itertuples(index=False)
        ) / len(frame)

    optimized_exact = {}
    static_forward_gates = {}
    mismatch_examples: dict[str, list[dict[str, int | None]]] = {}
    for policy in policies:
        matches = 0
        docs_with_equal_forwards = 0
        policy_mismatches = []
        for doc in frame.itertuples(index=False):
            result = _decode(model_rows, doc, policy, eos, mode, static=True,
                             compile_steps=not args.static_eager)
            generated, counters = result[:2]
            reference = dynamic_runs[(int(doc.pool_index), policy, 0)]
            matches += int(generated == reference)
            if generated != reference:
                common = min(len(generated), len(reference))
                first_diff = next((index for index in range(common)
                                   if generated[index] != reference[index]), common)
                policy_mismatches.append({
                    "pool_index": int(doc.pool_index),
                    "first_mismatch_token_index": first_diff,
                    "dynamic_token_id": reference[first_diff] if first_diff < len(reference) else None,
                    "static_token_id": generated[first_diff] if first_diff < len(generated) else None,
                    "dynamic_length": len(reference),
                    "static_length": len(generated),
                })
            if policy != "target_only":
                docs_with_equal_forwards += int(counters["target_forwards"] == counters["blocks"])
        optimized_exact[policy] = matches / len(frame)
        mismatch_examples[policy] = policy_mismatches[:10]
        static_forward_gates[policy] = (docs_with_equal_forwards == len(frame)
                                        if policy != "target_only" else None)
        if optimized_exact[policy] < repeat_rates[policy]:
            raise RuntimeError(
                f"{pair}/{policy}: static exact={optimized_exact[policy]:.4f} is below dynamic repeat agreement {repeat_rates[policy]:.4f}; "
                f"first mismatch examples={mismatch_examples[policy]}"
            )
        if policy != "target_only" and not static_forward_gates[policy]:
            counters_ok = False
    result = {
        "pair": pair,
        "documents": len(frame),
        "pool_indices": frame.pool_index.astype(int).tolist(),
        "margin_threshold": margin,
        "compile_mode": mode,
        "dynamic_repeat_exact_match_rate": repeat_rates,
        "static_vs_dynamic_exact_match_rate": optimized_exact,
        "static_target_forwards_equal_blocks": static_forward_gates,
        "static_mismatch_examples": mismatch_examples,
        "target_forwards_gate_pass": counters_ok,
        "equivalence_gate_pass": all(optimized_exact[p] >= repeat_rates[p] for p in policies) and counters_ok,
        "policies": list(policies),
    }
    suffix = "eager" if args.static_eager else "compiled"
    atomic_json(out / f"gate_{mode.replace('-', '_')}_{suffix}.json", result)
    clear_compiled_models()
    del model_rows
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/rebuttal/wallclock_opt")
    parser.add_argument("--pairs", nargs="+", choices=("Q2", "Q5"), default=("Q2", "Q5"))
    parser.add_argument("--documents", type=int, default=50)
    parser.add_argument("--compile-mode", choices=("reduce-overhead", "default"), default="reduce-overhead")
    parser.add_argument("--policies", nargs="+", choices=("target_only", "fixed_k4"),
                        help="Restrict the diagnostic run to selected policies")
    parser.add_argument("--static-eager", action="store_true",
                        help="Run the static decoder without torch.compile")
    parser.add_argument("--margin-q2", type=float)
    parser.add_argument("--margin-q5", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_gpu_metadata(output / "equivalence_run_metadata.json", job="static_compiled_equivalence",
                       require_idle=True,
                       details={"pairs": args.pairs, "documents": args.documents,
                                "compile_mode": args.compile_mode,
                                "model_revisions": {key: config["models"][key]["revision"]
                                                    for key in ("qwen06", "qwen4", "qwen8")}})
    api = HfApi(token=get_token())
    mode = args.compile_mode
    try:
        results = [_run_pair(pair, mode, args, config, api, get_token()) for pair in args.pairs]
    except Exception as error:
        if mode != "reduce-overhead":
            raise
        print(f"COMPILE_MODE_FALLBACK from=reduce-overhead to=default error={type(error).__name__}: {error}",
              flush=True)
        clear_compiled_models()
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        mode = "default"
        results = [_run_pair(pair, mode, args, config, api, get_token()) for pair in args.pairs]
    atomic_json(output / "equivalence_summary.json", {"results": results,
                                                        "all_pass": all(row["equivalence_gate_pass"] for row in results)})
    return 0 if all(row["equivalence_gate_pass"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
