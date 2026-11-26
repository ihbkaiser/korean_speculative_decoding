#!/usr/bin/env python3
"""Run the paired T5 instruction-tuned and base-Q2 analyses on fixed documents."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from expC_helpers import verified_k4
from expD_lib import CONTRASTS, boot_ci, keys_for, ratio
from ksdlib import load_pair, read_aligned
from policysim import doc_starts, simulate
from src.naacl_experiment import atomic_json, load_config, load_tokenizer

SEED = 20261125
BOOTSTRAPS = 500
DOC_LIMIT = 5000
INSTRUCT_ROOT = ROOT / "results/rebuttal/instruct"
BASE_ROOT = ROOT / "results/naacl_experiments"
BASE_MODEL_ROOT = Path(os.environ.get("CROSS_FAMILY_MODEL_DIR", "/dev/shm/cross_family_models"))
PAIR_CONFIG = {
    "instruct": ("qwen4it", "qwen06it", INSTRUCT_ROOT, "common"),
    "base_Q2": ("qwen4", "qwen06", BASE_ROOT, "all"),
}


def _hangul_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
        or 0xAC00 <= code <= 0xD7A3
        or 0xD7B0 <= code <= 0xD7FF
    )


def _continuation_files(root: Path, target: str, split: str) -> list[Path]:
    directory = root / "continuations" / target / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No continuation output at {directory}")
    return files


def _continuations(root: Path, target: str, split: str) -> pd.DataFrame:
    files = _continuation_files(root, target, split)
    frame = pd.concat([
        pd.read_parquet(file, columns=["doc_id", "pool_index", "prompt_ids", "continuation_ids", "continuation_length", "eos_hit"])
        for file in files
    ], ignore_index=True)
    frame = frame[frame.pool_index < DOC_LIMIT].sort_values("pool_index", kind="stable")
    frame = frame.drop_duplicates("doc_id", keep="first").reset_index(drop=True)
    if len(frame) != DOC_LIMIT:
        raise RuntimeError(f"Expected {DOC_LIMIT} frozen continuation rows at {root}; found {len(frame)}")
    return frame


def _alignment_gate(root: Path, target: str, split: str) -> dict:
    directory = root / "aligned" / "kiwi" / target / split
    files = sorted(directory.glob("**/shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No morphology alignment output at {directory}")
    parts = []
    for file in files:
        part = pd.read_parquet(file, columns=["pool_index", "alignment_status"])
        parts.append(part[part.pool_index < DOC_LIMIT])
    raw = pd.concat(parts, ignore_index=True)
    aligned_count = int((raw.alignment_status == "ALIGNED").sum())
    total = int(len(raw))
    return {"aligned_tokens": aligned_count, "total_tokens": total,
            "aligned_share": aligned_count / total if total else float("nan")}


def _repeated_4gram(frame: pd.DataFrame) -> dict:
    repeated = total = 0
    distinct_ratios = []
    for row in frame.itertuples(index=False):
        tokens = list(map(int, row.continuation_ids))
        if len(tokens) < 32:
            continue
        seen = set()
        doc_repeated = 0
        for index in range(len(tokens) - 3):
            gram = tuple(tokens[index:index + 4])
            if gram in seen:
                doc_repeated += 1
            seen.add(gram)
        repeated += doc_repeated
        total += len(tokens) - 3
        distinct_ratios.append(len(set(tokens)) / len(tokens))
    return {
        "repeated_4gram_rate": repeated / total if total else float("nan"),
        "repeated_4gram_count": repeated,
        "eligible_4gram_count": total,
        "median_distinct_token_ratio": float(np.median(distinct_ratios)) if distinct_ratios else float("nan"),
    }


def _quality_gate(root: Path, target: str, split: str, tokenizer) -> tuple[dict, pd.DataFrame]:
    frame = _continuations(root, target, split)
    lengths = frame.continuation_length.to_numpy(dtype=np.int64)
    hangul_shares = []
    think_docs = 0
    second_turn_docs = 0
    for row in frame.itertuples(index=False):
        ids = list(map(int, row.continuation_ids))
        raw_text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        visible_text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        think_docs += int("<think>" in raw_text or "</think>" in raw_text)
        second_turn_docs += int(any(marker in raw_text for marker in ("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>system")))
        hangul_count = sum(_hangul_char(ch) for ch in visible_text)
        hangul_shares.append(hangul_count / len(visible_text) if visible_text else 0.0)
    align = _alignment_gate(root, target, split)
    quality = {
        "documents": len(frame),
        "continuation_length_tokens": {
            "min": int(lengths.min()), "p10": float(np.quantile(lengths, .1)),
            "median": float(np.median(lengths)), "mean": float(lengths.mean()),
            "p90": float(np.quantile(lengths, .9)), "max": int(lengths.max()),
        },
        "eos_hit_rate": float(frame.eos_hit.mean()),
        "documents_with_reasoning_block": int(think_docs),
        "documents_with_second_chat_turn": int(second_turn_docs),
        "hangul_share_by_document": {
            "median": float(np.median(hangul_shares)),
            "p10": float(np.quantile(hangul_shares, .1)),
        },
        "alignment": align,
        "repetition": _repeated_4gram(frame),
    }
    return quality, frame


def _merge_pair(pair: str, include_docs: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    target, draft, base, split = PAIR_CONFIG[pair]
    pair_rows = load_pair("generated", target, draft, base=base)
    pair_rows = pair_rows[pair_rows.doc < DOC_LIMIT].sort_values(["doc", "pos"], kind="stable").reset_index(drop=True)
    aligned = read_aligned(target, split, base=base)
    aligned = aligned[aligned.doc < DOC_LIMIT]
    pair_rows = pair_rows.merge(aligned, on=["doc", "pos"], how="left", validate="one_to_one")
    pair_rows["ptype"] = pair_rows.ptype.fillna("excluded")
    pair_rows["aclass"] = pair_rows.aclass.fillna("OTHER")
    pair_rows["rej"] = (pair_rows.d_arg.to_numpy() != pair_rows.y.to_numpy()).astype(np.int8)
    if include_docs is not None:
        pair_rows = pair_rows[pair_rows.doc.isin(include_docs)].reset_index(drop=True)
    full = pair_rows.copy()
    pair_rows["verified_k4"] = verified_k4(pair_rows)
    verified = pair_rows[pair_rows.verified_k4].reset_index(drop=True)
    return full, verified


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _main_metrics(frame: pd.DataFrame) -> dict:
    n = len(frame)
    rejections = int(frame.rej.sum())
    target_dp = frame.mT.to_numpy(dtype=float) < 1.0
    draft_dp = frame.mD.to_numpy(dtype=float) < 1.0
    either_dp = target_dp | draft_dp
    low20 = frame.mT.to_numpy(dtype=float) <= float(frame.mT.quantile(.2))
    margin_auc = _safe_auc(frame.rej.to_numpy(dtype=np.int8), -frame.mT.to_numpy(dtype=float))
    draft_auc = _safe_auc(frame.rej.to_numpy(dtype=np.int8), -frame.mD.to_numpy(dtype=float))
    boundary_frame = frame[frame.ptype != "excluded"]
    boundary_auc = _safe_auc(
        boundary_frame.rej.to_numpy(dtype=np.int8),
        (boundary_frame.ptype.to_numpy() != "internal").astype(np.int8),
    )
    by_type = {}
    for col in ("ptype", "aclass"):
        groups = frame.groupby(col, dropna=False)
        by_type[col] = {
            str(name): {"positions": int(len(group)), "rejections": int(group.rej.sum()),
                        "rejection_rate": float(group.rej.mean())}
            for name, group in groups
        }

    keys = keys_for(frame)
    zero_key = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    ratios = {}
    for name, (numerator, denominator, column) in CONTRASTS.items():
        ratios[name] = {}
        for label, key in (
            ("raw", zero_key),
            ("target_absolute_bins", keys["target abs bins"]),
            ("draft_absolute_bins", keys["draft abs bins"]),
            ("both_absolute_bins", keys["both abs bins"]),
        ):
            point, coverage = ratio(frame, numerator, denominator, column, key)
            ratios[name][label] = {
                "ratio": point,
                "coverage": coverage,
                "ci95": boot_ci(frame, numerator, denominator, column, key, rng, reps=BOOTSTRAPS),
            }

    return {
        "positions_verified_k4": n,
        "documents": int(frame.doc.nunique()),
        "rejection_count": rejections,
        "rejection_rate": float(frame.rej.mean()) if n else float("nan"),
        "decision_point_target_margin_lt_1": {
            "position_share": float(target_dp.mean()) if n else float("nan"),
            "rejection_mass_share": float(frame.rej.to_numpy()[target_dp].sum() / rejections) if rejections else float("nan"),
        },
        "lowest_target_margin_20pct": {
            "position_share": float(low20.mean()) if n else float("nan"),
            "rejection_mass_share": float(frame.rej.to_numpy()[low20].sum() / rejections) if rejections else float("nan"),
        },
        "auc_target_margin": margin_auc,
        "auc_draft_margin": draft_auc,
        "auc_boundary_label": boundary_auc,
        "either_margin_lt_1": {
            "position_share": float(either_dp.mean()) if n else float("nan"),
            "rejection_mass_share": float(frame.rej.to_numpy()[either_dp].sum() / rejections) if rejections else float("nan"),
        },
        "rejection_by_position_type": by_type,
        "position_type_ratios": ratios,
    }


def _aggregate(calls: np.ndarray, drafted: np.ndarray, lengths: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "tokens_per_target_call": float(lengths[mask].sum() / calls[mask].sum()),
        "draft_tokens_per_target_call": float(drafted[mask].sum() / calls[mask].sum()),
        "target_calls": int(calls[mask].sum()),
        "drafted_tokens": int(drafted[mask].sum()),
        "generated_tokens": int(lengths[mask].sum()),
    }


def _policy_replay(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values(["doc", "pos"], kind="stable").reset_index(drop=True)
    docs = frame.doc.to_numpy(dtype=np.int64)
    starts = doc_starts(docs)
    lengths = np.diff(starts).astype(np.int64)
    doc_ids = docs[starts[:-1]]
    dev_docs = (doc_ids % 2) == 0
    test_docs = ~dev_docs
    if not dev_docs.any() or not test_docs.any():
        raise RuntimeError("Policy replay requires both even-index dev and odd-index test documents")
    match = (frame.d_arg.to_numpy() == frame.y.to_numpy())
    never = np.zeros(len(frame), dtype=np.bool_)
    fixed_calls, fixed_drafted, _ = simulate(match, never, starts, 4)
    base_dev = _aggregate(fixed_calls, fixed_drafted, lengths, dev_docs)
    base_test = _aggregate(fixed_calls, fixed_drafted, lengths, test_docs)
    test_indices = np.flatnonzero(test_docs)
    rng = np.random.default_rng(SEED)
    signals = {
        "entropy": -frame.HD.to_numpy(dtype=float),
        "top1_probability": frame.d_lp1.to_numpy(dtype=float),
        "margin": frame.mD.to_numpy(dtype=float),
    }
    token_dev = np.repeat(dev_docs, lengths)
    out = {"fixed_k4": {"dev": base_dev, "test": base_test, "gain_over_k4_pct": 0.0, "gain_ci95_pct": [0.0, 0.0]}}
    for name, signal in signals.items():
        dev_signal = signal[token_dev]
        thresholds = np.unique(np.quantile(dev_signal, np.linspace(.01, .75, 60)))
        candidates = []
        for threshold in thresholds:
            calls, drafted, _ = simulate(match, signal < threshold, starts, 16)
            dev = _aggregate(calls, drafted, lengths, dev_docs)
            candidates.append((abs(dev["draft_tokens_per_target_call"] - base_dev["draft_tokens_per_target_call"]), float(threshold), calls, drafted, dev))
        _, selected_threshold, calls, drafted, selected_dev = min(candidates, key=lambda item: (item[0], item[1]))
        test = _aggregate(calls, drafted, lengths, test_docs)
        bootstrap_gain = []
        n_test = len(test_indices)
        for _ in range(BOOTSTRAPS):
            sampled = rng.choice(test_indices, size=n_test, replace=True)
            weights = np.bincount(sampled, minlength=len(doc_ids)).astype(float)
            policy_calls = float(weights[test_docs] @ calls[test_docs])
            fixed_target_calls = float(weights[test_docs] @ fixed_calls[test_docs])
            tokens = float(weights[test_docs] @ lengths[test_docs])
            policy_l = tokens / policy_calls if policy_calls else float("nan")
            fixed_l = tokens / fixed_target_calls if fixed_target_calls else float("nan")
            if np.isfinite(policy_l) and fixed_l > 0:
                bootstrap_gain.append(100.0 * (policy_l / fixed_l - 1.0))
        point_gain = 100.0 * (test["tokens_per_target_call"] / base_test["tokens_per_target_call"] - 1.0)
        out[name] = {
            "selected_threshold": selected_threshold,
            "dev": selected_dev,
            "test": test,
            "gain_over_k4_pct": point_gain,
            "gain_ci95_pct": [float(np.quantile(bootstrap_gain, .025)), float(np.quantile(bootstrap_gain, .975))],
        }
    return out


def _quality_gates(tokenizer, config) -> dict:
    instruct_quality, instruct_cont = _quality_gate(INSTRUCT_ROOT, "qwen4it", "common", tokenizer)
    base_tokenizer = load_tokenizer(config, "qwen4", snapshot=BASE_MODEL_ROOT / config["models"]["qwen4"]["id"].replace("/", "--") / config["models"]["qwen4"]["revision"])
    base_quality, base_cont = _quality_gate(BASE_ROOT, "qwen4", "all", base_tokenizer)
    return {
        "instruct": instruct_quality,
        "base_Q2_same_5000_documents": base_quality,
        "instruct_hangul_median_gate_pass": instruct_quality["hangul_share_by_document"]["median"] >= .6,
        "base_Q2_hangul_median_gate_pass": base_quality["hangul_share_by_document"]["median"] >= .6,
    }


def _comparison_metrics(tokenizer, config) -> dict:
    # The long-document subset is based on stored generated continuation lengths.
    q, qcont = _quality_gate(INSTRUCT_ROOT, "qwen4it", "common", tokenizer)
    btokenizer = load_tokenizer(config, "qwen4", snapshot=BASE_MODEL_ROOT / config["models"]["qwen4"]["id"].replace("/", "--") / config["models"]["qwen4"]["revision"])
    b, bcont = _quality_gate(BASE_ROOT, "qwen4", "all", btokenizer)
    long_i = set(qcont.loc[qcont.continuation_length >= 64, "pool_index"].astype(int))
    long_b = set(bcont.loc[bcont.continuation_length >= 64, "pool_index"].astype(int))
    groups = {}
    for pair in ("instruct", "base_Q2"):
        full, verified_all = _merge_pair(pair)
        long_docs = long_i if pair == "instruct" else long_b
        verified_long = verified_all[verified_all.doc.isin(long_docs)].reset_index(drop=True)
        groups[pair] = {
            "all_documents": {
                "verified_k4": _main_metrics(verified_all),
                "policy_replay": _policy_replay(full),
            },
            "continuations_at_least_64_tokens": {
                "document_count": len(long_i if pair == "instruct" else long_b),
                "verified_k4": _main_metrics(verified_long),
                "policy_replay": _policy_replay(full[full.doc.isin(long_docs)]),
            },
        }
    return groups


def _markdown(payload: dict) -> str:
    gates = payload["quality_gates"]
    lines = [
        "# T5: instruction-tuned Qwen pair",
        "",
        "Model revisions: `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`; `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`.",
        "The base comparison is Q2 on the same first 5,000 common documents.",
        "",
        "## Sample quality and alignment gates",
        "",
        "| Measure | Instruct | Base Q2 |",
        "|---|---:|---:|",
    ]
    for label, accessor in (
        ("Continuation length median (tokens)", lambda x: x["continuation_length_tokens"]["median"]),
        ("Continuation length p10 (tokens)", lambda x: x["continuation_length_tokens"]["p10"]),
        ("EOS rate", lambda x: x["eos_hit_rate"]),
        ("Hangul share median", lambda x: x["hangul_share_by_document"]["median"]),
        ("Hangul share p10", lambda x: x["hangul_share_by_document"]["p10"]),
        ("ALIGNED token share", lambda x: x["alignment"]["aligned_share"]),
        ("Repeated 4-gram rate", lambda x: x["repetition"]["repeated_4gram_rate"]),
    ):
        a = accessor(gates["instruct"])
        b = accessor(gates["base_Q2_same_5000_documents"])
        lines.append(f"| {label} | {a:.4f} | {b:.4f} |")
    lines.extend([
        "",
        f"Reasoning-block continuations: {gates['instruct']['documents_with_reasoning_block']}; second chat turn: {gates['instruct']['documents_with_second_chat_turn']}; Hangul median gate: {gates['instruct_hangul_median_gate_pass']}.",
        "",
        "## Core analysis on k=4-verified proposals",
        "",
        "| Pair / subset | Rejection | DP positions | DP rejection share | Lowest 20% rejection share | AUC target | AUC draft | AUC boundary | Either-margin rejection share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair, label in (("instruct", "Instruct"), ("base_Q2", "Base Q2")):
        for subset, subset_label in (("all_documents", "all"), ("continuations_at_least_64_tokens", "≥64 tokens")):
            metric = payload["results"][pair][subset]["verified_k4"]
            lines.append(
                f"| {label} / {subset_label} | {metric['rejection_rate']:.4f} | "
                f"{metric['decision_point_target_margin_lt_1']['position_share']:.4f} | "
                f"{metric['decision_point_target_margin_lt_1']['rejection_mass_share']:.4f} | "
                f"{metric['lowest_target_margin_20pct']['rejection_mass_share']:.4f} | "
                f"{metric['auc_target_margin']:.4f} | {metric['auc_draft_margin']:.4f} | "
                f"{metric['auc_boundary_label']:.4f} | {metric['either_margin_lt_1']['rejection_mass_share']:.4f} |"
            )
    lines.extend(["", "## Policy replay (odd-index test documents)", "",
                  "Tokens per target call at the draft-token cost selected on even-index documents near fixed k=4.",
                  "Gains are relative to fixed k=4; intervals are paired document-bootstrap 95% intervals (500 resamples).", "",
                  "| Pair / subset | Policy | Threshold | Draft tokens/call | Tokens/call | Gain vs k=4 [95% CI]% |",
                  "|---|---|---:|---:|---:|---:|"])
    for pair, label in (("instruct", "Instruct"), ("base_Q2", "Base Q2")):
        for subset, subset_label in (("all_documents", "all"), ("continuations_at_least_64_tokens", "≥64 tokens")):
            policies = payload["results"][pair][subset]["policy_replay"]
            for name in ("fixed_k4", "entropy", "top1_probability", "margin"):
                row = policies[name]
                test = row["test"]
                threshold = "—" if "selected_threshold" not in row else f"{row['selected_threshold']:.4g}"
                lines.append(
                    f"| {label} / {subset_label} | {name} | {threshold} | "
                    f"{test['draft_tokens_per_target_call']:.3f} | {test['tokens_per_target_call']:.3f} | "
                    f"{row['gain_over_k4_pct']:.2f} [{row['gain_ci95_pct'][0]:.2f}, {row['gain_ci95_pct'][1]:.2f}] |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rebuttal_instruct.yaml")
    parser.add_argument("--output-dir", type=Path, default=INSTRUCT_ROOT / "analysis")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config, "qwen4it")
    quality = _quality_gates(tokenizer, config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not quality["instruct_hangul_median_gate_pass"]:
        payload = {"seed": SEED, "quality_gates": quality, "analysis_stopped": "instruct Hangul-share median is below 0.6"}
        atomic_json(output_dir / "summary.json", payload)
        (output_dir / "summary.md").write_text(
            "# T5: instruction-tuned Qwen pair\n\n"
            "Analysis stopped because the continuation Hangul-share median is below 0.6. "
            "The quality gates are in `summary.json`.\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 0
    results = _comparison_metrics(tokenizer, config)
    payload = {
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "same_document_ids": DOC_LIMIT,
        "quality_gates": quality,
        "results": results,
    }
    atomic_json(output_dir / "summary.json", payload)
    (output_dir / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
