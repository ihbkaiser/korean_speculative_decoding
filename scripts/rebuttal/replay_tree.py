#!/usr/bin/env python3
"""Replay comb-tree rescues from frozen draft top-k teacher-forced scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from ksdlib import load_pair
from policysim import doc_starts

SEED = 20261125
BOOTSTRAPS = 500
PAIRS = {
    "Q2": ("qwen4", "qwen06"),
    "Q5": ("qwen8", "qwen06"),
    "M1": ("ministral8", "ministral3"),
    "G1": ("gemma12", "gemma4"),
}


def _topk_path(pair: str, root: Path) -> Path:
    target, draft = PAIRS[pair]
    return root / "teacher_forced/generated" / target / draft / "common"


def _tune_margin(frame: pd.DataFrame, dev_docs: set[int], cost: float) -> float:
    part = frame[frame.doc.isin(dev_docs)]
    signal = part.mD.to_numpy(dtype=np.float64)
    values = np.unique(np.quantile(signal, np.linspace(.005, .9, 80)))
    match = part.d_arg.to_numpy() == part.y.to_numpy()
    starts = doc_starts(part.doc.to_numpy())
    best_threshold, best_score = float(values[0]), -np.inf
    for threshold in values:
        calls, drafted, _ = _simulate_arrays(match, signal < threshold, starts, 16)
        tokens = len(part)
        l_value = tokens / calls.sum()
        d_value = drafted.sum() / calls.sum()
        score = l_value / (1.0 + cost * d_value)
        if score > best_score:
            best_score, best_threshold = float(score), float(threshold)
    return best_threshold


def _simulate_arrays(match: np.ndarray, stop: np.ndarray, starts: np.ndarray, kmax: int):
    n_docs = len(starts) - 1
    calls = np.zeros(n_docs, dtype=np.int64)
    drafted = np.zeros(n_docs, dtype=np.int64)
    for doc in range(n_docs):
        begin, end = int(starts[doc]), int(starts[doc + 1])
        cursor = begin
        while cursor < end:
            j = 0
            while cursor + j < end:
                j += 1
                if stop[cursor + j - 1] or j >= kmax:
                    break
            calls[doc] += 1
            drafted[doc] += j
            first_mismatch = -1
            for offset in range(j):
                if not match[cursor + offset]:
                    first_mismatch = cursor + offset
                    break
            cursor = first_mismatch + 1 if first_mismatch >= 0 else cursor + j + 1
    return calls, drafted, None


def _run_variant(frame: pd.DataFrame, policy: str, threshold: float | None, width: int, mode: str) -> pd.DataFrame:
    records = []
    top1 = frame.d_arg.to_numpy(dtype=np.int64)
    target = frame.y.to_numpy(dtype=np.int64)
    margins_d = frame.mD.to_numpy(dtype=np.float64)
    margins_t = frame.mT.to_numpy(dtype=np.float64)
    topk = frame.top8_ids.to_list()
    docs = frame.doc.to_numpy(dtype=np.int64)
    starts = doc_starts(docs)
    for di in range(len(starts) - 1):
        begin, end = int(starts[di]), int(starts[di + 1])
        cursor = begin
        calls = drafted = extra = 0
        first_rejections = {
            "all": 0, "target_dp": 0, "target_non_dp": 0,
            "draft_dp": 0, "draft_non_dp": 0,
        }
        rescues = {
            "all": 0, "target_dp": 0, "target_non_dp": 0,
            "draft_dp": 0, "draft_non_dp": 0,
        }
        while cursor < end:
            block_positions = []
            while cursor + len(block_positions) < end and len(block_positions) < 16:
                pos = cursor + len(block_positions)
                block_positions.append(pos)
                if policy == "fixed_k4":
                    stop_now = len(block_positions) >= 4
                else:
                    stop_now = margins_d[pos] < float(threshold)
                if stop_now:
                    break
            calls += 1
            drafted += len(block_positions)
            widths = []
            for pos in block_positions:
                enabled = mode == "everywhere" or (mode == "draft_dp_only" and margins_d[pos] < 1.0)
                slot_width = width if enabled else 1
                widths.append(slot_width)
                extra += slot_width - 1
            mismatch_offset = next((offset for offset, pos in enumerate(block_positions)
                                    if target[pos] != top1[pos]), None)
            if mismatch_offset is None:
                cursor += len(block_positions) + 1
                continue
            reject_pos = block_positions[mismatch_offset]
            first_rejections["all"] += 1
            target_dp = margins_t[reject_pos] < 1.0
            draft_dp = margins_d[reject_pos] < 1.0
            if target_dp:
                first_rejections["target_dp"] += 1
            else:
                first_rejections["target_non_dp"] += 1
            if draft_dp:
                first_rejections["draft_dp"] += 1
            else:
                first_rejections["draft_non_dp"] += 1
            rescued = False
            if widths[mismatch_offset] > 1:
                rescued = int(target[reject_pos]) in set(map(int, topk[reject_pos][:widths[mismatch_offset]]))
            if rescued:
                rescues["all"] += 1
                if target_dp:
                    rescues["target_dp"] += 1
                else:
                    rescues["target_non_dp"] += 1
                if draft_dp:
                    rescues["draft_dp"] += 1
                else:
                    rescues["draft_non_dp"] += 1
                cursor = reject_pos + 2
            else:
                cursor = reject_pos + 1
        records.append({
            "doc": int(docs[begin]), "tokens": int(end - begin), "calls": calls,
            "drafted": drafted, "extra_nodes": extra,
            **{f"first_rejections_{key}": value for key, value in first_rejections.items()},
            **{f"rescues_{key}": value for key, value in rescues.items()},
        })
    return pd.DataFrame(records)


def _summarize(frame: pd.DataFrame, policy: str, threshold: float | None, width: int, mode: str,
               rng: np.random.Generator) -> dict[str, float | str | int | None]:
    tokens = frame.tokens.to_numpy(dtype=np.float64)
    calls = frame.calls.to_numpy(dtype=np.float64)
    drafted = frame.drafted.to_numpy(dtype=np.float64)
    extra = frame.extra_nodes.to_numpy(dtype=np.float64)
    n = len(frame)

    def calc(mult: np.ndarray):
        sum_calls = np.sum(mult * calls)
        results = {
            "tokens_per_target_call": float(np.sum(mult * tokens) / sum_calls),
            "drafted_tokens_per_target_call": float(np.sum(mult * drafted) / sum_calls),
            "extra_verified_nodes_per_target_call": float(np.sum(mult * extra) / sum_calls),
        }
        for category in ("all", "target_dp", "target_non_dp", "draft_dp", "draft_non_dp"):
            denominator = np.sum(mult * frame[f"first_rejections_{category}"].to_numpy())
            numerator = np.sum(mult * frame[f"rescues_{category}"].to_numpy())
            results[f"rescue_rate_{category}"] = float(numerator / denominator) if denominator else float("nan")
        return results

    point = calc(np.ones(n))
    samples = {key: [] for key in point}
    for _ in range(BOOTSTRAPS):
        sample_ids = rng.integers(0, n, size=n)
        mult = np.bincount(sample_ids, minlength=n).astype(np.float64)
        for key, value in calc(mult).items():
            if np.isfinite(value):
                samples[key].append(value)
    out = {"policy": policy, "threshold": threshold, "width": width, "branch_mode": mode,
           "documents": n, "tokens": int(tokens.sum()), "target_calls": int(calls.sum()),
           **point}
    for key, values in samples.items():
        if values:
            out[f"{key}_ci_low"] = float(np.percentile(values, 2.5))
            out[f"{key}_ci_high"] = float(np.percentile(values, 97.5))
    return out


def _load_frame(pair: str, topk_root: Path) -> pd.DataFrame:
    target, draft = PAIRS[pair]
    directory = _topk_path(pair, topk_root)
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Draft top-k shards missing for {pair} at {directory}")
    top = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    pair_data = load_pair("generated", target, draft, common_only=True)
    merged = pair_data.merge(
        top[["pool_index", "position", "target_token_id", "top8_ids"]],
        left_on=["doc", "pos"], right_on=["pool_index", "position"],
        how="inner", validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError(f"No common positions joined for {pair}")
    merged = merged.rename(columns={"target_token_id": "topk_target"})
    if not np.array_equal(merged.y.to_numpy(dtype=np.int64), merged.topk_target.to_numpy(dtype=np.int64)):
        raise RuntimeError(f"Frozen target continuations differ from top-k logs for {pair}")
    if not np.array_equal(merged.d_arg.to_numpy(dtype=np.int64), np.asarray([row[0] for row in merged.top8_ids])):
        raise RuntimeError(f"Draft top-1 IDs differ from teacher-forced logs for {pair}")
    return merged.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs="+", choices=tuple(PAIRS), default=list(PAIRS))
    parser.add_argument("--topk-root", type=Path, default=ROOT / "results/rebuttal/draft_topk")
    parser.add_argument("--latency", type=Path, default=ROOT / "results/rebuttal/latency/cost_ratio.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/tree")
    args = parser.parse_args()
    cost_data = pd.read_csv(args.latency)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gain_rows = []
    for pair in args.pair:
        frame = _load_frame(pair, args.topk_root.resolve())
        dev_docs = set(frame.loc[frame.doc % 2 == 0, "doc"].astype(int))
        test = frame[frame.doc % 2 == 1].reset_index(drop=True)
        cost_row = cost_data[cost_data.pair == pair]
        if cost_row.empty:
            raise RuntimeError(f"No measured cost ratio for {pair} in {args.latency}")
        threshold = _tune_margin(frame, dev_docs, float(cost_row.iloc[0].c_measured))
        print(f"TREE_THRESHOLD pair={pair} margin={threshold:.9g} dev_docs={len(dev_docs)} test_docs={test.doc.nunique()}", flush=True)
        rng = np.random.default_rng(SEED)
        results = {}
        per_doc_variants = {}
        for policy, tau in (("fixed_k4", None), ("margin_tuned", threshold)):
            for width in (1, 2, 3, 4, 8):
                modes = ("everywhere",) if width == 1 else ("everywhere", "draft_dp_only")
                for mode in modes:
                    per_doc = _run_variant(test, policy, tau, width, mode)
                    result = _summarize(per_doc, policy, tau, width, mode, rng)
                    result["pair"] = pair
                    rows.append(result)
                    results[(policy, width, mode)] = result
                    per_doc_variants[(policy, width, mode)] = per_doc
        for policy in ("fixed_k4", "margin_tuned"):
            base_frame = per_doc_variants[(policy, 1, "everywhere")]
            base = results[(policy, 1, "everywhere")]["tokens_per_target_call"]
            bootstrap_ids = rng.integers(0, len(base_frame), size=(BOOTSTRAPS, len(base_frame)))
            for width in (2, 3, 4, 8):
                all_row = results[(policy, width, "everywhere")]
                dp_row = results[(policy, width, "draft_dp_only")]
                all_gain = all_row["tokens_per_target_call"] - base
                dp_gain = dp_row["tokens_per_target_call"] - base
                all_frame = per_doc_variants[(policy, width, "everywhere")]
                dp_frame = per_doc_variants[(policy, width, "draft_dp_only")]
                base_tokens = base_frame.tokens.to_numpy(dtype=np.float64)
                base_calls = base_frame.calls.to_numpy(dtype=np.float64)
                all_extra = all_frame.extra_nodes.to_numpy(dtype=np.float64)
                all_calls = all_frame.calls.to_numpy(dtype=np.float64)
                dp_extra = dp_frame.extra_nodes.to_numpy(dtype=np.float64)
                dp_calls = dp_frame.calls.to_numpy(dtype=np.float64)
                all_tokens = all_frame.tokens.to_numpy(dtype=np.float64)
                dp_tokens = dp_frame.tokens.to_numpy(dtype=np.float64)
                gain_samples, node_samples = [], []
                for selected in bootstrap_ids:
                    mult = np.bincount(selected, minlength=len(base_frame)).astype(np.float64)
                    base_l = np.sum(mult * base_tokens) / np.sum(mult * base_calls)
                    all_l = np.sum(mult * all_tokens) / np.sum(mult * all_calls)
                    dp_l = np.sum(mult * dp_tokens) / np.sum(mult * dp_calls)
                    all_d = np.sum(mult * all_extra) / np.sum(mult * all_calls)
                    dp_d = np.sum(mult * dp_extra) / np.sum(mult * dp_calls)
                    all_delta = all_l - base_l
                    if all_delta:
                        gain_samples.append((dp_l - base_l) / all_delta)
                    if all_d:
                        node_samples.append(dp_d / all_d)
                gain_rows.append({
                    "pair": pair, "policy": policy, "width": width,
                    "everywhere_gain_tokens_per_call": all_gain,
                    "draft_dp_only_gain_tokens_per_call": dp_gain,
                    "gain_fraction_kept": dp_gain / all_gain if all_gain else float("nan"),
                    "gain_fraction_kept_ci_low": float(np.percentile(gain_samples, 2.5)) if gain_samples else float("nan"),
                    "gain_fraction_kept_ci_high": float(np.percentile(gain_samples, 97.5)) if gain_samples else float("nan"),
                    "extra_nodes_fraction": dp_row["extra_verified_nodes_per_target_call"] /
                                            all_row["extra_verified_nodes_per_target_call"]
                                            if all_row["extra_verified_nodes_per_target_call"] else float("nan"),
                    "extra_nodes_fraction_ci_low": float(np.percentile(node_samples, 2.5)) if node_samples else float("nan"),
                    "extra_nodes_fraction_ci_high": float(np.percentile(node_samples, 97.5)) if node_samples else float("nan"),
                })
    summary = pd.DataFrame(rows)
    gains = pd.DataFrame(gain_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    gains.to_csv(output_dir / "gain_fractions.csv", index=False)
    lines = ["# Comb-tree replay", "",
             "Intervals are 95% paired document-bootstrap intervals (500 resamples). Width 1 is the no-tree baseline.",
             "The logs support only a comb tree: sibling branches are not extended beyond the mismatch position.", "",
             "| Pair | Policy | Width | Branching | Tokens/call [95% CI] | Extra nodes/call [95% CI] | Rescue all | Target DP / non-DP | Draft DP / non-DP |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['pair']} | {row['policy']} | {row['width']} | {row['branch_mode']} | "
            f"{row['tokens_per_target_call']:.3f} [{row['tokens_per_target_call_ci_low']:.3f}, {row['tokens_per_target_call_ci_high']:.3f}] | "
            f"{row['extra_verified_nodes_per_target_call']:.3f} [{row['extra_verified_nodes_per_target_call_ci_low']:.3f}, {row['extra_verified_nodes_per_target_call_ci_high']:.3f}] | "
            f"{row['rescue_rate_all']:.3f} | {row['rescue_rate_target_dp']:.3f} / {row['rescue_rate_target_non_dp']:.3f} | "
            f"{row['rescue_rate_draft_dp']:.3f} / {row['rescue_rate_draft_non_dp']:.3f} |"
        )
    lines += ["", "## Draft-decision-point branching retention", "",
              "| Pair | Policy | Width | Gain kept [95% CI] | Fraction of extra nodes [95% CI] |", "|---|---|---:|---:|---:|"]
    for row in gain_rows:
        lines.append(f"| {row['pair']} | {row['policy']} | {row['width']} | {row['gain_fraction_kept']:.3f} [{row['gain_fraction_kept_ci_low']:.3f}, {row['gain_fraction_kept_ci_high']:.3f}] | {row['extra_nodes_fraction']:.3f} [{row['extra_nodes_fraction_ci_low']:.3f}, {row['extra_nodes_fraction_ci_high']:.3f}] |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
