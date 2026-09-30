#!/usr/bin/env python3
"""Create CPU-only diagnostics, paper figures, and the guard method report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/morphology_aware_boundary_guard"
PAIRS = ["P1", "P2", "P3"]
WORKLOADS = ["WIKIPEDIA", "FLORES"]
VARIANTS = [
    "TARGET_ONLY", "FIXED_K_SD", "NP_BOUNDARY_GUARD", "ALL_MORPH_BOUNDARY_GUARD",
    "RANDOM_MATCHED_GUARD", "ENTROPY_MATCHED_GUARD",
]
COLORS = {
    "TARGET_ONLY": "#666666", "FIXED_K_SD": "#0072B2", "NP_BOUNDARY_GUARD": "#D55E00",
    "ALL_MORPH_BOUNDARY_GUARD": "#56B4E9", "RANDOM_MATCHED_GUARD": "#E69F00",
    "ENTROPY_MATCHED_GUARD": "#009E73",
}
MARKERS = {"P1": "o", "P2": "s", "P3": "^"}
SEED = 3091
BOOTSTRAP_REPS = 2000


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    headers = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    return "\n".join([
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
        *["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows],
    ])


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    if "pair" in frame and "model_pair" not in frame:
        frame = frame.rename(columns={"pair": "model_pair"})
    return frame


def _read_prompt_rows(stage: str) -> pd.DataFrame:
    directory = OUT / ("pilot" if stage == "pilot" else "full_benchmark")
    if stage == "full" and (directory / "per_prompt_metrics.parquet").exists():
        frame = pd.read_parquet(directory / "per_prompt_metrics.parquet")
    else:
        checkpoints = directory / "checkpoints"
        parts = [pd.read_csv(path) for path in sorted(checkpoints.glob("*.csv"))] if checkpoints.exists() else []
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if frame.empty:
        return frame
    frame = _standardize(frame)
    for col in ["prompt_id", "output_tokens", "prompt_wall_seconds"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "exact_output_match" in frame:
        frame["exact_output_match"] = frame["exact_output_match"].astype(str).str.lower().isin(["true", "1", "yes"])
    keys = [c for c in ["workload", "model_pair", "variant", "repetition", "prompt_id"] if c in frame]
    if keys:
        frame = frame.drop_duplicates(keys, keep="last")
    return frame


def _read_results(stage: str) -> pd.DataFrame:
    path = OUT / ("pilot/benchmark_results.csv" if stage == "pilot" else "full_benchmark/benchmark_results.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = _standardize(pd.read_csv(path))
    if "all_outputs_exact" in frame:
        frame["all_outputs_exact"] = frame["all_outputs_exact"].astype(str).str.lower().isin(["true", "1", "yes"])
    return frame


def _bootstrap_rate(frame: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    by_prompt = frame.groupby("prompt_id", as_index=False).agg(
        tokens=("output_tokens", "sum"), seconds=("prompt_wall_seconds", "sum"),
    )
    if by_prompt.empty:
        return float("nan"), float("nan"), float("nan")
    tokens = by_prompt.tokens.to_numpy(float)
    seconds = by_prompt.seconds.to_numpy(float)
    estimate = float(tokens.sum() / max(seconds.sum(), 1e-12))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS)
    n = len(by_prompt)
    for b in range(BOOTSTRAP_REPS):
        indices = rng.integers(0, n, size=n)
        draws[b] = tokens[indices].sum() / max(seconds[indices].sum(), 1e-12)
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_speed(
    frame: pd.DataFrame, workload: str, pair: str, variant: str, seed: int,
    baseline_variant: str = "FIXED_K_SD",
) -> dict[str, float]:
    subset = frame.loc[frame.workload.eq(workload) & frame.model_pair.eq(pair)]
    candidate = subset.loc[subset.variant.eq(variant)].groupby("prompt_id", as_index=False).agg(
        tokens_v=("output_tokens", "sum"), seconds_v=("prompt_wall_seconds", "sum"),
    )
    baseline = subset.loc[subset.variant.eq(baseline_variant)].groupby("prompt_id", as_index=False).agg(
        tokens_f=("output_tokens", "sum"), seconds_f=("prompt_wall_seconds", "sum"),
    )
    joined = candidate.merge(baseline, on="prompt_id", how="inner")
    if joined.empty:
        return {"speedup": float("nan"), "delta_tokens_s": float("nan"), "delta_low": float("nan"), "delta_high": float("nan"), "speedup_low": float("nan"), "speedup_high": float("nan")}
    tv, sv = joined.tokens_v.to_numpy(float), joined.seconds_v.to_numpy(float)
    tf, sf = joined.tokens_f.to_numpy(float), joined.seconds_f.to_numpy(float)
    rate_v = tv.sum() / max(sv.sum(), 1e-12)
    rate_f = tf.sum() / max(sf.sum(), 1e-12)
    rng = np.random.default_rng(seed)
    delta_draws = np.empty(BOOTSTRAP_REPS)
    ratio_draws = np.empty(BOOTSTRAP_REPS)
    n = len(joined)
    for b in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        rv = tv[idx].sum() / max(sv[idx].sum(), 1e-12)
        rf = tf[idx].sum() / max(sf[idx].sum(), 1e-12)
        delta_draws[b] = rv - rf
        ratio_draws[b] = rv / max(rf, 1e-12)
    return {
        "speedup": float(rate_v / max(rate_f, 1e-12)),
        "delta_tokens_s": float(rate_v - rate_f),
        "delta_low": float(np.quantile(delta_draws, 0.025)),
        "delta_high": float(np.quantile(delta_draws, 0.975)),
        "speedup_low": float(np.quantile(ratio_draws, 0.025)),
        "speedup_high": float(np.quantile(ratio_draws, 0.975)),
    }


def build_summary(prompt_rows: pd.DataFrame, result_rows: pd.DataFrame) -> pd.DataFrame:
    if prompt_rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (workload, pair, variant), group in prompt_rows.groupby(["workload", "model_pair", "variant"], sort=False):
        rate, low, high = _bootstrap_rate(group, SEED + PAIRS.index(pair) * 100 + WORKLOADS.index(workload) * 10 + VARIANTS.index(variant))
        result_group = result_rows.loc[
            result_rows.workload.eq(workload) & result_rows.model_pair.eq(pair) & result_rows.variant.eq(variant)
        ] if not result_rows.empty else pd.DataFrame()
        record: dict[str, Any] = {
            "workload": workload, "model_pair": pair, "variant": variant,
            "prompts": int(group.prompt_id.nunique()),
            "repetitions": int(group.repetition.nunique()) if "repetition" in group else 1,
            "generated_target_tokens": int(group.output_tokens.sum()),
            "wall_clock_seconds": float(group.prompt_wall_seconds.sum()),
            "throughput_tokens_s": rate, "throughput_ci_low": low, "throughput_ci_high": high,
            "mean_prompt_latency_s": float(group.prompt_wall_seconds.mean()),
            "median_prompt_latency_s": float(group.prompt_wall_seconds.median()),
            "seconds_per_target_token": float(group.prompt_wall_seconds.sum() / max(1, group.output_tokens.sum())),
            "exact": bool(group.exact_output_match.astype(bool).all()) if "exact_output_match" in group else False,
        }
        for col in [
            "mean_accepted_tokens_per_verification_call", "rejection_rate",
            "target_forward_calls_per_output_token", "draft_tokens_proposed_per_output_token",
            "target_proposal_positions_verified_per_output_token", "guard_activation_rate",
            "invalid_or_ambiguous_candidate_fraction", "morphology_detection_seconds",
            "controller_seconds", "draft_resynchronization_seconds",
        ]:
            if col in group:
                numeric = pd.to_numeric(group[col], errors="coerce")
                record[col] = float(numeric.sum()) if col.endswith("_seconds") else float(numeric.mean())
        if not result_group.empty:
            record["mean_repeat_throughput_tokens_s"] = float(result_group.target_tokens_per_second.mean())
            record["median_repeat_throughput_tokens_s"] = float(result_group.target_tokens_per_second.median())
            record["max_peak_allocated_vram_gb"] = float(result_group.peak_allocated_vram_gb.max())
            record["max_peak_reserved_vram_gb"] = float(result_group.peak_reserved_vram_gb.max())
        position_counts: Counter[int] = Counter()
        if "guard_position_histogram_json" in group:
            for value in group.guard_position_histogram_json.dropna():
                try:
                    position_counts.update({int(k): int(v) for k, v in json.loads(value).items()})
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        total_guard_positions = sum(position_counts.values())
        record["guard_position_distribution_json"] = json.dumps({
            str(slot): {"count": count, "share": count / total_guard_positions}
            for slot, count in sorted(position_counts.items())
        }, separators=(",", ":")) if total_guard_positions else "{}"
        if variant != "FIXED_K_SD" and variant != "TARGET_ONLY":
            record.update(_paired_speed(prompt_rows, workload, pair, variant, SEED + 500 + PAIRS.index(pair) * 100 + WORKLOADS.index(workload) * 10 + VARIANTS.index(variant)))
            target_speed = _paired_speed(
                prompt_rows, workload, pair, variant,
                SEED + 750 + PAIRS.index(pair) * 100 + WORKLOADS.index(workload) * 10 + VARIANTS.index(variant),
                baseline_variant="TARGET_ONLY",
            )
            record["speedup_vs_target_only"] = target_speed["speedup"]
            record["targetonly_speedup_low"] = target_speed["speedup_low"]
            record["targetonly_speedup_high"] = target_speed["speedup_high"]
        elif variant == "FIXED_K_SD":
            record.update({"speedup": 1.0, "delta_tokens_s": 0.0, "delta_low": 0.0, "delta_high": 0.0, "speedup_low": 1.0, "speedup_high": 1.0})
            target_speed = _paired_speed(prompt_rows, workload, pair, variant, SEED + 900, baseline_variant="TARGET_ONLY")
            record["speedup_vs_target_only"] = target_speed["speedup"]
            record["targetonly_speedup_low"] = target_speed["speedup_low"]
            record["targetonly_speedup_high"] = target_speed["speedup_high"]
        else:
            record.update({"speedup": float("nan"), "delta_tokens_s": float("nan"), "delta_low": float("nan"), "delta_high": float("nan"), "speedup_low": float("nan"), "speedup_high": float("nan")})
            record.update({"speedup_vs_target_only": 1.0, "targetonly_speedup_low": 1.0, "targetonly_speedup_high": 1.0})
        records.append(record)
    return pd.DataFrame(records)


def _format_interval(mean: Any, low: Any, high: Any, digits: int = 2) -> str:
    if pd.isna(mean):
        return "—"
    return f"{float(mean):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _make_figures(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.fontsize": 7.5, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
        "grid.linestyle": "-", "savefig.bbox": "tight", "figure.dpi": 300,
    })
    order = VARIANTS
    labels = {
        "TARGET_ONLY": "Target only", "FIXED_K_SD": "Fixed K",
        "NP_BOUNDARY_GUARD": "NP guard", "ALL_MORPH_BOUNDARY_GUARD": "All-boundary",
        "RANDOM_MATCHED_GUARD": "Random matched", "ENTROPY_MATCHED_GUARD": "Entropy matched",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=False)
    for r, workload in enumerate(WORKLOADS):
        for c, pair in enumerate(PAIRS):
            ax = axes[r, c]
            subset = summary.loc[summary.workload.eq(workload) & summary.model_pair.eq(pair)].set_index("variant")
            x = np.arange(len(order))
            for i, variant in enumerate(order):
                if variant not in subset.index:
                    continue
                row = subset.loc[variant]
                if pd.isna(row.throughput_tokens_s):
                    continue
                low = max(0.0, float(row.throughput_tokens_s - row.throughput_ci_low))
                high = max(0.0, float(row.throughput_ci_high - row.throughput_tokens_s))
                ax.errorbar(
                    x[i], row.throughput_tokens_s, yerr=np.array([[low], [high]]),
                    fmt="o", markersize=4.2, capsize=2.5, color=COLORS[variant],
                    ecolor=COLORS[variant], linewidth=1.1, zorder=3,
                )
            ax.set_title(f"{workload.title()} · {pair}")
            ax.set_xticks(x)
            ax.set_xticklabels([labels[v] for v in order], rotation=32, ha="right", fontsize=7)
            ax.set_ylabel("Target tokens / second")
            ax.grid(axis="x", visible=False)
    fig.suptitle("End-to-end throughput by decoding policy (prompt-cluster bootstrap 95% CI)", y=1.01, fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "throughput_by_variant.png", dpi=300)
    fig.savefig(figdir / "throughput_by_variant.pdf")
    plt.close(fig)

    sd = summary.loc[summary.variant.ne("TARGET_ONLY")].copy()
    components = ["morphology_detection_seconds", "controller_seconds", "draft_resynchronization_seconds"]
    if not sd.empty:
        sd["morphology_only_s_per_token"] = sd.get("morphology_detection_seconds", 0.0) / sd.generated_target_tokens.clip(lower=1)
        sd["controller_selection_s_per_token"] = (
            sd.get("controller_seconds", 0.0) - sd.get("morphology_detection_seconds", 0.0)
        ).clip(lower=0) / sd.generated_target_tokens.clip(lower=1)
        sd["draft_prefill_s_per_token"] = sd.get("draft_resynchronization_seconds", 0.0) / sd.generated_target_tokens.clip(lower=1)
        sd["other_end_to_end_s_per_token"] = (
            sd.seconds_per_target_token - sd.morphology_only_s_per_token
            - sd.controller_selection_s_per_token - sd.draft_prefill_s_per_token
        ).clip(lower=0)
        plot_rows = sd.groupby(["workload", "variant"], as_index=False)[[
            "morphology_only_s_per_token", "controller_selection_s_per_token",
            "draft_prefill_s_per_token", "other_end_to_end_s_per_token",
        ]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=False)
        component_colors = ["#0072B2", "#E69F00", "#009E73", "#B8B8B8"]
        component_labels = ["Morphology projection", "Controller selection", "Draft prefix prefill", "Other end-to-end time"]
        for ax, workload in zip(axes, WORKLOADS):
            part = plot_rows.loc[plot_rows.workload.eq(workload)].set_index("variant").reindex([v for v in order if v != "TARGET_ONLY"])
            bottom = np.zeros(len(part))
            for col, color, label in zip([
                "morphology_only_s_per_token", "controller_selection_s_per_token",
                "draft_prefill_s_per_token", "other_end_to_end_s_per_token",
            ], component_colors, component_labels):
                values = part[col].fillna(0).to_numpy(float)
                ax.bar(np.arange(len(part)), values, bottom=bottom, color=color, label=label, width=0.72)
                bottom += values
            ax.set_title(workload.title())
            ax.set_xticks(np.arange(len(part)))
            ax.set_xticklabels([labels[v] for v in part.index], rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Seconds / generated target token")
            ax.grid(axis="x", visible=False)
        axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
        fig.suptitle("Instrumented end-to-end time breakdown", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / "guard_overhead_breakdown.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, workload in zip(axes, WORKLOADS):
        part = summary.loc[summary.workload.eq(workload) & summary.variant.isin(order[1:])]
        for variant in order[1:]:
            rows = part.loc[part.variant.eq(variant)]
            for row in rows.itertuples(index=False):
                accepted = getattr(row, "mean_accepted_tokens_per_verification_call", float("nan"))
                speedup = getattr(row, "speedup", float("nan"))
                if pd.isna(accepted) or pd.isna(speedup):
                    continue
                ax.scatter(
                    accepted, speedup,
                    color=COLORS[variant], marker=MARKERS[row.model_pair], s=34,
                    edgecolor="white", linewidth=0.5,
                    label=labels[variant] if row.model_pair == PAIRS[0] else None,
                )
                ax.annotate(row.model_pair, (row.mean_accepted_tokens_per_verification_call, row.speedup),
                            xytext=(3, 3), textcoords="offset points", fontsize=6.5)
        ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
        ax.set_title(workload.title())
        ax.set_xlabel("Accepted draft tokens / verification call")
        ax.set_ylabel("Throughput speedup vs fixed K")
        ax.grid(alpha=0.18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(figdir / "accepted_length_vs_speed.png", dpi=300)
    plt.close(fig)


def _correctness_summary() -> tuple[pd.DataFrame, bool, int]:
    path = OUT / "correctness/greedy_equality.csv"
    if not path.exists():
        return pd.DataFrame(), False, 0
    frame = _standardize(pd.read_csv(path))
    if "exact_output_match" in frame:
        frame["exact_output_match"] = frame["exact_output_match"].astype(str).str.lower().isin(["true", "1", "yes"])
    groups = frame.groupby(["workload", "model_pair", "variant"]).agg(
        prompts=("prompt_id", "nunique"), exact=("exact_output_match", "all"),
    ).reset_index()
    expected = len(WORKLOADS) * len(PAIRS) * len(VARIANTS)
    passed = len(groups) == expected and bool((groups.prompts >= 100).all()) and bool(groups.exact.all())
    return groups, passed, int(len(frame))


def _decision(config: dict[str, Any], stage: str, correctness_pass: bool, summary: pd.DataFrame) -> tuple[str, str]:
    stage_text = str(config.get("stage", ""))
    if not correctness_pass:
        return "INCONCLUSIVE", "The required 100-prompt-per-cell greedy equality suite is incomplete or contains a mismatch."
    pilot = _read_results("pilot")
    if not pilot.empty and "NP_BOUNDARY_GUARD" in set(pilot.variant):
        n = pilot.loc[pilot.variant.eq("NP_BOUNDARY_GUARD")]
        f = pilot.loc[pilot.variant.eq("FIXED_K_SD"), ["workload", "model_pair", "repetition", "wall_clock_seconds"]].rename(columns={"wall_clock_seconds": "fixed_wall"})
        joined = n.merge(f, on=["workload", "model_pair", "repetition"], how="inner")
        if not joined.empty:
            ratios = joined.assign(ratio=joined.wall_clock_seconds / joined.fixed_wall).groupby(["workload", "model_pair"]).ratio.mean()
            if len(ratios) == len(WORKLOADS) * len(PAIRS) and bool((ratios >= 1.02).all()):
                return "NEGATIVE", "The prespecified pilot gate stopped full evaluation: NP guard was more than 2% slower in every workload/model-pair cell."
    full_path = OUT / "full_benchmark/benchmark_results.csv"
    full = _read_results("full")
    if full_path.exists() and len(full) >= len(WORKLOADS) * len(PAIRS) * len(VARIANTS) * 3:
        if summary.empty:
            return "INCONCLUSIVE", "Full benchmark rows exist but per-prompt timing data are missing."
        np_rows = summary.loc[summary.variant.eq("NP_BOUNDARY_GUARD")]
        positive_cells = np_rows.loc[(np_rows.delta_low > 0) & (np_rows.speedup > 1.0)]
        positive_workload = any(positive_cells.loc[positive_cells.workload.eq(w)].model_pair.nunique() >= 2 for w in WORKLOADS)
        random_rows = summary.loc[summary.variant.eq("RANDOM_MATCHED_GUARD"), ["workload", "model_pair", "speedup"]].rename(columns={"speedup": "random_speedup"})
        all_rows = summary.loc[summary.variant.eq("ALL_MORPH_BOUNDARY_GUARD"), ["workload", "model_pair", "speedup", "delta_low"]].rename(columns={"speedup": "all_speedup", "delta_low": "all_delta_low"})
        compare = positive_cells.merge(random_rows, on=["workload", "model_pair"], how="left").merge(all_rows, on=["workload", "model_pair"], how="left")
        not_matched = bool(len(compare) and (compare.speedup > compare.random_speedup).all())
        all_not_equally_convincing = not any(
            ((all_rows.workload == w) & (all_rows.all_delta_low > 0)).sum() >= 2
            and ((all_rows.workload == w) & (all_rows.all_speedup >= np_rows.loc[np_rows.workload.eq(w), "speedup"].max())).any()
            for w in WORKLOADS
        )
        if positive_workload and not_matched and all_not_equally_convincing:
            return "POSITIVE", "NP guard improved measured end-to-end throughput with prompt-clustered uncertainty in the prespecified multi-pair comparison."
        if np_rows.empty or bool((np_rows.speedup <= 1.0).all()):
            return "NEGATIVE", "NP guard did not improve end-to-end throughput in any full-benchmark workload/model-pair cell."
        return "INCONCLUSIVE", "Some point estimates favored NP guard, but the full evidence did not satisfy every prespecified positive decision criterion."
    if "PILOT_NEGATIVE" in stage_text:
        return "NEGATIVE", "The prespecified pilot gate stopped the full run because every NP-versus-fixed-K cell was more than 2% slower."
    return "INCONCLUSIVE", f"The staged run is incomplete ({stage_text or 'no stage metadata'}); no final speed conclusion is available yet."


def generate_report(stage: str) -> None:
    config_path = OUT / "method_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    prompt_rows = _read_prompt_rows(stage) if stage in {"pilot", "full"} else pd.DataFrame()
    result_rows = _read_results(stage) if stage in {"pilot", "full"} else pd.DataFrame()
    summary = build_summary(prompt_rows, result_rows)
    if not summary.empty:
        summary.to_csv(OUT / f"{stage}_analysis_summary.csv", index=False)
        _make_figures(summary)
    correctness, correctness_pass, correctness_n = _correctness_summary()
    if len(correctness):
        correctness.to_csv(OUT / "correctness/correctness_coverage_summary.csv", index=False)
    result, conclusion = _decision(config, stage, correctness_pass, summary)

    if not summary.empty:
        diag = summary.copy()
        diag["throughput [95% CI]"] = [
            _format_interval(r.throughput_tokens_s, r.throughput_ci_low, r.throughput_ci_high)
            for r in diag.itertuples(index=False)
        ]
        diag["speedup [95% CI]"] = [
            _format_interval(r.speedup, r.speedup_low, r.speedup_high, 3)
            for r in diag.itertuples(index=False)
        ]
        cols = ["workload", "model_pair", "variant", "prompts", "repetitions", "throughput [95% CI]", "median_prompt_latency_s", "speedup [95% CI]", "speedup_vs_target_only", "mean_accepted_tokens_per_verification_call", "rejection_rate", "target_forward_calls_per_output_token", "draft_tokens_proposed_per_output_token", "target_proposal_positions_verified_per_output_token", "guard_activation_rate", "guard_position_distribution_json", "invalid_or_ambiguous_candidate_fraction", "morphology_detection_seconds", "controller_seconds", "draft_resynchronization_seconds", "exact"]
        cols = [c for c in cols if c in diag]
        diag[cols].to_csv(OUT / f"{stage}_analysis_diagnostics.csv", index=False)
        (OUT / ("full_benchmark/variant_diagnostics.md" if stage == "full" else "pilot/variant_diagnostics.md")).write_text(
            f"# {stage.title()} variant diagnostics\n\n"
            "Rates are computed from synchronized end-to-end prompt timings. Confidence intervals resample prompt IDs as clusters, preserving repeated measurements within prompt. The decomposition timers are component instrumentation; total end-to-end wall time is the authoritative throughput measure.\n\n"
            + markdown_table(diag[cols].round(4)) + "\n", encoding="utf-8",
        )

    cfg_gpu = config.get("gpu_metadata") or {}
    correctness_text = "No full correctness coverage table is available yet."
    if len(correctness):
        correctness_text = markdown_table(correctness)
    artifact_parity = config.get("batched_verification_experiment", {})
    off_path = OUT / "feasibility/offline_guard_opportunity.csv"
    off_text = "Offline opportunity audit not yet available."
    if off_path.exists():
        off = pd.read_csv(off_path)
        if len(off):
            group_cols = [c for c in ["workload", "pair", "policy"] if c in off]
            off_summary = off.groupby(group_cols, dropna=False).agg(
                candidate_blocks=("prompt_id", "size"), prompts=("prompt_id", "nunique"),
                reachable_share=("reachable_before_or_at_first_rejection", "mean"),
                mean_positions_after=("verifier_positions_after_candidate", "mean"),
            ).reset_index()
            off_text = markdown_table(off_summary.round(3))
    audit_path = OUT / "correctness/guard_decision_audit.csv"
    guard_text = "No guard decision audit has been generated yet."
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if len(audit):
            cols = [c for c in ["guard_variant", "detector_status", "guard_reason", "selected"] if c in audit]
            guard_text = markdown_table(audit.groupby(cols, dropna=False).size().rename("rows").reset_index()) if cols else f"{len(audit):,} guard audit rows."
    peak_alloc = float(result_rows.peak_allocated_vram_gb.max()) if not result_rows.empty and "peak_allocated_vram_gb" in result_rows else float("nan")
    peak_reserved = float(result_rows.peak_reserved_vram_gb.max()) if not result_rows.empty and "peak_reserved_vram_gb" in result_rows else float("nan")
    if pd.isna(peak_alloc):
        peak_alloc = float("nan")
    if pd.isna(peak_reserved):
        peak_reserved = float("nan")

    lines = [
        f"Method result: {result} — {conclusion}", "",
        "## Implementation and exactness", "",
        f"The run uses greedy, cached speculative decoding with K=4 and the pinned model pairs. Exact target token IDs are the invariant; no logits or weights are changed. The official path uses sequential target verification. Batched verification was rejected after an exact parity mismatch ({artifact_parity.get('cases_passed', '—')}/{artifact_parity.get('cases_completed', '—')} comparisons passed; first mismatch WIKIPEDIA/P1/FIXED_K_SD prompt 25) and is not used for benchmark results.", "",
        f"Greedy equality rows: {correctness_n:,}; required full correctness passed: **{correctness_pass}**. The table shows the current prompt coverage by workload, model pair and variant.", "",
        correctness_text, "",
        "## Stage and offline opportunity", "",
        f"Current stage: `{config.get('stage', 'unknown')}`. The offline audit uses saved baseline traces only; reachable share indicates a candidate appeared before or at the first baseline rejection, not an online speed result.", "",
        off_text, "",
        "## Benchmark results", "",
        f"Analyzed stage: `{stage if not summary.empty else 'none'}`. No throughput claim is made until the correctness gate and timing stage complete.", "",
    ]
    if summary.empty:
        lines.append("No complete pilot/full benchmark timing table is available yet.")
    else:
        display = summary.copy()
        display["tokens/s [95% CI]"] = [
            _format_interval(r.throughput_tokens_s, r.throughput_ci_low, r.throughput_ci_high)
            for r in display.itertuples(index=False)
        ]
        display["speedup vs fixed K [95% CI]"] = [
            _format_interval(r.speedup, r.speedup_low, r.speedup_high, 3)
            for r in display.itertuples(index=False)
        ]
        display["speedup vs target-only"] = display.speedup_vs_target_only
        cols = ["workload", "model_pair", "variant", "prompts", "repetitions", "tokens/s [95% CI]", "median_prompt_latency_s", "speedup vs fixed K [95% CI]", "speedup vs target-only", "mean_accepted_tokens_per_verification_call", "rejection_rate", "target_forward_calls_per_output_token", "draft_tokens_proposed_per_output_token", "target_proposal_positions_verified_per_output_token", "guard_activation_rate", "guard_position_distribution_json", "invalid_or_ambiguous_candidate_fraction", "morphology_detection_seconds", "controller_seconds", "draft_resynchronization_seconds", "exact"]
        cols = [c for c in cols if c in display]
        lines.append(markdown_table(display[cols].round(4)))
    lines += [
        "", "## Guard audit and runtime", "", guard_text, "",
        f"GPU metadata: physical index `{cfg_gpu.get('physical_gpu_index', '—')}`, `{cfg_gpu.get('GPU_name', '—')}`, `CUDA_VISIBLE_DEVICES={cfg_gpu.get('CUDA_VISIBLE_DEVICES', '—')}`, framework device `{cfg_gpu.get('framework_local_device', '—')}`, driver `{cfg_gpu.get('driver_version', '—')}`, CUDA `{cfg_gpu.get('torch_cuda_runtime_version', cfg_gpu.get('nvidia_smi_cuda_version', '—'))}`, PyTorch `{cfg_gpu.get('torch_version', '—')}`, Transformers `{cfg_gpu.get('transformers_version', '—')}`, attention `{cfg_gpu.get('attention_backend', '—')}`, dtype `{cfg_gpu.get('dtype', '—')}`.",
        f"Peak allocated/reserved VRAM in analyzed benchmark rows: {peak_alloc:.2f}/{peak_reserved:.2f} GiB. GPU utilization samples are in `gpu3_utilization.csv`.", "",
        "## Limitations", "",
        "The correctness suite and end-to-end timing gate determine whether the policy is useful. Projection/controller/re-prefill timers are instrumentation, while synchronized total prompt wall time is authoritative. The optional batched target verifier failed discrete parity and was disabled. Wikipedia and FLORES are reported separately; observational H3/H4 morphology associations are not reinterpreted as causal evidence.", "",
        "Paper-ready interpretation: The experiment evaluates whether an exact morphology-aware scheduling rule changes end-to-end speculative-decoding throughput under fixed model, prompt, and hardware settings. Any conclusion is limited to the tested Korean Wikipedia and English-to-Korean FLORES workloads and uses the observed decoding schedule as the intervention.", "",
    ]
    (OUT / "method_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["auto", "pilot", "full"], default="auto")
    args = parser.parse_args()
    if args.stage == "auto":
        stage = "full" if (OUT / "full_benchmark/benchmark_results.csv").exists() else "pilot"
    else:
        stage = args.stage
    generate_report(stage)
    print(f"Wrote {OUT / 'method_summary.md'}")


if __name__ == "__main__":
    main()
