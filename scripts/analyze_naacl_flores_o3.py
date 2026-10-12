#!/usr/bin/env python3
"""Analyze the optional NAACL O3 FLORES target/draft grid.

The estimates are saved-continuation k=4 replay associations. They are kept
separate from the Korean FineWeb2 primary sample and are not live latency or
causal estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from analyze_naacl_cpu import eligible_cross_within, fit_cross_ame, load_pair_frame
from analyze_naacl_rq2 import pairwise_stability, summarize_boundaries
from src.naacl_experiment import atomic_json, load_config


PAIR_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "G1", "M1"]


def _fit(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    try:
        return {"fit_status": "COMPLETE", **fit_cross_ame(frame, label=label)}
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return {
            "fit_status": "NOT_ESTIMABLE",
            "status": "NOT_ESTIMABLE",
            "error": f"{type(exc).__name__}: {exc}",
            "n_rows_before_fit": int(len(frame)),
            "n_docs_before_fit": int(frame.doc_id.nunique()) if "doc_id" in frame else 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_flores_o3.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/naacl_experiments/flores_optional")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/naacl_experiments/flores_optional/analysis")
    args = parser.parse_args()

    config = load_config(args.config)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    global_rows: list[dict[str, Any]] = []
    boundary_frames: list[pd.DataFrame] = []
    join_meta: list[dict[str, Any]] = []

    for pair_id in PAIR_ORDER:
        print(f"FLORES_O3_LOAD pair={pair_id}", flush=True)
        frame, meta = load_pair_frame(args.output_root, pair_id, config, split_override="common")
        usable = eligible_cross_within(frame)
        join_meta.append(meta)
        pair = config["pairs"][pair_id]

        for estimand, subset in (
            ("global", usable),
            ("nominal_to_particle", usable.loc[
                (usable.alignment_class.eq("CROSS") & usable.coarse_boundary.eq("NOMINAL_TO_PARTICLE"))
                | (usable.alignment_class.eq("WITHIN_SPLIT") & usable.nearest_coarse_boundary.eq("NOMINAL_TO_PARTICLE"))
            ]),
        ):
            fitted = _fit(subset, f"FLORES-O3-{pair_id}-{estimand}")
            global_rows.append({
                "workload": "FLORES-200 en→ko devtest",
                "pair_id": pair_id,
                "draft": pair["draft"],
                "target": pair["target"],
                "estimand": estimand,
                "documents_joined": meta["documents"],
                "eligible_k4_proposals": meta["eligible_k4_proposals"],
                "aligned_event_positions": meta["event_positions_with_alignment"],
                "AME_pp": 100 * fitted["AME_cross_minus_within"] if pd.notna(fitted.get("AME_cross_minus_within")) else None,
                "CI_low_95_pp": 100 * fitted["CI_low_95"] if pd.notna(fitted.get("CI_low_95")) else None,
                "CI_high_95_pp": 100 * fitted["CI_high_95"] if pd.notna(fitted.get("CI_high_95")) else None,
                **fitted,
            })

        setting = f"flores_generated_k4:{pair_id}"
        for boundary_level in ("coarse", "fine"):
            effects = summarize_boundaries(
                usable,
                pair_id=pair_id,
                setting=setting,
                outcome_type="k4_replay_rejection",
                boundary_level=boundary_level,
            )
            if not effects.empty:
                boundary_frames.append(effects)
        print(
            f"FLORES_O3_READY pair={pair_id} docs={meta['documents']:,} "
            f"eligible={meta['eligible_k4_proposals']:,} aligned={meta['event_positions_with_alignment']:,}",
            flush=True,
        )

    global_path = args.analysis_dir / "rq2_flores_o3_cross_within.csv"
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(global_path, index=False)
    boundary_path = args.analysis_dir / "rq2_flores_o3_boundary_effects.csv"
    boundary_frame = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else pd.DataFrame()
    boundary_frame.to_csv(boundary_path, index=False)

    core_path = ROOT / "results/naacl_experiments/analysis/rq2_boundary_effects.csv"
    stability_path = args.analysis_dir / "rq2_flores_o3_setting_stability.csv"
    stability_meta: dict[str, Any] = {"status": "CORE_BOUNDARY_EFFECTS_MISSING"}
    if core_path.is_file() and not boundary_frame.empty:
        core = pd.read_csv(core_path)
        core = core.loc[core.setting.astype(str).str.startswith("core_generated_k4:")]
        selected_core_pairs = set(PAIR_ORDER)
        core = core.loc[core.pair_id.astype(str).isin(selected_core_pairs)]
        combined = pd.concat([core, boundary_frame], ignore_index=True)
        stability = pairwise_stability(combined)
        stability.to_csv(stability_path, index=False)
        stability_meta = {
            "status": "COMPLETE",
            "core_settings": int(core.setting.nunique()),
            "flores_settings": int(boundary_frame.setting.nunique()),
            "comparison_rows": int(len(stability)),
        }

    metadata = {
        "experiment": "NAACL optional O3 FLORES-200 English-to-Korean domain extension",
        "status": "COMPLETE" if len(global_frame) == len(PAIR_ORDER) * 2 else "PARTIAL",
        "pair_order": PAIR_ORDER,
        "estimands": ["global CROSS-vs-WITHIN", "NOMINAL_TO_PARTICLE CROSS-vs-nearest-WITHIN"],
        "population": "Target greedy continuations on all 1,012 official FLORES-200 devtest English prompts; the Korean reference is not used in generation, alignment, or inference.",
        "outcome": "k=4 offline greedy speculative-decoding replay rejection; fitted with the primary global controls and document-clustered uncertainty.",
        "scope": "Saved-continuation observational domain analysis. It does not measure live decoding latency, validate diverged contexts, or establish causality.",
        "join_metadata": join_meta,
        "stability": stability_meta,
        "files": {
            "cross_within": str(global_path.relative_to(ROOT)),
            "boundary_effects": str(boundary_path.relative_to(ROOT)),
            "setting_stability": str(stability_path.relative_to(ROOT)),
        },
    }
    atomic_json(args.analysis_dir / "rq2_flores_o3_metadata.json", metadata)
    print(f"FLORES_O3_ANALYSIS_COMPLETE pairs={len(PAIR_ORDER)} fits={len(global_frame)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
