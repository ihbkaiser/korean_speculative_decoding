#!/usr/bin/env python3
"""Analyze the optional Gemma-3-1B to Gemma-3-12B scale comparison."""

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
from src.naacl_experiment import atomic_json, load_config, output_root


PAIR_ID = "O4_G1_12"


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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_o4_gemma1.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if PAIR_ID not in config["pairs"]:
        parser.error(f"config does not define {PAIR_ID}")
    root = output_root(config, args.output_root)
    analysis_dir = args.analysis_dir or root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pair = config["pairs"][PAIR_ID]
    print(f"O4_LOAD pair={PAIR_ID}", flush=True)
    frame, join_meta = load_pair_frame(root, PAIR_ID, config, split_override="common")
    usable = eligible_cross_within(frame)
    subsets = (
        ("global", usable),
        (
            "nominal_to_particle",
            usable.loc[
                (usable.alignment_class.eq("CROSS")
                 & usable.coarse_boundary.eq("NOMINAL_TO_PARTICLE"))
                | (usable.alignment_class.eq("WITHIN_SPLIT")
                   & usable.nearest_coarse_boundary.eq("NOMINAL_TO_PARTICLE"))
            ],
        ),
    )

    result_rows: list[dict[str, Any]] = []
    for estimand, subset in subsets:
        fitted = _fit(subset, f"O4-{PAIR_ID}-{estimand}")
        ame = fitted.get("AME_cross_minus_within")
        low, high = fitted.get("CI_low_95"), fitted.get("CI_high_95")
        result_rows.append({
            "workload": "Korean FineWeb2 common 20k; Gemma-3-12B-pt continuation",
            "pair_id": PAIR_ID,
            "draft": pair["draft"],
            "target": pair["target"],
            "estimand": estimand,
            "documents_joined": join_meta["documents"],
            "eligible_k4_proposals": join_meta["eligible_k4_proposals"],
            "aligned_event_positions": join_meta["event_positions_with_alignment"],
            "AME_pp": 100 * ame if pd.notna(ame) else None,
            "CI_low_95_pp": 100 * low if pd.notna(low) else None,
            "CI_high_95_pp": 100 * high if pd.notna(high) else None,
            **fitted,
        })
    del frame, usable

    result = pd.DataFrame(result_rows)
    result_path = analysis_dir / "rq2_o4_gemma1_cross_within.csv"
    result.to_csv(result_path, index=False)
    complete = len(result) == 2 and result.fit_status.eq("COMPLETE").all()
    metadata = {
        "experiment": "NAACL optional O4 Gemma-3-1B draft scale comparison",
        "status": "COMPLETE" if complete else "PARTIAL",
        "pair_id": PAIR_ID,
        "estimands": ["global CROSS-vs-WITHIN", "NOMINAL_TO_PARTICLE CROSS-vs-nearest-WITHIN"],
        "population": "All 20,000 prompts in the frozen Korean FineWeb2 common split.",
        "outcome": "k=4 offline greedy speculative-decoding replay rejection; fitted with the primary global controls and document-clustered uncertainty.",
        "scope": "Saved-continuation Gemma draft-size analysis. It does not measure live decoding latency, validate diverged contexts, or establish causality.",
        "join_metadata": join_meta,
        "files": {"cross_within": str(result_path.relative_to(ROOT))},
    }
    atomic_json(analysis_dir / "rq2_o4_gemma1_metadata.json", metadata)
    print(f"O4_ANALYSIS_{'COMPLETE' if complete else 'PARTIAL'} fits={len(result)}", flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
