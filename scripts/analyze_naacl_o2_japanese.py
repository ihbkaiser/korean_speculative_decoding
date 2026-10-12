#!/usr/bin/env python3
"""Analyze topic は and subject が fusion risk in the optional O2 control."""

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


PAIR_ORDER = ["O2_Q06", "O2_G4_12"]
PARTICLE_ROLES = {
    "topic_ha": "TOPIC_HA",
    "subject_ga": "SUBJECT_GA",
}


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


def _role_subset(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    cross = frame.loc[
        frame.alignment_class.eq("CROSS")
        & frame.japanese_particle_roles_crossed.map(
            lambda values: role in values if values is not None else False
        )
    ]
    within = frame.loc[
        frame.alignment_class.eq("WITHIN_SPLIT")
        & frame.japanese_particle_role_nearest.eq(role)
    ]
    return pd.concat([cross, within], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_o2_japanese.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_root(config, args.output_root)
    analysis_dir = args.analysis_dir or root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    join_metadata: list[dict[str, Any]] = []
    analyzer_metadata: dict[str, Any] = {}

    for pair_id in PAIR_ORDER:
        print(f"O2_LOAD pair={pair_id}", flush=True)
        frame, meta = load_pair_frame(root, pair_id, config, split_override="common")
        usable = eligible_cross_within(frame)
        join_metadata.append(meta)
        pair = config["pairs"][pair_id]
        analyzer_path = root / "aligned" / "fugashi" / pair["target"] / "common" / "run_metadata.json"
        if analyzer_path.exists():
            analyzer_metadata[pair["target"]] = json.loads(analyzer_path.read_text(encoding="utf-8"))

        subsets = [("global", usable)]
        subsets.extend((estimand, _role_subset(usable, role)) for estimand, role in PARTICLE_ROLES.items())
        for estimand, subset in subsets:
            fitted = _fit(subset, f"O2-{pair_id}-{estimand}")
            ame, low, high = (
                fitted.get("AME_cross_minus_within"),
                fitted.get("CI_low_95"),
                fitted.get("CI_high_95"),
            )
            result_rows.append({
                "workload": "FineWeb2 jpn_Jpan train, frozen Japanese 5k",
                "pair_id": pair_id,
                "draft": pair["draft"],
                "target": pair["target"],
                "estimand": estimand,
                "documents_joined": meta["documents"],
                "eligible_k4_proposals": meta["eligible_k4_proposals"],
                "aligned_event_positions": meta["event_positions_with_alignment"],
                "AME_pp": 100 * ame if pd.notna(ame) else None,
                "CI_low_95_pp": 100 * low if pd.notna(low) else None,
                "CI_high_95_pp": 100 * high if pd.notna(high) else None,
                **fitted,
            })
        print(
            f"O2_READY pair={pair_id} docs={meta['documents']:,} "
            f"eligible={meta['eligible_k4_proposals']:,} "
            f"aligned={meta['event_positions_with_alignment']:,}",
            flush=True,
        )
        del frame, usable

    result = pd.DataFrame(result_rows)
    result_path = analysis_dir / "rq2_o2_japanese_cross_within.csv"
    result.to_csv(result_path, index=False)
    complete = len(result) == len(PAIR_ORDER) * 3
    metadata = {
        "experiment": "NAACL optional O2 FineWeb2 Japanese control",
        "status": "COMPLETE" if complete else "PARTIAL",
        "pair_order": PAIR_ORDER,
        "estimands": ["global CROSS-vs-WITHIN", "nominal→topic は boundary", "nominal→subject が boundary"],
        "population": "All 5,000 documents in the frozen FineWeb2 jpn_Jpan train pool.",
        "outcome": "k=4 offline greedy speculative-decoding replay rejection; fits use the primary global controls and document-clustered uncertainty.",
        "role_definition": "TOPIC_HA is a UniDic 係助詞 は immediately following a nominal; SUBJECT_GA is a UniDic 格助詞 が immediately following a nominal. CROSS rows use crossed boundaries and WITHIN_SPLIT rows use the unique nearest boundary.",
        "scope": "Saved-continuation Japanese control. It does not measure live decoding latency, validate diverged contexts, or establish causality.",
        "join_metadata": join_metadata,
        "alignment_metadata": analyzer_metadata,
        "files": {"cross_within": str(result_path.relative_to(ROOT))},
    }
    atomic_json(analysis_dir / "rq2_o2_japanese_metadata.json", metadata)
    print(f"O2_ANALYSIS_{'COMPLETE' if complete else 'PARTIAL'} pairs={len(PAIR_ORDER)} fits={len(result)}", flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
