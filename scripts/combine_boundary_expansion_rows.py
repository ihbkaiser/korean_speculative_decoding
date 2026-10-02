#!/usr/bin/env python3
"""Combine the validated original proposal rows with newly projected rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/proposal_side_boundary_validation/proposal_boundary_rows_all.parquet"
EXPANSION = ROOT / "runs/proposal_boundary_data_expansion/analysis/proposal_rows_new.parquet"
OUT = ROOT / "runs/proposal_boundary_data_expansion/analysis"
KEYS = ["workload", "pair", "prompt_id", "round_index", "proposal_slot", "output_token_position"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--expansion", type=Path, default=EXPANSION)
    parser.add_argument("--out", type=Path, default=OUT / "proposal_rows_combined.parquet")
    args = parser.parse_args()
    baseline = pd.read_parquet(args.baseline)
    expansion = pd.read_parquet(args.expansion)
    baseline["trace_source"] = "baseline"
    expansion["trace_source"] = "expansion"
    baseline_keys = pd.MultiIndex.from_frame(baseline[KEYS])
    expansion_keys = pd.MultiIndex.from_frame(expansion[KEYS])
    overlap = baseline_keys.intersection(expansion_keys)
    if len(overlap):
        raise AssertionError(f"Found {len(overlap)} duplicate proposal keys across old/new traces")
    combined = pd.concat([baseline, expansion], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    manifest = {
        "baseline_file": str(args.baseline), "baseline_sha256": sha256(args.baseline),
        "expansion_file": str(args.expansion), "expansion_sha256": sha256(args.expansion),
        "combined_file": str(args.out), "combined_sha256": sha256(args.out),
        "baseline_rows": len(baseline), "expansion_rows": len(expansion),
        "combined_rows": len(combined), "overlap_keys": int(len(overlap)),
        "rows_by_source_workload_pair": combined.groupby(["trace_source", "workload", "pair"], observed=True).size().rename("rows").reset_index().to_dict("records"),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
