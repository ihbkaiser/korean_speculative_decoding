#!/usr/bin/env python3
"""Build E2 tables, models, figures, and report from completed checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e2_model_pairs import PAIRS, _pair_dir, analyze_e2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/e2_model_pair_replication")
    args = parser.parse_args()
    output_dir = Path(args.run_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    metadata = json.loads((output_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    tokenizer_report = json.loads((output_dir / "tokenizer_compatibility.json").read_text(encoding="utf-8"))
    pair_tables = {}
    pair_audits = {}
    pair_continuations = {}
    pair_teachers = {}
    expected_prompts = int(metadata["num_prompts"])
    expected_ids = set(range(expected_prompts))
    for pair_name in ("P1", "P2"):
        pair_dir = _pair_dir(output_dir, pair_name)
        continuations = pd.read_parquet(pair_dir / "continuations.parquet")
        events = pd.read_parquet(pair_dir / "sd_events.parquet")
        teachers = pd.read_parquet(pair_dir / "teacher_forced_tokens.parquet")
        table = pd.read_parquet(pair_dir / "token_table.parquet")
        audit = pd.read_csv(pair_dir / "alignment_audit.csv")
        observed = set(map(int, continuations.prompt_id))
        if observed != expected_ids or continuations.prompt_id.duplicated().any():
            raise RuntimeError(f"{pair_name}: final continuations do not cover every prompt exactly once")
        if not continuations.exact_match.astype(bool).all():
            raise RuntimeError(f"{pair_name}: at least one speculative output differs from target greedy")
        if set(map(int, audit.prompt_id)) != expected_ids or len(audit) != expected_prompts:
            raise RuntimeError(f"{pair_name}: alignment audit is incomplete")
        if set(map(int, teachers.prompt_id)) != expected_ids or teachers.duplicated(
            ["prompt_id", "output_token_position"]
        ).any():
            raise RuntimeError(f"{pair_name}: teacher-forced score coverage is incomplete or duplicated")
        if set(map(int, events.prompt_id)) != expected_ids:
            raise RuntimeError(f"{pair_name}: SD event coverage is incomplete")
        if set(map(int, table.prompt_id)) != expected_ids or table.duplicated(
            ["prompt_id", "generation_pos"]
        ).any():
            raise RuntimeError(f"{pair_name}: morphology table is incomplete or duplicated")
        pair_tables[pair_name] = table
        pair_audits[pair_name] = audit
        pair_continuations[pair_name] = continuations
        pair_teachers[pair_name] = teachers

    result = analyze_e2(
        output_dir,
        pair_tables,
        pair_audits,
        pair_continuations,
        pair_teachers,
        metadata,
        tokenizer_report,
    )
    print(json.dumps({
        "decision": result["decision"],
        "report": str(result["report"]),
        "summary_csv": str(output_dir / "combined/e2_model_pair_summary.csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
