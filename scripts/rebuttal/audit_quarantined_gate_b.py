#!/usr/bin/env python3
"""Recompute Gate B for quarantined outputs and compare with the rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.score_cross_family_judge import (
    OUTPUT_ROOT,
    _ensure_snapshot,
    _gate_b,
    _read_continuations,
    _source_doc,
)
from src.naacl_experiment import atomic_json, load_config, load_tokenizer


PAIRS = (("qwen4", "ministral8"), ("qwen8", "ministral8"), ("ministral8", "qwen8"))
QUARANTINE_ROOT = ROOT / "results/rebuttal/cross_family_invalid_20261126"


def _legacy_gate_b(source: str, judge: str, config: dict, api: HfApi, token: str | None) -> dict:
    source_path, _, _ = _ensure_snapshot(config, source, api, token)
    judge_path, _, _ = _ensure_snapshot(config, judge, api, token)
    source_tokenizer = load_tokenizer(config, source, snapshot=source_path)
    judge_tokenizer = load_tokenizer(config, judge, snapshot=judge_path)
    docs = [_source_doc(row, source_tokenizer) for row in _read_continuations(source, 5000)]
    ids_by_pool = {
        row["pool_index"]: list(map(int, judge_tokenizer(row["text"], add_special_tokens=False)["input_ids"]))
        for row in docs
    }
    output_dir = QUARANTINE_ROOT / source / judge
    files = sorted(output_dir.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No quarantined score shards under {output_dir}")
    rows = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    actual = []
    for row in rows.itertuples(index=False):
        token_ids = ids_by_pool.get(int(row.pool_index), [])
        index = int(row.judge_index)
        if index >= len(token_ids):
            raise RuntimeError(f"Cannot re-tokenize legacy score at pool={row.pool_index}, judge_index={index}")
        actual.append(int(token_ids[index]) == int(row.argmax_id))
    rows["is_top1"] = actual
    judge_config = AutoConfig.from_pretrained(judge_path, local_files_only=True)
    vocabulary_size = int(getattr(judge_config, "vocab_size", len(judge_tokenizer)))
    gate = _gate_b(rows, vocabulary_size)
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    gate.update({
        "source_target": source,
        "judge": judge,
        "model_storage_mode": metadata.get("model_storage_mode"),
        "judge_snapshot": metadata.get("judge_snapshot"),
        "top1_accuracy_recomputed_from_retokenized_text": True,
    })
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output", type=Path, default=QUARANTINE_ROOT / "sanity_gate_comparison.json")
    args = parser.parse_args()
    config = load_config(args.config)
    token = get_token()
    api = HfApi(token=token)
    comparison = []
    for source, judge in PAIRS:
        before = _legacy_gate_b(source, judge, config, api, token)
        after_path = OUTPUT_ROOT / source / judge / "sanity_gate.json"
        after = json.loads(after_path.read_text(encoding="utf-8"))
        if before["passed"]:
            raise RuntimeError(f"Quarantined output unexpectedly passes Gate B: {source}/{judge}: {before}")
        if not after.get("passed"):
            raise RuntimeError(f"Rerun output does not pass Gate B: {source}/{judge}: {after}")
        comparison.append({"source_target": source, "judge": judge, "before_invalid_run": before, "after_rerun": after})
    payload = {"pairs": comparison}
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
