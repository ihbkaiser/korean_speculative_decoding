#!/usr/bin/env python3
"""Verify tokenizer IDs, special tokens, and character offsets for a model pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import compare_tokenizers, load_table1_tokenizer
from src.naacl_experiment import atomic_json, load_config, model_path, output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    pair = config["pairs"][args.pair]
    draft_spec = config["models"][pair["draft"]]
    target_spec = config["models"][pair["target"]]
    draft = load_table1_tokenizer(
        model_path(config, pair["draft"]),
        backend=draft_spec.get("tokenizer_backend", "auto"),
        fix_mistral_regex=bool(draft_spec.get("fix_mistral_regex", False)),
        local_files_only=True,
    )
    target = load_table1_tokenizer(
        model_path(config, pair["target"]),
        backend=target_spec.get("tokenizer_backend", "auto"),
        fix_mistral_regex=bool(target_spec.get("fix_mistral_regex", False)),
        local_files_only=True,
    )
    report = compare_tokenizers(draft, target, require_offsets=True)
    output = output_root(config, args.output_dir) / "provenance" / "tokenizer_audits" / f"{args.pair}.json"
    atomic_json(output, {
        "pair": args.pair,
        "draft_key": pair["draft"],
        "draft_id": draft_spec["id"],
        "draft_revision": draft_spec["revision"],
        "target_key": pair["target"],
        "target_id": target_spec["id"],
        "target_revision": target_spec["revision"],
        **report,
    })
    status = "PASS" if report["compatible"] else "FAIL"
    print(f"TOKENIZER_AUDIT {args.pair} {status} report={output}", flush=True)
    if not report["compatible"]:
        print("Reasons: " + "; ".join(report["reasons"]), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
