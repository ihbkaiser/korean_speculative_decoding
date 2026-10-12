#!/usr/bin/env python3
"""Freeze the optional 5,000-document FineWeb2 Japanese O2 workload."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import atomic_json, sha256_file
from src.table1_data import normalize_document_text, text_sha256


DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASET_CONFIG = "jpn_Jpan"
DATASET_SPLIT = "train"
TARGET_COUNT = 5_000
SOURCE_LIMIT = 250_000
MIN_CHARS = 64
MIN_JAPANESE_RATIO = 0.35
SEED = 42

PROMPT_PATH = ROOT / "data/naacl_fineweb2_jpn_5000.parquet"
ID_PATH = ROOT / "data/naacl_fineweb2_jpn_5000_ids.txt"
CONFIG_PATH = ROOT / "configs/naacl_o2_japanese.yaml"
MANIFEST_PATH = ROOT / "results/naacl_experiments/provenance/o2_japanese_manifest.json"


def is_japanese_script(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xFF65 <= code <= 0xFF9F
    )


def has_kana(text: str) -> bool:
    return any(
        0x3041 <= ord(char) <= 0x3096
        or 0x309D <= ord(char) <= 0x309F
        or 0x30A1 <= ord(char) <= 0x30FA
        or 0x30FD <= ord(char) <= 0x30FF
        or 0x31F0 <= ord(char) <= 0x31FF
        or 0xFF66 <= ord(char) <= 0xFF9D
        for char in text
    )


def main() -> int:
    revision_path = ROOT / "metadata/dataset_revision.json"
    revision_meta = json.loads(revision_path.read_text(encoding="utf-8"))
    revision = str(revision_meta["resolved_revision"])
    print(
        f"LOAD_FINEWEB2 dataset={DATASET_ID} config={DATASET_CONFIG} split={DATASET_SPLIT} "
        f"revision={revision} source_range=[0,{SOURCE_LIMIT})",
        flush=True,
    )
    stream = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
        revision=revision,
    )

    accepted = []
    seen_hashes: set[str] = set()
    counts = {"scanned": 0, "empty_or_nontext": 0, "short": 0, "no_kana": 0,
              "low_japanese_ratio": 0, "duplicate": 0}
    for source_index, item in enumerate(stream):
        if source_index >= SOURCE_LIMIT:
            break
        counts["scanned"] += 1
        raw = item.get("text")
        if not isinstance(raw, str) or not raw.strip():
            counts["empty_or_nontext"] += 1
            continue
        text = normalize_document_text(raw)
        if len(text) < MIN_CHARS:
            counts["short"] += 1
            continue
        visible = [char for char in text if not char.isspace()]
        if not has_kana(text):
            counts["no_kana"] += 1
            continue
        ratio = sum(is_japanese_script(char) for char in visible) / len(visible)
        if ratio < MIN_JAPANESE_RATIO:
            counts["low_japanese_ratio"] += 1
            continue
        digest = text_sha256(text)
        if digest in seen_hashes:
            counts["duplicate"] += 1
            continue
        seen_hashes.add(digest)
        accepted.append({
            "doc_id": f"fineweb2:{DATASET_CONFIG}:{DATASET_SPLIT}:{source_index}",
            "source_index": int(source_index),
            "text": text,
            "raw_text_hash": digest,
            "text_chars": len(text),
            "japanese_script_ratio": ratio,
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "title": str(item.get("title", "")),
            "source_revision": revision,
        })
    if len(accepted) < TARGET_COUNT:
        raise RuntimeError(
            f"Only {len(accepted):,} unique Japanese documents passed the frozen filters; "
            f"need {TARGET_COUNT:,}. Counts={counts}"
        )

    random.Random(SEED).shuffle(accepted)
    selected = accepted[:TARGET_COUNT]
    for index, row in enumerate(selected):
        row["pool_index"] = index
        row["pool_split"] = "common_japanese_5k"
    if len({row["doc_id"] for row in selected}) != TARGET_COUNT:
        raise AssertionError("O2 Japanese document IDs are not unique")
    if len({row["raw_text_hash"] for row in selected}) != TARGET_COUNT:
        raise AssertionError("O2 Japanese normalized texts are not unique")

    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROMPT_PATH.with_suffix(".parquet.tmp")
    pd.DataFrame(selected).to_parquet(
        temporary, index=False, engine="pyarrow", compression="zstd", compression_level=3
    )
    temporary.replace(PROMPT_PATH)
    ID_PATH.write_text("".join(f"{row['doc_id']}\n" for row in selected), encoding="utf-8")

    base = yaml.safe_load((ROOT / "configs/naacl_experiments.yaml").read_text(encoding="utf-8"))
    base["data"]["prompts"] = str(PROMPT_PATH.relative_to(ROOT))
    base["data"]["common_ids"] = str(ID_PATH.relative_to(ROOT))
    base["data"]["extra_ids"] = str(ID_PATH.relative_to(ROOT))
    base["paths"]["outputs"] = "results/naacl_experiments/o2_japanese"
    base["paths"]["alignment_analyzer"] = "fugashi"
    base["pairs"] = {
        "O2_Q06": {
            "family": "Qwen3", "draft": "qwen06", "target": "qwen4",
            "split": "common", "legacy_name": "Qwen3-0.6B→4B Japanese",
        },
        "O2_G4_12": {
            "family": "Gemma 3", "draft": "gemma4", "target": "gemma12",
            "split": "common", "legacy_name": "Gemma-3-4B→12B Japanese",
        },
    }
    CONFIG_PATH.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")

    ordered_ids_hash = hashlib.sha256(
        "\n".join(row["doc_id"] for row in selected).encode("utf-8")
    ).hexdigest()
    manifest = {
        "experiment": "NAACL optional O2: FineWeb2 Japanese control",
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "resolved_revision": revision,
        "source_range": [0, SOURCE_LIMIT],
        "selection": {
            "target_documents": TARGET_COUNT,
            "min_chars": MIN_CHARS,
            "min_japanese_script_ratio": MIN_JAPANESE_RATIO,
            "requires_hiragana_or_katakana": True,
            "seed": SEED,
            "eligible_unique_before_shuffle": len(accepted),
            "counts": counts,
        },
        "prompt_pool": str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_pool_sha256": sha256_file(PROMPT_PATH),
        "ids_path": str(ID_PATH.relative_to(ROOT)),
        "ordered_doc_ids_sha256": ordered_ids_hash,
        "ordered_text_hashes_sha256": hashlib.sha256(
            "\n".join(row["raw_text_hash"] for row in selected).encode("utf-8")
        ).hexdigest(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "pair_keys": list(base["pairs"]),
        "language_alignment": "Fugashi with full UniDic; topic は and case が retain their own particle role labels.",
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(
        f"O2_JAPANESE_PROMPTS_FROZEN rows={len(selected)} "
        f"eligible={len(accepted):,} id_sha256={ordered_ids_hash}",
        flush=True,
    )
    print(json.dumps(counts, sort_keys=True), flush=True)
    print(f"O2_JAPANESE_CONFIG {CONFIG_PATH.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
