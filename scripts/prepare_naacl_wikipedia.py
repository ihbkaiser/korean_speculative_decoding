#!/usr/bin/env python3
"""Freeze the plan-approved 2,000-document Korean Wikipedia workload."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.naacl_experiment import atomic_json, sha256_file
from src.table1_data import document_hangul_ratio, normalize_document_text, text_sha256


DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.ko"
SPLIT = "train"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SEED_PATH = ROOT / "data/korean_wikipedia_20231101_ko.jsonl"
PARQUET_PATH = ROOT / "data/naacl_wikipedia_2k.parquet"
IDS_PATH = ROOT / "data/naacl_wikipedia_2k_ids.txt"
MANIFEST_PATH = ROOT / "results/naacl_experiments/provenance/wikipedia_2k_manifest.json"
SEED_COUNT = 1_000
TOTAL_COUNT = 2_000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_seed() -> list[dict]:
    rows = [
        json.loads(line)
        for line in SEED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != SEED_COUNT:
        raise RuntimeError(f"Expected {SEED_COUNT} cached seed rows, found {len(rows)}")
    for row in rows:
        if (row.get("dataset"), row.get("config"), row.get("split")) != (
            DATASET,
            CONFIG,
            SPLIT,
        ):
            raise RuntimeError("Cached seed metadata does not match the planned corpus")
    return rows


def _record(item: dict, source_index: int, pool_index: int) -> dict:
    text = normalize_document_text(str(item["text"]))
    if not text:
        raise RuntimeError(f"Empty Wikipedia text at source index {source_index}")
    return {
        "doc_id": f"wikipedia:{CONFIG}:{SPLIT}:{source_index}",
        "source_index": int(source_index),
        "text": text,
        "raw_text_hash": text_sha256(text),
        "text_chars": len(text),
        "hangul_ratio": document_hangul_ratio(text),
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "title": str(item.get("title", "")),
        "pool_index": pool_index,
        "pool_split": "common_wikipedia_2k",
        "source_revision": REVISION,
    }


def main() -> int:
    seed = _read_seed()
    records: list[dict] = []
    seen_hashes: set[str] = set()
    valid_ordinal = 0
    duplicate_extensions = 0

    dataset = load_dataset(
        DATASET,
        CONFIG,
        split=SPLIT,
        streaming=True,
        revision=REVISION,
    )
    for source_index, item in enumerate(dataset):
        raw_text = item.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        if valid_ordinal < SEED_COUNT:
            cached = seed[valid_ordinal]
            source_text = normalize_document_text(raw_text)
            if int(cached["source_index"]) != source_index:
                raise RuntimeError(
                    f"Seed source index mismatch at ordinal {valid_ordinal}: "
                    f"cache={cached['source_index']} pinned={source_index}"
                )
            if normalize_document_text(str(cached["text"])) != source_text:
                raise RuntimeError(
                    f"Cached seed text differs from pinned revision at ordinal {valid_ordinal}"
                )
            if str(cached.get("title", "")) != str(item.get("title", "")):
                raise RuntimeError(
                    f"Cached seed title differs from pinned revision at ordinal {valid_ordinal}"
                )
            row = _record(item, source_index, len(records))
            if row["raw_text_hash"] in seen_hashes:
                raise RuntimeError(f"Duplicate seed text at source index {source_index}")
            records.append(row)
            seen_hashes.add(row["raw_text_hash"])
            valid_ordinal += 1
            continue

        row = _record(item, source_index, len(records))
        if row["raw_text_hash"] in seen_hashes:
            duplicate_extensions += 1
            valid_ordinal += 1
            continue
        records.append(row)
        seen_hashes.add(row["raw_text_hash"])
        valid_ordinal += 1
        if len(records) == TOTAL_COUNT:
            break

    if len(records) != TOTAL_COUNT:
        raise RuntimeError(f"Collected {len(records)} unique rows; expected {TOTAL_COUNT}")
    if len({row["doc_id"] for row in records}) != TOTAL_COUNT:
        raise RuntimeError("Wikipedia prompt IDs are not unique")

    import pandas as pd

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PARQUET_PATH.with_suffix(".parquet.tmp")
    pd.DataFrame(records).to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
    )
    temporary.replace(PARQUET_PATH)
    IDS_PATH.write_text("".join(f"{row['doc_id']}\n" for row in records), encoding="utf-8")

    ordered_ids_hash = _sha256_bytes("\n".join(row["doc_id"] for row in records).encode())
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "prompt_count": len(records),
        "seed_count": SEED_COUNT,
        "extension_count": TOTAL_COUNT - SEED_COUNT,
        "extension_duplicate_texts_skipped": duplicate_extensions,
        "cached_seed_path": str(SEED_PATH.relative_to(ROOT)),
        "cached_seed_sha256": sha256_file(SEED_PATH),
        "seed_matches_pinned_revision": True,
        "parquet_path": str(PARQUET_PATH.relative_to(ROOT)),
        "parquet_sha256": sha256_file(PARQUET_PATH),
        "ids_path": str(IDS_PATH.relative_to(ROOT)),
        "ordered_doc_ids_sha256": ordered_ids_hash,
        "ordered_text_hashes_sha256": _sha256_bytes(
            "\n".join(row["raw_text_hash"] for row in records).encode()
        ),
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
