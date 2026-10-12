#!/usr/bin/env python3
"""Build a pinned, disjoint FineWeb2 token-frequency cache for Q2 RQ4."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from src.naacl_experiment import atomic_json, load_config, output_root
from src.table1_data import document_hangul_ratio, normalize_document_text, text_sha256, _stable_doc_id


DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASET_CONFIG = "kor_Hang"
DATASET_SPLIT = "train"
TARGET_DOCUMENTS = 40_000


def sha256_texts(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-documents", type=int, default=TARGET_DOCUMENTS)
    parser.add_argument("--source-start", type=int)
    parser.add_argument("--source-stop", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target_documents <= 0:
        raise ValueError("--target-documents must be positive")

    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    data_metadata_path = ROOT / "metadata/dataset_revision.json"
    frozen_pool_path = ROOT / config["data"]["prompts"]
    frozen_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    revision = str(frozen_metadata["resolved_revision"])
    source_start = int(args.source_start if args.source_start is not None else frozen_metadata["scan_limit"])
    source_stop = int(args.source_stop if args.source_stop is not None else source_start + frozen_metadata["scan_limit"])
    if source_start < int(frozen_metadata["scan_limit"]) or source_stop <= source_start:
        raise ValueError("The frequency slice must begin after the 40k pool's raw source scan and have a positive range")

    frequency_path = out / "fineweb2_disjoint_qwen4_token_frequency.parquet"
    documents_path = out / "fineweb2_disjoint_frequency_documents.parquet"
    metadata_path = out / "fineweb2_disjoint_frequency_metadata.json"
    existing = [path for path in (frequency_path, documents_path, metadata_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Refusing to overwrite existing frequency outputs without --force: " + ", ".join(str(x) for x in existing))

    print(f"LOAD_FROZEN_POOL path={frozen_pool_path}", flush=True)
    frozen_pool = pd.read_parquet(frozen_pool_path, columns=["doc_id", "raw_text_hash"])
    pool_hashes = set(frozen_pool.raw_text_hash.astype(str))
    if len(pool_hashes) != len(frozen_pool):
        raise AssertionError("The frozen 40k pool contains duplicate normalized text hashes")

    tokenizer_spec = config["models"]["qwen4"]
    print(f"LOAD_PINNED_TOKENIZER id={tokenizer_spec['id']} revision={tokenizer_spec['revision']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_spec["id"], revision=tokenizer_spec["revision"], use_fast=True,
        local_files_only=False,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The frequency cache requires the pinned fast tokenizer")
    tokenizer_backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    tokenizer_sha256 = hashlib.sha256(tokenizer_backend).hexdigest()

    print(
        f"LOAD_FINEWEB2 dataset={DATASET_ID} config={DATASET_CONFIG} split={DATASET_SPLIT} "
        f"revision={revision} source_range=[{source_start},{source_stop})",
        flush=True,
    )
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT,
        streaming=True, revision=revision,
    )
    iterator = iter(dataset.skip(source_start))
    counts: Counter[int] = Counter()
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    docs: list[dict[str, Any]] = []
    total_tokens = 0
    scanned = 0
    empty_or_nontext = 0
    short_text = 0
    low_hangul = 0
    duplicate_pool = 0
    duplicate_slice = 0
    print("COUNT_DISJOINT_TOKENS", flush=True)
    for offset, item in enumerate(iterator):
        source_index = source_start + offset
        if source_index >= source_stop or len(docs) >= args.target_documents:
            break
        scanned += 1
        raw_text = item.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            empty_or_nontext += 1
            continue
        text = normalize_document_text(raw_text)
        if len(text) < int(frozen_metadata["min_chars"]):
            short_text += 1
            continue
        hangul_ratio = document_hangul_ratio(text)
        if hangul_ratio < float(frozen_metadata["min_hangul_ratio"]):
            low_hangul += 1
            continue
        digest = text_sha256(text)
        if digest in pool_hashes:
            duplicate_pool += 1
            continue
        if digest in seen_hashes:
            duplicate_slice += 1
            continue
        doc_id = _stable_doc_id(item, source_index)
        if doc_id in seen_ids:
            raise RuntimeError(f"Duplicate FineWeb2 document ID in disjoint slice: {doc_id}")
        input_ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        counts.update(int(token_id) for token_id in input_ids)
        total_tokens += len(input_ids)
        seen_hashes.add(digest)
        seen_ids.add(doc_id)
        docs.append({
            "doc_id": doc_id,
            "source_index": source_index,
            "raw_text_hash": digest,
            "text_chars": len(text),
            "hangul_ratio": hangul_ratio,
            "token_count": len(input_ids),
        })
        if scanned % 25_000 == 0 or len(docs) == args.target_documents:
            print(
                f"FREQUENCY_PROGRESS source_index={source_index} scanned={scanned:,} "
                f"documents={len(docs):,}/{args.target_documents:,} tokens={total_tokens:,}",
                flush=True,
            )

    close_iterator = getattr(iterator, "close", None)
    if callable(close_iterator):
        close_iterator()
    del iterator, dataset
    gc.collect()

    if len(docs) != args.target_documents:
        raise RuntimeError(
            f"Found {len(docs):,}/{args.target_documents:,} eligible unique documents "
            f"in source range [{source_start},{source_stop}); outputs not written"
        )
    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_hashes = [str(row["raw_text_hash"]) for row in docs]
    if set(doc_hashes) & pool_hashes:
        raise AssertionError("Disjoint FineWeb2 frequency texts overlap the frozen Q2 pool")

    all_token_ids = sorted({int(token_id) for token_id in tokenizer.get_vocab().values()})
    frequency = pd.DataFrame({"token_id": all_token_ids})
    frequency["token_count"] = frequency.token_id.map(counts).fillna(0).astype("int64")
    frequency["token_frequency"] = frequency.token_count / max(total_tokens, 1)
    frequency["log_token_count"] = np.log1p(frequency.token_count)
    frequency["token_is_unseen"] = frequency.token_count.eq(0).astype("int8")
    frequency["token_frequency_rank"] = frequency.token_count.rank(method="min", ascending=False).astype("int64")
    frequency["token_rank_percentile"] = frequency.token_frequency_rank / len(frequency)

    tmp_frequency = frequency_path.with_suffix(frequency_path.suffix + ".tmp")
    frequency.to_parquet(tmp_frequency, index=False, compression="zstd", compression_level=3)
    tmp_frequency.replace(frequency_path)
    docs_frame = pd.DataFrame(docs)
    tmp_documents = documents_path.with_suffix(documents_path.suffix + ".tmp")
    docs_frame.to_parquet(tmp_documents, index=False, compression="zstd", compression_level=3)
    tmp_documents.replace(documents_path)

    metadata = {
        "status": "COMPLETE",
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "revision": revision,
        "source_scan": {
            "disjoint_from_frozen_pool": True,
            "source_start_inclusive": source_start,
            "source_stop_exclusive": source_stop,
            "source_first_index": int(docs[0]["source_index"]),
            "source_last_index": int(docs[-1]["source_index"]),
            "source_records_scanned": scanned,
            "target_documents": args.target_documents,
            "documents_collected": len(docs),
            "filters": {
                "min_chars": int(frozen_metadata["min_chars"]),
                "min_hangul_ratio": float(frozen_metadata["min_hangul_ratio"]),
                "normalization": "NFC then strip",
                "deduplicate_normalized_text": True,
                "deduplicate_against_frozen_40k_pool": True,
            },
            "rejections": {
                "empty_or_nontext": empty_or_nontext,
                "too_short": short_text,
                "below_hangul_threshold": low_hangul,
                "duplicate_of_pool": duplicate_pool,
                "duplicate_within_slice": duplicate_slice,
            },
        },
        "frozen_pool": {
            "path": str(frozen_pool_path.relative_to(ROOT)),
            "documents": len(frozen_pool),
            "raw_text_hashes_sha256": sha256_texts(sorted(pool_hashes)),
            "source_revision": frozen_metadata["resolved_revision"],
            "source_scan_limit": int(frozen_metadata["scan_limit"]),
        },
        "tokenizer": {
            "model_id": tokenizer_spec["id"],
            "revision": tokenizer_spec["revision"],
            "backend_sha256": tokenizer_sha256,
            "vocab_size": int(len(tokenizer)),
            "fast": bool(tokenizer.is_fast),
            "add_special_tokens": False,
        },
        "frequency": {
            "documents": len(docs),
            "total_tokens": total_tokens,
            "vocabulary_rows": len(frequency),
            "observed_token_ids": len(counts),
            "zero_count_token_ids": int(frequency.token_count.eq(0).sum()),
        },
        "documents_manifest": {
            "path": str(documents_path.relative_to(ROOT)),
            "ordered_doc_ids_sha256": sha256_texts(doc_ids),
            "ordered_text_hashes_sha256": sha256_texts(doc_hashes),
        },
        "frequency_path": str(frequency_path.relative_to(ROOT)),
        "scope": "FineWeb2 token counts on a deterministic source-index slice after the full Q2 source scan; exact normalized-text hashes are checked against all 40k frozen prompts.",
    }
    atomic_json(metadata_path, metadata)
    del tokenizer, frequency, docs_frame, counts
    gc.collect()
    print(
        f"FINEWEB2_FREQUENCY_COMPLETE docs={len(docs):,} tokens={total_tokens:,} "
        f"vocab={metadata['frequency']['vocabulary_rows']:,} overlap={len(set(doc_hashes) & pool_hashes)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
