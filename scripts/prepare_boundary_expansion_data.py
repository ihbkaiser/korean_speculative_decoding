#!/usr/bin/env python3
"""Prepare disjoint Korean Wikipedia and FLORES-dev prompts for trace expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer

from src.e2_model_pairs import MODELS
from src.data import encode_prompt


OUT = ROOT / "runs/proposal_boundary_data_expansion"
FLORES_ARCHIVE = ROOT / "runs/flores200_en_ko_replication/data/flores200_dataset.tar.gz"
WIKI_CACHE = ROOT / "data/korean_wikipedia_20231101_ko.jsonl"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ko"
WIKI_SPLIT = "train"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_EXTRA_COUNT = 1000
MAX_PROMPT_TOKENS = 128


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def old_wiki_hashes() -> set[str]:
    rows = [json.loads(line) for line in WIKI_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000:
        raise RuntimeError(f"Expected the original 1,000-prompt cache; found {len(rows)}")
    return {sha256(str(row["text"]).strip().encode("utf-8")) for row in rows}


def flores_dev() -> tuple[list[str], list[str], dict[str, str]]:
    if not FLORES_ARCHIVE.exists():
        raise FileNotFoundError(FLORES_ARCHIVE)
    with tarfile.open(FLORES_ARCHIVE, "r:gz") as archive:
        by_suffix = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        required = ["eng_Latn.dev", "kor_Hang.dev", "metadata_dev.tsv"]
        missing = [name for name in required if name not in by_suffix]
        if missing:
            raise RuntimeError(f"FLORES archive lacks dev files: {missing}")
        payloads = {name: archive.extractfile(by_suffix[name]).read() for name in required}
    en = payloads["eng_Latn.dev"].decode("utf-8").splitlines()
    ko = payloads["kor_Hang.dev"].decode("utf-8").splitlines()
    if len(en) != 997 or len(ko) != 997:
        raise RuntimeError(f"Expected 997 FLORES dev rows; found EN={len(en)}, KO={len(ko)}")
    if any(not a.strip() or not b.strip() for a, b in zip(en, ko)):
        raise RuntimeError("FLORES dev contains an empty source or reference")
    return en, ko, {name: sha256(payload) for name, payload in payloads.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["0.6B"]["id"], revision=MODELS["0.6B"]["revision"],
        use_fast=True, local_files_only=True,
    )
    previous_hashes = old_wiki_hashes()
    ds = load_dataset(
        WIKI_DATASET, WIKI_CONFIG, split=WIKI_SPLIT, streaming=True,
        revision=WIKI_REVISION,
    )
    wiki_records = []
    valid_ordinal = 0
    for source_index, item in enumerate(ds):
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if valid_ordinal < 1000:
            valid_ordinal += 1
            continue
        local_index = valid_ordinal - 1000
        text_hash = sha256(text.encode("utf-8"))
        if text_hash in previous_hashes:
            raise RuntimeError(f"Wiki extension duplicates an original article at source index {source_index}")
        prompt_id = 1000 + local_index
        ids = encode_prompt(tokenizer, text, MAX_PROMPT_TOKENS)
        canonical = {"workload": "WIKIPEDIA", "prompt_id": prompt_id, "title": str(item.get("title", "")), "text": text}
        prompt_hash = sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        wiki_records.append({
            **canonical, "source_split": "train_extension", "source_ordinal": valid_ordinal,
            "source_index": source_index, "source_revision": WIKI_REVISION,
            "source_text_sha256": text_hash, "prompt_hash": prompt_hash,
            "input_ids": ids, "prompt_token_count": len(ids),
        })
        valid_ordinal += 1
        if len(wiki_records) >= WIKI_EXTRA_COUNT:
            break
    if len(wiki_records) != WIKI_EXTRA_COUNT:
        raise RuntimeError(f"Could only collect {len(wiki_records)} extra Wiki articles")

    en, ko, flores_hashes = flores_dev()
    devtest_en = (ROOT / "runs/flores200_en_ko_replication/data/flores200_dataset/devtest/eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    devtest_hashes = {sha256(row.strip().encode("utf-8")) for row in devtest_en}
    overlap = [i + 1 for i, source in enumerate(en) if sha256(source.strip().encode("utf-8")) in devtest_hashes]
    if overlap:
        raise RuntimeError(f"FLORES dev and devtest share source rows: {overlap[:10]}")
    flores_records = []
    for i, (source, reference) in enumerate(zip(en, ko), start=1):
        rendered = f"Translate the following sentence into Korean.\nEnglish: {source}\nKorean:"
        prompt_id = 100000 + i
        ids = encode_prompt(tokenizer, rendered, MAX_PROMPT_TOKENS)
        canonical = {"workload": "FLORES", "prompt_id": prompt_id, "rendered_prompt": rendered}
        prompt_hash = sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        flores_records.append({
            **canonical, "source_split": "dev", "source_ordinal": i, "english_source": source,
            "reference_sha256_provenance_only": sha256(reference.encode("utf-8")),
            "source_revision": "FLORES-200 original release; dev split from pinned local archive",
            "source_text_sha256": sha256(source.encode("utf-8")), "prompt_hash": prompt_hash,
            "input_ids": ids, "prompt_token_count": len(ids),
        })

    records = wiki_records + flores_records
    jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    atomic_text(OUT / "prompts.jsonl", jsonl)
    manifest_path = OUT / "prompt_manifest.csv"
    columns = ["workload", "prompt_id", "source_split", "source_ordinal", "source_index", "source_revision", "source_text_sha256", "reference_sha256_provenance_only", "prompt_hash", "prompt_token_count"]
    with manifest_path.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    manifest_path.with_suffix(".csv.tmp").replace(manifest_path)
    meta = {
        "wiki": {"dataset": WIKI_DATASET, "config": WIKI_CONFIG, "split": WIKI_SPLIT,
                 "revision": WIKI_REVISION, "additional_prompts": len(wiki_records),
                 "first_source_ordinal": wiki_records[0]["source_ordinal"], "last_source_ordinal": wiki_records[-1]["source_ordinal"],
                 "original_prompt_cache_sha256": sha256(WIKI_CACHE.read_bytes())},
        "flores": {"release": "FLORES-200 original release", "split": "dev", "prompt_count": len(flores_records),
                   "source_archive_sha256": sha256(FLORES_ARCHIVE.read_bytes()), "source_file_sha256": flores_hashes,
                   "overlap_with_devtest_source_rows": overlap},
        "tokenizer": {"model_id": MODELS["0.6B"]["id"], "revision": MODELS["0.6B"]["revision"],
                      "prompt_encoding": "add_special_tokens=False; truncation to 128 tokens"},
        "total_prompts": len(records),
        "prompts_jsonl_sha256": sha256((OUT / "prompts.jsonl").read_bytes()),
        "prompt_manifest_sha256": sha256(manifest_path.read_bytes()),
    }
    atomic_text(OUT / "data_manifest.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
