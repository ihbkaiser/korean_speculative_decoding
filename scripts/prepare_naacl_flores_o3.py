#!/usr/bin/env python3
"""Freeze the plan's 1,012-row FLORES O3 prompt pool and run config.

Only English devtest source sentences are copied into the prompt pool. Korean
reference translations remain provenance-only and are not made available to
the generation, alignment, or analysis stages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/flores200_en_ko_replication"
EN_PATH = RUN / "data/flores200_dataset/devtest/eng_Latn.devtest"
KOR_PATH = RUN / "data/flores200_dataset/devtest/kor_Hang.devtest"
METADATA_PATH = RUN / "data/flores200_dataset/metadata_devtest.tsv"
ARCHIVE_PATH = RUN / "data/flores200_dataset.tar.gz"
PROMPT_PATH = ROOT / "data/naacl_flores_1012.parquet"
ID_PATH = ROOT / "data/naacl_flores_1012_ids.txt"
CONFIG_PATH = ROOT / "configs/naacl_flores_o3.yaml"
PROVENANCE_PATH = ROOT / "results/naacl_experiments/flores_optional/o3_prompt_manifest.json"

EXPECTED_SHA256 = {
    ARCHIVE_PATH: "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
    EN_PATH: "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88",
    KOR_PATH: "540972696230a56e888f1fbd04b7000135c436351f5e467f99ce67fdccd8df0f",
    METADATA_PATH: "8edac47f861fcf6b54dad2068c10ba8f741da8b8676861c6bc27fcd9f5483b36",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"FLORES source checksum mismatch for {path}: {actual}")

    english = EN_PATH.read_text(encoding="utf-8").splitlines()
    if len(english) != 1012 or any(not row.strip() for row in english):
        raise RuntimeError(f"Expected 1,012 non-empty English devtest rows; found {len(english)}")

    rows = []
    for index, sentence in enumerate(english, start=1):
        rendered = (
            "Translate the following sentence into Korean.\n"
            f"English: {sentence}\nKorean:"
        )
        rows.append({
            "doc_id": f"FLORES-200:devtest:{index:04d}",
            "source_index": index,
            "pool_index": index - 1,
            "pool_split": "flores_devtest_1012",
            "raw_text_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "text": rendered,
            "text_chars": len(rendered),
            "dataset": "FLORES-200",
            "config": "devtest:eng_Latn->kor_Hang",
            "split": "devtest",
            "title": "",
        })
    if len({row["doc_id"] for row in rows}) != 1012:
        raise AssertionError("FLORES prompt IDs are not unique")
    if len({row["raw_text_hash"] for row in rows}) != 1012:
        raise AssertionError("FLORES rendered prompts are not unique")

    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROMPT_PATH.with_name(PROMPT_PATH.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(PROMPT_PATH)
    ID_PATH.write_text("".join(f"{row['doc_id']}\n" for row in rows), encoding="utf-8")

    base_path = ROOT / "configs/naacl_experiments.yaml"
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config["data"]["prompts"] = "data/naacl_flores_1012.parquet"
    config["data"]["common_ids"] = "data/naacl_flores_1012_ids.txt"
    config["data"]["extra_ids"] = "data/naacl_flores_1012_ids.txt"
    config["paths"]["outputs"] = "results/naacl_experiments/flores_optional"
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "NAACL optional O3: FLORES-200 English-to-Korean target and draft grid",
        "source_release": "FLORES-200 original 2022 release, devtest",
        "source_languages": {"source": "eng_Latn", "target_reference": "kor_Hang"},
        "row_count": len(rows),
        "prompt_template": "Translate the following sentence into Korean.\\nEnglish: <English sentence>\\nKorean:",
        "reference_use": "Korean translations are checksum-verified provenance only; no reference text is present in the frozen prompt pool or inference inputs.",
        "source_files_sha256": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED_SHA256.items()},
        "prompt_pool": str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_pool_sha256": sha256_file(PROMPT_PATH),
        "prompt_ids_sha256": hashlib.sha256("\n".join(row["doc_id"] for row in rows).encode("utf-8")).hexdigest(),
        "prompt_template_hashes_sha256": hashlib.sha256("\n".join(row["raw_text_hash"] for row in rows).encode("ascii")).hexdigest(),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "target_keys": ["qwen17", "qwen4", "qwen8", "gemma12", "ministral8"],
        "pair_keys": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "G1", "M1"],
    }
    PROVENANCE_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FLORES_O3_PROMPTS_FROZEN rows={len(rows)} pool_sha256={manifest['prompt_pool_sha256']}")
    print(f"FLORES_O3_CONFIG {CONFIG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
