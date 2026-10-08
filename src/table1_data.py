"""Deterministic FineWeb2 Korean prompt-pool preparation for Table 1.

The module intentionally keeps sampling separate from model inference.  A
frozen parquet manifest is the source of truth for every model pair.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


HANGUL_RANGES = (
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3130, 0x318F),   # Compatibility Jamo
    (0xA960, 0xA97F),   # Jamo Extended-A
    (0xAC00, 0xD7AF),   # Syllables
    (0xD7B0, 0xD7FF),   # Jamo Extended-B
)


def normalize_document_text(text: str) -> str:
    """Apply the only content normalization allowed by the experiment."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    return unicodedata.normalize("NFC", text).strip()


def document_hangul_ratio(text: str) -> float:
    """Return Hangul codepoints divided by non-whitespace codepoints."""
    non_whitespace = [char for char in text if not char.isspace()]
    if not non_whitespace:
        return 0.0
    hangul = sum(
        any(start <= ord(char) <= end for start, end in HANGUL_RANGES)
        for char in non_whitespace
    )
    return hangul / len(non_whitespace)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_doc_id(record: dict[str, Any], source_index: int) -> str:
    for key in ("doc_id", "id", "document_id", "url"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return f"source_index:{source_index}"


def _iter_records(records: Iterable[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    for source_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        yield source_index, record


def build_prompt_pool(
    records: Iterable[dict[str, Any]],
    *,
    limit: int = 40_000,
    seed: int = 42,
    min_chars: int = 64,
    min_hangul_ratio: float = 0.35,
    dataset_id: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
) -> list[dict[str, Any]]:
    """Filter, exact-deduplicate, seed-shuffle, and freeze prompt records.

    ``records`` is deliberately an iterable so a streaming Hugging Face
    dataset can be consumed without materializing the entire corpus.  The
    caller controls how large the deterministic prefix scan is; the returned
    records are the only records used by inference.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    if not 0 <= min_hangul_ratio <= 1:
        raise ValueError("min_hangul_ratio must be between 0 and 1")

    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for source_index, record in _iter_records(records):
        raw_text = record.get("text")
        if not isinstance(raw_text, str):
            continue
        text = normalize_document_text(raw_text)
        if len(text) < min_chars:
            continue
        hangul_ratio = document_hangul_ratio(text)
        if hangul_ratio < min_hangul_ratio:
            continue
        raw_text_hash = text_sha256(text)
        if raw_text_hash in seen_hashes:
            continue
        seen_hashes.add(raw_text_hash)
        accepted.append({
            "doc_id": _stable_doc_id(record, source_index),
            "source_index": source_index,
            "text": text,
            "raw_text_hash": raw_text_hash,
            "text_chars": len(text),
            "hangul_ratio": hangul_ratio,
            "dataset": dataset_id,
            "config": dataset_config,
            "split": dataset_split,
        })

    if len(accepted) < limit:
        raise ValueError(
            f"Only {len(accepted)} eligible unique documents were found; "
            f"requested {limit}"
        )
    random.Random(seed).shuffle(accepted)
    selected = accepted[:limit]
    if len({row["raw_text_hash"] for row in selected}) != limit:
        raise AssertionError("prompt pool contains duplicate normalized text hashes")
    if len({row["doc_id"] for row in selected}) != limit:
        # Source identifiers are expected to be unique.  If a dataset exposes
        # duplicate IDs, make the collision explicit instead of silently
        # overwriting a downstream artifact.
        raise AssertionError("prompt pool contains duplicate document IDs")
    for index, row in enumerate(selected):
        row["pool_index"] = index
        row["pool_split"] = "common_20k" if index < 20_000 else "extra_20k"
    return selected


def iter_hf_documents(
    dataset_id: str,
    dataset_config: str,
    split: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw records from a pinned streaming Hugging Face dataset."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "name": dataset_config,
        "split": split,
        "streaming": True,
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    dataset = load_dataset(**kwargs)
    yield from dataset


def resolve_dataset_revision(
    dataset_id: str,
    *,
    requested_revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Resolve a branch/tag to an immutable dataset commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(dataset_id, revision=requested_revision)
    resolved = getattr(info, "sha", None)
    if not resolved:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "requested_revision": requested_revision or "main",
        "resolved_revision": resolved,
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False)),
    }


def freeze_prompt_pool(
    records: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
    common_ids_path: str | Path,
    extra_ids_path: str | Path,
    dataset_metadata_path: str | Path,
    dataset_metadata: dict[str, Any],
    **pool_kwargs: Any,
) -> list[dict[str, Any]]:
    """Write all durable data-stage artifacts atomically where practical."""
    import pandas as pd

    selected = build_prompt_pool(records, **pool_kwargs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    pd.DataFrame(selected).to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(output)

    common = selected[:20_000]
    extra = selected[20_000:]
    Path(common_ids_path).parent.mkdir(parents=True, exist_ok=True)
    Path(common_ids_path).write_text(
        "".join(f"{row['doc_id']}\n" for row in common), encoding="utf-8"
    )
    Path(extra_ids_path).write_text(
        "".join(f"{row['doc_id']}\n" for row in extra), encoding="utf-8"
    )
    metadata_path = Path(dataset_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset_metadata)
    metadata.update({
        "pool_size": len(selected),
        "common_size": len(common),
        "extra_size": len(extra),
        "seed": pool_kwargs.get("seed", 42),
        "min_chars": pool_kwargs.get("min_chars", 64),
        "min_hangul_ratio": pool_kwargs.get("min_hangul_ratio", 0.35),
        "prompt_ids_sha256": hashlib.sha256(
            "\n".join(row["doc_id"] for row in selected).encode("utf-8")
        ).hexdigest(),
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected


def load_prompt_pool(path: str | Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    rows = frame.to_dict(orient="records")
    if len({row["doc_id"] for row in rows}) != len(rows):
        raise AssertionError("frozen prompt pool contains duplicate doc_id")
    if len({row["raw_text_hash"] for row in rows}) != len(rows):
        raise AssertionError("frozen prompt pool contains duplicate raw_text_hash")
    return rows
