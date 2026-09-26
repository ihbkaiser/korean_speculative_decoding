"""Deterministic Korean Wikipedia prompt preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_prompt_cache(
    path: str | Path,
    dataset_id: str = "wikimedia/wikipedia",
    config_name: str = "20231101.ko",
    split: str = "train",
    limit: int = 1000,
) -> int:
    """Write the first `limit` non-empty article texts in published dataset order."""
    from datasets import load_dataset

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, config_name, split=split, streaming=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for source_index, item in enumerate(dataset):
            text = item.get("text")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            handle.write(json.dumps({
                "source_index": source_index,
                "title": item.get("title", ""),
                "text": text,
                "dataset": dataset_id,
                "config": config_name,
                "split": split,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    if count < limit:
        raise RuntimeError(f"Dataset produced only {count} non-empty prompts; requested {limit}")
    return count


def load_prompt_cache(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    if limit is not None and len(records) < limit:
        raise RuntimeError(f"Prompt cache contains {len(records)} prompts; requested {limit}")
    return records


def encode_prompt(tokenizer: Any, text: str, max_prompt_tokens: int) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    ids = list(encoded["input_ids"])
    if not ids:
        raise ValueError("Prompt became empty after tokenization")
    return ids
