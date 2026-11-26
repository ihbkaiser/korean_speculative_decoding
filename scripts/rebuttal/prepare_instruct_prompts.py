#!/usr/bin/env python3
"""Freeze the Qwen3 instruction-tuned prompt pool used by rebuttal task T5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import atomic_json, atomic_parquet, load_config

USER_INSTRUCTION = (
    "다음 글의 뒷부분을 자연스럽게 이어서 써 주세요. "
    "설명이나 머리말 없이, 이어지는 내용만 한국어로 쓰세요."
)
ASSISTANT_HEADER = "<|im_start|>assistant\n"
NO_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_prompt_rows(
    source_rows: list[dict[str, Any]],
    *,
    base_tokenizer: Any,
    chat_tokenizer: Any,
    max_prompt_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Decode base prompts, wrap them in a non-thinking chat turn, and validate."""
    if not getattr(chat_tokenizer, "chat_template", None):
        raise RuntimeError("The instruction-tuned tokenizer has no chat template")
    result: list[dict[str, Any]] = []
    for row in source_rows:
        prefix = base_tokenizer.decode(
            list(map(int, row["prompt_ids"])),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        content = f"{USER_INSTRUCTION}\n\n{prefix}"
        messages = [{"role": "user", "content": content}]
        rendered = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text_ids = list(chat_tokenizer(rendered, add_special_tokens=False)["input_ids"])
        template_output = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(template_output, dict) or hasattr(template_output, "input_ids"):
            template_output = template_output["input_ids"]
        if hasattr(template_output, "tolist"):
            template_output = template_output.tolist()
        if template_output and isinstance(template_output[0], list):
            template_output = template_output[0]
        template_ids = list(template_output)
        if text_ids != template_ids:
            raise AssertionError(f"Chat template tokenization differs for {row['doc_id']}")
        if len(text_ids) >= max_prompt_tokens:
            raise AssertionError(
                f"Rendered prompt for {row['doc_id']} has {len(text_ids)} tokens; "
                f"expected fewer than {max_prompt_tokens}"
            )
        header_at = rendered.rfind(ASSISTANT_HEADER)
        suffix = rendered[header_at + len(ASSISTANT_HEADER) :] if header_at >= 0 else None
        if suffix not in ("", NO_THINKING_PREFIX):
            raise AssertionError(
                f"Rendered prompt does not end with the assistant generation header: {row['doc_id']}"
            )

        text = str(rendered)
        result.append(
            {
                "doc_id": str(row["doc_id"]),
                "source_index": int(row["source_index"]),
                "pool_index": int(row["pool_index"]),
                "pool_split": str(row["pool_split"]),
                "raw_text_hash": sha256_text(text),
                "text_chars": len(text),
                "text": text,
            }
        )
    return result


def load_source_rows(directory: Path, max_prompts: int) -> tuple[list[dict[str, Any]], list[str]]:
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("shard-*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"No source continuation shards under {directory}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["pool_split"].astype(str) == "common_20k"]
    frame = frame.sort_values(["pool_index", "doc_id"], kind="stable").head(max_prompts)
    if len(frame) != max_prompts:
        raise RuntimeError(f"Found {len(frame)} common source rows; expected {max_prompts}")
    rows = frame.to_dict(orient="records")
    ids = [str(row["doc_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError("Source continuation pool contains duplicate doc_id values")
    if [int(row["pool_index"]) for row in rows] != list(range(max_prompts)):
        raise AssertionError("Source rows are not the first ascending common pool indices")
    return rows, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results/naacl_experiments/continuations/qwen4/all")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--chat-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--chat-revision", required=True)
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--output-pool", type=Path, default=ROOT / "data/rebuttal_instruct_5000.parquet")
    parser.add_argument("--ids-file", type=Path, default=ROOT / "data/rebuttal_instruct_5000_ids.txt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/rebuttal/instruct/prompt_pool_manifest.json")
    parser.add_argument("--force", action="store_true", help="Replace only the named prompt-pool outputs")
    args = parser.parse_args()
    if args.max_prompts < 1 or args.max_prompt_tokens < 1:
        parser.error("prompt counts and token limits must be positive")
    outputs = [args.output_pool, args.ids_file, args.manifest]
    if not args.force and any(path.exists() for path in outputs):
        parser.error("An output already exists; choose new paths or pass --force explicitly")

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision, token=token, use_fast=True
    )
    chat_tokenizer = AutoTokenizer.from_pretrained(
        args.chat_model, revision=args.chat_revision, token=token, use_fast=True
    )
    config = load_config(args.config)
    source_rows, ids = load_source_rows(args.source_dir, args.max_prompts)
    rows = render_prompt_rows(
        source_rows,
        base_tokenizer=base_tokenizer,
        chat_tokenizer=chat_tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    args.output_pool.parent.mkdir(parents=True, exist_ok=True)
    args.ids_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(rows, args.output_pool)
    args.ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    manifest = {
        "status": "COMPLETE",
        "source": str(args.source_dir.relative_to(ROOT)) if args.source_dir.is_relative_to(ROOT) else str(args.source_dir),
        "source_target": "qwen4",
        "source_split": "all",
        "prompt_count": len(rows),
        "distinct_rendered_prompt_hashes": len({row["raw_text_hash"] for row in rows}),
        "duplicate_rendered_prompt_hash_rows": len(rows) - len({row["raw_text_hash"] for row in rows}),
        "distinct_doc_ids": len({row["doc_id"] for row in rows}),
        "pool_index_min": min(int(row["pool_index"]) for row in rows),
        "pool_index_max": max(int(row["pool_index"]) for row in rows),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "chat_model": args.chat_model,
        "chat_revision": args.chat_revision,
        "enable_thinking": False,
        "max_prompt_tokens": args.max_prompt_tokens,
        "output_pool": str(args.output_pool.relative_to(ROOT)) if args.output_pool.is_relative_to(ROOT) else str(args.output_pool),
        "ids_file": str(args.ids_file.relative_to(ROOT)) if args.ids_file.is_relative_to(ROOT) else str(args.ids_file),
        "pool_sha256": __import__("src.naacl_experiment", fromlist=["sha256_file"]).sha256_file(args.output_pool),
        "id_file_sha256": sha256_text(args.ids_file.read_text(encoding="utf-8")),
        "source_prompt_pool": config["data"]["prompts"],
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
