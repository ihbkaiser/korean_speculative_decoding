#!/usr/bin/env python3
"""Audit the two Qwen3 chat checkpoints and tokenizer compatibility for T5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import atomic_json

CHAT_REPOS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")
BASE_REPO = "Qwen/Qwen3-4B-Base"
BASE_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
SOURCE = ROOT / "results/naacl_experiments/continuations/qwen4/all"
USER_INSTRUCTION = (
    "다음 글의 뒷부분을 자연스럽게 이어서 써 주세요. "
    "설명이나 머리말 없이, 이어지는 내용만 한국어로 쓰세요."
)


def _input_ids(value):
    if isinstance(value, dict) or hasattr(value, "input_ids"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _vocab_hash(tokenizer) -> tuple[dict[str, int], str]:
    vocab = tokenizer.get_vocab()
    encoded = json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return vocab, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal/instruct/model_audit.json")
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()
    if args.sample_count != 10:
        parser.error("T5 requires exactly ten tokenizer parity prompts")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing audit file: {args.output}")

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    audit = {"sample_count": args.sample_count, "models": {}, "vocabularies_identical": None, "sample_ids_identical": None}
    tokenizers = []
    for repo in CHAT_REPOS:
        info = api.model_info(repo, files_metadata=True)
        card_path = hf_hub_download(repo_id=repo, filename="README.md", revision=info.sha, token=token)
        card = Path(card_path).read_text(encoding="utf-8")
        config_path = hf_hub_download(repo_id=repo, filename="tokenizer_config.json", revision=info.sha, token=token)
        tokenizer_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        template = tokenizer_config.get("chat_template")
        if isinstance(template, list):
            template = "\n".join(str(item.get("template", "")) for item in template)
        if not isinstance(template, str) or not template.strip():
            raise RuntimeError(f"{repo}@{info.sha} has no tokenizer chat template")
        if "enable_thinking" not in template:
            raise RuntimeError(f"{repo}@{info.sha} chat template has no enable_thinking switch")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=info.sha, token=token, use_fast=True)
        messages = [{"role": "user", "content": "검증용 문장입니다."}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assistant_header = "<|im_start|>assistant\n"
        header_start = rendered.rfind(assistant_header)
        if header_start < 0:
            raise RuntimeError(f"{repo}@{info.sha} did not render an assistant header")
        tokenized_render = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        tokenized_template = _input_ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        )
        if list(tokenized_render) != tokenized_template:
            raise RuntimeError(f"{repo}@{info.sha} tokenize=False/True paths disagree")
        vocab, vocab_hash = _vocab_hash(tokenizer)
        weight_bytes = sum(
            int(sibling.size or 0)
            for sibling in info.siblings
            if sibling.rfilename.endswith(".safetensors")
            and (sibling.rfilename == "model.safetensors" or sibling.rfilename.startswith("model-"))
        )
        audit["models"][repo] = {
            "revision": info.sha,
            "gated": bool(info.gated),
            "card_url": f"https://huggingface.co/{repo}/tree/{info.sha}",
            "model_card_mentions_instruction_or_chat": bool(re.search(r"instruction|chat", card, re.I)),
            "tokenizer_config_has_chat_template": True,
            "template_accepts_enable_thinking_false": True,
            "no_thinking_prompt_suffix": rendered[header_start + len(assistant_header) :],
            "vocab_size": len(vocab),
            "vocab_sha256": vocab_hash,
            "weight_bytes": weight_bytes,
            "weight_gib": round(weight_bytes / 1024**3, 4),
        }
        tokenizers.append(tokenizer)

    vocab_a, _ = _vocab_hash(tokenizers[0])
    vocab_b, _ = _vocab_hash(tokenizers[1])
    audit["vocabularies_identical"] = vocab_a == vocab_b
    if not audit["vocabularies_identical"]:
        raise RuntimeError("The two instruction-tuned checkpoints have different tokenizer vocabularies")

    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, token=token, use_fast=True
    )
    shards = [pd.read_parquet(path, columns=["pool_index", "prompt_ids"]) for path in sorted(SOURCE.glob("shard-*.parquet"))]
    if not shards:
        raise FileNotFoundError(f"No Qwen4 source continuations under {SOURCE}")
    prompts = pd.concat(shards, ignore_index=True)
    prompts = prompts[prompts.pool_index < 20_000].sort_values("pool_index", kind="stable").head(args.sample_count)
    if len(prompts) != args.sample_count:
        raise RuntimeError(f"Only {len(prompts)} source prompts are available")
    for row in prompts.itertuples(index=False):
        prefix = base_tokenizer.decode(
            list(map(int, row.prompt_ids)),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        messages = [{"role": "user", "content": f"{USER_INSTRUCTION}\n\n{prefix}"}]
        ids = []
        for tokenizer in tokenizers:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids.append(list(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
        if ids[0] != ids[1]:
            raise RuntimeError(f"Chat tokenizer IDs differ at pool_index={int(row.pool_index)}")
    audit["sample_ids_identical"] = True
    audit["sample_pool_indices"] = list(map(int, prompts.pool_index.tolist()))
    audit["base_tokenizer"] = {"repo": BASE_REPO, "revision": BASE_REVISION}
    audit["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
