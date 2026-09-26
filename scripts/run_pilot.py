#!/usr/bin/env python3
"""Run the target reference, audited speculative decoding, and morphology pilot."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.alignment import map_token_positions_to_spans, token_ids_to_character_offsets
from src.data import encode_prompt, load_prompt_cache, prepare_prompt_cache
from src.models import assert_tokenizer_compatible, load_models
from src.morphology import extract_eojeol_rows, make_kiwi
from src.speculative_decoding import greedy_generate, speculative_greedy, teacher_forced_scores, verify_greedy_equivalence


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.yaml"))
    parser.add_argument("--limit", type=int, help="Number of prompts; use 5 or 20 for the requested pilot sequence")
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["experiment"]
    limit = int(args.limit or config["pilot_prompts"])
    if limit < 1:
        raise SystemExit("--limit must be positive")
    device = args.device or config["device"]
    seed = int(config["seed"])
    _set_seed(seed)

    data_path = ROOT / config["data_cache"]
    if not data_path.exists():
        prepare_prompt_cache(data_path, config["dataset_id"], config["dataset_config"], config["dataset_split"], max(limit, int(config["pilot_prompts"])))
    try:
        prompt_records = load_prompt_cache(data_path, limit=limit)
    except RuntimeError:
        prepare_prompt_cache(data_path, config["dataset_id"], config["dataset_config"], config["dataset_split"], max(limit, int(config["pilot_prompts"])))
        prompt_records = load_prompt_cache(data_path, limit=limit)

    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_pilot{limit}"
    run_dir = ROOT / config["runs_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = dict(config)
    resolved.update({"limit": limit, "device": device, "seed": seed, "run_id": run_id})
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": resolved}, sort_keys=False), encoding="utf-8")

    from transformers import AutoTokenizer

    draft_tokenizer = AutoTokenizer.from_pretrained(config["draft_model"], use_fast=True)
    target_tokenizer = AutoTokenizer.from_pretrained(config["target_model"], use_fast=True)
    compatibility = assert_tokenizer_compatible(draft_tokenizer, target_tokenizer)
    (run_dir / "tokenizer_compatibility.json").write_text(json.dumps(compatibility, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = target_tokenizer
    draft_model, target_model = load_models(config["draft_model"], config["target_model"], device=device)
    max_token_id = max(tokenizer.get_vocab().values())
    if max_token_id >= draft_model.config.vocab_size or max_token_id >= target_model.config.vocab_size:
        raise RuntimeError("A tokenizer token ID exceeds a model's output vocabulary")
    kiwi = make_kiwi()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sd_rows: list[dict] = []
    teacher_rows: list[dict] = []
    eojeol_rows: list[dict] = []
    reference_rows: list[dict] = []
    roundtrip_fallback_prompts: list[int] = []
    started = time.time()
    eos_token_id = tokenizer.eos_token_id

    for index, record in enumerate(prompt_records):
        prompt_id = int(record["source_index"])
        prompt_ids = encode_prompt(tokenizer, record["text"], int(config["max_prompt_tokens"]))
        reference_ids = greedy_generate(target_model, prompt_ids, int(config["max_new_tokens"]), eos_token_id)
        sd_ids, events = speculative_greedy(
            draft_model,
            target_model,
            prompt_ids,
            int(config["max_new_tokens"]),
            eos_token_id,
            int(config["speculative_k"]),
            prompt_id=prompt_id,
        )
        verify_greedy_equivalence(reference_ids, sd_ids, prompt_id)

        teacher = teacher_forced_scores(draft_model, target_model, prompt_ids, reference_ids, prompt_id=prompt_id)
        complete_ids = prompt_ids + reference_ids
        full_alignment = token_ids_to_character_offsets(tokenizer, complete_ids)
        if not full_alignment["roundtrip_exact"]:
            roundtrip_fallback_prompts.append(prompt_id)
        prompt_offsets = [offset for offset in full_alignment["offsets"][:len(prompt_ids)] if offset[1] > offset[0]]
        prompt_char_end = max((end for _, end in prompt_offsets), default=0)
        continuation_text = full_alignment["text"][prompt_char_end:]
        per_prompt_eojeols = extract_eojeol_rows(continuation_text, kiwi, tokenizer, prompt_id)
        token_to_eojeol = map_token_positions_to_spans(
            full_alignment["offsets"],
            token_start=len(prompt_ids),
            span_rows=per_prompt_eojeols,
            text_origin=prompt_char_end,
        )
        for row in per_prompt_eojeols:
            row["output_token_positions"] = [position for position, eojeol_index in token_to_eojeol.items() if eojeol_index == row["eojeol_index"]]
        eojeol_rows.extend(per_prompt_eojeols)

        for row in events:
            row["eojeol_index"] = token_to_eojeol.get(int(row["output_token_position"]))
            row["reference_target_token_id"] = reference_ids[row["output_token_position"]] if row["output_token_position"] < len(reference_ids) else None
            sd_rows.append(row)
        for row in teacher:
            row["eojeol_index"] = token_to_eojeol.get(int(row["output_token_position"]))
            teacher_rows.append(row)
        reference_rows.append({
            "prompt_id": prompt_id,
            "source_index": record["source_index"],
            "title": record.get("title", ""),
            "prompt_token_count": len(prompt_ids),
            "reference_token_ids": reference_ids,
            "reference_text": continuation_text,
            "speculative_token_ids": sd_ids,
            "exact_match": reference_ids == sd_ids,
            "tokenizer_roundtrip_exact": full_alignment["roundtrip_exact"],
        })
        elapsed = time.time() - started
        print(f"[{index + 1}/{limit}] prompt={prompt_id} ref_tokens={len(reference_ids)} events={len(events)} eojeols={len(per_prompt_eojeols)} elapsed={elapsed:.1f}s", flush=True)

    _write_parquet_atomic(pd.DataFrame(sd_rows), run_dir / "sd_events.parquet")
    _write_parquet_atomic(pd.DataFrame(teacher_rows), run_dir / "teacher_forced_tokens.parquet")
    _write_parquet_atomic(pd.DataFrame(eojeol_rows), run_dir / "eojeols.parquet")
    with (run_dir / "reference_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in reference_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
    metadata = {
        "run_id": run_id,
        "prompt_count": limit,
        "all_target_sd_outputs_equal": all(row["exact_match"] for row in reference_rows),
        "tokenizer_compatible": compatibility["compatible"],
        "tokenizer_roundtrip_fallback_prompt_ids": roundtrip_fallback_prompts,
        "peak_vram_allocated_gb": peak_allocated,
        "peak_vram_reserved_gb": peak_reserved,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "COMPLETE").write_text("All prompts passed target-greedy/speculative token-ID equivalence.\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
