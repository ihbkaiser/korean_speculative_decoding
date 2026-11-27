#!/usr/bin/env python3
"""Compare the historical streamed loader with the standard model loader."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.rebuttal.score_cross_family_judge as cross_family
import scripts.rebuttal.streamed_model_loader as streamed_loader
from scripts.rebuttal.streamed_model_loader import _load_streamed_model
from src.naacl_experiment import atomic_json, load_config, load_model, load_tokenizer, model_path


DEFAULT_CONFIG = ROOT / "configs/naacl_experiments.yaml"
SOURCE_TARGET = "qwen4"
DIAGNOSIS_DOCS = 4


def _weight_file(path: Path) -> bool:
    return path.name.endswith((".safetensors", ".bin", ".pt", ".pth"))


def _copy_metadata_snapshot(source: Path, destination: Path) -> None:
    """Copy config/tokenizer files so the old loader takes its streamed branch."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        if not path.is_file() or _weight_file(path):
            continue
        relative = path.relative_to(source)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)


def _uncheckpointed_buffers(model) -> dict[str, torch.Tensor]:
    state_keys = set(model.state_dict())
    return {name: tensor.detach().cpu() for name, tensor in model.named_buffers() if name not in state_keys}


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all().item()) if tensor.is_floating_point() or tensor.is_complex() else True
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite, "min": None, "max": None}
    if not finite:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "max_abs": None,
        }
    values = tensor.detach().to(torch.float64)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "max_abs": float(values.abs().max().item()),
    }


def _compare_buffers(standard, streamed) -> list[dict[str, Any]]:
    standard_buffers = _uncheckpointed_buffers(standard)
    streamed_buffers = _uncheckpointed_buffers(streamed)
    rows = []
    for name in sorted(set(standard_buffers) | set(streamed_buffers)):
        expected = standard_buffers.get(name)
        observed = streamed_buffers.get(name)
        present = expected is not None and observed is not None
        equal = bool(present and expected.shape == observed.shape and torch.equal(expected, observed))
        summary = _tensor_summary(observed) if observed is not None else None
        finite = bool(summary["finite"]) if summary else False
        rows.append({
            "name": name,
            "present_in_standard": expected is not None,
            "present_in_streamed": observed is not None,
            "equal_to_standard": equal,
            "streamed": summary,
            # These buffers are initialized by the architecture, not checkpoint weights.
            "streamed_implausible_relative_to_standard": not equal,
            "streamed_nonfinite": not finite,
        })
    return rows


def _load_source_texts(config: dict[str, Any], qwen4_path: Path) -> list[tuple[int, str]]:
    rows = cross_family._read_continuations(SOURCE_TARGET, 5000)
    tokenizer = load_tokenizer(config, SOURCE_TARGET, snapshot=qwen4_path)
    result = []
    for row in rows[:DIAGNOSIS_DOCS]:
        ids = list(map(int, row["prompt_ids"])) + list(map(int, row["continuation_ids"]))
        result.append((int(row["pool_index"]), tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )))
    return result


@torch.inference_mode()
def _compare_logits(standard, streamed, texts: list[tuple[int, str]], tokenizer) -> dict[str, Any]:
    max_abs_difference = 0.0
    total_positions = 0
    equal_argmax = 0
    entropy_sum_standard = 0.0
    entropy_sum_streamed = 0.0
    device = next(standard.parameters()).device
    for pool_index, text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            raise RuntimeError(f"Diagnosis document {pool_index} tokenized to fewer than two tokens")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        standard_logits = standard(input_ids=input_ids, use_cache=False).logits.float()[0]
        streamed_logits = streamed(input_ids=input_ids, use_cache=False).logits.float()[0]
        positions = min(standard_logits.shape[0], streamed_logits.shape[0])
        if positions != len(ids):
            raise RuntimeError(f"Unexpected logit sequence length for pool_index={pool_index}")
        max_abs_difference = max(
            max_abs_difference,
            float((standard_logits[:positions] - streamed_logits[:positions]).abs().max().item()),
        )
        standard_top = standard_logits[:positions].argmax(dim=-1)
        streamed_top = streamed_logits[:positions].argmax(dim=-1)
        equal_argmax += int((standard_top == streamed_top).sum().item())
        total_positions += positions
        entropy_sum_standard += float(torch.distributions.Categorical(logits=standard_logits[:positions]).entropy().sum().item())
        entropy_sum_streamed += float(torch.distributions.Categorical(logits=streamed_logits[:positions]).entropy().sum().item())
        del input_ids, standard_logits, streamed_logits, standard_top, streamed_top
    return {
        "documents": [pool_index for pool_index, _ in texts],
        "positions": total_positions,
        "max_abs_logit_difference": max_abs_difference,
        "argmax_agreement": equal_argmax / total_positions if total_positions else None,
        "mean_entropy_standard": entropy_sum_standard / total_positions if total_positions else None,
        "mean_entropy_streamed": entropy_sum_streamed / total_positions if total_positions else None,
    }


def diagnose_model(config: dict[str, Any], key: str, api: HfApi, token: str) -> dict[str, Any]:
    standard_path = model_path(config, key)
    standard = load_model(config, key, snapshot=standard_path)
    source_path, _, _ = cross_family._ensure_snapshot(config, SOURCE_TARGET, api, token)
    source_texts = _load_source_texts(config, source_path)
    tokenizer = load_tokenizer(config, key, snapshot=standard_path)

    stream_root = ROOT / ".t6b-streamed-diagnosis"
    stream_path = stream_root / config["models"][key]["id"].replace("/", "--") / config["models"][key]["revision"]
    _copy_metadata_snapshot(standard_path, stream_path)
    original_model_root = cross_family.MODEL_ROOT
    original_naacl_root = os.environ.get("NAACL_MODEL_DIR")
    original_hf_hub_download = streamed_loader.hf_hub_download

    def _local_pinned_shard(repo_id: str, filename: str, **kwargs):
        """Use a hard link to the exact pinned shard while preserving one-shard loading."""
        source = standard_path / filename
        if not source.is_file():
            return original_hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / filename
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.unlink(missing_ok=True)
        os.link(source, local_file)
        return str(local_file)

    try:
        cross_family.MODEL_ROOT = stream_root
        os.environ["NAACL_MODEL_DIR"] = str(stream_root)
        streamed_loader.hf_hub_download = _local_pinned_shard
        streamed, actual_path, storage_mode = _load_streamed_model(config, key, api, token)
    finally:
        cross_family.MODEL_ROOT = original_model_root
        streamed_loader.hf_hub_download = original_hf_hub_download
        if original_naacl_root is None:
            os.environ.pop("NAACL_MODEL_DIR", None)
        else:
            os.environ["NAACL_MODEL_DIR"] = original_naacl_root

    buffers = _compare_buffers(standard, streamed)
    logits = _compare_logits(standard, streamed, source_texts, tokenizer)
    result = {
        "model_key": key,
        "model_id": config["models"][key]["id"],
        "revision": config["models"][key]["revision"],
        "standard_snapshot": str(standard_path),
        "streamed_snapshot": str(actual_path),
        "streamed_storage_mode": storage_mode,
        "streamed_weight_source": "hard-linked one shard at a time from the same pinned local snapshot",
        "uncheckpointed_buffer_count": len(buffers),
        "uncheckpointed_buffers": buffers,
        "logit_comparison_on_qwen4_generated_text": logits,
    }
    del streamed, standard
    cross_family.cleanup_cuda(torch)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("qwen06", "ministral3"), default="qwen06")
    parser.add_argument("--include-ministral3", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal/cross_family/loader_diagnosis.json")
    args = parser.parse_args()
    config = load_config(args.config)
    token = get_token()
    api = HfApi(token=token)
    keys = [args.model]
    if args.include_ministral3 and "ministral3" not in keys:
        keys.append("ministral3")
    results = []
    for key in keys:
        print(f"LOADER_DIAGNOSIS_START model={key}", flush=True)
        results.append(diagnose_model(config, key, api, token))
        print(f"LOADER_DIAGNOSIS_COMPLETE model={key}", flush=True)
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        merged = {row["model_key"]: row for row in previous.get("models", [])}
        merged.update({row["model_key"]: row for row in results})
        results = [merged[key] for key in ("qwen06", "ministral3") if key in merged]
    payload = {
        "historical_commit": "939a52f",
        "source_target": SOURCE_TARGET,
        "documents_per_model": DIAGNOSIS_DOCS,
        "buffer_implausibility_definition": "A streamed non-persistent buffer is implausible when it differs from the same model's standard-loader initialization.",
        "models": results,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
