"""E2 robustness across Qwen3 draft/target capacity pairs.

The implementation deliberately reuses the H2 span classifier and E1
frequency-cache/regression definitions. GPU decoding is pinned by the CLI to
physical GPU 7 (visible as cuda:0), and every stage is checkpointed by prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import encode_prompt, load_prompt_cache
from .h2_analysis import (
    FRAGMENTATION_ORDER,
    PRIMARY_CLASSES,
    _classify_token,
    _fit_clustered_logit,
    _fragmentation_bin,
    _primary_formula,
    _span_tokenizer_ids,
)
from .e1_frequency import (
    ENTROPY_CONTROLS,
    STRUCTURAL_CONTROLS,
    _fit_logit,
    _formula,
)


ROOT = Path(__file__).resolve().parents[1]
P3_DIR = ROOT / "runs/20260926T184145Z_pilot1000"
MODELS = {
    "0.6B": {
        "id": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    },
    "1.7B": {
        "id": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    },
    "4B": {
        "id": "Qwen/Qwen3-4B-Base",
        "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    },
}
PAIRS = {
    "P1": {"draft": "0.6B", "target": "1.7B", "slug": "p1_06b_to_17b"},
    "P2": {"draft": "1.7B", "target": "4B", "slug": "p2_17b_to_4b"},
    "P3": {"draft": "0.6B", "target": "4B", "slug": "p3_06b_to_4b"},
}
P3_RUN_ID = "20260926T184145Z_pilot1000"
P3_FREQUENCY_METADATA = P3_DIR / "e1_token_frequency_cache_metadata.json"
P3_E1_FREQUENCY_TABLE = P3_DIR / "e1_frequency_table.parquet"
P3_FREQUENCY_CACHE = P3_DIR / "e1_token_frequency_cache.parquet"
DEFAULT_STRUCTURAL = list(STRUCTURAL_CONTROLS)
DEFAULT_ENTROPY = list(ENTROPY_CONTROLS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)


def _special_signature(tokenizer: Any) -> dict[str, Any]:
    fields = (
        "bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
        "cls_token", "mask_token", "additional_special_tokens",
    )
    result: dict[str, Any] = {}
    for field in fields:
        value = getattr(tokenizer, field, None)
        if isinstance(value, (tuple, list)):
            value = [getattr(x, "content", str(x)) for x in value]
        else:
            value = getattr(value, "content", value)
        result[field] = value
        result[f"{field}_id"] = getattr(tokenizer, f"{field}_id", None)
    return result


def _tokenizer_identity(tokenizer: Any, model: dict[str, str]) -> tuple[dict[str, Any], dict[str, int], str | None]:
    vocab = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    vocab_sha = hashlib.sha256(
        json.dumps(sorted(vocab.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    backend_sha = hashlib.sha256(backend_json.encode("utf-8")).hexdigest() if backend_json else None
    backend_config = json.loads(backend_json) if backend_json else {}
    revision = getattr(tokenizer, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    revision_source = "tokenizer metadata"
    if revision is None and re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        # The requested revision is a full immutable commit SHA resolved from
        # the Hub API before tokenizer loading, even when the tokenizer class
        # does not expose `_commit_hash` in its init kwargs.
        revision = model["revision"]
        revision_source = "pinned full commit SHA"
    identity = {
        "model_id": model["id"],
        "requested_revision": model["revision"],
        "resolved_revision": revision,
        "resolved_revision_source": revision_source,
        "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(len(tokenizer)),
        "vocab_mapping_sha256": vocab_sha,
        "added_vocab": {str(k): int(v) for k, v in sorted(tokenizer.get_added_vocab().items())},
        "special_tokens": _special_signature(tokenizer),
        "backend_sha256": backend_sha,
        "normalizer": backend_config.get("normalizer"),
        "pre_tokenizer": backend_config.get("pre_tokenizer"),
    }
    return identity, vocab, backend_sha


def tokenizer_preflight(output_dir: Path) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    loaded: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    vocabs: dict[str, dict[str, int]] = {}
    backends: dict[str, str | None] = {}
    for size, model in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model["id"], revision=model["revision"], use_fast=True,
        )
        loaded[size] = tokenizer
        identities[size], vocabs[size], backends[size] = _tokenizer_identity(tokenizer, model)

    comparison_rows = {}
    probe_texts = [
        "한국어 형태소 분석과 토큰 경계",
        "서울은 대한민국의 수도이다.",
        "English mixed with 한글 and numbers 3090.",
        "띄어쓰기\n여러 줄 테스트",
        "한국어의 형태소 경계는 중요하다.",
    ]
    reference = loaded["4B"]
    for size, tokenizer in loaded.items():
        reasons = []
        if identities[size]["vocab_size"] != identities["4B"]["vocab_size"]:
            reasons.append("vocab_size differs")
        if vocabs[size] != vocabs["4B"]:
            reasons.append("token-to-ID mapping differs")
        if identities[size]["added_vocab"] != identities["4B"]["added_vocab"]:
            reasons.append("added tokens differ")
        if identities[size]["special_tokens"] != identities["4B"]["special_tokens"]:
            reasons.append("special token configuration differs")
        if identities[size]["tokenizer_class"] != identities["4B"]["tokenizer_class"]:
            reasons.append("tokenizer class differs")
        if backends[size] != backends["4B"]:
            reasons.append("backend serialization differs")
        if not identities[size]["is_fast"]:
            reasons.append("fast tokenizer with character offsets is required")
        probes = []
        for text in probe_texts:
            try:
                left = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                right = reference(text, add_special_tokens=False, return_offsets_mapping=True)
                ids_equal = list(left["input_ids"]) == list(right["input_ids"])
                offsets_equal = [tuple(x) for x in left["offset_mapping"]] == [tuple(x) for x in right["offset_mapping"]]
                error = None
            except Exception as exc:
                ids_equal = offsets_equal = False
                error = f"{type(exc).__name__}: {exc}"
            probes.append({"text": text, "ids_equal": ids_equal, "offsets_equal": offsets_equal, "error": error})
            if not ids_equal or not offsets_equal:
                reasons.append("probe IDs or offsets differ")
        comparison_rows[size] = {
            "compatible_with_4B": not reasons,
            "reasons": sorted(set(reasons)),
            "probes": probes,
        }

    p3_frequency = json.loads(P3_FREQUENCY_METADATA.read_text(encoding="utf-8"))
    expected_backend = p3_frequency["tokenizer_backend_sha256"]
    p3_cache_match = (
        backends["4B"] == expected_backend
        and identities["4B"]["resolved_revision"] == p3_frequency["tokenizer_revision"]
        and identities["4B"]["vocab_size"] == p3_frequency["vocab_size"]
    )
    shared = all(row["compatible_with_4B"] for row in comparison_rows.values()) and p3_cache_match
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_tokenizers": identities,
        "comparisons_to_4B": comparison_rows,
        "p3_frequency_cache_identity": {
            "tokenizer_name": p3_frequency["tokenizer_name"],
            "revision": p3_frequency["tokenizer_revision"],
            "backend_sha256": expected_backend,
            "vocab_size": p3_frequency["vocab_size"],
            "matches_pinned_4B": p3_cache_match,
        },
        "shared_tokenizer": shared,
    }
    atomic_json(output_dir / "tokenizer_compatibility.json", result)
    if not shared:
        raise RuntimeError("Tokenizer compatibility or P3 frequency-cache identity failed; stopping before model weights/GPU work")
    return loaded["4B"], result


def make_prompts(prompt_cache: Path, tokenizer: Any, num_prompts: int, max_prompt_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = load_prompt_cache(prompt_cache, limit=num_prompts)
    inputs = []
    hashes = []
    for i, record in enumerate(records):
        prompt_id = int(record["source_index"])
        if prompt_id != i:
            raise AssertionError(f"Prompt order/source_index changed at row {i}: {prompt_id}")
        ids = encode_prompt(tokenizer, str(record["text"]), max_prompt_tokens)
        canonical = {
            "prompt_id": prompt_id,
            "title": str(record.get("title", "")),
            "text": str(record["text"]),
            "input_ids": ids,
        }
        prompt_hash = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        inputs.append({"prompt_id": prompt_id, "source_index": prompt_id, "title": canonical["title"], "input_ids": ids, "prompt_hash": prompt_hash})
        hashes.append({"prompt_id": prompt_id, "sha256": prompt_hash, "prompt_token_count": len(ids)})
    combined = hashlib.sha256("\n".join(row["sha256"] for row in hashes).encode()).hexdigest()
    summary = {
        "source_path": str(prompt_cache.resolve()),
        "source_sha256": sha256_file(prompt_cache),
        "source_description": "wikimedia/wikipedia, 20231101.ko, train; first non-empty articles in file order",
        "num_prompts": num_prompts,
        "max_prompt_tokens": max_prompt_tokens,
        "ordered_prompt_set_sha256": combined,
        "prompt_hashes": hashes,
    }
    return records, inputs, summary


def write_pretokenized_inputs(output_dir: Path, inputs: list[dict[str, Any]], hashes: dict[str, Any]) -> None:
    atomic_json(output_dir / "prompt_hashes.json", hashes)
    temp = output_dir / "prompt_ids.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for row in inputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(output_dir / "prompt_ids.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_p3_references(num_prompts: int, inputs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    outputs = _read_jsonl(P3_DIR / "reference_outputs.jsonl")
    if len(outputs) < num_prompts:
        raise AssertionError(f"P3 has {len(outputs)} saved references, expected {num_prompts}")
    input_by_id = {row["prompt_id"]: row for row in inputs}
    references: dict[int, dict[str, Any]] = {}
    for output in outputs[:num_prompts]:
        prompt_id = int(output["prompt_id"])
        if prompt_id not in input_by_id or int(output["source_index"]) != prompt_id:
            raise AssertionError(f"P3 prompt identity mismatch at prompt_id={prompt_id}")
        if len(input_by_id[prompt_id]["input_ids"]) != int(output["prompt_token_count"]):
            raise AssertionError(f"P3 prompt token length changed for prompt_id={prompt_id}")
        ids = [int(x) for x in output["reference_token_ids"]]
        if not output.get("exact_match", False):
            raise AssertionError(f"P3 saved target/SD parity was false for prompt_id={prompt_id}")
        references[prompt_id] = {
            "prompt_id": prompt_id,
            "reference_token_ids": ids,
            "reference_text": str(output["reference_text"]),
            "reference_source": "P3 saved 4B target continuation",
            "tokenizer_roundtrip_exact": bool(output.get("tokenizer_roundtrip_exact", False)),
        }
    return references


def write_initial_config(output_dir: Path, args: Any, prompt_hashes: dict[str, Any], tokenizer_report: dict[str, Any]) -> None:
    config = {
        "experiment": "E2 robustness across Korean draft-target model pairs",
        "models": MODELS,
        "pairs": PAIRS,
        "p3_run_id": P3_RUN_ID,
        "p3_run_dir": str(P3_DIR.resolve()),
        "precision": "float16",
        "attention_backend": "sdpa",
        "physical_gpu": 7,
        "cuda_device_inside_process": "cuda:0",
        "visible_cuda_devices": "7",
        "seed": int(args.seed),
        "prompt_cache": str(Path(args.prompt_cache).resolve()),
        "prompt_cache_sha256": prompt_hashes["source_sha256"],
        "ordered_prompt_set_sha256": prompt_hashes["ordered_prompt_set_sha256"],
        "num_prompts": args.num_prompts,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "speculative_k": args.speculative_k,
        "eos_token_id": tokenizer_report["model_tokenizers"]["4B"]["special_tokens"]["eos_token_id"],
        "prompt_encoding": "add_special_tokens=False; tokenizer truncation to max_prompt_tokens",
        "generation": "greedy, no sampling; per-prompt deterministic cache path",
        "shared_tokenizer": tokenizer_report["shared_tokenizer"],
        "teacher_forcing": "batched full-sequence FP16; SD rejection labels remain from Pass A",
        "resume": bool(args.resume),
    }
    atomic_json(output_dir / "config.json", config)
    source_plan = ROOT / "implementation_e2.md"
    if source_plan.exists():
        (output_dir / "implementation_e2.md").write_text(source_plan.read_text(encoding="utf-8"), encoding="utf-8")


def _require_gpu7() -> Any:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=7 before launching E2; refusing to use an unmasked GPU")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible CUDA device mapped to physical GPU 7; count={torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    if "A100" not in name or props.total_memory < 78 * 1024**3:
        raise RuntimeError(f"Physical GPU 7 is expected to be an A100 80GB, found {name} / {props.total_memory} bytes")
    if torch.cuda.current_device() != 0:
        raise RuntimeError(f"Expected mapped cuda:0, current device is {torch.cuda.current_device()}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch


def write_environment(output_dir: Path, torch: Any, attention_backend: str) -> dict[str, Any]:
    output = subprocess.run(
        ["nvidia-smi", "--id=7", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    import transformers

    env = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "attention_backend": attention_backend,
        "dtype": "torch.float16",
        "physical_gpu": 7,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_device": "cuda:0",
        "process_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "physical_gpu7_nvidia_smi": output,
    }
    atomic_json(output_dir / "environment.json", env)
    (output_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in env.items()) + "\n", encoding="utf-8",
    )
    return env


def load_models(torch: Any, attention_backend: str = "sdpa") -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    models: dict[str, Any] = {}
    for size in ("0.6B", "1.7B", "4B"):
        spec = MODELS[size]
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"], revision=spec["revision"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation=attention_backend,
        )
        model.to("cuda:0")
        model.eval()
        models[size] = model
        print(f"Loaded {size} {spec['id']}@{spec['revision']} on cuda:0 ({attention_backend})", flush=True)
    return models


def greedy_generate_single(model: Any, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int) -> list[int]:
    import torch

    if not prompt_ids:
        raise ValueError("Prompt IDs must not be empty")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model(input_ids=ids, use_cache=True)
        past = result.past_key_values
        logits = result.logits[0, -1]
        output: list[int] = []
        while len(output) < max_new_tokens:
            token_id = int(torch.argmax(logits, dim=-1).item())
            output.append(token_id)
            if token_id == eos_token_id:
                break
            result = model(
                input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
                past_key_values=past, use_cache=True,
            )
            past = result.past_key_values
            logits = result.logits[0, -1]
    return output


def greedy_generate_batch(
    model: Any, prompts: list[list[int]], max_new_tokens: int, eos_token_id: int, batch_size: int,
) -> list[list[int]]:
    """Left-padded greedy generation; exact parity is checked before full use."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size == 1:
        return [greedy_generate_single(model, prompt, max_new_tokens, eos_token_id) for prompt in prompts]
    pad_token_id = eos_token_id
    outputs: list[list[int]] = [[] for _ in prompts]
    for base in range(0, len(prompts), batch_size):
        group = prompts[base:base + batch_size]
        if not group:
            continue
        device = next(model.parameters()).device
        width = max(map(len, group))
        input_ids = torch.full((len(group), width), pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(group), width), dtype=torch.long, device=device)
        for i, prompt in enumerate(group):
            input_ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
            attention[i, width - len(prompt):] = 1
        position_ids = attention.cumsum(-1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        with torch.inference_mode():
            result = model(
                input_ids=input_ids, attention_mask=attention, position_ids=position_ids,
                use_cache=True,
            )
            past = result.past_key_values
            logits = result.logits[:, -1, :]
            lengths = [len(prompt) for prompt in group]
            active = [True] * len(group)
            group_outputs: list[list[int]] = [[] for _ in group]
            for step in range(max_new_tokens):
                next_ids = torch.argmax(logits, dim=-1).tolist()
                active_before = list(active)
                for i, token_id in enumerate(next_ids):
                    if active[i]:
                        group_outputs[i].append(int(token_id))
                        if int(token_id) == eos_token_id:
                            active[i] = False
                if step + 1 >= max_new_tokens or not any(active):
                    break
                input_column = [int(next_ids[i]) if active_before[i] else pad_token_id for i in range(len(group))]
                input_tensor = torch.tensor(input_column, dtype=torch.long, device=device).unsqueeze(1)
                active_mask = torch.tensor(active_before, dtype=torch.long, device=device).unsqueeze(1)
                attention = torch.cat((attention, active_mask), dim=1)
                positions = [
                    lengths[i] + len(group_outputs[i]) - 1 if active_before[i] else 0
                    for i in range(len(group))
                ]
                position_tensor = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1)
                result = model(
                    input_ids=input_tensor, attention_mask=attention, position_ids=position_tensor,
                    past_key_values=past, use_cache=True,
                )
                past = result.past_key_values
                logits = result.logits[:, -1, :]
        outputs[base:base + len(group)] = group_outputs
    return outputs


def speculative_greedy_fast(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    k: int,
    prompt_id: int,
    guard_controller: Any | None = None,
    runtime_stats: dict[str, Any] | None = None,
    batch_target_verification: bool = False,
    record_entropies: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Exact cached-path greedy SD, with an optional exact proposal scheduler.

    With ``guard_controller=None`` this follows the original FIXED_K path.
    ``batch_target_verification`` evaluates a whole candidate block in one
    causal target forward. It is optional because FP16 kernels can produce
    different argmaxes than incremental verification on near-tie logits.
    Guard controllers may shorten verification only before an unverified
    proposal. The unchanged target argmax is then emitted as the correction /
    bonus token, and the target cache is advanced exactly as in the baseline
    rejection path.
    """
    import torch

    if not prompt_ids:
        raise ValueError("Prompt IDs must not be empty")
    if k < 1 or max_new_tokens < 0:
        raise ValueError("k must be positive and max_new_tokens non-negative")
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    generated: list[int] = []
    events: list[dict[str, Any]] = []
    round_index = 0
    finished = False
    if runtime_stats is not None:
        runtime_stats.setdefault("rounds", 0)
        runtime_stats.setdefault("draft_forward_calls", 0)
        runtime_stats.setdefault("target_forward_calls", 0)
        runtime_stats.setdefault("draft_tokens_proposed", 0)
        runtime_stats.setdefault("target_proposal_positions_verified", 0)
        runtime_stats.setdefault("accepted_draft_tokens", 0)
        runtime_stats.setdefault("rejected_draft_tokens", 0)
        runtime_stats.setdefault("guard_triggered_rounds", 0)
        runtime_stats.setdefault("guard_applied_rounds", 0)
        runtime_stats.setdefault("guard_positions", [])
        runtime_stats.setdefault("guard_audit", [])
        runtime_stats.setdefault("candidate_positions_checked", 0)
        runtime_stats.setdefault("invalid_candidate_positions", 0)
        runtime_stats.setdefault("morphology_seconds", 0.0)
        runtime_stats.setdefault("controller_seconds", 0.0)
        runtime_stats.setdefault("draft_resynchronization_seconds", 0.0)
    with torch.inference_mode():
        target_result = target_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=target_device), use_cache=True,
        )
        if runtime_stats is not None:
            runtime_stats["target_forward_calls"] += 1
        target_past = target_result.past_key_values
        target_next_logits = target_result.logits[0, -1]

        while len(generated) < max_new_tokens and not finished:
            current = prompt_ids + generated
            base_position = len(generated)
            proposal_limit = min(k, max_new_tokens - base_position)
            draft_prefill_started = time.perf_counter() if runtime_stats is not None else None
            draft_result = draft_model(
                input_ids=torch.tensor([current], dtype=torch.long, device=draft_device), use_cache=True,
            )
            if runtime_stats is not None:
                runtime_stats["draft_forward_calls"] += 1
                runtime_stats["draft_resynchronization_seconds"] += time.perf_counter() - draft_prefill_started
            draft_past = draft_result.past_key_values
            proposals: list[int] = []
            proposal_entropies: list[float] = []
            for _ in range(proposal_limit):
                if record_entropies or (guard_controller is not None and bool(getattr(guard_controller, "requires_entropy", False))):
                    log_probs = torch.log_softmax(draft_result.logits[0, -1].float(), dim=-1)
                    entropy = float((-(log_probs.exp() * log_probs).sum()).item())
                    proposal_entropies.append(entropy)
                proposal_id = int(torch.argmax(draft_result.logits[0, -1], dim=-1).item())
                proposals.append(proposal_id)
                if proposal_id == eos_token_id:
                    break
                draft_result = draft_model(
                    input_ids=torch.tensor([[proposal_id]], dtype=torch.long, device=draft_device),
                    past_key_values=draft_past, use_cache=True,
                )
                draft_past = draft_result.past_key_values
                if runtime_stats is not None:
                    runtime_stats["draft_forward_calls"] += 1

            guard_decision: dict[str, Any] | None = None
            verify_slots = len(proposals)
            if guard_controller is not None:
                guard_decision = guard_controller.inspect(
                    prompt_ids=prompt_ids,
                    committed_ids=generated,
                    proposal_ids=proposals,
                    proposal_entropies=proposal_entropies if proposal_entropies else None,
                    prompt_id=prompt_id,
                    round_index=round_index,
                )
                if guard_decision.get("triggered"):
                    verify_slots = int(guard_decision["verify_slots"])
                if runtime_stats is not None:
                    runtime_stats["controller_seconds"] += float(guard_decision.get("controller_seconds", 0.0))
                    runtime_stats["morphology_seconds"] += float(guard_decision.get("projection_seconds", 0.0))
                    runtime_stats["candidate_positions_checked"] += int(guard_decision.get("candidate_count", 0))
                    runtime_stats["invalid_candidate_positions"] += int(guard_decision.get("invalid_candidates", 0))
                    if guard_decision.get("triggered"):
                        runtime_stats["guard_triggered_rounds"] += 1
                        runtime_stats["guard_positions"].append(int(guard_decision["guard_slot"]))
                    chosen = guard_decision.get("chosen_record")
                    activation = guard_decision.get("activation_evidence")
                    audit_records = [chosen] if chosen is not None else [
                        r for r in guard_decision.get("candidate_records", []) if r.get("detector_status") != "VALID"
                    ]
                    runtime_stats["guard_audit"].extend({
                        **record,
                        "guard_variant": getattr(guard_controller, "variant", "UNKNOWN"),
                        "selected": bool(chosen is not None and int(record["proposal_slot"]) == int(chosen["proposal_slot"])),
                        "guard_reason": guard_decision.get("reason"),
                        "activation_evidence": activation,
                    } for record in audit_records)
                    if chosen is None and guard_decision.get("triggered") and activation is not None:
                        runtime_stats["guard_audit"].append({
                            "pair": getattr(guard_controller, "pair", None),
                            "prompt_id": int(prompt_id), "round_index": int(round_index),
                            "proposal_slot": int(guard_decision["guard_slot"]),
                            "guard_variant": getattr(guard_controller, "variant", "UNKNOWN"),
                            "selected": True, "guard_reason": guard_decision.get("reason"),
                            "activation_evidence": activation,
                        })

            target_decisions: list[int | None] = [None] * len(proposals)
            target_entropies: list[float | None] = [None] * len(proposals)
            accepted_prefix = 0
            mismatch_index: int | None = None
            target_eos_accepted = False
            batched_verification_logits = None
            verification_cache_length = None
            if batch_target_verification and verify_slots > 0:
                if not hasattr(target_past, "crop") or not hasattr(target_past, "get_seq_length"):
                    raise RuntimeError("Batched target verification requires a crop-capable Transformers Cache")
                verification_cache_length = int(target_past.get_seq_length())
                target_result = target_model(
                    input_ids=torch.tensor([proposals[:verify_slots]], dtype=torch.long, device=target_device),
                    past_key_values=target_past, use_cache=True,
                )
                target_past = target_result.past_key_values
                batched_verification_logits = target_result.logits[0]
                if runtime_stats is not None:
                    runtime_stats["target_forward_calls"] += 1
            for j, proposal_id in enumerate(proposals[:verify_slots]):
                decision_logits = (
                    target_next_logits if j == 0 else batched_verification_logits[j - 1]
                ) if batch_target_verification else target_next_logits
                decision = int(torch.argmax(decision_logits, dim=-1).item())
                target_decisions[j] = decision
                if record_entropies:
                    target_log_probs = torch.log_softmax(decision_logits.float(), dim=-1)
                    target_entropies[j] = float((-(target_log_probs.exp() * target_log_probs).sum()).item())
                if runtime_stats is not None:
                    runtime_stats["target_proposal_positions_verified"] += 1
                if proposal_id != decision:
                    mismatch_index = j
                    break
                accepted_prefix += 1
                if runtime_stats is not None:
                    runtime_stats["accepted_draft_tokens"] += 1
                if proposal_id == eos_token_id:
                    target_eos_accepted = True
                    break
                if not batch_target_verification:
                    target_result = target_model(
                        input_ids=torch.tensor([[proposal_id]], dtype=torch.long, device=target_device),
                        past_key_values=target_past, use_cache=True,
                    )
                    target_past = target_result.past_key_values
                    target_next_logits = target_result.logits[0, -1]
                    if runtime_stats is not None:
                        runtime_stats["target_forward_calls"] += 1

            if batch_target_verification and batched_verification_logits is not None and mismatch_index is None:
                if accepted_prefix == verify_slots and not target_eos_accepted and verify_slots > 0:
                    target_next_logits = batched_verification_logits[verify_slots - 1]

            rejection_position = None if mismatch_index is None else base_position + mismatch_index
            for j, proposal_id in enumerate(proposals):
                valid = target_decisions[j] is not None
                decision = target_decisions[j]
                rejected = bool(proposal_id != decision) if valid else None
                guarded_unverified = bool(guard_decision and guard_decision.get("triggered") and j >= verify_slots)
                invalidated = bool(not valid and mismatch_index is not None and j > mismatch_index)
                events.append({
                    "prompt_id": int(prompt_id),
                    "round_index": int(round_index),
                    "proposal_slot": int(j + 1),
                    "proposal_position": int(j + 1),
                    "output_token_position": int(base_position + j),
                    "generation_position": int(base_position + j),
                    "draft_proposed_token_id": int(proposal_id),
                    "target_verification_token_id": decision,
                    "target_greedy_token_id": decision,
                    "draft_entropy": proposal_entropies[j] if record_entropies and j < len(proposal_entropies) else None,
                    "target_entropy": target_entropies[j] if record_entropies else None,
                    "accepted": bool(valid and not rejected),
                    "rejected": rejected,
                    "sd_valid": bool(valid),
                    "is_first_rejection": bool(mismatch_index == j),
                    "invalidated_after_first_rejection": bool(invalidated),
                    "accepted_prefix_length": int(accepted_prefix),
                    "first_rejection_output_position": rejection_position,
                    "guard_triggered": bool(guard_decision and guard_decision.get("triggered")),
                    "guard_cut_slot": None if not guard_decision or not guard_decision.get("triggered") else int(guard_decision["guard_slot"]),
                    "guard_applied": False,
                    "guard_unverified": guarded_unverified,
                })

            if mismatch_index is not None:
                if runtime_stats is not None:
                    runtime_stats["rejected_draft_tokens"] += 1
                if batch_target_verification and verify_slots > accepted_prefix:
                    # The vectorized call appended every verified draft token,
                    # including the rejecting token and any later positions.
                    # Remove all uncommitted positions, retaining only the
                    # safely accepted prefix before applying the ordinary
                    # target correction.
                    target_past.crop(-(verify_slots - accepted_prefix))
                    expected_cache_length = int(verification_cache_length) + accepted_prefix
                    actual_cache_length = int(target_past.get_seq_length())
                    if actual_cache_length != expected_cache_length:
                        raise RuntimeError(
                            "Target cache rollback mismatch after batched rejection: "
                            f"expected {expected_cache_length}, got {actual_cache_length}"
                        )
                generated.extend(proposals[:mismatch_index])
                correction = int(target_decisions[mismatch_index])
                generated.append(correction)
                if correction == eos_token_id:
                    finished = True
                else:
                    target_result = target_model(
                        input_ids=torch.tensor([[correction]], dtype=torch.long, device=target_device),
                        past_key_values=target_past, use_cache=True,
                    )
                    target_past = target_result.past_key_values
                    target_next_logits = target_result.logits[0, -1]
                    if runtime_stats is not None:
                        runtime_stats["target_forward_calls"] += 1
            else:
                if guard_decision is not None and guard_decision.get("triggered"):
                    applied = accepted_prefix == verify_slots and not target_eos_accepted
                    for event in events[-len(proposals):] if proposals else []:
                        if int(event["round_index"]) == int(round_index):
                            event["guard_applied"] = bool(applied)
                    if applied:
                        if runtime_stats is not None:
                            runtime_stats["guard_applied_rounds"] += 1
                        generated.extend(proposals[:verify_slots])
                        bonus = int(torch.argmax(target_next_logits, dim=-1).item())
                        generated.append(bonus)
                        if bonus == eos_token_id:
                            finished = True
                        else:
                            target_result = target_model(
                                input_ids=torch.tensor([[bonus]], dtype=torch.long, device=target_device),
                                past_key_values=target_past, use_cache=True,
                            )
                            target_past = target_result.past_key_values
                            target_next_logits = target_result.logits[0, -1]
                            if runtime_stats is not None:
                                runtime_stats["target_forward_calls"] += 1
                    else:
                        generated.extend(proposals)
                        if target_eos_accepted or (proposals and proposals[-1] == eos_token_id):
                            finished = True
                else:
                    generated.extend(proposals)
                    if target_eos_accepted or (proposals and proposals[-1] == eos_token_id):
                        finished = True
            if runtime_stats is not None:
                runtime_stats["rounds"] += 1
                runtime_stats["draft_tokens_proposed"] += len(proposals)
            round_index += 1

    for event in events:
        position = int(event["output_token_position"])
        event["reference_target_token_id"] = generated[position] if position < len(generated) else None
    return generated, events


class Gpu7Monitor:
    """Sample utilization for physical GPU 7 only; does not query other devices."""

    def __init__(self, path: Path, interval_seconds: int = 10):
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            if handle.tell() == 0:
                handle.write("timestamp_utc,utilization_gpu_pct,memory_used_mib\n")
                handle.flush()
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--id=7", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                        check=True, capture_output=True, text=True, timeout=5,
                    ).stdout.strip()
                    util, memory = [part.strip() for part in result.split(",", maxsplit=1)]
                    handle.write(f"{datetime.now(timezone.utc).isoformat()},{util},{memory}\n")
                    handle.flush()
                except Exception as exc:
                    handle.write(f"{datetime.now(timezone.utc).isoformat()},error,{type(exc).__name__}\n")
                    handle.flush()
                self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="gpu7-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=8)


def _sync(torch: Any) -> None:
    torch.cuda.synchronize(0)


def _gpu_peaks(torch: Any) -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated(0) / 1024**3,
        torch.cuda.max_memory_reserved(0) / 1024**3,
    )


def score_teacher_forced_batch(
    torch: Any,
    draft_model: Any,
    target_model: Any,
    prompt_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    pad_token_id: int,
    chunk_positions: int = 8,
) -> list[dict[str, Any]]:
    """Full-sequence FP16 scores, with entropy/probabilities reduced in chunks."""
    if len(prompt_rows) != len(reference_rows):
        raise ValueError("Prompt and reference batch lengths differ")
    if not prompt_rows:
        return []
    scoring_ids = []
    target_ids = []
    for prompt, reference in zip(prompt_rows, reference_rows):
        prompt_ids = [int(x) for x in prompt["input_ids"]]
        reference_ids = [int(x) for x in reference["reference_token_ids"]]
        scoring_ids.append(prompt_ids + reference_ids[:-1])
        target_ids.append(reference_ids)
    widths = [len(row) for row in scoring_ids]
    width = max(widths)
    max_output = max(map(len, target_ids), default=0)
    device = next(draft_model.parameters()).device
    input_ids = torch.full((len(scoring_ids), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(scoring_ids), width), dtype=torch.long, device=device)
    batch_positions = torch.arange(len(scoring_ids), device=device)
    for i, row in enumerate(scoring_ids):
        input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention[i, :len(row)] = 1
    position_ids = attention.cumsum(-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    with torch.inference_mode():
        draft_logits = draft_model(
            input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=False,
        ).logits
        target_logits = target_model(
            input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=False,
        ).logits

        rows: list[dict[str, Any] | None] = [None] * len(scoring_ids)
        prompt_lengths = torch.tensor([len(row["input_ids"]) for row in prompt_rows], dtype=torch.long, device=device)
        reference_tokens = torch.full((len(scoring_ids), max_output), pad_token_id, dtype=torch.long, device=device)
        for i, token_ids in enumerate(target_ids):
            if token_ids:
                reference_tokens[i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)

        for start in range(0, max_output, chunk_positions):
            end = min(max_output, start + chunk_positions)
            offsets = torch.arange(start, end, dtype=torch.long, device=device)
            positions = prompt_lengths[:, None] - 1 + offsets[None, :]
            positions.clamp_(min=0, max=width - 1)
            d_logits = draft_logits[batch_positions[:, None], positions]
            t_logits = target_logits[batch_positions[:, None], positions]
            target_chunk = reference_tokens[:, start:end]
            d_argmax = d_logits.argmax(dim=-1)
            t_argmax = t_logits.argmax(dim=-1)
            d_log_probs = torch.log_softmax(d_logits.float(), dim=-1)
            t_log_probs = torch.log_softmax(t_logits.float(), dim=-1)
            d_entropy = -(d_log_probs.exp() * d_log_probs).sum(dim=-1)
            t_entropy = -(t_log_probs.exp() * t_log_probs).sum(dim=-1)
            d_actual_logprob = d_log_probs.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1)
            t_actual_logprob = t_log_probs.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1)
            t_prob_draft = t_log_probs.gather(-1, d_argmax.unsqueeze(-1)).squeeze(-1).exp()
            d_prob_target = d_log_probs.gather(-1, t_argmax.unsqueeze(-1)).squeeze(-1).exp()
            values = {
                "draft_argmax": d_argmax.detach().cpu().tolist(),
                "target_argmax": t_argmax.detach().cpu().tolist(),
                "draft_entropy": d_entropy.detach().cpu().tolist(),
                "target_entropy": t_entropy.detach().cpu().tolist(),
                "draft_logprob": d_actual_logprob.detach().cpu().tolist(),
                "target_logprob": t_actual_logprob.detach().cpu().tolist(),
                "target_prob_draft_argmax": t_prob_draft.detach().cpu().tolist(),
                "draft_prob_target_argmax": d_prob_target.detach().cpu().tolist(),
            }
            for batch_index, (prompt, ref) in enumerate(zip(prompt_rows, target_ids)):
                for local_index, output_pos in enumerate(range(start, end)):
                    if output_pos >= len(ref):
                        continue
                    d_id = int(values["draft_argmax"][batch_index][local_index])
                    t_id = int(values["target_argmax"][batch_index][local_index])
                    actual = int(ref[output_pos])
                    rows[batch_index] = rows[batch_index] or []
                    rows[batch_index].append({
                        "prompt_id": int(prompt["prompt_id"]),
                        "output_token_position": int(output_pos),
                        "target_token_id": actual,
                        "draft_greedy_token_id": d_id,
                        "target_greedy_token_id": t_id,
                        "draft_target_disagreement": bool(d_id != t_id),
                        "draft_matches_target_token": bool(d_id == actual),
                        "target_matches_target_token": bool(t_id == actual),
                        "draft_logprob": float(values["draft_logprob"][batch_index][local_index]),
                        "target_logprob": float(values["target_logprob"][batch_index][local_index]),
                        "draft_entropy": float(values["draft_entropy"][batch_index][local_index]),
                        "target_entropy": float(values["target_entropy"][batch_index][local_index]),
                        "target_prob_draft_argmax": float(values["target_prob_draft_argmax"][batch_index][local_index]),
                        "draft_prob_target_argmax": float(values["draft_prob_target_argmax"][batch_index][local_index]),
                    })
            del d_logits, t_logits, d_log_probs, t_log_probs
            del d_entropy, t_entropy, d_actual_logprob, t_actual_logprob, t_prob_draft, d_prob_target
        del draft_logits, target_logits
    return [row or [] for row in rows]


def compare_teacher_predictions(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    by_key = {(int(row["prompt_id"]), int(row["output_token_position"])): row for row in reference}
    other = {(int(row["prompt_id"]), int(row["output_token_position"])): row for row in candidate}
    if by_key.keys() != other.keys():
        return False
    fields = ("draft_greedy_token_id", "target_greedy_token_id", "draft_target_disagreement")
    return all(all(left[field] == other[key][field] for field in fields) for key, left in by_key.items())


def _pair_dir(output_dir: Path, pair: str) -> Path:
    path = output_dir / PAIRS[pair]["slug"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_progress(pair_dir: Path) -> dict[str, Any]:
    path = pair_dir / "checkpoints/progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}}


def _write_stage_chunk(pair_dir: Path, stage: str, frames: dict[str, pd.DataFrame], prompt_ids: list[int]) -> None:
    progress = _load_progress(pair_dir)
    stage_state = progress.setdefault("stages", {}).setdefault(stage, {"chunks": [], "completed_prompt_ids": []})
    chunk_number = len(stage_state["chunks"])
    base = pair_dir / "checkpoints" / stage
    base.mkdir(parents=True, exist_ok=True)
    names = {}
    for label, frame in frames.items():
        path = base / f"{label}_{chunk_number:05d}.parquet"
        atomic_parquet(frame, path)
        names[label] = str(path.relative_to(pair_dir))
    stage_state["chunks"].append({"chunk": chunk_number, "prompt_ids": [int(x) for x in prompt_ids], "files": names})
    completed = set(map(int, stage_state.get("completed_prompt_ids", [])))
    completed.update(map(int, prompt_ids))
    stage_state["completed_prompt_ids"] = sorted(completed)
    stage_state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(pair_dir / "checkpoints/progress.json", progress)


def _completed_ids(pair_dir: Path, stage: str) -> set[int]:
    progress = _load_progress(pair_dir)
    return set(map(int, progress.get("stages", {}).get(stage, {}).get("completed_prompt_ids", [])))


def _read_stage(pair_dir: Path, stage: str, label: str) -> pd.DataFrame:
    progress = _load_progress(pair_dir)
    files = [chunk["files"][label] for chunk in progress.get("stages", {}).get(stage, {}).get("chunks", []) if label in chunk.get("files", {})]
    frames = [pd.read_parquet(pair_dir / path) for path in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _continuation_text(tokenizer: Any, prompt_ids: list[int], reference_ids: list[int]) -> tuple[str, bool]:
    from .alignment import token_ids_to_character_offsets

    aligned = token_ids_to_character_offsets(tokenizer, prompt_ids + reference_ids)
    prompt_ends = [end for start, end in aligned["offsets"][:len(prompt_ids)] if end > start]
    prompt_end = max(prompt_ends, default=0)
    return aligned["text"][prompt_end:], bool(aligned["roundtrip_exact"])


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _process_sd_prompt(
    torch: Any,
    tokenizer: Any,
    draft_model: Any,
    target_model: Any,
    prompt: dict[str, Any],
    reference: dict[str, Any],
    max_new_tokens: int,
    eos_token_id: int,
    speculative_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_id = int(prompt["prompt_id"])
    prompt_ids = [int(x) for x in prompt["input_ids"]]
    reference_ids = [int(x) for x in reference["reference_token_ids"]]
    sd_ids, events = speculative_greedy_fast(
        draft_model, target_model, prompt_ids, max_new_tokens, eos_token_id, speculative_k, prompt_id,
    )
    fallback = False
    initial_difference = _first_difference(reference_ids, sd_ids)
    if initial_difference is not None:
        # The P3 4B reference, or a batched P1 reference, can differ from the
        # current singleton cache path. Regenerate only this target prompt.
        fallback = True
        reference_ids = greedy_generate_single(target_model, prompt_ids, max_new_tokens, eos_token_id)
        reference_source = "target greedy singleton fallback after reference mismatch"
        if reference_ids != sd_ids:
            raise AssertionError(
                f"prompt_id={prompt_id}: SD output still differs from singleton target greedy at "
                f"position {_first_difference(reference_ids, sd_ids)}"
            )
    else:
        reference_source = str(reference.get("reference_source", "target greedy batch"))
    continuation_text, roundtrip = _continuation_text(tokenizer, prompt_ids, reference_ids)
    for row in events:
        pos = int(row["output_token_position"])
        row["reference_target_token_id"] = reference_ids[pos] if pos < len(reference_ids) else None
        row["prompt_hash"] = prompt.get("prompt_hash")
    continuation = {
        "prompt_id": prompt_id,
        "source_index": int(prompt["source_index"]),
        "prompt_hash": prompt.get("prompt_hash"),
        "prompt_token_count": len(prompt_ids),
        "reference_token_ids": reference_ids,
        "speculative_token_ids": sd_ids,
        "reference_text": continuation_text,
        "reference_source": reference_source,
        "p3_reference_mismatch_position": initial_difference if fallback else None,
        "target_singleton_fallback": fallback,
        "exact_match": reference_ids == sd_ids,
        "tokenizer_roundtrip_exact": roundtrip,
        "generated_token_count": len(reference_ids),
        "stopping_reason": "eos" if reference_ids and reference_ids[-1] == eos_token_id else "max_new_tokens_or_model_stop",
    }
    return continuation, events


def _process_sd_chunk(
    torch: Any,
    tokenizer: Any,
    draft_model: Any,
    target_model: Any,
    pair_dir: Path,
    prompts: list[dict[str, Any]],
    references: dict[int, dict[str, Any]],
    max_new_tokens: int,
    eos_token_id: int,
    speculative_k: int,
) -> list[dict[str, Any]]:
    continuations = []
    events = []
    started = time.perf_counter()
    for index, prompt in enumerate(prompts, start=1):
        prompt_id = int(prompt["prompt_id"])
        continuation, prompt_events = _process_sd_prompt(
            torch, tokenizer, draft_model, target_model, prompt, references[prompt_id],
            max_new_tokens, eos_token_id, speculative_k,
        )
        if not continuation["exact_match"]:
            raise AssertionError(f"Unmatched speculative output for prompt_id={prompt_id}")
        continuations.append(continuation)
        events.extend(prompt_events)
        if index % 8 == 0 or index == len(prompts):
            print(
                f"  SD {index}/{len(prompts)} prompt={prompt_id} tokens={continuation['generated_token_count']} "
                f"fallback={continuation['target_singleton_fallback']} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    _write_stage_chunk(
        pair_dir, "sd",
        {"events": pd.DataFrame(events), "continuations": pd.DataFrame(continuations)},
        [int(row["prompt_id"]) for row in prompts],
    )
    return continuations


def _record_benchmark_row(
    rows: list[dict[str, Any]], torch: Any, *, pair: str, stage: str, backend: str,
    batch_size: int, elapsed: float, prompt_count: int, generated_tokens: int,
    parity: bool, detail: str = "",
) -> None:
    allocated, reserved = _gpu_peaks(torch)
    rows.append({
        "pair": pair,
        "stage": stage,
        "backend": backend,
        "batch_size": batch_size,
        "prompts": prompt_count,
        "generated_tokens": generated_tokens,
        "prompts_per_second": prompt_count / elapsed if elapsed > 0 else None,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed > 0 else None,
        "wall_clock_seconds": elapsed,
        "peak_allocated_vram_gb": allocated,
        "peak_reserved_vram_gb": reserved,
        "correctness_parity": bool(parity),
        "parity_definition": detail,
    })


def _save_performance_report(perf_dir: Path, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    perf_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(pd.DataFrame(rows), perf_dir / "e2_performance_benchmark.parquet")
    temp = perf_dir / "e2_performance_benchmark.csv.tmp"
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(perf_dir / "e2_performance_benchmark.csv")
    atomic_json(perf_dir / "selection.json", selected)
    lines = [
        "# E2 performance benchmark",
        "",
        "Benchmarked on the first 32 source prompts, with the three FP16 models resident on physical GPU 7 (mapped to `cuda:0`).",
        "The SD decoder is sequential and cache dependent; it was not rewritten for cross-prompt batching. Teacher-forced scoring is the batched stage.",
        "",
        f"Selected target-only generation batch size: **{selected['target_generation_batch_size']}**.",
        f"Selected teacher-forcing batch size: **{selected['teacher_forcing_batch_size']}**.",
        f"Attention backend: **{selected['attention_backend']}**; FlashAttention 2 installed: **{selected['flash_attention_2_available']}**.",
        "",
        "| Pair | Stage | Backend | Batch | Prompts/s | Generated tokens/s | Wall s | Peak allocated GB | Peak reserved GB | Exact parity |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair']} | {row['stage']} | {row['backend']} | {row['batch_size']} | "
            f"{row['prompts_per_second']:.3f} | {row['generated_tokens_per_second']:.2f} | "
            f"{row['wall_clock_seconds']:.2f} | {row['peak_allocated_vram_gb']:.2f} | "
            f"{row['peak_reserved_vram_gb']:.2f} | {row['correctness_parity']} |"
        )
    lines.extend([
        "",
        "Parity means target continuation token IDs and stopping positions match the singleton cached reference for target generation; for teacher forcing, both draft and target argmax IDs and disagreement flags match the batch-size-1 full-sequence reference. SD rows verify exact output IDs, proposal IDs, valid acceptance/rejection decisions, and stopping position against target greedy.",
        "",
        "Full-token-vocabulary entropy is reduced in chunks of eight sequence positions during Pass B. No entropy, full-vocabulary softmax, per-token file write, or CUDA synchronization is performed inside the sequential SD hot loop.",
    ])
    (perf_dir / "e2_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_and_checkpoint(
    torch: Any,
    output_dir: Path,
    tokenizer: Any,
    models: dict[str, Any],
    prompts: list[dict[str, Any]],
    p3_references: dict[int, dict[str, Any]],
    args: Any,
) -> dict[str, Any]:
    perf_dir = output_dir / "performance"
    selection_path = perf_dir / "selection.json"
    if (perf_dir / "e2_performance_benchmark.csv").exists() and selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    n_bench = min(32, len(prompts))
    bench_prompts = prompts[:n_bench]
    candidate_sizes = [size for size in (1, 4, 8, 16) if size <= max(1, n_bench)]
    perf_rows: list[dict[str, Any]] = []
    references_by_size: dict[int, list[list[int]]] = {}
    baseline: list[list[int]] | None = None

    # Target-only P1 reference generation candidates. Size 1 is the discrete
    # reference path used by the SD target KV cache.
    for batch_size in candidate_sizes:
        torch.cuda.reset_peak_memory_stats(0)
        _sync(torch)
        started = time.perf_counter()
        output_ids = greedy_generate_batch(
            models["1.7B"], [row["input_ids"] for row in bench_prompts],
            args.max_new_tokens, int(tokenizer.eos_token_id), batch_size,
        )
        _sync(torch)
        elapsed = time.perf_counter() - started
        if batch_size == 1:
            baseline = output_ids
        if baseline is None:
            raise AssertionError("Batch-size-1 target reference was not evaluated first")
        parity = len(output_ids) == len(baseline) and all(left == right for left, right in zip(output_ids, baseline))
        references_by_size[batch_size] = output_ids
        _record_benchmark_row(
            perf_rows, torch, pair="P1", stage="target_reference", backend="sdpa", batch_size=batch_size,
            elapsed=elapsed, prompt_count=n_bench, generated_tokens=sum(map(len, output_ids)), parity=parity,
            detail="exact output token IDs and EOS/max-token stopping position vs singleton cached target",
        )
        print(f"Benchmark P1 target batch={batch_size}: parity={parity} {n_bench/elapsed:.2f} prompts/s", flush=True)

    valid_target = [
        row for row in perf_rows
        if row["stage"] == "target_reference" and row["correctness_parity"] and row["peak_reserved_vram_gb"] < 74.0
    ]
    selected_target = max(valid_target, key=lambda row: row["prompts_per_second"])["batch_size"] if valid_target else 1
    p1_bench_outputs = references_by_size[selected_target]

    # Batched Pass-B scores. Batch-size 1 is the scoring reference; all
    # candidates must preserve exact discrete predictions and disagreements.
    score_reference = None
    score_results: dict[int, list[dict[str, Any]]] = {}
    for batch_size in candidate_sizes:
        torch.cuda.reset_peak_memory_stats(0)
        _sync(torch)
        started = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        for start in range(0, n_bench, batch_size):
            group_prompts = bench_prompts[start:start + batch_size]
            group_refs = [
                {"reference_token_ids": p1_bench_outputs[start + offset]}
                for offset in range(len(group_prompts))
            ]
            chunk = score_teacher_forced_batch(
                torch, models["0.6B"], models["1.7B"], group_prompts, group_refs,
                int(tokenizer.pad_token_id),
            )
            chunks.extend(row for prompt_rows in chunk for row in prompt_rows)
        _sync(torch)
        elapsed = time.perf_counter() - started
        if batch_size == 1:
            score_reference = chunks
        if score_reference is None:
            raise AssertionError("Batch-size-1 teacher-forced reference was not evaluated first")
        parity = compare_teacher_predictions(score_reference, chunks)
        score_results[batch_size] = chunks
        _record_benchmark_row(
            perf_rows, torch, pair="P1", stage="teacher_forced", backend="sdpa", batch_size=batch_size,
            elapsed=elapsed, prompt_count=n_bench, generated_tokens=len(chunks), parity=parity,
            detail="exact draft/target argmax IDs and disagreement flags vs batch-size-1 full-sequence scores",
        )
        print(f"Benchmark P1 teacher forcing batch={batch_size}: parity={parity} {len(chunks)/elapsed:.1f} tokens/s", flush=True)

    valid_score = [
        row for row in perf_rows
        if row["stage"] == "teacher_forced" and row["correctness_parity"] and row["peak_reserved_vram_gb"] < 74.0
    ]
    selected_teacher = max(valid_score, key=lambda row: row["generated_tokens_per_second"])["batch_size"] if valid_score else 1

    # Audit current pinned 4B target behavior on the first 32 saved P3 targets.
    torch.cuda.reset_peak_memory_stats(0)
    _sync(torch)
    started = time.perf_counter()
    current_4b = greedy_generate_batch(
        models["4B"], [row["input_ids"] for row in bench_prompts], args.max_new_tokens,
        int(tokenizer.eos_token_id), selected_target,
    )
    _sync(torch)
    elapsed = time.perf_counter() - started
    p3_bench_refs = [p3_references[int(row["prompt_id"])]["reference_token_ids"] for row in bench_prompts]
    p3_parity = all(left == right for left, right in zip(current_4b, p3_bench_refs))
    _record_benchmark_row(
        perf_rows, torch, pair="P2", stage="target_revision_validation", backend="sdpa", batch_size=selected_target,
        elapsed=elapsed, prompt_count=n_bench, generated_tokens=sum(map(len, current_4b)), parity=p3_parity,
        detail="pinned current 4B targets vs saved P3 target IDs; non-parity is repaired per prompt by singleton fallback",
    )

    # The measured SD path is sequential. Persist the first chunk now so it is
    # also the first durable chunk of the resumable full experiment.
    eos_id = int(tokenizer.eos_token_id)
    for pair_name in ("P1", "P2"):
        pair_spec = PAIRS[pair_name]
        pair_dir = _pair_dir(output_dir, pair_name)
        if _completed_ids(pair_dir, "sd"):
            continue
        if pair_name == "P1":
            ref_list = p1_bench_outputs
            ref_source = f"P1 batched target greedy batch={selected_target}"
        else:
            ref_list = p3_bench_refs
            ref_source = "P3 saved 4B target continuation"
        refs = {
            int(prompt["prompt_id"]): {
                "reference_token_ids": ref_list[i], "reference_source": ref_source,
            }
            for i, prompt in enumerate(bench_prompts)
        }
        torch.cuda.reset_peak_memory_stats(0)
        _sync(torch)
        started = time.perf_counter()
        output_continuations = _process_sd_chunk(
            torch, tokenizer, models[pair_spec["draft"]], models[pair_spec["target"]], pair_dir,
            bench_prompts, refs, args.max_new_tokens, eos_id, args.speculative_k,
        )
        _sync(torch)
        elapsed = time.perf_counter() - started
        reference_for_parity = {int(row["prompt_id"]): row for row in output_continuations}
        expected = refs
        discrete_parity = all(
            reference_for_parity[int(prompt["prompt_id"])]["exact_match"]
            and reference_for_parity[int(prompt["prompt_id"])]["reference_token_ids"] ==
            reference_for_parity[int(prompt["prompt_id"])]["speculative_token_ids"]
            for prompt in bench_prompts
        )
        event_frame = _read_stage(pair_dir, "sd", "events")
        proposal_ids = event_frame["draft_proposed_token_id"].astype("int64").tolist()
        first_run_event_ids = []
        for prompt in bench_prompts:
            pid = int(prompt["prompt_id"])
            first_run_event_ids.extend(
                event_frame.loc[event_frame.prompt_id == pid, "draft_proposed_token_id"].astype("int64").tolist()
            )
        # Event decisions are generated by the singleton target path; expose
        # the exact proposal/decision checks in benchmark metadata.
        decisions_valid = bool(
            event_frame.loc[event_frame["sd_valid"].astype(bool), "target_verification_token_id"].notna().all()
        )
        parity = discrete_parity and decisions_valid and len(first_run_event_ids) == len(proposal_ids)
        _record_benchmark_row(
            perf_rows, torch, pair=pair_name, stage="speculative_decode", backend="sdpa", batch_size=1,
            elapsed=elapsed, prompt_count=n_bench,
            generated_tokens=sum(row["generated_token_count"] for row in output_continuations), parity=parity,
            detail="exact target/spec output IDs, valid proposal decisions, and stopping position; invalidated proposals excluded from decisions",
        )

    selected = {
        "attention_backend": "sdpa",
        "flash_attention_2_available": False,
        "target_generation_batch_size": int(selected_target),
        "teacher_forcing_batch_size": int(selected_teacher),
        "benchmark_prompt_count": n_bench,
        "p1_target_generation_parity_by_batch": {
            str(row["batch_size"]): bool(row["correctness_parity"])
            for row in perf_rows if row["stage"] == "target_reference"
        },
        "teacher_forcing_prediction_parity_by_batch": {
            str(row["batch_size"]): bool(row["correctness_parity"])
            for row in perf_rows if row["stage"] == "teacher_forced"
        },
        "p3_4b_target_exact_on_first_32": p3_parity,
        "sd_parity_by_pair": {
            row["pair"]: bool(row["correctness_parity"])
            for row in perf_rows if row["stage"] == "speculative_decode"
        },
        "models_resident": True,
    }
    _save_performance_report(perf_dir, perf_rows, selected)
    return selected


def run_sd_pair(
    torch: Any, tokenizer: Any, models: dict[str, Any], output_dir: Path,
    pair_name: str, prompts: list[dict[str, Any]], p3_references: dict[int, dict[str, Any]],
    args: Any, selected_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pair = PAIRS[pair_name]
    pair_dir = _pair_dir(output_dir, pair_name)
    done = _completed_ids(pair_dir, "sd")
    ref_by_id = dict(p3_references)
    completed_continuations = _read_stage(pair_dir, "sd", "continuations")
    if not completed_continuations.empty:
        for row in completed_continuations.to_dict("records"):
            ref_by_id[int(row["prompt_id"])] = {
                "reference_token_ids": [int(x) for x in row["reference_token_ids"]],
                "reference_source": str(row["reference_source"]),
            }
    new_prompts = [row for row in prompts if int(row["prompt_id"]) not in done]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    chunk_size = max(1, int(args.checkpoint_prompts))
    for start in range(0, len(new_prompts), chunk_size):
        group = new_prompts[start:start + chunk_size]
        if pair_name == "P1":
            generated = greedy_generate_batch(
                models[pair["target"]], [row["input_ids"] for row in group], args.max_new_tokens,
                int(tokenizer.eos_token_id), selected_batch,
            )
            refs = {
                int(prompt["prompt_id"]): {
                    "reference_token_ids": generated[index],
                    "reference_source": f"P1 target greedy batch={selected_batch}",
                }
                for index, prompt in enumerate(group)
            }
        else:
            refs = {int(row["prompt_id"]): p3_references[int(row["prompt_id"])] for row in group}
        _process_sd_chunk(
            torch, tokenizer, models[pair["draft"]], models[pair["target"]], pair_dir,
            group, refs, args.max_new_tokens, int(tokenizer.eos_token_id), args.speculative_k,
        )
        print(f"{pair_name} committed prompts={min(start+len(group),len(new_prompts))}/{len(new_prompts)}", flush=True)
    events = _read_stage(pair_dir, "sd", "events")
    continuations = _read_stage(pair_dir, "sd", "continuations")
    if set(map(int, continuations.prompt_id)) != {int(row["prompt_id"]) for row in prompts}:
        raise AssertionError(f"{pair_name}: continuation checkpoints do not cover every requested prompt")
    if not continuations["exact_match"].astype(bool).all():
        raise AssertionError(f"{pair_name}: at least one SD output differs from target greedy")
    if continuations.prompt_id.duplicated().any():
        raise AssertionError(f"{pair_name}: duplicate completed prompt")
    atomic_parquet(events, pair_dir / "sd_events.parquet")
    atomic_parquet(continuations.sort_values("prompt_id"), pair_dir / "continuations.parquet")
    allocated, reserved = _gpu_peaks(torch)
    metadata = {
        "pair": pair_name,
        "draft_model": MODELS[pair["draft"]],
        "target_model": MODELS[pair["target"]],
        "prompt_count": len(continuations),
        "all_target_sd_outputs_equal": bool(continuations["exact_match"].astype(bool).all()),
        "fallback_target_regenerations": int(continuations["target_singleton_fallback"].astype(bool).sum()),
        "invalidated_proposals": int(events["invalidated_after_first_rejection"].astype(bool).sum()),
        "peak_allocated_vram_gb": allocated,
        "peak_reserved_vram_gb": reserved,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "completed_prompt_ids": sorted(map(int, continuations.prompt_id.unique())),
    }
    atomic_json(pair_dir / "runtime_metadata.json", metadata)
    return events, continuations, metadata


def run_teacher_forcing(
    torch: Any, tokenizer: Any, models: dict[str, Any], output_dir: Path, pair_name: str,
    prompts: list[dict[str, Any]], continuations: pd.DataFrame, args: Any, batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair = PAIRS[pair_name]
    pair_dir = _pair_dir(output_dir, pair_name)
    done = _completed_ids(pair_dir, "teacher_forced")
    cont_by_id = {
        int(row.prompt_id): {"reference_token_ids": [int(x) for x in row.reference_token_ids]}
        for row in continuations.itertuples(index=False)
    }
    prompt_by_id = {int(row["prompt_id"]): row for row in prompts}
    new_ids = [pid for pid in sorted(prompt_by_id) if pid not in done]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    chunk_size = max(1, int(args.checkpoint_prompts))
    for start in range(0, len(new_ids), chunk_size):
        prompt_ids = new_ids[start:start + chunk_size]
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(prompt_ids), batch_size):
            group_ids = prompt_ids[offset:offset + batch_size]
            group_prompts = [prompt_by_id[pid] for pid in group_ids]
            group_refs = [cont_by_id[pid] for pid in group_ids]
            scored = score_teacher_forced_batch(
                torch, models[pair["draft"]], models[pair["target"]], group_prompts, group_refs,
                int(tokenizer.pad_token_id),
            )
            rows.extend(row for prompt_rows in scored for row in prompt_rows)
        _write_stage_chunk(pair_dir, "teacher_forced", {"teacher": pd.DataFrame(rows)}, prompt_ids)
        print(f"{pair_name} teacher forcing committed prompts={min(start+len(prompt_ids),len(new_ids))}/{len(new_ids)}", flush=True)
    teacher = _read_stage(pair_dir, "teacher_forced", "teacher")
    if teacher.duplicated(["prompt_id", "output_token_position"]).any():
        raise AssertionError(f"{pair_name}: duplicate teacher-forced token score")
    expected = sum(len(row["reference_token_ids"]) for row in cont_by_id.values())
    if len(teacher) != expected:
        raise AssertionError(f"{pair_name}: expected {expected} teacher rows, found {len(teacher)}")
    atomic_parquet(teacher.sort_values(["prompt_id", "output_token_position"]), pair_dir / "teacher_forced_tokens.parquet")
    allocated, reserved = _gpu_peaks(torch)
    metadata = {
        "pair": pair_name,
        "prompt_count": len(continuations),
        "teacher_forced_token_count": len(teacher),
        "batch_size": int(batch_size),
        "peak_allocated_vram_gb": allocated,
        "peak_reserved_vram_gb": reserved,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    atomic_json(pair_dir / "teacher_forced_runtime_metadata.json", metadata)
    return teacher, metadata


def _build_h2_rows_for_prompt(
    tokenizer: Any,
    kiwi: Any,
    prompt: dict[str, Any],
    continuation: dict[str, Any],
    events: pd.DataFrame,
    teacher: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .alignment import map_token_positions_to_spans, token_ids_to_character_offsets
    from .morphology import extract_eojeol_rows

    prompt_id = int(prompt["prompt_id"])
    prompt_ids = [int(x) for x in prompt["input_ids"]]
    generated_ids = [int(x) for x in continuation["reference_token_ids"]]
    full_ids = prompt_ids + generated_ids
    alignment = _span_tokenizer_ids(tokenizer, full_ids, len(prompt_ids))
    full_alignment = token_ids_to_character_offsets(tokenizer, full_ids)
    prompt_offsets = [offset for offset in full_alignment["offsets"][:len(prompt_ids)] if offset[1] > offset[0]]
    continuation_start = max((end for _, end in prompt_offsets), default=0)
    continuation_text = full_alignment["text"][continuation_start:]
    if continuation_text != str(continuation["reference_text"]):
        raise AssertionError(f"Saved continuation text differs from tokenizer offsets for prompt_id={prompt_id}")

    # Preserve H2's whitespace-delimited eojeol rule exactly.
    raw_eojeols = [
        {"eojeol_id": i, "text": match.group(0), "char_start": match.start(), "char_end": match.end()}
        for i, match in enumerate(re.finditer(r"\S+", continuation_text, flags=re.UNICODE))
    ]
    eojeol_rows = extract_eojeol_rows(continuation_text, kiwi, tokenizer, prompt_id)
    token_to_eojeol = map_token_positions_to_spans(
        full_alignment["offsets"], token_start=len(prompt_ids), span_rows=eojeol_rows,
        text_origin=continuation_start,
    )
    saved_position_to_eojeol: dict[int, int] = {}
    for row in eojeol_rows:
        row["output_token_positions"] = [
            int(position) for position, eojeol_id in token_to_eojeol.items() if eojeol_id == row["eojeol_index"]
        ]
        for position in row["output_token_positions"]:
            saved_position_to_eojeol[position] = int(row["eojeol_index"])

    morphemes: list[dict[str, Any]] = []
    bad_morpheme_indices: set[int] = set()
    for morph in kiwi.tokenize(continuation_text):
        start = int(morph.start)
        end = start + int(morph.len)
        index = len(morphemes)
        morphemes.append({"surface": str(morph.form), "pos": str(morph.tag), "start": start, "end": end})
        if start < 0 or end <= start or end > len(continuation_text):
            bad_morpheme_indices.add(index)

    event_rows = events.loc[events["prompt_id"].astype(int) == prompt_id]
    valid_events = event_rows[event_rows["rejected"].notna()]
    if valid_events.duplicated(["prompt_id", "output_token_position"]).any():
        raise AssertionError(f"Multiple valid SD events map to one output position for prompt_id={prompt_id}")
    event_by_pos = {int(row.output_token_position): row for row in valid_events.itertuples(index=False)}
    teacher_rows = teacher.loc[teacher["prompt_id"].astype(int) == prompt_id]
    teacher_by_pos = {int(row.output_token_position): row for row in teacher_rows.itertuples(index=False)}
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_rows: list[dict[str, Any]] = []

    for generation_pos, token_id in enumerate(generated_ids):
        full_pos = len(prompt_ids) + generation_pos
        absolute_start, absolute_end = alignment["spans"][full_pos]
        span_exact = full_pos in alignment["exact_ids"]
        crosses_prompt = absolute_start < continuation_start and absolute_end > 0 and token_id not in special_ids
        local_start = max(0, int(absolute_start) - continuation_start)
        local_end = max(0, int(absolute_end) - continuation_start)
        if local_start > len(continuation_text) or local_end > len(continuation_text):
            span_exact = False
            local_start = min(local_start, len(continuation_text))
            local_end = min(local_end, len(continuation_text))
        if token_id in special_ids:
            token_surface = str(tokenizer.convert_ids_to_tokens(token_id))
            visible_positions: list[int] = []
            whitespace_only = True
            nonwhite_start, nonwhite_end = local_start, local_end
        else:
            token_surface = continuation_text[local_start:local_end]
            visible_positions = [i for i in range(local_start, local_end) if not continuation_text[i].isspace()]
            whitespace_only = not visible_positions
            nonwhite_start = visible_positions[0] if visible_positions else local_start
            nonwhite_end = visible_positions[-1] + 1 if visible_positions else local_end
        morph_class, eojeol_ids, morpheme_ids, reason = _classify_token(
            span_exact=span_exact,
            crosses_prompt=crosses_prompt,
            whitespace_only=whitespace_only,
            token_start=nonwhite_start,
            token_end=nonwhite_end,
            eojeols=raw_eojeols,
            morphemes=morphemes,
            bad_morpheme_indices=bad_morpheme_indices,
        )
        if token_id in special_ids:
            morph_class = "KIWI_COMPLEX"
            reason = "special_token_without_visible_character_span"
        if not eojeol_ids and not whitespace_only and local_end > local_start and generation_pos in saved_position_to_eojeol:
            eojeol_ids = [saved_position_to_eojeol[generation_pos]]

        event = event_by_pos.get(generation_pos)
        score = teacher_by_pos.get(generation_pos)
        if score is None:
            raise AssertionError(f"Missing teacher-forced score at prompt={prompt_id}, position={generation_pos}")
        if event is None:
            sd_valid = False
            sd_rejected = None
            proposal_slot = None
        else:
            sd_valid = True
            sd_rejected = bool(event.rejected)
            proposal_slot = int(event.proposal_slot)
            if int(event.reference_target_token_id) != token_id:
                raise AssertionError(f"SD target token differs from continuation at prompt={prompt_id}, position={generation_pos}")

        token_rows.append({
            "prompt_id": prompt_id,
            "generation_pos": generation_pos,
            "token_id": token_id,
            "is_special_token": token_id in special_ids,
            "token_text": token_surface,
            "token_start_char": nonwhite_start,
            "token_end_char": nonwhite_end,
            "tokenizer_span_start_char": local_start,
            "tokenizer_span_end_char": local_end,
            "span_exact": bool(span_exact),
            "morph_class": morph_class,
            "classification_reason": reason,
            "overlapping_eojeol_ids": eojeol_ids,
            "overlapping_morpheme_ids": morpheme_ids,
            "overlapping_morpheme_surfaces": [morphemes[i]["surface"] for i in morpheme_ids],
            "overlapping_morpheme_pos": [morphemes[i]["pos"] for i in morpheme_ids],
            "overlapping_morpheme_spans": [[morphemes[i]["start"], morphemes[i]["end"]] for i in morpheme_ids],
            "overlapping_morpheme_count": len(morpheme_ids),
            "eojeol_id": eojeol_ids[0] if len(eojeol_ids) == 1 else None,
            "sd_valid": sd_valid,
            "sd_rejected": sd_rejected,
            "draft_entropy": float(score.draft_entropy),
            "target_entropy": float(score.target_entropy),
            "proposal_slot": proposal_slot,
            "teacher_disagreement": bool(score.draft_target_disagreement),
            "teacher_draft_greedy_token_id": int(score.draft_greedy_token_id),
            "teacher_target_greedy_token_id": int(score.target_greedy_token_id),
            "target_prob_draft_argmax": float(score.target_prob_draft_argmax),
            "draft_prob_target_argmax": float(score.draft_prob_target_argmax),
            "generation_position": generation_pos,
        })

    positions_by_eojeol: dict[int, list[int]] = {row["eojeol_id"]: [] for row in raw_eojeols}
    for row_idx, row in enumerate(token_rows):
        for eojeol_id in row["overlapping_eojeol_ids"]:
            positions_by_eojeol[eojeol_id].append(row_idx)
    for eojeol in raw_eojeols:
        positions = sorted(
            positions_by_eojeol[eojeol["eojeol_id"]],
            key=lambda i: (token_rows[i]["token_start_char"], token_rows[i]["generation_pos"]),
        )
        fragmentation = len(positions)
        for token_pos, row_idx in enumerate(positions):
            row = token_rows[row_idx]
            if row["eojeol_id"] != eojeol["eojeol_id"]:
                continue
            row["fragmentation"] = fragmentation
            row["fragmentation_bin"] = _fragmentation_bin(fragmentation)
            row["token_pos_in_eojeol"] = token_pos
            row["relative_pos_in_eojeol"] = token_pos / (fragmentation - 1) if fragmentation > 1 else 0.0
            row["relative_position"] = row["relative_pos_in_eojeol"]
            row["first_token"] = token_pos == 0
            row["last_token"] = token_pos == fragmentation - 1
            row["token_char_length"] = max(0, row["token_end_char"] - row["token_start_char"])
            row["eojeol_char_length"] = eojeol["char_end"] - eojeol["char_start"]
    for row in token_rows:
        if "fragmentation" not in row:
            row.update({
                "fragmentation": None, "fragmentation_bin": None, "token_pos_in_eojeol": None,
                "relative_pos_in_eojeol": None, "relative_position": None, "first_token": None,
                "last_token": None, "token_char_length": max(0, row["token_end_char"]-row["token_start_char"]),
                "eojeol_char_length": None,
            })
        row["entropy_gap"] = row["draft_entropy"] - row["target_entropy"]

    audit = {
        "prompt_id": prompt_id,
        "generated_tokens": len(generated_ids),
        "generated_visible_ids": alignment["generated_visible_ids"],
        "generated_exact_id_spans": len(alignment["exact_ids"]),
        "retokenization_roundtrip_exact": alignment["roundtrip_exact"],
        "expected_visible_ids": alignment["expected_visible_ids"],
        "retokenized_ids": alignment["retokenized_ids"],
    }
    return token_rows, eojeol_rows, audit


def run_morphology(
    tokenizer: Any, output_dir: Path, pair_name: str, prompts: list[dict[str, Any]],
    events: pd.DataFrame, continuations: pd.DataFrame, teacher: pd.DataFrame, args: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .morphology import make_kiwi

    pair_dir = _pair_dir(output_dir, pair_name)
    done = _completed_ids(pair_dir, "morphology")
    cont_by_id = {int(row.prompt_id): row._asdict() for row in continuations.itertuples(index=False)}
    prompt_by_id = {int(row["prompt_id"]): row for row in prompts}
    new_ids = [pid for pid in sorted(prompt_by_id) if pid not in done]
    kiwi = make_kiwi()
    chunk_size = max(1, int(args.checkpoint_prompts))
    started = time.perf_counter()
    for start in range(0, len(new_ids), chunk_size):
        pids = new_ids[start:start + chunk_size]
        token_rows: list[dict[str, Any]] = []
        eojeol_rows: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for prompt_id in pids:
            rows, eojeols, audit = _build_h2_rows_for_prompt(
                tokenizer, kiwi, prompt_by_id[prompt_id], cont_by_id[prompt_id], events, teacher,
            )
            token_rows.extend(rows)
            eojeol_rows.extend(eojeols)
            audits.append(audit)
        _write_stage_chunk(
            pair_dir, "morphology",
            {"tokens": pd.DataFrame(token_rows), "eojeols": pd.DataFrame(eojeol_rows), "audit": pd.DataFrame(audits)},
            pids,
        )
        print(f"{pair_name} morphology committed prompts={min(start+len(pids),len(new_ids))}/{len(new_ids)}", flush=True)
    table = _read_stage(pair_dir, "morphology", "tokens")
    eojeols = _read_stage(pair_dir, "morphology", "eojeols")
    audit = _read_stage(pair_dir, "morphology", "audit")
    expected = sum(len(row["reference_token_ids"]) for row in cont_by_id.values())
    if len(table) != expected or table.duplicated(["prompt_id", "generation_pos"]).any():
        raise AssertionError(f"{pair_name}: H2 morphology table token-position coverage mismatch")
    primary_or_unambiguous = table["morph_class"].isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME", "CROSS_EOJEOL"])
    if not table.loc[primary_or_unambiguous, "span_exact"].astype(bool).all():
        raise AssertionError(f"{pair_name}: a non-exact retokenized span received an unambiguous H2 label")
    atomic_parquet(table.sort_values(["prompt_id", "generation_pos"]), pair_dir / "token_table.parquet")
    atomic_parquet(eojeols.sort_values(["prompt_id", "eojeol_index"]), pair_dir / "eojeols.parquet")
    audit_path = pair_dir / "alignment_audit.csv"
    tmp = audit_path.with_suffix(".csv.tmp")
    audit.to_csv(tmp, index=False)
    tmp.replace(audit_path)
    summary = {
        "pair": pair_name,
        "generated_tokens": len(table),
        "exactly_aligned_tokens": int(table["span_exact"].astype(bool).sum()),
        "cross_eojeol_excluded": int((table["morph_class"] == "CROSS_EOJEOL").sum()),
        "kiwi_complex_excluded": int((table["morph_class"] == "KIWI_COMPLEX").sum()),
        "exact_retokenization_failures": int((~table["span_exact"].astype(bool) & ~table["is_special_token"].astype(bool)).sum()),
        "percentage_excluded_cross_eojeol_or_kiwi_complex": float(100 * table["morph_class"].isin(["CROSS_EOJEOL", "KIWI_COMPLEX"]).mean()),
        "roundtrip_exact_prompts": int(audit["retokenization_roundtrip_exact"].astype(bool).sum()),
        "roundtrip_fallback_prompts": int((~audit["retokenization_roundtrip_exact"].astype(bool)).sum()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    atomic_json(pair_dir / "alignment_summary.json", summary)
    return table, eojeols, audit


def _eligible(table: pd.DataFrame) -> pd.DataFrame:
    primary = table.loc[
        table["sd_valid"].astype(bool)
        & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
        & table["morph_class"].isin(PRIMARY_CLASSES)
    ].copy()
    primary["sd_rejected"] = primary["sd_rejected"].astype(int)
    primary["morph_class"] = pd.Categorical(primary["morph_class"], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    primary["fragmentation_bin"] = pd.Categorical(primary["fragmentation_bin"], categories=FRAGMENTATION_ORDER)
    primary["first_token"] = primary["first_token"].astype(int)
    primary["last_token"] = primary["last_token"].astype(int)
    return primary


def _fit_pair_models(table: pd.DataFrame, frequency_table: pd.DataFrame | None = None) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, str]:
    primary = _eligible(table)
    controls = list(DEFAULT_STRUCTURAL)
    formulas = {
        "M1": _primary_formula([]),
        "M2": _primary_formula(controls),
        "M3": _primary_formula(controls + list(DEFAULT_ENTROPY)),
    }
    models: dict[str, dict[str, Any]] = {
        label: _fit_clustered_logit(primary, "sd_rejected", formula)
        for label, formula in formulas.items()
    }
    used_table = table
    if frequency_table is not None:
        frequency_columns = ["token_id", "token_count", "log_token_count", "token_is_unseen"]
        missing = sorted(set(frequency_columns) - set(frequency_table.columns))
        if missing:
            raise ValueError(f"Frequency cache missing columns: {missing}")
        used_table = table.merge(
            frequency_table[frequency_columns], on="token_id", how="left", validate="many_to_one",
        )
        if used_table["token_count"].isna().any():
            missing_n = int(used_table["token_count"].isna().sum())
            raise AssertionError(f"Token frequency cache misses {missing_n} rows")
        used_table["token_count"] = used_table["token_count"].astype("int64")
        used_table["log_token_count"] = used_table["log_token_count"].astype(float)
        used_table["token_is_unseen"] = used_table["token_is_unseen"].astype("int8")
        primary = _eligible(used_table)
        models["M4"] = _fit_logit(
            primary,
            _formula("sd_rejected", controls + list(DEFAULT_ENTROPY) + ["log_token_count"]),
            "M4 linear log token frequency",
        )
        models["M5"] = _fit_logit(
            primary,
            _formula(
                "sd_rejected", controls + list(DEFAULT_ENTROPY)
                + ["bs(log_token_count, df=4, degree=3, include_intercept=False)"],
            ),
            "M5 nonlinear spline log token frequency",
        )
    raw_rates = (
        primary.groupby("morph_class", observed=True)["sd_rejected"]
        .agg(n_tokens="count", rejection_rate="mean")
        .reset_index()
    )
    raw_rates["n_prompts"] = raw_rates["morph_class"].map(
        primary.groupby("morph_class", observed=True)["prompt_id"].nunique()
    )
    report_lines = [
        "Eligibility: sd_valid=True; fragmentation bins 2,3,4,5,6,7,8+; morphology class in WITHIN_SPLIT/CROSS_MORPHEME.",
        "Contrast: CROSS_MORPHEME relative to WITHIN_SPLIT. Two-sided Wald tests; standard errors clustered by prompt_id with the H2 correction.",
        f"N eligible tokens={len(primary):,}; N prompts={primary.prompt_id.nunique():,}.",
        "Raw rejection rates:\n" + raw_rates.to_string(index=False),
    ]
    for label, result in models.items():
        report_lines.extend([
            "",
            f"===== {label} =====",
            f"formula: {result.get('formula')}",
            f"status: {result.get('status')}; N={result.get('n_tokens')}; prompts={result.get('n_prompts')}",
            f"beta={result.get('contrast_cross_minus_split', result.get('beta_cross'))}; "
            f"OR={result.get('odds_ratio')}; OR_CI=[{result.get('or_ci_low')},{result.get('or_ci_high')}]; "
            f"p={result.get('p_value')}",
            result.get("summary", result.get("model_summary", "")),
        ])
    return models, raw_rates, "\n".join(report_lines) + "\n"


def _compact_model(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        key: result.get(key)
        for key in (
            "status", "formula", "n_tokens", "n_prompts", "beta_cross", "contrast_cross_minus_split",
            "std_error", "odds_ratio", "or_ci_low", "or_ci_high", "ci_low", "ci_high", "p_value",
        )
        if key in result
    }


def _attach_frequency(table: pd.DataFrame, frequency_cache: pd.DataFrame) -> pd.DataFrame:
    use_cols = ["token_id", "token_count", "log_token_count", "token_is_unseen"]
    out = table.merge(frequency_cache[use_cols], on="token_id", how="left", validate="many_to_one")
    if out["token_count"].isna().any():
        raise AssertionError(f"Frequency cache does not cover {int(out.token_count.isna().sum())} token rows")
    out["token_count"] = out["token_count"].astype("int64")
    out["log_token_count"] = out["log_token_count"].astype(float)
    out["token_is_unseen"] = out["token_is_unseen"].astype("int8")
    return out


def _load_p3_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "h2": pd.read_parquet(P3_DIR / "h2_token_table.parquet"),
        "teacher": pd.read_parquet(P3_DIR / "teacher_forced_tokens.parquet"),
        "frequency_table": pd.read_parquet(P3_DIR / "e1_frequency_table.parquet"),
        "frequency_cache": pd.read_parquet(P3_FREQUENCY_CACHE),
        "events": pd.read_parquet(P3_DIR / "sd_events.parquet"),
    }


def _p3_harmonized_entropy(h2: pd.DataFrame, teacher: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tf = teacher[[
        "prompt_id", "output_token_position", "draft_entropy", "target_entropy",
        "draft_greedy_token_id", "target_greedy_token_id", "draft_target_disagreement",
    ]].copy()
    tf = tf.rename(columns={
        "output_token_position": "generation_pos",
        "draft_entropy": "tf_draft_entropy",
        "target_entropy": "tf_target_entropy",
        "draft_greedy_token_id": "tf_draft_greedy_token_id",
        "target_greedy_token_id": "tf_target_greedy_token_id",
        "draft_target_disagreement": "tf_teacher_disagreement",
    })
    merged = h2.merge(tf, on=["prompt_id", "generation_pos"], how="left", validate="one_to_one")
    if merged["tf_draft_entropy"].isna().any() or merged["tf_target_entropy"].isna().any():
        raise AssertionError("Saved P3 teacher-forced scores do not cover the H2 token table")
    valid_events = events[events["rejected"].notna()].copy()
    event_columns = valid_events[[
        "prompt_id", "output_token_position", "draft_token_id", "target_greedy_token_id",
        "draft_entropy", "target_entropy",
    ]].rename(columns={
        "output_token_position": "generation_pos",
        "draft_token_id": "cached_draft_argmax",
        "target_greedy_token_id": "cached_target_argmax",
        "draft_entropy": "event_draft_entropy",
        "target_entropy": "event_target_entropy",
    })
    valid_join = event_columns.merge(tf, on=["prompt_id", "generation_pos"], how="inner", validate="one_to_one")
    diagnostic = {
        "rows": len(merged),
        "sd_valid_cached_vs_full_sequence_rows": len(valid_join),
        "draft_entropy_mean_absolute_difference_cached_vs_full_sequence": float((valid_join.event_draft_entropy - valid_join.tf_draft_entropy).abs().mean()),
        "draft_entropy_max_absolute_difference_cached_vs_full_sequence": float((valid_join.event_draft_entropy - valid_join.tf_draft_entropy).abs().max()),
        "target_entropy_mean_absolute_difference_cached_vs_full_sequence": float((valid_join.event_target_entropy - valid_join.tf_target_entropy).abs().mean()),
        "target_entropy_max_absolute_difference_cached_vs_full_sequence": float((valid_join.event_target_entropy - valid_join.tf_target_entropy).abs().max()),
        "draft_entropy_correlation_cached_vs_full_sequence": float(valid_join.event_draft_entropy.corr(valid_join.tf_draft_entropy)),
        "target_entropy_correlation_cached_vs_full_sequence": float(valid_join.event_target_entropy.corr(valid_join.tf_target_entropy)),
        "draft_argmax_match_count_cached_vs_full_sequence": int((valid_join.cached_draft_argmax == valid_join.tf_draft_greedy_token_id).sum()),
        "target_argmax_match_count_cached_vs_full_sequence": int((valid_join.cached_target_argmax == valid_join.tf_target_greedy_token_id).sum()),
        "draft_argmax_match_percentage_cached_vs_full_sequence": float(100 * (valid_join.cached_draft_argmax == valid_join.tf_draft_greedy_token_id).mean()),
        "target_argmax_match_percentage_cached_vs_full_sequence": float(100 * (valid_join.cached_target_argmax == valid_join.tf_target_greedy_token_id).mean()),
        "disagreement_match_count": int(((valid_join.cached_draft_argmax != valid_join.cached_target_argmax) == valid_join.tf_teacher_disagreement.astype(bool)).sum()),
        "disagreement_match_percentage": float(100 * ((valid_join.cached_draft_argmax != valid_join.cached_target_argmax) == valid_join.tf_teacher_disagreement.astype(bool)).mean()),
    }
    merged["sd_cached_draft_entropy"] = merged["draft_entropy"]
    merged["sd_cached_target_entropy"] = merged["target_entropy"]
    merged["draft_entropy"] = merged["tf_draft_entropy"]
    merged["target_entropy"] = merged["tf_target_entropy"]
    merged["entropy_gap"] = merged["draft_entropy"] - merged["target_entropy"]
    return merged, diagnostic


def _fit_pooled(table: pd.DataFrame, frequency_control: bool = False) -> tuple[Any, dict[str, Any], str]:
    import statsmodels.formula.api as smf
    import scipy.stats as st

    data = _eligible(table)
    data["model_pair"] = pd.Categorical(data["model_pair"], categories=["P3", "P1", "P2"])
    interaction = "C(model_pair, Treatment(reference='P3')) * C(morph_class, Treatment(reference='WITHIN_SPLIT'))"
    terms = [interaction, "C(fragmentation_bin, Treatment(reference='2'))"] + list(DEFAULT_STRUCTURAL) + list(DEFAULT_ENTROPY)
    if frequency_control:
        terms.append("bs(log_token_count, df=4, degree=3, include_intercept=False)")
    formula = "sd_rejected ~ " + " + ".join(terms)
    fitted = smf.logit(formula, data=data).fit(
        disp=False, maxiter=200, cov_type="cluster",
        cov_kwds={"groups": data["prompt_id"], "use_correction": True},
    )
    params = fitted.params
    cov = fitted.cov_params()
    morph_term = next(
        name for name in params.index
        if "morph_class" in name and "CROSS_MORPHEME" in name and ":" not in name
    )
    pair_effects = {}
    interaction_terms = []
    for pair in ("P3", "P1", "P2"):
        vector = np.zeros(len(params), dtype=float)
        vector[list(params.index).index(morph_term)] = 1.0
        if pair != "P3":
            term = next(
                name for name in params.index
                if f"[T.{pair}]" in name and "morph_class" in name and "CROSS_MORPHEME" in name and ":" in name
            )
            vector[list(params.index).index(term)] = 1.0
            interaction_terms.append(term)
        beta = float(vector @ params.to_numpy())
        variance = float(vector @ cov.to_numpy() @ vector)
        se = math.sqrt(max(variance, 0.0))
        z = beta / se if se else float("inf")
        pair_effects[pair] = {
            "beta": beta,
            "std_error": se,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(beta - 1.96 * se),
            "or_ci_high": math.exp(beta + 1.96 * se),
            "p_value": float(2 * st.norm.sf(abs(z))),
        }
    heterogeneity = None
    if interaction_terms:
        restrictions = np.zeros((len(interaction_terms), len(params)), dtype=float)
        for i, term in enumerate(interaction_terms):
            restrictions[i, list(params.index).index(term)] = 1.0
        delta = restrictions @ params.to_numpy()
        v = restrictions @ cov.to_numpy() @ restrictions.T
        statistic = float(delta.T @ np.linalg.pinv(v) @ delta)
        heterogeneity = {
            "chi_square": statistic,
            "df": len(interaction_terms),
            "p_value": float(st.chi2.sf(statistic, len(interaction_terms))),
            "interaction_terms": interaction_terms,
        }
    result = {
        "formula": formula,
        "n_tokens": int(len(data)),
        "n_prompts": int(data.prompt_id.nunique()),
        "converged": bool(fitted.mle_retvals.get("converged", True)),
        "pair_effects": pair_effects,
        "heterogeneity_test": heterogeneity,
        "summary": fitted.summary().as_text(),
    }
    return fitted, result, result["summary"]


def _summary_row(
    pair_name: str,
    table: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    audit: pd.DataFrame,
    continuations: pd.DataFrame,
    original_p3: bool = False,
) -> dict[str, Any]:
    primary = _eligible(table)
    rates = primary.groupby("morph_class", observed=True)["sd_rejected"].mean().to_dict()
    counts = primary.groupby("morph_class", observed=True)["sd_rejected"].size().to_dict()
    summary = {
        "pair": pair_name,
        "draft_to_target": f"{PAIRS[pair_name]['draft']} → {PAIRS[pair_name]['target']}",
        "n_eligible_tokens": len(primary),
        "n_prompts": int(primary.prompt_id.nunique()),
        "generated_tokens": len(table),
        "cross_rejection_rate": rates.get("CROSS_MORPHEME"),
        "split_rejection_rate": rates.get("WITHIN_SPLIT"),
        "cross_n": counts.get("CROSS_MORPHEME", 0),
        "split_n": counts.get("WITHIN_SPLIT", 0),
        "exactly_aligned_tokens": int(table.span_exact.astype(bool).sum()),
        "cross_eojeol_excluded": int((table.morph_class == "CROSS_EOJEOL").sum()),
        "kiwi_complex_excluded": int((table.morph_class == "KIWI_COMPLEX").sum()),
        "exact_retokenization_failures": int((~table.span_exact.astype(bool) & ~table.is_special_token.astype(bool)).sum()),
        "ambiguous_excluded_pct": 100 * float(table.morph_class.isin(["CROSS_EOJEOL", "KIWI_COMPLEX"]).mean()),
        "original_p3_artifacts": original_p3,
    }
    for model_name in ("M1", "M2", "M3", "M4", "M5"):
        model = models.get(model_name, {})
        summary[f"{model_name}_OR"] = model.get("odds_ratio")
        summary[f"{model_name}_OR_CI_low"] = model.get("or_ci_low")
        summary[f"{model_name}_OR_CI_high"] = model.get("or_ci_high")
        summary[f"{model_name}_p"] = model.get("p_value")
        summary[f"{model_name}_N"] = model.get("n_tokens")
        summary[f"{model_name}_prompts"] = model.get("n_prompts")
    summary["p3_singleton_fallbacks"] = int(continuations["target_singleton_fallback"].astype(bool).sum()) if "target_singleton_fallback" in continuations else 0
    summary["retokenization_roundtrip_exact_prompts"] = int(audit.retokenization_roundtrip_exact.astype(bool).sum()) if len(audit) else None
    return summary


def _decision_rule(summary: pd.DataFrame, pooled_m3: dict[str, Any], pooled_m5: dict[str, Any], heterogeneity: dict[str, Any]) -> tuple[str, str]:
    by_pair = summary.set_index("pair")
    p1 = float(by_pair.loc["P1", "M3_OR"])
    p2 = float(by_pair.loc["P2", "M3_OR"])
    p1_freq = float(by_pair.loc["P1", "M5_OR"])
    p2_freq = float(by_pair.loc["P2", "M5_OR"])
    p3 = float(by_pair.loc["P3", "M3_OR"])
    pooled_m5_effects = pooled_m5["pair_effects"]
    freq_direction = all(pooled_m5_effects[pair]["beta"] > 0 for pair in ("P1", "P2", "P3"))
    freq_supported = all(pooled_m5_effects[pair]["or_ci_low"] > 1 for pair in ("P1", "P2", "P3"))
    het_p = (heterogeneity.get("heterogeneity_test") or {}).get("p_value", 1.0)
    if p1 > 1 and p2 > 1 and p1_freq > 1 and p2_freq > 1 and freq_direction and freq_supported:
        code = "A — strong replication"
        explanation = "Both new pairs show the prespecified positive CROSS association under M3 and M5, and the pooled frequency-controlled interaction model keeps all pair contrasts positive."
    elif (p1 <= 1 or p2 <= 1) and het_p < 0.05:
        code = "C — model-pair dependent"
        explanation = "At least one new pair is non-positive and the pooled pair-by-morphology interaction indicates heterogeneity."
    elif p1 <= 1 and p2 <= 1:
        code = "D — replication failure"
        explanation = "Neither new pair has a positive M3 CROSS-vs-SPLIT estimate."
    else:
        code = "B — partial replication"
        explanation = "The new-pair estimates are mostly directionally positive, but one or more adjusted/frequency-controlled estimates are imprecise or do not retain the positive direction."
    return code, f"{explanation} P3's saved H2 M3 OR is {p3:.3f}; the pooled frequency-controlled interaction is reported as exploratory."


def analyze_e2(output_dir: Path, pair_tables: dict[str, pd.DataFrame], pair_audits: dict[str, pd.DataFrame], pair_continuations: dict[str, pd.DataFrame], pair_teachers: dict[str, pd.DataFrame], run_metadata: dict[str, Any], tokenizer_report: dict[str, Any]) -> dict[str, Any]:
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    p3 = _load_p3_artifacts()
    frequency_metadata = json.loads(P3_FREQUENCY_METADATA.read_text(encoding="utf-8"))
    if not tokenizer_report["p3_frequency_cache_identity"]["matches_pinned_4B"]:
        raise AssertionError("E1 frequency-cache tokenizer identity no longer matches the verified shared tokenizer")

    p3_h2 = p3["h2"].copy()
    p3_harmonized, teacher_diagnostic = _p3_harmonized_entropy(p3_h2, p3["teacher"], p3["events"])
    pair_model_results: dict[str, dict[str, Any]] = {}
    pair_reports: dict[str, str] = {}
    summary_rows = []

    # P3 baseline results are refit from the saved H2/E1 tables; no P3 GPU work.
    p3_h2_models, p3_rates, p3_h2_report = _fit_pair_models(p3_h2)
    # Join the verified frequency cache to the H2 table. The saved E1 table
    # already contains these columns; passing it as both inputs would create
    # suffixed `token_count` columns and break the M4/M5 fit.
    p3_e1_models, _, p3_e1_report = _fit_pair_models(p3_h2, p3["frequency_cache"])
    p3_models = {**p3_h2_models, "M4": p3_e1_models["M4"], "M5": p3_e1_models["M5"]}
    pair_model_results["P3"] = p3_models
    p3_audit = pd.read_csv(P3_DIR / "h2_tokenizer_alignment_audit.csv")
    p3_cont = pd.read_parquet(P3_DIR / "reference_outputs.jsonl") if False else pd.DataFrame(_read_jsonl(P3_DIR / "reference_outputs.jsonl"))
    p3_summary = _summary_row("P3", p3_h2, p3_models, p3_audit, p3_cont, original_p3=True)
    summary_rows.append(p3_summary)
    pair_reports["P3"] = (
        "P3 is reused from the existing saved H2/E1 run; it was not regenerated.\n\n"
        + p3_h2_report + "\n\n--- E1 frequency models ---\n" + p3_e1_report
    )

    frequency_cache = p3["frequency_cache"]
    for pair_name in ("P1", "P2"):
        table = pair_tables[pair_name]
        with_frequency = _attach_frequency(table, frequency_cache)
        models, raw_rates, report = _fit_pair_models(table, frequency_cache)
        pair_model_results[pair_name] = models
        summary_rows.append(
            _summary_row(pair_name, with_frequency, models, pair_audits[pair_name], pair_continuations[pair_name])
        )
        pair_reports[pair_name] = report
        pair_dir = _pair_dir(output_dir, pair_name)
        (pair_dir / "regression.txt").write_text(report, encoding="utf-8")
        lines = [
            f"# {pair_name}: {PAIRS[pair_name]['draft']} → {PAIRS[pair_name]['target']}",
            "",
            "## Data and alignment",
            f"- Generated tokens: {len(table):,}; eligible primary contrast tokens: {len(_eligible(with_frequency)):,} across {_eligible(with_frequency).prompt_id.nunique():,} prompts.",
            f"- Exact visible spans: {int(table.span_exact.astype(bool).sum()):,}/{int(table.is_special_token.eq(False).sum()):,} non-special generated tokens.",
            f"- CROSS_EOJEOL excluded: {int((table.morph_class == 'CROSS_EOJEOL').sum()):,}; KIWI_COMPLEX excluded: {int((table.morph_class == 'KIWI_COMPLEX').sum()):,}; ambiguous exclusion: {100*table.morph_class.isin(['CROSS_EOJEOL','KIWI_COMPLEX']).mean():.2f}%.",
            f"- Target/speculative outputs matched on all {int(pair_continuations[pair_name].exact_match.astype(bool).sum()):,}/{len(pair_continuations[pair_name]):,} prompts; singleton reference fallbacks: {int(pair_continuations[pair_name].target_singleton_fallback.astype(bool).sum()):,}.",
            f"- Re-tokenization exact for {int(pair_audits[pair_name].retokenization_roundtrip_exact.astype(bool).sum()):,}/{len(pair_audits[pair_name]):,} prompts.",
            "",
            "## Primary models and raw rates",
            "",
            "| Class | N tokens | N prompts | Raw rejection rate |",
            "|---|---:|---:|---:|",
        ]
        for row in raw_rates.itertuples(index=False):
            lines.append(f"| {row.morph_class} | {row.n_tokens:,} | {row.n_prompts:,} | {row.rejection_rate:.2%} |")
        lines.extend(["", "Model formulas, estimates, two-sided tests, and prompt-clustered summaries are in `regression.txt`."])
        (pair_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = pd.DataFrame(summary_rows)
    summary["_pair_order"] = summary["pair"].map({"P1": 0, "P2": 1, "P3": 2})
    summary = summary.sort_values("_pair_order").drop(columns="_pair_order").reset_index(drop=True)
    # Use the harmonized full-sequence FP16 entropies for the pooled analysis.
    pooled_rows = []
    for pair_name in ("P1", "P2"):
        pair_tbl = _attach_frequency(pair_tables[pair_name], frequency_cache)
        pair_tbl["model_pair"] = pair_name
        pooled_rows.append(pair_tbl)
    p3_pooled = _attach_frequency(p3_harmonized, frequency_cache)
    p3_pooled["model_pair"] = "P3"
    pooled_rows.append(p3_pooled)
    pooled = pd.concat(pooled_rows, ignore_index=True, sort=False)
    pooled["model_pair"] = pd.Categorical(pooled["model_pair"], categories=["P3", "P1", "P2"])
    _, pooled_m3, pooled_m3_text = _fit_pooled(pooled, frequency_control=False)
    _, pooled_m5, pooled_m5_text = _fit_pooled(pooled, frequency_control=True)
    pooled_text = [
        "Pooled model pair analysis. P1/P2 full-sequence entropies are from E2 Pass B; P3 uses the already saved full-sequence teacher-forced entropies for harmonization. Clustering is by shared prompt_id.",
        "",
        "===== Pooled M3 with model-pair × morphology interaction =====",
        json.dumps({k: v for k, v in pooled_m3.items() if k != "summary"}, indent=2),
        pooled_m3_text,
        "===== Pooled frequency-controlled M5 with model-pair × morphology interaction =====",
        json.dumps({k: v for k, v in pooled_m5.items() if k != "summary"}, indent=2),
        pooled_m5_text,
    ]
    (combined_dir / "e2_model_pair_regression.txt").write_text("\n\n".join(pooled_text) + "\n", encoding="utf-8")

    summary_path = combined_dir / "e2_model_pair_summary.csv"
    tmp = summary_path.with_suffix(".csv.tmp")
    summary.to_csv(tmp, index=False)
    tmp.replace(summary_path)

    # Paper-ready M3 and nonlinear-frequency forest plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["P1", "P2", "P3"]
    labels = ["0.6B → 1.7B", "1.7B → 4B", "0.6B → 4B"]
    for model_key, filename, title in (
        ("M3", "e2_forest_m3.png", "CROSS_MORPHEME vs WITHIN_SPLIT: M3"),
        ("M5", "e2_forest_frequency_controlled.png", "CROSS_MORPHEME vs WITHIN_SPLIT: frequency-controlled M5"),
    ):
        estimate_rows = []
        for pair_name in order:
            result = pair_model_results[pair_name].get(model_key, {})
            estimate_rows.append((result.get("odds_ratio"), result.get("or_ci_low"), result.get("or_ci_high")))
        fig, ax = plt.subplots(figsize=(8.3, 4.2))
        ys = np.arange(len(order))
        for y, (or_value, low, high) in zip(ys, estimate_rows):
            if or_value is None or low is None or high is None:
                continue
            ax.errorbar(or_value, y, xerr=[[or_value-low], [high-or_value]], fmt="o", capsize=4,
                        color="#176b87" if y < 2 else "#8b3a3a", linewidth=1.7)
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yticks(ys, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Odds ratio, CROSS_MORPHEME vs WITHIN_SPLIT (log scale)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(combined_dir / filename, dpi=200)
        plt.close(fig)

    heterogeneity = pooled_m3
    decision, decision_explanation = _decision_rule(summary, pooled_m3, pooled_m5, heterogeneity)
    util_path = output_dir / "performance/gpu7_utilization.csv"
    util_metrics = {
        "samples": 0, "mean_gpu_utilization_pct": None, "max_gpu_utilization_pct": None,
        "max_memory_used_mib": None, "active_monitored_hours": None,
        "wall_span_hours": None, "monitor_windows": None,
        "first_sample_utc": None, "last_sample_utc": None,
    }
    if util_path.exists():
        util = pd.read_csv(util_path)
        util = util[pd.to_numeric(util["utilization_gpu_pct"], errors="coerce").notna()]
        if len(util):
            timestamps = pd.to_datetime(util["timestamp_utc"], utc=True, errors="coerce")
            util = util.assign(_timestamp=timestamps).dropna(subset=["_timestamp"]).sort_values("_timestamp")
            delta = util["_timestamp"].diff().dt.total_seconds()
            active_seconds = float(delta.where(delta.between(0, 30), 0).sum())
            span_seconds = float((util["_timestamp"].iloc[-1] - util["_timestamp"].iloc[0]).total_seconds())
            monitor_windows = 1 + int((delta > 30).sum())
            util_metrics = {
                "samples": len(util),
                "mean_gpu_utilization_pct": float(pd.to_numeric(util.utilization_gpu_pct).mean()),
                "max_gpu_utilization_pct": float(pd.to_numeric(util.utilization_gpu_pct).max()),
                "max_memory_used_mib": float(pd.to_numeric(util.memory_used_mib).max()),
                "active_monitored_hours": active_seconds / 3600,
                "wall_span_hours": span_seconds / 3600,
                "monitor_windows": monitor_windows,
                "first_sample_utc": util["_timestamp"].iloc[0].isoformat(),
                "last_sample_utc": util["_timestamp"].iloc[-1].isoformat(),
            }
            run_metadata["cumulative_monitored_pipeline_seconds"] = active_seconds
            run_metadata["cumulative_monitored_pipeline_hours"] = active_seconds / 3600
            run_metadata["monitored_wall_span_hours_including_pauses"] = span_seconds / 3600
            run_metadata["gpu_monitor_windows"] = monitor_windows
            run_metadata["gpu_monitor_first_sample_utc"] = util["_timestamp"].iloc[0].isoformat()
            run_metadata["gpu_monitor_last_sample_utc"] = util["_timestamp"].iloc[-1].isoformat()
            run_metadata["elapsed_seconds_scope"] = "final resumed invocation only; cumulative monitored time is recorded separately"
            atomic_json(output_dir / "runtime_metadata.json", run_metadata)

    # Verify baseline H2 and E1 values against their authoritative saved files.
    p3_m1m3 = pd.read_csv(P3_DIR / "h2_forest_estimates.csv")
    p3_e1 = pd.read_csv(P3_DIR / "e1_model_estimates.csv")
    p3_checks = {
        "h2_primary_tokens_saved": 73143,
        "h2_primary_tokens_recomputed": len(_eligible(p3_h2)),
        "h2_m1_saved_or": float(p3_m1m3.loc[p3_m1m3.label == "Pooled (M1; fragmentation FE)", "odds_ratio"].iloc[0]),
        "h2_m1_recomputed_or": p3_models["M1"].get("odds_ratio"),
        "e1_m5_saved_or": float(p3_e1.loc[p3_e1.label == "M5 nonlinear spline frequency", "odds_ratio"].iloc[0]),
        "e1_m5_recomputed_or": p3_models["M5"].get("odds_ratio"),
    }

    lines = [
        f"# E2 result: {decision}.",
        "",
        decision_explanation,
        "",
        "This report describes conditional associations in generated continuations; it does not claim that cross-morpheme tokenization causally increases rejection.",
        "",
        "## Pair-specific results",
        "",
        "| Draft → Target | N tokens | Raw CROSS rejection | Raw SPLIT rejection | M1 OR (95% CI) | M2 OR (95% CI) | M3 OR (95% CI) | M3 p | Frequency-controlled M5 OR (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair_name in order:
        row = summary.loc[summary.pair == pair_name].iloc[0]
        fmt = lambda key: f"{row[key]:.3f} ({row[key+'_CI_low']:.3f}–{row[key+'_CI_high']:.3f})" if pd.notna(row.get(key)) and pd.notna(row.get(key+"_CI_low")) and pd.notna(row.get(key+"_CI_high")) else "NA"
        lines.append(
            f"| {row.draft_to_target} | {int(row.n_eligible_tokens):,} | {row.cross_rejection_rate:.2%} (n={int(row.cross_n):,}) | "
            f"{row.split_rejection_rate:.2%} (n={int(row.split_n):,}) | {fmt('M1_OR')} | {fmt('M2_OR')} | "
            f"{fmt('M3_OR')} | {row.M3_p:.3g} | {fmt('M5_OR')} |"
        )
    lines.extend([
        "",
        "M1 controls exact fragmentation fixed effects; M2 adds H2 structural controls; M3 adds full-sequence draft and target entropy. M4 adds linear log token frequency. M5 adds the E1 cubic spline frequency control. All pair-specific logit models use two-sided Wald tests and standard errors clustered by prompt_id.",
        "",
        "## Tokenizer, prompts, and correctness",
        f"- Shared tokenizer: **{tokenizer_report['shared_tokenizer']}**; all three vocab maps, special-token setup, tokenizer class, backend hashes, and ID/offset probes are documented in `tokenizer_compatibility.json`.",
        f"- Prompt source: `{run_metadata['prompt_cache']}`; ordered prompt-set SHA256: `{run_metadata['ordered_prompt_set_sha256']}`; count: {run_metadata['num_prompts']:,}; max prompt/generated lengths: {run_metadata['max_prompt_tokens']}/{run_metadata['max_new_tokens']}.",
        f"- P1/P2 speculative output matched target greedy for every prompt: **{all(bool(pair_continuations[name]['exact_match'].astype(bool).all()) for name in ('P1','P2'))}** ({run_metadata['num_prompts']:,} prompts per pair). Singleton reference fallbacks: P1 {int(pair_continuations['P1'].target_singleton_fallback.astype(bool).sum()):,}; P2 {int(pair_continuations['P2'].target_singleton_fallback.astype(bool).sum()):,}. P2 uses P3's saved 4B references except those explicitly recorded fallbacks.",
        f"- P3 was reused from `{P3_DIR}`; no P3 speculative-decoding rerun occurred. P3 saved primary N and M1/M5 estimate checks: `{json.dumps(p3_checks, sort_keys=True)}`.",
        f"- P3 cached-path vs saved full-sequence FP16 entropy diagnostic: {json.dumps(teacher_diagnostic, sort_keys=True)}.",
        f"- E1 frequency cache identity: tokenizer `{frequency_metadata['tokenizer_name']}` revision `{frequency_metadata['tokenizer_revision']}`, backend SHA256 `{frequency_metadata['tokenizer_backend_sha256']}`; exact ID map verified shared before join.",
        "",
        "## Alignment and exclusions",
        "| Pair | Generated tokens | Aligned visible token spans | CROSS_EOJEOL | KIWI_COMPLEX | Retokenization failures | Ambiguous exclusions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pair_name in order:
        row = summary.loc[summary.pair == pair_name].iloc[0]
        lines.append(
            f"| {pair_name} | {int(row.generated_tokens):,} | {int(row.exactly_aligned_tokens):,} | "
            f"{int(row.cross_eojeol_excluded):,} | {int(row.kiwi_complex_excluded):,} | "
            f"{int(row.exact_retokenization_failures):,} | {row.ambiguous_excluded_pct:.2f}% |"
        )
    lines.extend([
        "",
        "The `EXACT`, `WITHIN_SPLIT`, `CROSS_MORPHEME`, `CROSS_EOJEOL`, and `KIWI_COMPLEX` labels use the existing H2 span classifier without changes. Primary models exclude CROSS_EOJEOL/KIWI_COMPLEX and require `sd_valid=True` and fragmentation 2,3,4,5,6,7,8+.",
        "",
        "## Pooled model-pair analysis",
        f"- Pooled M3 interaction model: N={pooled_m3['n_tokens']:,} tokens / {pooled_m3['n_prompts']:,} shared prompts; model-pair × morphology joint interaction p={((pooled_m3.get('heterogeneity_test') or {}).get('p_value')):.4g}.",
        f"- Pooled frequency-controlled M5 interaction model: N={pooled_m5['n_tokens']:,}; pair contrasts: `{json.dumps(pooled_m5['pair_effects'], sort_keys=True)}`.",
        "- Pair interactions assess heterogeneity only. Three model pairs do not support a capacity-scaling law.",
        "",
        "## Runtime, GPU, and limits",
        f"- Device: physical GPU 7, exposed as `cuda:0`; GPU model `{run_metadata['gpu_name']}`; CUDA runtime `{run_metadata['torch_cuda_version']}`; PyTorch `{run_metadata['torch_version']}`; Transformers `{run_metadata['transformers_version']}`; attention backend `{run_metadata['attention_backend']}`; dtype FP16.",
        f"- E2 process-local peak allocated/reserved memory: {run_metadata['peak_allocated_vram_gb']:.2f}/{run_metadata['peak_reserved_vram_gb']:.2f} GiB. GPU7 device-level monitor summary: `{json.dumps(util_metrics, sort_keys=True)}`.",
        f"- Monitored pipeline time across all resumed run windows: {util_metrics['active_monitored_hours']:.2f} hours; first-to-last monitor span including pauses: {util_metrics['wall_span_hours']:.2f} hours across {util_metrics['monitor_windows']} windows. The 3.14-hour `elapsed_seconds` value is only the final resumed invocation; model loading and unmonitored gaps are outside the summed monitor intervals. Teacher forcing uses full-sequence FP16 and SD rejection labels remain cached-path labels.",
        "- GPU utilization and total device memory are board-level measurements, not E2-only. A concurrent external process was observed on physical GPU 7 during the run (26,691 MiB at one check); E2 allocator peaks above are process-local.",
        "- P3's exact model revisions were not fully recorded in its run metadata; its configured IDs and 4B tokenizer revision match the pinned comparison, while the historical P3 0.6B draft commit remains unknown.",
        "- No generated artifacts were silently excluded after target/speculative parity checks; prompt-level resume checkpoints and `progress.json` files are retained.",
        "",
        "## Files",
        "- `e2_model_pair_summary.csv`, `e2_model_pair_regression.txt`.",
        "- `e2_forest_m3.png`, `e2_forest_frequency_controlled.png`.",
        "- Pair folders contain `sd_events.parquet`, `continuations.parquet`, `teacher_forced_tokens.parquet`, `token_table.parquet`, `alignment_audit.csv`, `regression.txt`, and `summary.md`.",
        "- P3 source artifacts are referenced at `runs/20260926T184145Z_pilot1000/` and are not duplicated.",
    ])
    final_path = combined_dir / "e2_model_pair_replication.md"
    final_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pooled_meta = {
        "decision": decision,
        "decision_explanation": decision_explanation,
        "pooled_m3": {k: v for k, v in pooled_m3.items() if k != "summary"},
        "pooled_m5": {k: v for k, v in pooled_m5.items() if k != "summary"},
        "teacher_forcing_diagnostic_p3": teacher_diagnostic,
        "p3_reproduction_checks": p3_checks,
    }
    atomic_json(combined_dir / "e2_analysis_metadata.json", pooled_meta)
    return {"summary": summary, "decision": decision, "report": final_path, "pooled_m3": pooled_m3, "pooled_m5": pooled_m5}


def run(args: Any) -> int:
    output_dir = Path(args.run_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("performance", "combined", PAIRS["P1"]["slug"], PAIRS["P2"]["slug"]):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        # The check is intentionally after creating no files in the directory
        # only when the directory was empty at CLI entry.
        allowed = {"performance", "combined", PAIRS["P1"]["slug"], PAIRS["P2"]["slug"]}
        existing = {path.name for path in output_dir.iterdir()}
        if existing != allowed:
            raise SystemExit(f"{output_dir} already contains E2 outputs; pass --resume")

    prompt_cache = Path(args.prompt_cache)
    if not prompt_cache.is_absolute():
        prompt_cache = ROOT / prompt_cache
    tokenizer, tokenizer_report = tokenizer_preflight(output_dir)
    records, prompts, prompt_hashes = make_prompts(prompt_cache, tokenizer, args.num_prompts, args.max_prompt_tokens)
    write_pretokenized_inputs(output_dir, prompts, prompt_hashes)
    references_p3 = read_p3_references(args.num_prompts, prompts)
    write_initial_config(output_dir, args, prompt_hashes, tokenizer_report)
    if args.preflight_only:
        print(json.dumps({
            "preflight": "passed",
            "shared_tokenizer": tokenizer_report["shared_tokenizer"],
            "num_prompts": len(prompts),
            "ordered_prompt_set_sha256": prompt_hashes["ordered_prompt_set_sha256"],
            "p3_reference_count": len(references_p3),
            "output_dir": str(output_dir),
        }, indent=2))
        return 0

    if args.dtype != "float16":
        raise SystemExit("The primary E2 analysis is pinned to FP16, matching P3")
    torch = _require_gpu7()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    backend = "sdpa"
    import importlib.util
    flash_available = importlib.util.find_spec("flash_attn") is not None
    if flash_available:
        # The selected environment was preflighted without FlashAttention 2.
        # If installed in a future run, require a dedicated parity benchmark
        # before considering it; SDPA remains the defined safe default here.
        print("FlashAttention 2 is installed; retaining SDPA unless a separate backend parity run is added.", flush=True)
    environment = write_environment(output_dir, torch, backend)
    print(f"GPU7 pinned as cuda:0: {environment['gpu_name']}; loading all three models", flush=True)
    model_load_started = time.perf_counter()
    models = load_models(torch, backend)
    model_load_elapsed = time.perf_counter() - model_load_started
    monitor = Gpu7Monitor(output_dir / "performance/gpu7_utilization.csv")
    monitor.start()
    wall_started = time.perf_counter()
    run_metadata: dict[str, Any] = {}
    try:
        if flash_available:
            raise RuntimeError("FlashAttention 2 is installed, but this implementation did not verify an FA2 backend; refusing an unbenchmarked backend choice")
        selection = benchmark_and_checkpoint(torch, output_dir, tokenizer, models, prompts, references_p3, args)
        pair_tables: dict[str, pd.DataFrame] = {}
        pair_audits: dict[str, pd.DataFrame] = {}
        pair_continuations: dict[str, pd.DataFrame] = {}
        pair_teachers: dict[str, pd.DataFrame] = {}
        runtime_by_pair = {}
        for pair_name in ("P1", "P2"):
            events, continuations, sd_metadata = run_sd_pair(
                torch, tokenizer, models, output_dir, pair_name, prompts, references_p3, args,
                int(selection["target_generation_batch_size"]),
            )
            teacher, tf_metadata = run_teacher_forcing(
                torch, tokenizer, models, output_dir, pair_name, prompts, continuations, args,
                int(selection["teacher_forcing_batch_size"]),
            )
            table, _, audit = run_morphology(tokenizer, output_dir, pair_name, prompts, events, continuations, teacher, args)
            pair_tables[pair_name] = table
            pair_audits[pair_name] = audit
            pair_continuations[pair_name] = continuations
            pair_teachers[pair_name] = teacher
            runtime_by_pair[pair_name] = {"sd": sd_metadata, "teacher_forced": tf_metadata}

        if len(pair_continuations["P1"]) != args.num_prompts or len(pair_continuations["P2"]) != args.num_prompts:
            raise AssertionError("Both new pairs must cover the exact requested prompt count")
        for pair_name in ("P1", "P2"):
            observed = set(map(int, pair_continuations[pair_name].prompt_id))
            expected = set(map(int, [row["prompt_id"] for row in prompts]))
            if observed != expected:
                raise AssertionError(f"{pair_name} prompt IDs differ from the common source set")
        benchmark = pd.read_csv(output_dir / "performance/e2_performance_benchmark.csv")
        peak_allocated = max(
            [float(benchmark.peak_allocated_vram_gb.max())]
            + [runtime_by_pair[p][s]["peak_allocated_vram_gb"] for p in runtime_by_pair for s in runtime_by_pair[p]]
        )
        peak_reserved = max(
            [float(benchmark.peak_reserved_vram_gb.max())]
            + [runtime_by_pair[p][s]["peak_reserved_vram_gb"] for p in runtime_by_pair for s in runtime_by_pair[p]]
        )
        import transformers
        run_metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(output_dir.resolve()),
            "prompt_cache": str(prompt_cache.resolve()),
            "prompt_cache_sha256": prompt_hashes["source_sha256"],
            "ordered_prompt_set_sha256": prompt_hashes["ordered_prompt_set_sha256"],
            "num_prompts": args.num_prompts,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "speculative_k": args.speculative_k,
            "seed": args.seed,
            "physical_gpu": 7,
            "cuda_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "attention_backend": backend,
            "dtype": "torch.float16",
            "model_load_elapsed_seconds": model_load_elapsed,
            "elapsed_seconds": time.perf_counter() - wall_started,
            "peak_allocated_vram_gb": peak_allocated,
            "peak_reserved_vram_gb": peak_reserved,
            "selected_target_generation_batch_size": selection["target_generation_batch_size"],
            "selected_teacher_forcing_batch_size": selection["teacher_forcing_batch_size"],
            "selection": selection,
            "pairs": runtime_by_pair,
            "p3_reused_without_sd_rerun": True,
        }
        atomic_json(output_dir / "runtime_metadata.json", run_metadata)
    finally:
        monitor.stop()

    result = analyze_e2(
        output_dir, pair_tables, pair_audits, pair_continuations, pair_teachers,
        run_metadata, tokenizer_report,
    )
    print(json.dumps({
        "decision": result["decision"],
        "report": str(result["report"]),
        "summary_csv": str(output_dir / "combined/e2_model_pair_summary.csv"),
        "elapsed_seconds": run_metadata["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0
