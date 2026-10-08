#!/usr/bin/env python3
"""Modal entrypoint for the complete Table 1 artifact pipeline.

The persistent cache volume holds Hugging Face snapshots; the artifact volume
holds the frozen data, audit logs, run shards, tables, and final zip.  The
functions intentionally call the same local CLI used outside Modal so every
stage has one reproducible implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/root/repo")
REMOTE_WORKSPACE = Path("/root/artifacts/workspace")
REMOTE_CACHE = Path("/root/cache")
REMOTE_MODELS = REMOTE_CACHE / "models"
REMOTE_BUNDLES = Path("/root/artifacts/bundles")
DEFAULT_ZIP_NAME = "korean_speculative_decoding_table1_bundle.zip"


def _read_local_token() -> str:
    token_path = ROOT / "hf_token"
    if not token_path.exists():
        raise RuntimeError(f"Missing Hugging Face token file: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Hugging Face token file is empty")
    return token


def _local_hf_secret() -> modal.Secret:
    """Create a local Modal secret only when an HF token is actually present.

    Test/benchmark stages use persistent Modal volumes and do not need Hub
    authentication.  Keeping import-time secret creation optional lets those
    stages run from a source snapshot without copying credentials anywhere.
    Stages that access the Hub still fail explicitly when ``HF_TOKEN`` is
    absent inside the remote function.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})
    token_path = ROOT / "hf_token"
    if not token_path.exists() or not token_path.read_text(encoding="utf-8").strip():
        return modal.Secret.from_dict({})
    return modal.Secret.from_dict({"HF_TOKEN": _read_local_token()})


HF_SECRET = _local_hf_secret()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5,<3",
        "transformers>=4.51,<6",
        "mistral-common>=1.8.6",
        "datasets>=3.3,<5",
        "huggingface-hub>=0.30,<2",
        "kiwipiepy>=0.21,<1",
        "pyarrow>=19",
        "pandas>=2.2",
        "PyYAML>=6",
        "tabulate>=0.9",
        "pytest>=8",
    )
    .add_local_dir(str(ROOT), remote_path="/root/repo", ignore=lambda p: p.name in {
        "hf_token", ".git", ".hf_home", ".venv", "__pycache__", ".pytest_cache"
    })
)

cache_volume = modal.Volume.from_name("korean-speculative-decoding-hf-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("korean-speculative-decoding-table1-artifacts", create_if_missing=True)
app = modal.App("korean-speculative-decoding-table1", image=image)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(REMOTE_CACHE / "huggingface"),
        "HF_HUB_CACHE": str(REMOTE_CACHE / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(REMOTE_CACHE / "huggingface" / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def _workspace() -> Path:
    REMOTE_WORKSPACE.mkdir(parents=True, exist_ok=True)
    # Copy source files only on first use.  Subsequent stages reuse durable
    # outputs on the artifact volume and receive code changes only when the
    # Modal image is rebuilt.
    marker = REMOTE_WORKSPACE / ".repo_snapshot_ready"
    if not marker.exists():
        shutil.copytree(REMOTE_REPO, REMOTE_WORKSPACE, dirs_exist_ok=True)
        marker.write_text("repo copied from image\n", encoding="utf-8")
    else:
        # Refresh source/config files when the image is rebuilt, but never copy
        # the local repo's output directories over the durable artifact volume.
        for name in ("src", "scripts", "configs", "tests"):
            source = REMOTE_REPO / name
            if source.exists():
                shutil.copytree(source, REMOTE_WORKSPACE / name, dirs_exist_ok=True)
        for source in REMOTE_REPO.iterdir():
            if source.is_file() and source.name not in {"hf_token"}:
                shutil.copy2(source, REMOTE_WORKSPACE / source.name)
    return REMOTE_WORKSPACE


def _run_cli(args: list[str]) -> dict[str, Any]:
    workspace = _workspace()
    command = [sys.executable, str(workspace / "scripts" / "table1_pipeline.py"), "--root", str(workspace), *args]
    print("TABLE1_COMMAND=" + json.dumps(command), flush=True)
    completed = subprocess.run(command, cwd=str(workspace), env=_env(), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Table 1 stage failed with exit code {completed.returncode}: {args}")
    return {"command": command, "return_code": completed.returncode}


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@app.function(image=image, cpu=4, memory=32768, timeout=1800)
def test_remote() -> dict[str, Any]:
    """Run the repository test suite inside the pinned Modal image."""
    test_root = Path(tempfile.mkdtemp(prefix="table1-repo-tests-"))
    try:
        shutil.copytree(REMOTE_REPO, test_root, dirs_exist_ok=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(test_root),
            env=_env(),
            text=True,
            check=False,
        )
        print(f"TABLE1_REMOTE_TEST_EXIT_CODE={completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Remote test suite failed with exit code {completed.returncode}")
        return {"status": "passed", "return_code": completed.returncode}
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


@app.function(
    image=image,
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=8,
    memory=32768,
    timeout=86400,
)
def prepare_data_remote() -> dict[str, Any]:
    result = _run_cli(["prepare-data"])
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=8,
    memory=32768,
    timeout=86400,
)
def download_models_remote() -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    workspace = _workspace()
    import yaml

    config = yaml.safe_load((workspace / "configs/table1_pipeline.yaml").read_text(encoding="utf-8"))
    model_meta_path = workspace / config["paths"]["model_metadata"]
    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {"pairs": {}}
    model_meta.setdefault("models", {})
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    model_ids = {}
    for pair in config["pairs"].values():
        model_ids[pair["draft"]] = None
        model_ids[pair["target"]] = None
    downloaded = []
    for model_id in sorted(model_ids):
        existing = model_meta["models"].get(model_id, {})
        info = api.model_info(model_id, revision=existing.get("requested_revision"))
        revision = getattr(info, "sha", None)
        if not revision:
            raise RuntimeError(f"No immutable revision returned for {model_id}")
        destination = REMOTE_MODELS / _model_slug(model_id) / revision
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / ".DOWNLOAD_COMPLETE"
        if not marker.exists():
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                local_dir=str(destination),
            )
            marker.write_text("snapshot_download complete\n", encoding="utf-8")
        model_meta["models"][model_id] = {
            "id": model_id,
            "requested_revision": existing.get("requested_revision", "main"),
            "resolved_revision": revision,
            "local_path": str(destination),
            "gated": bool(getattr(info, "gated", False)),
            "private": bool(getattr(info, "private", False)),
        }
        downloaded.append({"id": model_id, "revision": revision, "path": str(destination)})

    for pair_id, pair in config["pairs"].items():
        model_meta["pairs"].setdefault(pair_id, {})
        model_meta["pairs"][pair_id]["draft"] = model_meta["models"][pair["draft"]]
        model_meta["pairs"][pair_id]["target"] = model_meta["models"][pair["target"]]
    _write_json(model_meta_path, model_meta)
    _write_json(workspace / "metadata/model_download_manifest.json", {
        "downloaded": downloaded,
        "cache_root": str(REMOTE_MODELS),
    })
    artifact_volume.commit()
    print("MODEL_DOWNLOAD_COUNT=" + str(len(downloaded)), flush=True)
    return {"status": "complete", "downloaded": downloaded}


def _audit(pair_id: str, device: str, smoke_limit: int = 200, smoke_max_new_tokens: int = 128) -> dict[str, Any]:
    result = _run_cli([
        "audit-models", "--pair", pair_id, "--device", device,
        "--smoke-limit", str(smoke_limit),
        "--smoke-max-new-tokens", str(smoke_max_new_tokens),
    ])
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=16,
    memory=131072,
    timeout=86400,
)
def audit_pair_b200(pair_id: str, smoke_limit: int = 200, smoke_max_new_tokens: int = 128) -> dict[str, Any]:
    return _audit(pair_id, "cuda", smoke_limit, smoke_max_new_tokens)


def _run_pair(pair_id: str, device: str, shard_index: int, num_shards: int) -> dict[str, Any]:
    result = _run_cli([
        "run-sd", "--pair", pair_id, "--device", device,
        "--shard-index", str(shard_index), "--num-shards", str(num_shards),
    ])
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=16,
    memory=131072,
    timeout=86400,
)
def run_pair_b200(pair_id: str, shard_index: int = 0, num_shards: int = 1) -> dict[str, Any]:
    return _run_pair(pair_id, "cuda", shard_index, num_shards)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=16,
    memory=131072,
    timeout=86400,
)
def benchmark_pair_b200(
    pair_id: str,
    prompts: int = 5,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Compare the legacy and persistent-draft-cache decoders on one B200."""
    import yaml
    import torch
    from transformers import AutoTokenizer

    workspace = _workspace()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from src.data import encode_prompt
    from src.speculative_decoding import (
        greedy_generate,
        greedy_generate_batch,
        speculative_greedy,
        speculative_greedy_cached,
        verify_greedy_equivalence,
    )
    from src.table1_runner import _model_kwargs, load_pair_records
    from src.models import load_models

    config = yaml.safe_load((workspace / "configs/table1_pipeline.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((workspace / config["paths"]["model_metadata"] ).read_text(encoding="utf-8"))
    pair_spec = config["pairs"][pair_id]
    model_meta = metadata["pairs"][pair_id]
    records = load_pair_records(workspace / config["paths"]["prompts"], pair_spec)[: int(prompts)]
    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    target_name, target_revision = _model_kwargs(model_meta, "target")
    token = os.environ.get("HF_TOKEN")
    pair_dtype = str(pair_spec.get("dtype", config["inference"].get("dtype", "float16")))
    pair_attention_backend = str(
        pair_spec.get("attention_backend", config["inference"].get("attention_backend", "sdpa"))
    )
    tokenizer = AutoTokenizer.from_pretrained(target_name, revision=target_revision, use_fast=True, token=token)
    draft_model, target_model = load_models(
        draft_name,
        target_name,
        device="cuda",
        draft_revision=draft_revision,
        target_revision=target_revision,
        token=token,
        dtype=pair_dtype,
        attention_backend=pair_attention_backend,
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_ids = [
        encode_prompt(tokenizer, str(record["text"]), int(config["inference"]["max_prompt_tokens"]))
        for record in records
    ]
    references = [
        greedy_generate(target_model, ids, int(max_new_tokens), eos_token_id)
        for ids in prompt_ids
    ]
    if not prompt_ids:
        raise ValueError("benchmark requires at least one prompt")

    # Warm up model kernels before timing either decoder.
    speculative_greedy(draft_model, target_model, prompt_ids[0], 4, eos_token_id, 4)
    speculative_greedy_cached(draft_model, target_model, prompt_ids[0], 4, eos_token_id, 4)
    torch.cuda.synchronize()

    def run(decoder):
        outputs = []
        signatures = []
        valid_signatures = []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for ids, reference in zip(prompt_ids, references):
            output, events = decoder(
                draft_model, target_model, ids, int(max_new_tokens), eos_token_id, 4,
            )
            verify_greedy_equivalence(reference, output)
            outputs.append(output)
            signatures.append([
                (event["draft_token_id"], event["target_greedy_token_id"], event["accepted"], event["rejected"])
                for event in events
            ])
            valid_signatures.append([
                (event["draft_token_id"], event["target_greedy_token_id"], event["accepted"], event["rejected"])
                for event in events
                if event["rejected"] is not None
            ])
        torch.cuda.synchronize()
        return (
            time.perf_counter() - started,
            outputs,
            signatures,
            valid_signatures,
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
        )

    (
        old_seconds,
        old_outputs,
        old_signatures,
        old_valid_signatures,
        old_peak_allocated,
        old_peak_reserved,
    ) = run(speculative_greedy)
    (
        new_seconds,
        new_outputs,
        new_signatures,
        new_valid_signatures,
        new_peak_allocated,
        new_peak_reserved,
    ) = run(speculative_greedy_cached)
    outputs_exact = old_outputs == new_outputs
    valid_audit_exact = old_valid_signatures == new_valid_signatures
    full_audit_exact = old_signatures == new_signatures
    if not outputs_exact or not valid_audit_exact:
        for index, (old_output, new_output, old_signature, new_signature) in enumerate(
            zip(old_outputs, new_outputs, old_valid_signatures, new_valid_signatures)
        ):
            if old_output != new_output or old_signature != new_signature:
                first_output_diff = next(
                    (
                        position
                        for position, (old_token, new_token) in enumerate(
                            zip(old_output, new_output)
                        )
                        if old_token != new_token
                    ),
                    min(len(old_output), len(new_output))
                    if len(old_output) != len(new_output)
                    else None,
                )
                first_signature_diff = next(
                    (
                        position
                        for position, (old_event, new_event) in enumerate(
                            zip(old_signature, new_signature)
                        )
                        if old_event != new_event
                    ),
                    min(len(old_signature), len(new_signature))
                    if len(old_signature) != len(new_signature)
                    else None,
                )
                print(
                    "TABLE1_BENCHMARK_MISMATCH=" + json.dumps({
                        "prompt_index": index,
                        "old_output_length": len(old_output),
                        "new_output_length": len(new_output),
                        "first_output_diff": first_output_diff,
                        "old_valid_signature_length": len(old_signature),
                        "new_valid_signature_length": len(new_signature),
                        "first_signature_diff": first_signature_diff,
                        "old_output_prefix": old_output[:16],
                        "new_output_prefix": new_output[:16],
                        "old_signature_context": old_signature[
                            max(0, (first_signature_diff or 0) - 2):
                            (first_signature_diff or 0) + 3
                        ],
                        "new_signature_context": new_signature[
                            max(0, (first_signature_diff or 0) - 2):
                            (first_signature_diff or 0) + 3
                        ],
                    }, sort_keys=True),
                    flush=True,
                )
                break
    result = {
        "status": "complete" if outputs_exact and valid_audit_exact else "parity_failed",
        "pair_id": pair_id,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "attention_backend": pair_attention_backend,
        "dtype": pair_dtype,
        "prompts": len(prompt_ids),
        "max_new_tokens": int(max_new_tokens),
        "legacy_seconds": old_seconds,
        "cached_seconds": new_seconds,
        "speedup": old_seconds / new_seconds if new_seconds else None,
        "legacy_peak_allocated_gib": old_peak_allocated / (1024 ** 3),
        "legacy_peak_reserved_gib": old_peak_reserved / (1024 ** 3),
        "cached_peak_allocated_gib": new_peak_allocated / (1024 ** 3),
        "cached_peak_reserved_gib": new_peak_reserved / (1024 ** 3),
        "outputs_exact": outputs_exact,
        "audit_events_exact": full_audit_exact,
        "valid_audit_events_exact": valid_audit_exact,
        "invalidated_suffixes_may_differ": not full_audit_exact,
    }
    output_path = Path("/root/artifacts/benchmarks") / f"table1_decoder_{pair_id.lower()}_b200.json"
    _write_json(output_path, result)
    artifact_volume.commit()
    print("TABLE1_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=16,
    memory=131072,
    timeout=86400,
)
def benchmark_reference_batch_b200(
    pair_id: str,
    prompts: int = 16,
    max_new_tokens: int = 32,
    attention_backend_override: str = "",
    dtype_override: str = "",
    disable_reduced_precision_matmul: bool = False,
    batch_size_override: int = 0,
) -> dict[str, Any]:
    """Measure exact same-length target-reference microbatching on B200."""
    import yaml
    import torch
    from collections import defaultdict
    from transformers import AutoTokenizer

    workspace = _workspace()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from src.data import encode_prompt
    from src.speculative_decoding import greedy_generate, greedy_generate_batch
    from src.table1_runner import _model_kwargs, load_pair_records
    from src.models import load_models

    if disable_reduced_precision_matmul:
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    config = yaml.safe_load((workspace / "configs/table1_pipeline.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((workspace / config["paths"]["model_metadata"]).read_text(encoding="utf-8"))
    pair_spec = config["pairs"][pair_id]
    model_meta = metadata["pairs"][pair_id]
    records = load_pair_records(workspace / config["paths"]["prompts"], pair_spec)[: int(prompts)]
    target_name, target_revision = _model_kwargs(model_meta, "target")
    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        target_name, revision=target_revision, use_fast=True, token=token,
    )
    pair_dtype = str(pair_spec.get("dtype", config["inference"].get("dtype", "float16")))
    pair_attention_backend = str(
        pair_spec.get("attention_backend", config["inference"].get("attention_backend", "sdpa"))
    )
    attention_backend = attention_backend_override or pair_attention_backend
    dtype = dtype_override or pair_dtype
    draft_model, target_model = load_models(
        draft_name,
        target_name,
        device="cuda",
        draft_revision=draft_revision,
        target_revision=target_revision,
        token=token,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    del draft_model
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_ids = [
        encode_prompt(tokenizer, str(record["text"]), int(config["inference"]["max_prompt_tokens"]))
        for record in records
    ]
    if not prompt_ids:
        raise ValueError("batch benchmark requires at least one prompt")

    # Equal-length buckets avoid padding/mask changes in this exactness pilot.
    buckets: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for index, ids in enumerate(prompt_ids):
        buckets[len(ids)].append((index, ids))

    greedy_generate(target_model, prompt_ids[0], int(max_new_tokens), eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    scalar_started = time.perf_counter()
    scalar_outputs = [
        greedy_generate(target_model, ids, int(max_new_tokens), eos_token_id)
        for ids in prompt_ids
    ]
    torch.cuda.synchronize()
    scalar_seconds = time.perf_counter() - scalar_started
    scalar_peak = int(torch.cuda.max_memory_allocated())

    rows = []
    batch_sizes = (
        (int(batch_size_override),)
        if int(batch_size_override) > 0
        else (1, 2, 4, 8, 16, 32, 64)
    )
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batched_outputs: list[list[int] | None] = [None] * len(prompt_ids)
        actual_batches = 0
        max_batch_seen = 0
        for bucket in buckets.values():
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start:start + batch_size]
                outputs = greedy_generate_batch(
                    target_model,
                    [ids for _, ids in chunk],
                    int(max_new_tokens),
                    eos_token_id,
                )
                actual_batches += 1
                max_batch_seen = max(max_batch_seen, len(chunk))
                for (index, _), output in zip(chunk, outputs):
                    batched_outputs[index] = output
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        exact = batched_outputs == scalar_outputs
        mismatches = []
        for index, (scalar, batched) in enumerate(zip(scalar_outputs, batched_outputs)):
            if scalar == batched:
                continue
            first_difference = next(
                (
                    position
                    for position, (left, right) in enumerate(zip(scalar, batched or []))
                    if left != right
                ),
                min(len(scalar), len(batched or []))
                if len(scalar) != len(batched or [])
                else None,
            )
            mismatches.append({
                "prompt_index": index,
                "first_difference": first_difference,
                "scalar_prefix": scalar[:8],
                "batched_prefix": (batched or [])[:8],
            })
        rows.append({
            "batch_size_requested": batch_size,
            "max_batch_seen": max_batch_seen,
            "actual_batches": actual_batches,
            "seconds": elapsed,
            "speedup_vs_scalar": scalar_seconds / elapsed if elapsed else None,
            "exact_vs_scalar": exact,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:8],
            "peak_allocated_gib": int(torch.cuda.max_memory_allocated()) / (1024 ** 3),
        })

    result = {
        "status": "complete",
        "pair_id": pair_id,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "attention_backend": attention_backend,
        "dtype": dtype,
        "disable_reduced_precision_matmul": disable_reduced_precision_matmul,
        "prompts": len(prompt_ids),
        "max_new_tokens": int(max_new_tokens),
        "batch_size_override": int(batch_size_override),
        "prompt_length_buckets": {str(length): len(items) for length, items in sorted(buckets.items())},
        "scalar_seconds": scalar_seconds,
        "scalar_peak_allocated_gib": scalar_peak / (1024 ** 3),
        "batch_rows": rows,
    }
    output_path = Path("/root/artifacts/benchmarks") / f"table1_reference_batch_{pair_id.lower()}_b200.json"
    _write_json(output_path, result)
    artifact_volume.commit()
    print("TABLE1_BATCH_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=16,
    memory=131072,
    timeout=86400,
)
def benchmark_reference_dtype_parity_b200(
    pair_id: str,
    prompts: int = 16,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Compare scalar reference IDs across the production and parity dtypes."""
    import yaml
    import torch
    from transformers import AutoTokenizer

    workspace = _workspace()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from src.data import encode_prompt
    from src.speculative_decoding import greedy_generate
    from src.table1_runner import _model_kwargs, load_pair_records
    from src.models import load_models

    config = yaml.safe_load((workspace / "configs/table1_pipeline.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((workspace / config["paths"]["model_metadata"]).read_text(encoding="utf-8"))
    pair_spec = config["pairs"][pair_id]
    model_meta = metadata["pairs"][pair_id]
    records = load_pair_records(workspace / config["paths"]["prompts"], pair_spec)[: int(prompts)]
    target_name, target_revision = _model_kwargs(model_meta, "target")
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        target_name, revision=target_revision, use_fast=True, token=token,
    )
    prompt_ids = [
        encode_prompt(tokenizer, str(record["text"]), int(config["inference"]["max_prompt_tokens"]))
        for record in records
    ]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    production_dtype = str(pair_spec.get("dtype", config["inference"].get("dtype", "float16")))
    production_backend = str(
        pair_spec.get("attention_backend", config["inference"].get("attention_backend", "sdpa"))
    )

    def generate(dtype: str, backend: str) -> tuple[list[list[int]], float]:
        started = time.perf_counter()
        draft_model, target_model = load_models(
            target_name,
            target_name,
            device="cuda",
            target_revision=target_revision,
            token=token,
            dtype=dtype,
            attention_backend=backend,
        )
        del draft_model
        outputs = [
            greedy_generate(target_model, ids, int(max_new_tokens), eos_token_id)
            for ids in prompt_ids
        ]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        del target_model
        torch.cuda.empty_cache()
        return outputs, elapsed

    production_outputs, production_seconds = generate(production_dtype, production_backend)
    parity_outputs, parity_seconds = generate("float32", "eager")
    mismatches = []
    for index, (left, right) in enumerate(zip(production_outputs, parity_outputs)):
        if left == right:
            continue
        first_difference = next(
            (
                position
                for position, (left_id, right_id) in enumerate(zip(left, right))
                if left_id != right_id
            ),
            min(len(left), len(right)) if len(left) != len(right) else None,
        )
        mismatches.append({
            "prompt_index": index,
            "first_difference": first_difference,
            "production_prefix": left[:8],
            "parity_prefix": right[:8],
        })
    result = {
        "status": "complete",
        "pair_id": pair_id,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "prompts": len(prompt_ids),
        "max_new_tokens": int(max_new_tokens),
        "production_dtype": production_dtype,
        "production_attention_backend": production_backend,
        "parity_dtype": "float32",
        "parity_attention_backend": "eager",
        "production_seconds": production_seconds,
        "parity_seconds": parity_seconds,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
    }
    output_path = Path("/root/artifacts/benchmarks") / f"table1_reference_dtype_parity_{pair_id.lower()}_b200.json"
    _write_json(output_path, result)
    artifact_volume.commit()
    print("TABLE1_DTYPE_PARITY_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=8,
    memory=32768,
    timeout=86400,
)
def align_pair_remote(pair_id: str, shard_index: int = 0, num_shards: int = 1) -> dict[str, Any]:
    result = _run_cli([
        "align-morphology", "--pair", pair_id,
        "--shard-index", str(shard_index), "--num-shards", str(num_shards),
    ])
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=8,
    memory=65536,
    timeout=86400,
)
def build_table1_remote() -> dict[str, Any]:
    result = _run_cli(["build-table1"])
    artifact_volume.commit()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    volumes={"/root/cache": cache_volume, "/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=8,
    memory=65536,
    timeout=86400,
)
def package_remote(zip_name: str = DEFAULT_ZIP_NAME) -> dict[str, Any]:
    workspace = _workspace()
    REMOTE_BUNDLES.mkdir(parents=True, exist_ok=True)
    if not REMOTE_MODELS.exists():
        raise RuntimeError("Model cache is not mounted; refusing to create a bundle without model weights")

    # Remove a previous staging tree from the legacy copy-then-zip
    # implementation. The new implementation streams files directly from the
    # persistent model cache into the archive and never creates a second full
    # copy of the model weights.
    legacy_stage = REMOTE_BUNDLES / "stage"
    if legacy_stage.exists():
        shutil.rmtree(legacy_stage)

    manifest = {
        "bundle_format": "table1-artifact-bundle-v1",
        "created_unix": time.time(),
        "contents": ["repo", "models"],
        "secret_files_excluded": ["hf_token"],
        "packaging_mode": "streaming_zip_from_modal_volumes",
    }
    zip_path = REMOTE_BUNDLES / zip_name
    if zip_path.exists():
        zip_path.unlink()
    checksum_rows = []

    def source_files(root: Path, prefix: str):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in {"hf_token", ".hf_token", ".repo_snapshot_ready"}:
                continue
            yield path, (Path(prefix) / path.relative_to(root)).as_posix()

    def stream_file(archive: zipfile.ZipFile, source: Path, arcname: str) -> dict[str, Any]:
        info = zipfile.ZipInfo(arcname)
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        digest = hashlib.sha256()
        byte_count = 0
        with source.open("rb") as handle, archive.open(info, "w", force_zip64=True) as destination:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
                destination.write(chunk)
        return {"path": arcname, "sha256": digest.hexdigest(), "bytes": byte_count}

    bundle_manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("BUNDLE_MANIFEST.json", bundle_manifest_bytes, compress_type=zipfile.ZIP_STORED)
        checksum_rows.append({
            "path": "BUNDLE_MANIFEST.json",
            "sha256": hashlib.sha256(bundle_manifest_bytes).hexdigest(),
            "bytes": len(bundle_manifest_bytes),
        })
        file_count = 0
        for source, arcname in source_files(workspace, "repo"):
            checksum_rows.append(stream_file(archive, source, arcname))
            file_count += 1
        for source, arcname in source_files(REMOTE_MODELS, "models"):
            checksum_rows.append(stream_file(archive, source, arcname))
            file_count += 1
            if file_count % 10 == 0:
                print(f"BUNDLE_FILES_WRITTEN={file_count}", flush=True)
    _write_json(zip_path.with_suffix(".manifest.json"), {
        **manifest,
        "zip": zip_path.name,
        "zip_sha256": _sha256(zip_path),
        "files": checksum_rows,
        "zip_bytes": zip_path.stat().st_size,
    })
    artifact_volume.commit()
    result = {"status": "complete", "zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": _sha256(zip_path)}
    print("BUNDLE_SUMMARY_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    volumes={"/root/artifacts": artifact_volume},
    secrets=[HF_SECRET],
    cpu=4,
    memory=16384,
    timeout=86400,
)
def upload_remote(
    zip_name: str = DEFAULT_ZIP_NAME,
    repo_id: str = "",
    private: bool = True,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    zip_path = REMOTE_BUNDLES / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Bundle does not exist: {zip_path}")
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    if not repo_id:
        whoami = api.whoami()
        namespace = whoami.get("name") or whoami.get("orgs", [{}])[0].get("name")
        if not namespace:
            raise RuntimeError("Could not determine Hugging Face namespace; pass --hf-repo-id")
        repo_id = f"{namespace}/korean-speculative-decoding-table1-bundle"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    # `create_repo(..., exist_ok=True)` does not change visibility when the
    # repository was created by an earlier attempt.  Apply the requested
    # visibility explicitly so a retry with --no-private really becomes public.
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private, token=token)
    uploaded = []
    for path in [zip_path, zip_path.with_suffix(".manifest.json")]:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload reproducible Korean Table 1 experiment bundle",
        )
        uploaded.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    result = {"status": "complete", "repo_id": repo_id, "repo_type": "dataset", "private": private, "uploaded": uploaded}
    print("HF_UPLOAD_SUMMARY_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _choose_function(pair_id: str):
    return {"Q1": run_pair_b200, "Q2": run_pair_b200, "Q3": run_pair_b200, "M1": run_pair_b200, "G1": run_pair_b200}[pair_id]


@app.local_entrypoint()
def main(
    stage: str = "all",
    pair: str = "",
    num_shards: int = 1,
    shard_index: int = 0,
    hf_repo_id: str = "",
    private: bool = True,
    quick_smoke: bool = False,
    smoke_limit: int = 200,
    smoke_max_new_tokens: int = 128,
    benchmark_prompts: int = 5,
    benchmark_max_new_tokens: int = 32,
    benchmark_attention_backend: str = "",
    benchmark_dtype: str = "",
    benchmark_disable_reduced_precision_matmul: bool = False,
    benchmark_batch_size: int = 0,
) -> None:
    """Run one stage or the entire sequential pipeline.

    For a long run, call this script with stage=run-sd and a pair, then use
    the same shard arguments to resume failed work.
    """
    if stage == "prepare-data":
        print(prepare_data_remote.remote())
        return
    if stage == "test":
        print(test_remote.remote())
        return
    if stage == "download-models":
        print(download_models_remote.remote())
        return
    if stage == "audit-models":
        pairs = [pair] if pair else ["Q1", "Q2", "Q3", "M1", "G1"]
        selected_smoke_limit = 5 if quick_smoke else smoke_limit
        selected_smoke_max_new_tokens = 32 if quick_smoke else smoke_max_new_tokens
        for pair_id in pairs:
            function = audit_pair_b200
            print(function.remote(pair_id, selected_smoke_limit, selected_smoke_max_new_tokens))
        return
    if stage == "run-sd":
        if not pair:
            raise ValueError("stage=run-sd requires --pair")
        print(_choose_function(pair).remote(pair, shard_index, num_shards))
        return
    if stage == "benchmark":
        if not pair:
            raise ValueError("stage=benchmark requires --pair")
        print(benchmark_pair_b200.remote(pair, benchmark_prompts, benchmark_max_new_tokens))
        return
    if stage == "benchmark-all":
        pair_ids = ["Q1", "Q2", "Q3", "M1", "G1"]
        print("TABLE1_PARALLEL_STAGE=benchmark pairs=" + json.dumps(pair_ids), flush=True)
        for result in benchmark_pair_b200.starmap(
            [(pair_id, benchmark_prompts, benchmark_max_new_tokens) for pair_id in pair_ids]
        ):
            print(result)
        return
    if stage == "benchmark-batch":
        if not pair:
            raise ValueError("stage=benchmark-batch requires --pair")
        print(benchmark_reference_batch_b200.remote(
            pair, benchmark_prompts, benchmark_max_new_tokens, benchmark_attention_backend,
            benchmark_dtype,
            benchmark_disable_reduced_precision_matmul,
            benchmark_batch_size,
        ))
        return
    if stage == "benchmark-batch-all":
        pair_ids = ["Q1", "Q2", "Q3", "M1", "G1"]
        print("TABLE1_PARALLEL_STAGE=benchmark-batch pairs=" + json.dumps(pair_ids), flush=True)
        for result in benchmark_reference_batch_b200.starmap(
            [
                (pair_id, benchmark_prompts, benchmark_max_new_tokens, benchmark_attention_backend)
                + (benchmark_dtype, benchmark_disable_reduced_precision_matmul)
                for pair_id in pair_ids
            ]
        ):
            print(result)
        return
    if stage == "benchmark-dtype-parity":
        if not pair:
            raise ValueError("stage=benchmark-dtype-parity requires --pair")
        print(benchmark_reference_dtype_parity_b200.remote(
            pair, benchmark_prompts, benchmark_max_new_tokens,
        ))
        return
    if stage == "align-morphology":
        if not pair:
            raise ValueError("stage=align-morphology requires --pair")
        print(align_pair_remote.remote(pair, shard_index, num_shards))
        return
    if stage == "build-table1":
        print(build_table1_remote.remote())
        return
    if stage == "package":
        print(package_remote.remote())
        return
    if stage == "upload":
        print(upload_remote.remote(repo_id=hf_repo_id, private=private))
        return
    if stage != "all":
        raise ValueError(f"Unknown stage: {stage}")

    if num_shards != 1:
        raise ValueError(
            "stage=all assumes one B200 per model pair and requires --num-shards 1; "
            "run additional shards explicitly with stage=run-sd"
        )
    print(prepare_data_remote.remote())
    print(download_models_remote.remote())
    pair_ids = ["Q1", "Q2", "Q3", "M1", "G1"]
    for pair_id in ["Q1", "Q2", "Q3", "M1", "G1"]:
        audit_function = audit_pair_b200
        selected_smoke_limit = 5 if quick_smoke and pair_id in {"Q3", "M1", "G1"} else smoke_limit
        selected_smoke_max_new_tokens = 32 if quick_smoke and pair_id in {"Q3", "M1", "G1"} else smoke_max_new_tokens
        print(audit_function.remote(pair_id, selected_smoke_limit, selected_smoke_max_new_tokens))
    # Each pair owns one B200.  Submit all five SD jobs together so Modal can
    # schedule them concurrently instead of serializing the GPU work.
    print("TABLE1_PARALLEL_STAGE=run-sd pairs=" + json.dumps(pair_ids), flush=True)
    for result in run_pair_b200.starmap([(pair_id, 0, 1) for pair_id in pair_ids]):
        print(result)
    print("TABLE1_PARALLEL_STAGE=align-morphology pairs=" + json.dumps(pair_ids), flush=True)
    for result in align_pair_remote.starmap([(pair_id, 0, 1) for pair_id in pair_ids]):
        print(result)
    print(build_table1_remote.remote())
    print(package_remote.remote())
    print(upload_remote.remote(repo_id=hf_repo_id, private=private))
