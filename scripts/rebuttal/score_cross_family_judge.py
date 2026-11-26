#!/usr/bin/env python3
"""Teacher-force fixed outside-family judges on frozen generated continuations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import get_token
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from scripts.tf_score import _body_and_head, _score_with_backoff
from src.naacl_experiment import atomic_json, atomic_parquet, cleanup_cuda, load_config, load_tokenizer, model_path
from src.rebuttal.runtime import write_gpu_metadata

SEED = 20261125
DEFAULT_CONFIG = ROOT / "configs/naacl_experiments.yaml"
MODEL_ROOT = Path(os.environ.get("CROSS_FAMILY_MODEL_DIR", "/dev/shm/cross_family_models"))
OUTPUT_ROOT = Path(os.environ.get("CROSS_FAMILY_OUTPUT_ROOT", ROOT / "results/rebuttal/cross_family"))
SHARD_DOCS = 250


def _weight_file(name: str) -> bool:
    return name.endswith((".safetensors", ".bin", ".pt", ".pth"))


def _snapshot_dir(config: dict, key: str) -> Path:
    spec = config["models"][key]
    return MODEL_ROOT / spec["id"].replace("/", "--") / spec["revision"]


def _ensure_snapshot(config: dict, key: str, api: HfApi, token: str) -> tuple[Path, Any, list[str]]:
    spec = config["models"][key]
    info = api.model_info(spec["id"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise RuntimeError(f"Pinned revision mismatch for {key}: {info.sha} != {spec['revision']}")
    weight_files = [
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith(".safetensors")
        and (sibling.rfilename == "model.safetensors" or sibling.rfilename.startswith("model-"))
    ]
    if not weight_files:
        raise RuntimeError(f"No Transformers safetensors weights listed for {key}")
    path = _snapshot_dir(config, key)
    path.mkdir(parents=True, exist_ok=True)
    allow = [sibling.rfilename for sibling in info.siblings if not _weight_file(sibling.rfilename)]
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        snapshot_download(
            repo_id=spec["id"],
            revision=spec["revision"],
            local_dir=path,
            allow_patterns=allow,
            token=token,
        )
    return path, info, weight_files


def _load_streamed_model(config: dict, key: str, api: HfApi, token: str):
    """Load pinned safetensors one shard at a time, keeping only the model in RAM."""
    path, info, weight_files = _ensure_snapshot(config, key, api, token)
    configured_path = model_path(config, key)
    if all((configured_path / filename).is_file() for filename in weight_files):
        model = _from_pretrained(configured_path)
        return model, configured_path, "local_checkpoint"
    existing = [path / filename for filename in weight_files]
    if all(file.is_file() for file in existing):
        model = _from_pretrained(path)
        return model, path, "local_checkpoint"

    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    kwargs: dict[str, Any] = {"attn_implementation": config["inference"].get("attention_backend", "sdpa")}
    major = int(__import__("transformers").__version__.split(".", 1)[0])
    kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    with torch.device("meta"):
        # Transformers 5.19 exposes the model-class constructor as `_from_config`
        # for Mistral3, while AutoModel classes expose `from_config`.
        constructor = getattr(model_class, "from_config", None)
        if constructor is None:
            constructor = model_class._from_config
        model = constructor(model_config, **kwargs)
    model.to_empty(device="cpu")

    loaded: set[str] = set()
    tmp_dir = path / ".streaming_weights"
    tmp_dir.mkdir(exist_ok=True)
    for shard_index, filename in enumerate(weight_files):
        print(f"STREAM_WEIGHT_START model={key} shard={shard_index + 1}/{len(weight_files)} name={filename}", flush=True)
        local_file = Path(hf_hub_download(
            repo_id=config["models"][key]["id"],
            filename=filename,
            revision=config["models"][key]["revision"],
            token=token,
            local_dir=tmp_dir,
        ))
        state = load_file(str(local_file), device="cpu")
        if getattr(model_config, "model_type", None) == "mistral3":
            # Ministral 3's published checkpoint uses unwrapped component
            # prefixes; Transformers 5.19 registers those components under
            # `model.*` in Mistral3ForConditionalGeneration.
            remapped = {}
            for name, value in state.items():
                if name.startswith("language_model.model."):
                    name = "model.language_model." + name[len("language_model.model."):]
                elif name.startswith("language_model.lm_head."):
                    name = name[len("language_model."):]
                elif name.startswith("vision_tower."):
                    name = "model." + name
                elif name.startswith("multi_modal_projector."):
                    name = "model." + name
                if name in remapped:
                    raise RuntimeError(f"Checkpoint key remap collision for {key}: {name}")
                remapped[name] = value
            state = remapped
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys for {key}: {incompatible.unexpected_keys[:8]}")
        loaded.update(state.keys())
        del state
        local_file.unlink(missing_ok=True)
        print(f"STREAM_WEIGHT_COMPLETE model={key} shard={shard_index + 1}/{len(weight_files)}", flush=True)
        gc.collect()

    tied_aliases: set[str] = set()
    if model.config.tie_word_embeddings:
        model.tie_weights()
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is not None and output_embeddings is not None and input_embeddings.weight is output_embeddings.weight:
            tied_aliases.add("lm_head.weight")
    expected = set(model.state_dict().keys())
    missing = expected - loaded - tied_aliases
    if missing:
        raise RuntimeError(f"Checkpoint for {key} is missing {len(missing)} model keys: {sorted(missing)[:12]}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(f"Checkpoint loading left meta parameters in {key}")
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model = model.to("cuda").eval()
    cleanup_cuda(torch)
    return model, path, "streamed_shards_to_cpu"


def _from_pretrained(path: Path):
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    major = int(__import__("transformers").__version__.split(".", 1)[0])
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    return model_class.from_pretrained(path, **kwargs).to("cuda").eval()


def _continuation_directory(target: str) -> tuple[str, str]:
    return target, "all" if target == "qwen4" else "common"


def _read_continuations(target: str, limit: int) -> list[dict]:
    _, split = _continuation_directory(target)
    directory = ROOT / "results/naacl_experiments/continuations" / target / split
    if not directory.is_dir():
        raise FileNotFoundError(f"No stored continuations at {directory}")
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No continuation shards at {directory}")
    frames = [pd.read_parquet(file, columns=["doc_id", "source_index", "pool_index", "prompt_ids", "continuation_ids"])
              for file in files]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.sort_values("pool_index", kind="stable")
    rows = rows[rows.pool_index < limit].drop_duplicates("doc_id", keep="first")
    if len(rows) != limit:
        raise RuntimeError(f"Expected first {limit} common continuations for {target}; found {len(rows)}")
    return rows.to_dict(orient="records")


def _read_target_logs(target: str, limit: int) -> pd.DataFrame:
    split = "all" if target == "qwen4" else "common"
    directory = ROOT / "results/naacl_experiments/teacher_forced/generated" / target / target / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No stored target logits at {directory}")
    frame = pd.concat([
        pd.read_parquet(file, columns=["doc_id", "pool_index", "position", "argmax_id", "target_token_id"])
        for file in files
    ], ignore_index=True)
    frame = frame[frame.pool_index < limit]
    return frame.sort_values(["pool_index", "position"], kind="stable").reset_index(drop=True)


def _read_alignment(target: str, limit: int) -> dict[tuple[int, int], int]:
    split = "all" if target == "qwen4" else "common"
    directory = ROOT / "results/naacl_experiments/aligned/kiwi" / target / split
    files = sorted(directory.glob("**/shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No source alignments at {directory}")
    parts = []
    for file in files:
        part = pd.read_parquet(file, columns=["pool_index", "generated_token_position", "global_token_char_span"])
        parts.append(part[part.pool_index < limit])
    frame = pd.concat(parts, ignore_index=True)
    result = {}
    for row in frame.itertuples(index=False):
        span = row.global_token_char_span
        if span is not None and len(span) == 2:
            result[(int(row.pool_index), int(row.generated_token_position))] = int(span[0])
    return result


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _source_doc(row: dict, tokenizer) -> dict:
    prompt_ids = list(map(int, row["prompt_ids"]))
    continuation_ids = list(map(int, row["continuation_ids"]))
    full_ids = prompt_ids + continuation_ids
    text = _decode(tokenizer, full_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    boundaries: list[int | None] = []
    for t in range(len(continuation_ids)):
        prefix = _decode(tokenizer, prompt_ids + continuation_ids[:t])
        boundaries.append(len(prefix) if text.startswith(prefix) else None)
    return {
        **row,
        "doc_id": str(row["doc_id"]),
        "pool_index": int(row["pool_index"]),
        "source_index": int(row["source_index"]),
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "text": text,
        "source_boundaries": boundaries,
        "source_special_mask": [token_id in special_ids for token_id in continuation_ids],
    }


def _boundary_audit(docs: list[dict], alignment: dict[tuple[int, int], int], audit_docs: int = 100) -> dict:
    comparable = equal = valid = total = special = 0
    for doc in docs[:audit_docs]:
        for t, boundary in enumerate(doc["source_boundaries"]):
            if doc["source_special_mask"][t]:
                special += 1
                continue
            total += 1
            if boundary is None:
                continue
            valid += 1
            aligned = alignment.get((doc["pool_index"], t))
            if aligned is not None:
                comparable += 1
                equal += int(aligned == boundary)
    agreement = equal / comparable if comparable else float("nan")
    valid_share = valid / total if total else float("nan")
    result = {
        "audit_documents": min(audit_docs, len(docs)),
        "positions_total": total,
        "special_token_positions_skipped": special,
        "positions_with_valid_prefix": valid,
        "valid_prefix_share": valid_share,
        "positions_compared_to_alignment": comparable,
        "exact_offset_agreements": equal,
        "agreement": agreement,
    }
    if comparable == 0:
        raise RuntimeError(f"Source tokenizer boundaries could not be checked against alignment: {result}")
    return result


def _judge_boundaries(text: str, ids: list[int], tokenizer) -> list[int | None]:
    # Position i predicts token i from the prefix ids[:i]; i=0 is intentionally omitted.
    result: list[int | None] = [None] * len(ids)
    for i in range(1, len(ids)):
        prefix = _decode(tokenizer, ids[:i])
        if text.startswith(prefix):
            result[i] = len(prefix)
    return result


def _hash_rows(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['doc_id']}\t{row['pool_index']}\n".encode())
    return digest.hexdigest()


def _manifest(path: Path, base: dict, completed: list[dict], status: str, extra: dict | None = None) -> None:
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {**base, **(extra or {}), "completed_shards": completed, "status": status}
    if "gpu_runtime" in old:
        payload["gpu_runtime"] = old["gpu_runtime"]
    atomic_json(path, payload)


def _score_source(
    target: str,
    judge: str,
    docs: list[dict],
    tokenizer,
    model,
    body,
    head,
    alignment: dict[tuple[int, int], int],
    args,
    config,
) -> dict:
    boundary_audit = _boundary_audit(docs, alignment)
    for doc in docs:
        offsets = []
        for t, boundary in enumerate(doc["source_boundaries"]):
            aligned = alignment.get((doc["pool_index"], t))
            if boundary is None:
                offsets.append(None)
            elif doc["source_special_mask"][t]:
                offsets.append(boundary)
            else:
                offsets.append(aligned if aligned is not None else boundary)
        doc["source_offsets"] = offsets
        doc["judge_ids"] = list(map(int, tokenizer(doc["text"], add_special_tokens=False)["input_ids"]))
        doc["judge_boundaries"] = _judge_boundaries(doc["text"], doc["judge_ids"], tokenizer)

    output_dir = OUTPUT_ROOT / target / judge
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    base = {
        "stage": "cross_family_judge_scoring",
        "source_target": target,
        "judge": judge,
        "source_model_id": config["models"][target]["id"],
        "source_revision": config["models"][target]["revision"],
        "judge_model_id": config["models"][judge]["id"],
        "judge_revision": config["models"][judge]["revision"],
        "split": "all" if target == "qwen4" else "common",
        "prompt_count": len(docs),
        "prompt_ids_sha256": _hash_rows(docs),
        "batch_size": args.batch_size,
        "seed": SEED,
        "source_boundary_audit": boundary_audit,
        "source_token_count": int(sum(len(row["continuation_ids"]) for row in docs)),
        "judge_token_count": int(sum(max(0, len(row["judge_ids"]) - 1) for row in docs)),
        "model_storage_mode": "streamed_shards_to_cpu_or_local_checkpoint",
    }
    completed_by_shard = {}
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(old.get(field) != base.get(field) for field in ("source_target", "judge", "prompt_count", "prompt_ids_sha256")):
            raise RuntimeError(f"Existing cross-family metadata differs at {metadata_path}; choose another output root")
        completed_by_shard = {int(row["shard"]): row for row in old.get("completed_shards", [])}
    write_gpu_metadata(
        metadata_path,
        job="t6b_cross_family_judge_scoring",
        details={"source_target": target, "judge": judge, "batch_size": args.batch_size},
        merge_existing=metadata_path.exists(),
    )
    _manifest(metadata_path, base, list(completed_by_shard.values()), "RUNNING")

    shard_total = (len(docs) + SHARD_DOCS - 1) // SHARD_DOCS
    scale_state = {"scale": None, "max_abs_error": None}
    batch_sizes: set[int] = set()
    for shard_index in range(shard_total):
        start = shard_index * SHARD_DOCS
        shard_docs = docs[start : start + SHARD_DOCS]
        shard_path = output_dir / f"shard-{shard_index:05d}.parquet"
        if shard_index in completed_by_shard and shard_path.is_file():
            print(f"CROSS_FAMILY_RESUME_SKIP target={target} judge={judge} shard={shard_index:05d}", flush=True)
            continue
        result_rows = []
        for batch_start in tqdm(range(0, len(shard_docs), args.batch_size), desc=f"{target} judged by {judge} shard {shard_index}", unit="batch", dynamic_ncols=True, mininterval=1):
            batch_docs = shard_docs[batch_start : batch_start + args.batch_size]
            score_rows = []
            for row in batch_docs:
                ids = row["judge_ids"]
                if len(ids) < 2:
                    continue
                score_rows.append({
                    "doc_id": row["doc_id"],
                    "source_index": row["source_index"],
                    "pool_index": row["pool_index"],
                    "prompt_ids": [ids[0]],
                    "continuation_ids": ids[1:],
                })
            if not score_rows:
                continue
            for per_doc in _score_with_backoff(
                model, body, head, score_rows, "target", 32, scale_state, batch_sizes
            ):
                source_row = next(item for item in batch_docs if item["doc_id"] == per_doc[0]["doc_id"]) if per_doc else None
                if source_row is None:
                    continue
                boundaries = source_row["judge_boundaries"]
                for score in per_doc:
                    judge_index = int(score["position"]) + 1
                    cstart = boundaries[judge_index]
                    if cstart is None:
                        continue
                    result_rows.append({
                        "doc_id": str(score["doc_id"]),
                        "pool_index": int(score["pool_index"]),
                        "judge_index": judge_index,
                        "cstart": int(cstart),
                        "top2_margin": float(score["top2_margin"]),
                        "entropy": float(score["entropy"]),
                        "argmax_id": int(score["argmax_id"]),
                    })
        atomic_parquet(result_rows, shard_path)
        completed_by_shard[shard_index] = {
            "shard": shard_index,
            "document_count": len(shard_docs),
            "scored_position_count": len(result_rows),
            "actual_batch_sizes": sorted(batch_sizes),
            "head_scale": scale_state["scale"],
            "head_check_max_abs_error": scale_state["max_abs_error"],
        }
        _manifest(metadata_path, base, [completed_by_shard[i] for i in sorted(completed_by_shard)], "RUNNING")
        print(f"CROSS_FAMILY_SHARD_COMPLETE target={target} judge={judge} shard={shard_index:05d} rows={len(result_rows)}", flush=True)
    _manifest(metadata_path, base, [completed_by_shard[i] for i in sorted(completed_by_shard)], "COMPLETE")
    return {"target": target, "judge": judge, "boundary_audit": boundary_audit, "manifest": str(metadata_path)}


def _self_gate(target: str, judge: str, limit: int) -> dict:
    if target != judge:
        return {}
    scores = []
    score_dir = OUTPUT_ROOT / target / judge
    for file in sorted(score_dir.glob("shard-*.parquet")):
        scores.append(pd.read_parquet(file))
    score = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    score = score[score.pool_index < limit]
    target_logs = _read_target_logs(target, limit)
    continuations = {int(row["pool_index"]): row for row in _read_continuations(target, limit)}
    tokenizer_path = _snapshot_dir(load_config(DEFAULT_CONFIG), target)
    source_tokenizer = load_tokenizer(load_config(DEFAULT_CONFIG), target, snapshot=tokenizer_path)
    score_by_index = {(int(r.pool_index), int(r.judge_index)): int(r.argmax_id) for r in score.itertuples(index=False)}
    logs = {(int(r.pool_index), int(r.position)): int(r.argmax_id) for r in target_logs.itertuples(index=False)}
    canonical = compared = agreement = total = 0
    for pool_index, row in continuations.items():
        prompt = list(map(int, row["prompt_ids"]))
        cont = list(map(int, row["continuation_ids"]))
        text = _decode(source_tokenizer, prompt + cont)
        reencoded_full = list(map(int, source_tokenizer(text, add_special_tokens=False)["input_ids"]))
        for t in range(len(cont)):
            total += 1
            prefix_ids = prompt + cont[:t]
            prefix_text = _decode(source_tokenizer, prefix_ids)
            if not text.startswith(prefix_text):
                continue
            judge_index = len(prefix_ids)
            if reencoded_full[:judge_index] != prefix_ids:
                continue
            canonical += 1
            judge_argmax = score_by_index.get((pool_index, judge_index))
            stored_argmax = logs.get((pool_index, t))
            if judge_argmax is not None and stored_argmax is not None:
                compared += 1
                agreement += int(judge_argmax == stored_argmax)
    result = {
        "source_target": target,
        "judge": judge,
        "documents": limit,
        "continuation_positions": total,
        "canonical_prefix_positions": canonical,
        "canonical_prefix_share": canonical / total if total else float("nan"),
        "argmax_compared_positions": compared,
        "argmax_agreement": agreement / compared if compared else float("nan"),
    }
    if compared == 0 or result["argmax_agreement"] < 0.985:
        raise RuntimeError(f"Same-model T6-b consistency gate failed: {result}")
    gate_path = score_dir / "consistency_gate.json"
    atomic_json(gate_path, result)
    print("CROSS_FAMILY_SELF_GATE " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-target", nargs="+", required=True, choices=("qwen4", "qwen8", "ministral8"))
    parser.add_argument("--judge", required=True, choices=("qwen4", "qwen8", "ministral8"))
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.max_prompts < 1 or args.batch_size < 1:
        parser.error("--max-prompts and --batch-size must be positive")
    if args.judge == "ministral8" and any(t not in {"qwen4", "qwen8"} for t in args.source_target):
        parser.error("Ministral-8B is the fixed judge for Qwen3 source targets")
    if args.judge == "qwen8" and any(t != "ministral8" for t in args.source_target):
        parser.error("Qwen3-8B is the fixed judge for the Ministral source target")
    config = load_config(args.config)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face token is unavailable; set HF_TOKEN from /workspace/hf_token")
    api = HfApi(token=token)

    source_data = {}
    for target in args.source_target:
        rows = _read_continuations(target, args.max_prompts)
        target_path, _, _ = _ensure_snapshot(config, target, api, token)
        source_tokenizer = load_tokenizer(config, target, snapshot=target_path)
        docs = [_source_doc(row, source_tokenizer) for row in rows]
        alignment = _read_alignment(target, args.max_prompts)
        audit = _boundary_audit(docs, alignment)
        source_data[target] = (docs, alignment, audit)
        print(f"SOURCE_BOUNDARY_AUDIT target={target} " + json.dumps(audit), flush=True)

    judge_path, _, _ = _ensure_snapshot(config, args.judge, api, token)
    judge_tokenizer = load_tokenizer(config, args.judge, snapshot=judge_path)
    # Load the judge after capturing the pre-inference GPU state in every output manifest.
    model = None
    pending = []
    for target in args.source_target:
        output_dir = OUTPUT_ROOT / target / args.judge
        manifest = output_dir / "run_metadata.json"
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "COMPLETE":
            pending.append(target)
        else:
            print(f"CROSS_FAMILY_ALREADY_COMPLETE target={target} judge={args.judge}", flush=True)
    if pending:
        model, model_path, storage_mode = _load_streamed_model(config, args.judge, api, token)
        body, head = _body_and_head(model)
        for target in pending:
            docs, alignment, _audit = source_data[target]
            result = _score_source(
                target, args.judge, docs, judge_tokenizer, model, body, head,
                alignment, args, config,
            )
            mpath = Path(result["manifest"])
            data = json.loads(mpath.read_text(encoding="utf-8"))
            data["model_storage_mode"] = storage_mode
            data["judge_snapshot"] = str(model_path)
            atomic_json(mpath, data)
            print(f"CROSS_FAMILY_COMPLETE target={target} judge={args.judge} output={OUTPUT_ROOT / target / args.judge}", flush=True)
        del body, head, model
        cleanup_cuda(torch)
    if len(args.source_target) == 1 and args.source_target[0] == args.judge:
        _self_gate(args.source_target[0], args.judge, args.max_prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
