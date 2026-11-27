#!/usr/bin/env python3
"""Teacher-force fixed outside-family judges on frozen generated continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import get_token
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from scripts.tf_score import _body_and_head, _score_with_backoff
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    load_tokenizer,
    model_path,
    require_model_snapshot,
)
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
        pd.read_parquet(file, columns=["doc_id", "pool_index", "position", "argmax_id", "target_token_id", "top2_margin"])
        for file in files
    ], ignore_index=True)
    frame = frame[frame.pool_index < limit]
    return frame.sort_values(["pool_index", "position"], kind="stable").reset_index(drop=True)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _gate_b(rows: pd.DataFrame, vocabulary_size: int) -> dict[str, Any]:
    """Evaluate the preregistered LM sanity gate without using rejection labels."""
    required = {"entropy", "top2_margin", "is_top1"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Gate B input is missing columns: {sorted(missing)}")
    if vocabulary_size < 2:
        raise ValueError("Gate B requires a vocabulary with at least two tokens")
    entropy = rows["entropy"].to_numpy(dtype=float)
    margin = rows["top2_margin"].to_numpy(dtype=float)
    is_top1 = rows["is_top1"].to_numpy(dtype=float)
    if not len(entropy) or not (np.isfinite(entropy).all() and np.isfinite(margin).all() and np.isfinite(is_top1).all()):
        raise ValueError("Gate B requires nonempty finite entropy, margin, and is_top1 values")
    threshold_high_entropy = 0.8 * math.log(vocabulary_size)
    mean_entropy = float(entropy.mean())
    high_entropy_share = float((entropy > threshold_high_entropy).mean())
    top1_accuracy = float(is_top1.mean())
    checks = {
        "mean_entropy": mean_entropy <= 4.0,
        "high_entropy_share": high_entropy_share <= 0.01,
        "top1_accuracy": top1_accuracy >= 0.25,
    }
    return {
        "positions": int(len(rows)),
        "vocabulary_size": int(vocabulary_size),
        "mean_entropy": mean_entropy,
        "high_entropy_threshold_nats": threshold_high_entropy,
        "high_entropy_share": high_entropy_share,
        "top1_accuracy": top1_accuracy,
        "entropy_quantiles": _quantiles(entropy),
        "top2_margin_quantiles": _quantiles(margin),
        "thresholds": {
            "mean_entropy_max": 4.0,
            "high_entropy_share_max": 0.01,
            "top1_accuracy_min": 0.25,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _require_complete_judge_snapshot(config: dict[str, Any], judge: str) -> Path:
    snapshot = model_path(config, judge)
    require_model_snapshot(snapshot)
    manifest_path = snapshot / "naacl_download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_key") != judge or manifest.get("revision") != config["models"][judge]["revision"]:
        raise RuntimeError(f"Judge download manifest does not match pinned model {judge}: {manifest_path}")
    files = manifest.get("weight_files") or []
    if not files:
        raise RuntimeError(f"Judge snapshot manifest contains no weight files: {manifest_path}")
    missing = []
    wrong_size = []
    for item in files:
        path = snapshot / item["filename"]
        if not path.is_file():
            missing.append(item["filename"])
        elif item.get("bytes") is not None and path.stat().st_size != int(item["bytes"]):
            wrong_size.append(item["filename"])
    if missing or wrong_size:
        raise FileNotFoundError(
            f"Judge snapshot is incomplete for {judge}: missing={missing[:8]}, size_mismatch={wrong_size[:8]}"
        )
    return snapshot


def _load_standard_judge(config: dict[str, Any], judge: str):
    snapshot = _require_complete_judge_snapshot(config, judge)
    model = load_model(config, judge)
    return model, snapshot, "standard_from_pretrained"


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
    storage_mode: str,
    snapshot: Path,
    scale_state: dict,
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
        "model_storage_mode": storage_mode,
        "judge_snapshot": str(snapshot),
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
                        "is_top1": int(score["argmax_id"]) == int(score["target_token_id"]),
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


def _self_gate(
    model,
    body,
    head,
    target: str,
    judge: str,
    tokenizer,
    limit: int,
    batch_size: int,
    scale_state: dict,
    storage_mode: str,
    snapshot: Path,
) -> dict:
    """Run Gate A with the same loaded judge object that will score other text."""
    if target != judge:
        raise ValueError("Gate A requires the judge's own frozen continuations")
    continuation_rows = _read_continuations(target, limit)
    target_logs = _read_target_logs(target, limit)
    stored = {
        (int(row.pool_index), int(row.position)): row
        for row in target_logs.itertuples(index=False)
    }
    canonical_map: dict[tuple[int, int], int] = {}
    prompt_lengths: dict[int, int] = {}
    score_rows = []
    total_positions = 0
    for row in continuation_rows:
        pool_index = int(row["pool_index"])
        prompt = list(map(int, row["prompt_ids"]))
        continuation = list(map(int, row["continuation_ids"]))
        prompt_lengths[pool_index] = len(prompt)
        text = _decode(tokenizer, prompt + continuation)
        reencoded = list(map(int, tokenizer(text, add_special_tokens=False)["input_ids"]))
        total_positions += len(continuation)
        for t in range(len(continuation)):
            prefix_ids = prompt + continuation[:t]
            prefix_text = _decode(tokenizer, prefix_ids)
            if not text.startswith(prefix_text):
                continue
            judge_index = len(prefix_ids)
            if reencoded[:judge_index] != prefix_ids:
                continue
            canonical_map[(pool_index, judge_index)] = t
        score_rows.append({
            "doc_id": str(row["doc_id"]),
            "source_index": int(row["source_index"]),
            "pool_index": pool_index,
            "prompt_ids": prompt,
            "continuation_ids": continuation,
        })

    canonical_positions = len(canonical_map)
    compared_argmax = 0
    matches = 0
    new_margins = []
    stored_margins = []
    entropies = []
    batch_sizes_seen: set[int] = set()
    rows_by_prompt_length: dict[int, list[dict]] = {}
    for row in score_rows:
        rows_by_prompt_length.setdefault(len(row["prompt_ids"]), []).append(row)
    for prompt_length in sorted(rows_by_prompt_length):
        prompt_rows = rows_by_prompt_length[prompt_length]
        for start in range(0, len(prompt_rows), batch_size):
            batch = prompt_rows[start : start + batch_size]
            per_doc_rows = _score_with_backoff(
                model, body, head, batch, "target", 32, scale_state, batch_sizes_seen
            )
            for per_doc in per_doc_rows:
                for score in per_doc:
                    pool_index = int(score["pool_index"])
                    # _score_with_backoff positions are continuation-relative;
                    # the canonical map is keyed by the full prefix length.
                    judge_index = prompt_lengths[pool_index] + int(score["position"])
                    target_position = canonical_map.get((pool_index, judge_index))
                    if target_position is None:
                        continue
                    target_log = stored.get((pool_index, target_position))
                    if target_log is None:
                        continue
                    compared_argmax += 1
                    matches += int(int(score["argmax_id"]) == int(target_log.argmax_id))
                    new_margins.append(float(score["top2_margin"]))
                    stored_margins.append(float(target_log.top2_margin))
                    entropies.append(float(score["entropy"]))

    margin_correlation = None
    if len(new_margins) > 1:
        correlation = float(spearmanr(new_margins, stored_margins).statistic)
        if np.isfinite(correlation):
            margin_correlation = correlation
    argmax_agreement = matches / compared_argmax if compared_argmax else None
    entropy_values = np.asarray(entropies, dtype=float)
    result = {
        "source_target": target,
        "judge": judge,
        "documents": len(continuation_rows),
        "continuation_positions": total_positions,
        "canonical_prefix_positions": canonical_positions,
        "canonical_prefix_share": canonical_positions / total_positions if total_positions else None,
        "argmax_compared_positions": compared_argmax,
        "comparison_coverage": compared_argmax / canonical_positions if canonical_positions else None,
        "argmax_agreement": argmax_agreement,
        "spearman_new_vs_stored_top2_margin": margin_correlation,
        "mean_entropy_canonical_positions": float(entropy_values.mean()) if entropy_values.size else None,
        "entropy_quantiles": _quantiles(entropy_values),
        "top2_margin_quantiles": _quantiles(np.asarray(new_margins, dtype=float)),
        "model_storage_mode": storage_mode,
        "judge_snapshot": str(snapshot),
        "head_scale": scale_state["scale"],
        "head_check_max_abs_error": scale_state["max_abs_error"],
        "actual_batch_sizes": sorted(batch_sizes_seen),
        "thresholds": {
            "argmax_agreement_min": 0.985,
            "spearman_new_vs_stored_top2_margin_min": 0.95,
        },
    }
    result["passed"] = bool(
        compared_argmax == canonical_positions
        and argmax_agreement is not None
        and argmax_agreement >= 0.985
        and margin_correlation is not None
        and margin_correlation >= 0.95
    )
    gate_path = OUTPUT_ROOT / target / judge / "consistency_gate.json"
    atomic_json(gate_path, result)
    print("CROSS_FAMILY_CONSISTENCY_GATE " + json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise RuntimeError(f"Gate A failed; not scoring cross-family text: {result}")
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
    allowed_targets = {
        "ministral8": {"qwen4", "qwen8", "ministral8"},
        "qwen8": {"ministral8", "qwen8"},
        "qwen4": {"qwen4"},
    }[args.judge]
    invalid_targets = set(args.source_target) - allowed_targets
    if invalid_targets:
        parser.error(f"Invalid source target(s) for judge {args.judge}: {sorted(invalid_targets)}")
    config = load_config(args.config)
    token = get_token()
    api = HfApi(token=token)

    # Validate the full pinned judge snapshot before any scoring begins. The
    # task's output must come from the standard repository loader.
    judge_snapshot = _require_complete_judge_snapshot(config, args.judge)
    judge_tokenizer = load_tokenizer(config, args.judge, snapshot=judge_snapshot)
    model, judge_snapshot, storage_mode = _load_standard_judge(config, args.judge)
    body, head = _body_and_head(model)
    scale_state = {"scale": None, "max_abs_error": None}

    _self_gate(
        model, body, head, args.judge, args.judge, judge_tokenizer, 200,
        args.batch_size, scale_state, storage_mode, judge_snapshot,
    )

    targets_to_score = [target for target in args.source_target if target != args.judge]
    source_data = {}
    for target in targets_to_score:
        rows = _read_continuations(target, args.max_prompts)
        target_path, _, _ = _ensure_snapshot(config, target, api, token)
        source_tokenizer = load_tokenizer(config, target, snapshot=target_path)
        docs = [_source_doc(row, source_tokenizer) for row in rows]
        alignment = _read_alignment(target, args.max_prompts)
        audit = _boundary_audit(docs, alignment)
        source_data[target] = (docs, alignment, audit)
        print(f"SOURCE_BOUNDARY_AUDIT target={target} " + json.dumps(audit), flush=True)

    pending = []
    for target in targets_to_score:
        output_dir = OUTPUT_ROOT / target / args.judge
        manifest = output_dir / "run_metadata.json"
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "COMPLETE":
            pending.append(target)
        else:
            print(f"CROSS_FAMILY_ALREADY_COMPLETE target={target} judge={args.judge}", flush=True)
    for target in pending:
        docs, alignment, _audit = source_data[target]
        result = _score_source(
            target, args.judge, docs, judge_tokenizer, model, body, head,
            alignment, args, config, storage_mode, judge_snapshot, scale_state,
        )
        print(f"CROSS_FAMILY_COMPLETE target={target} judge={args.judge} output={OUTPUT_ROOT / target / args.judge}", flush=True)

    gate_results = []
    for target in targets_to_score:
        output_dir = OUTPUT_ROOT / target / args.judge
        files = sorted(output_dir.glob("shard-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No scored rows available for Gate B: {output_dir}")
        scored = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
        vocabulary_size = int(getattr(model.config, "vocab_size", len(judge_tokenizer)))
        gate = _gate_b(scored, vocabulary_size)
        gate.update({
            "source_target": target,
            "judge": args.judge,
            "model_storage_mode": storage_mode,
            "judge_snapshot": str(judge_snapshot),
        })
        atomic_json(output_dir / "sanity_gate.json", gate)
        print("CROSS_FAMILY_SANITY_GATE " + json.dumps(gate, ensure_ascii=False), flush=True)
        gate_results.append(gate)
    failed = [result for result in gate_results if not result["passed"]]
    del body, head, model
    cleanup_cuda(torch)
    if failed:
        raise RuntimeError(f"Gate B failed; do not analyze these outputs: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
