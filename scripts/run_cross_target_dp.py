#!/usr/bin/env python3
"""Extract next-token margins on identical Korean text prefixes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_target_dp import prefix_hash
from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    cleanup_cuda,
    load_config,
    load_model,
    load_tokenizer,
    model_path,
    select_prompts,
    sha256_file,
)


def _inputs_for_batch(token_rows: list[list[int]], pad_id: int, device: str):
    width = max(len(row) for row in token_rows)
    ids = torch.full((len(token_rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(token_rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(token_rows):
        if not row:
            raise ValueError("Tokenizer returned an empty evaluation prefix")
        start = width - len(row)
        ids[index, start:] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, start:] = 1
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids


@torch.inference_mode()
def _final_logits(model, ids, mask, position_ids, *, compare_full: bool):
    """Return native model logits at the final non-padding prefix position."""
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": position_ids,
        "use_cache": False,
    }
    max_error = None
    try:
        output = model(**kwargs, logits_to_keep=1)
        selected = output.logits[:, -1, :]
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        output = model(**kwargs)
        selected = output.logits[:, -1, :]
    if selected.shape[0] != ids.shape[0]:
        raise RuntimeError("Model returned the wrong batch dimension")

    if compare_full:
        full_kwargs = {
            key: (value[:1] if isinstance(value, torch.Tensor) else value)
            for key, value in kwargs.items()
        }
        reduced_output = model(**full_kwargs, logits_to_keep=1)
        reduced_selected = reduced_output.logits[0, -1, :].float()
        full_output = model(**full_kwargs)
        full_selected = full_output.logits[0, -1, :].float()
        max_error = float((full_selected - reduced_selected).abs().max().item())
        top_reduced = torch.topk(reduced_selected, k=2)
        top_full = torch.topk(full_selected, k=2)
        margin_delta = float(
            ((top_reduced.values[0] - top_reduced.values[1])
             - (top_full.values[0] - top_full.values[1])).abs().item()
        )
        if not torch.equal(top_reduced.indices, top_full.indices) or margin_delta > 0.02:
            raise RuntimeError(
                "Reduced-logit path changed the top two or margin in a same-shape full-forward check: "
                f"max_abs_error={max_error:.6g}, margin_delta={margin_delta:.6g}, "
                f"top2_reduced={top_reduced.indices.tolist()}, top2_full={top_full.indices.tolist()}"
            )
        del reduced_output, reduced_selected, full_output, full_selected, top_reduced, top_full
    del output
    return selected, max_error


def _distribution_stats(logits: torch.Tensor):
    logits = logits.float()
    log_normalizer = torch.logsumexp(logits, dim=-1)
    log_probs = logits - log_normalizer.unsqueeze(-1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    top_logp, top_ids = torch.topk(log_probs, k=2, dim=-1)
    return (
        top_ids.cpu().numpy(),
        top_logp.cpu().numpy(),
        (top_logp[:, 0] - top_logp[:, 1]).cpu().numpy(),
        entropy.cpu().numpy(),
    )


def _all_positions(config, position_path: Path):
    if not position_path.is_file():
        raise FileNotFoundError(f"Run scripts/prepare_cross_target_dp.py first: {position_path}")
    positions = pd.read_parquet(position_path)
    wanted_ids = set(positions.doc_id.astype(str))
    documents = select_prompts(config, "common")
    document_by_id = {
        str(row["doc_id"]): row for row in documents if str(row["doc_id"]) in wanted_ids
    }
    missing = set(positions.doc_id.astype(str)) - set(document_by_id)
    if missing:
        raise RuntimeError(f"Position file refers to {len(missing)} documents outside the frozen sample")
    prefixes = []
    for row in positions.itertuples(index=False):
        document = document_by_id[str(row.doc_id)]
        text = str(document["text"])
        prefix = text[int(row.context_start) : int(row.char_offset)]
        if len(prefix) != int(row.context_chars) or prefix_hash(prefix) != row.shared_prefix_sha256:
            raise RuntimeError(f"Shared-prefix reconstruction mismatch at {row.position_id}")
        prefixes.append(prefix)
    positions["_prefix_text"] = prefixes
    positions["_position_order"] = np.arange(len(positions), dtype=np.int64)
    return positions, document_by_id


def _model_output_columns(position: pd.DataFrame, token_counts: list[int], spec: dict, key: str,
                          top_ids: np.ndarray, top_logp: np.ndarray, margins: np.ndarray,
                          entropy: np.ndarray, tokenizer) -> pd.DataFrame:
    decode = lambda value: tokenizer.decode(
        [int(value)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    output = position.drop(columns=["_prefix_text", "_position_order"]).copy()
    output["model_key"] = key
    output["model_name"] = spec["id"]
    output["model_revision"] = spec["revision"]
    output["model_family"] = spec["family"]
    output["context_model_tokens"] = np.asarray(token_counts, dtype=np.int32)
    output["target_top1_token_id"] = top_ids[:, 0].astype(np.int64)
    output["target_top2_token_id"] = top_ids[:, 1].astype(np.int64)
    output["target_top1_token"] = [decode(value) for value in top_ids[:, 0]]
    output["target_top2_token"] = [decode(value) for value in top_ids[:, 1]]
    output["target_top1_logp"] = top_logp[:, 0].astype(np.float32)
    output["target_top2_logp"] = top_logp[:, 1].astype(np.float32)
    output["target_top2_margin_nats"] = margins.astype(np.float32)
    output["target_entropy_nats"] = entropy.astype(np.float32)
    return output


def _evict_snapshot(config: dict, key: str) -> None:
    path = model_path(config, key)
    manifest_path = path / "naacl_download_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Refusing to evict unmanifested model directory {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != config["models"][key]["revision"]:
        raise RuntimeError(f"Refusing to evict a different revision at {path}")
    for item in manifest.get("weight_files", []):
        weight = path / item["filename"]
        if weight.is_symlink() or weight.is_file():
            weight.unlink()
    manifest["weights_evicted"] = True
    atomic_json(manifest_path, manifest)
    print(f"WEIGHTS_EVICTED {key}", flush=True)


def run_model(config: dict, key: str, positions: pd.DataFrame, *, batch_size: int, force: bool,
              evict_after: bool, run_root: Path | None = None) -> None:
    if key not in config["models"]:
        raise ValueError(f"Unknown model key: {key}")
    spec = config["models"][key]
    snapshot = model_path(config, key)
    manifest_path = snapshot / "naacl_download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Download the pinned model first: {snapshot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != spec["revision"] or manifest.get("weights_evicted"):
        raise RuntimeError(f"Pinned weights are not available for {key}; download {spec['revision']} first")

    base_output = run_root or (ROOT / config["paths"]["outputs"])
    result_root = base_output / "model_runs" / key
    result_root.mkdir(parents=True, exist_ok=True)
    run_manifest = result_root / "run_metadata.json"
    position_manifest = ROOT / config["paths"]["outputs"] / "position_metadata.json"
    position_sha = sha256_file(ROOT / config["paths"]["outputs"] / "evaluation_positions.parquet")
    expected_ids = positions["position_id"].astype(str).tolist()
    if run_manifest.exists() and not force:
        old = json.loads(run_manifest.read_text(encoding="utf-8"))
        if old.get("status") == "COMPLETE":
            if old.get("position_file_sha256") != position_sha:
                raise RuntimeError(f"Completed output for {key} used a different position sample")
            if evict_after:
                _evict_snapshot(config, key)
            print(f"MODEL_RUN_ALREADY_COMPLETE {key}", flush=True)
            return

    shard_size = 2000
    if force:
        for path in result_root.glob("margins-*.parquet"):
            path.unlink()
    started = time.monotonic()
    base_metadata = {
        "schema": config["schema"],
        "status": "RUNNING",
        "model_key": key,
        "model_name": spec["id"],
        "model_revision": spec["revision"],
        "model_family": spec["family"],
        "position_file_sha256": position_sha,
        "position_metadata_sha256": sha256_file(position_manifest),
        "positions_expected": len(positions),
        "batch_size": batch_size,
        "dtype": config["inference"]["dtype"],
        "attention_backend": config["inference"]["attention_backend"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(run_manifest, base_metadata)

    tokenizer = load_tokenizer(config, key)
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"Tokenizer for {key} has no pad or EOS token")
        pad_id = tokenizer.eos_token_id
    model = load_model(config, key, snapshot=snapshot)
    model.eval()

    processed_positions = 0
    total_tokens = 0
    smoke_max_error: float | None = None
    for start in range(0, len(positions), shard_size):
        shard_index = start // shard_size
        shard_path = result_root / f"margins-{shard_index:05d}.parquet"
        shard_positions = positions.iloc[start : start + shard_size]
        expected_shard_ids = shard_positions.position_id.astype(str).tolist()
        if shard_path.exists() and not force:
            stored = pd.read_parquet(shard_path, columns=["position_id", "model_revision"])
            if stored.position_id.astype(str).tolist() == expected_shard_ids and stored.model_revision.eq(spec["revision"]).all():
                processed_positions += len(stored)
                total_tokens += int(pd.read_parquet(shard_path, columns=["context_model_tokens"]).context_model_tokens.sum())
                print(f"RESUME_SKIP {key} shard={shard_index:05d}", flush=True)
                continue
            raise RuntimeError(f"Existing output shard does not match frozen positions: {shard_path}")

        prefix_texts = shard_positions["_prefix_text"].astype(str).tolist()
        encoded = tokenizer(prefix_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        if any(not row for row in encoded):
            raise RuntimeError(f"Tokenizer {key} produced an empty prefix")
        lengths = np.asarray([len(row) for row in encoded], dtype=np.int32)
        max_positions = getattr(model.config, "max_position_embeddings", None)
        text_cfg = getattr(model.config, "text_config", None)
        if max_positions is None and text_cfg is not None:
            max_positions = getattr(text_cfg, "max_position_embeddings", None)
        if max_positions is not None and int(lengths.max()) > int(max_positions):
            raise RuntimeError(f"Context length {int(lengths.max())} exceeds {key} limit {max_positions}")
        order = np.argsort(lengths, kind="stable")
        shard_top_ids = np.empty((len(encoded), 2), dtype=np.int64)
        shard_top_logp = np.empty((len(encoded), 2), dtype=np.float32)
        shard_margins = np.empty(len(encoded), dtype=np.float32)
        shard_entropy = np.empty(len(encoded), dtype=np.float32)
        for batch_start in tqdm(range(0, len(order), batch_size), desc=f"{key} shard {shard_index:05d}", leave=False):
            batch_indices = order[batch_start : batch_start + batch_size]
            token_rows = [encoded[int(i)] for i in batch_indices]
            ids, mask, position_ids = _inputs_for_batch(token_rows, int(pad_id), "cuda")
            logits, full_error = _final_logits(
                model,
                ids,
                mask,
                position_ids,
                compare_full=(smoke_max_error is None),
            )
            if full_error is not None:
                smoke_max_error = full_error
            top_ids, top_logp, margins, entropy = _distribution_stats(logits)
            for local_index, global_index in enumerate(batch_indices.tolist()):
                shard_top_ids[global_index] = top_ids[local_index]
                shard_top_logp[global_index] = top_logp[local_index]
                shard_margins[global_index] = margins[local_index]
                shard_entropy[global_index] = entropy[local_index]
            del ids, mask, position_ids, logits, top_ids, top_logp, margins, entropy
        token_counts = lengths.tolist()
        model_rows = _model_output_columns(
            shard_positions.reset_index(drop=True),
            token_counts,
            spec,
            key,
            shard_top_ids,
            shard_top_logp,
            shard_margins,
            shard_entropy,
            tokenizer,
        )
        atomic_parquet(model_rows.to_dict(orient="records"), shard_path)
        total_tokens += int(lengths.sum())
        processed_positions += len(model_rows)
        print(
            f"MODEL_SHARD_COMPLETE model={key} shard={shard_index:05d} "
            f"positions={len(model_rows)} token_sum={int(lengths.sum())} max_tokens={int(lengths.max())}",
            flush=True,
        )

    if processed_positions != len(positions):
        raise RuntimeError(f"Output position accounting mismatch for {key}")
    output_ids: list[str] = []
    shard_files = sorted(result_root.glob("margins-*.parquet"))
    for path in shard_files:
        output_ids.extend(pd.read_parquet(path, columns=["position_id"]).position_id.astype(str).tolist())
    if output_ids != expected_ids:
        raise RuntimeError(f"Saved output positions/order do not match the frozen sample for {key}")
    if smoke_max_error is None:
        first_prefix = str(positions.iloc[0]["_prefix_text"])
        token_rows = tokenizer([first_prefix], add_special_tokens=False, padding=False)["input_ids"]
        ids, mask, position_ids = _inputs_for_batch(token_rows, int(pad_id), "cuda")
        _, smoke_max_error = _final_logits(
            model, ids, mask, position_ids, compare_full=True
        )
        del ids, mask, position_ids
    elapsed = time.monotonic() - started
    completed = {
        **base_metadata,
        "status": "COMPLETE",
        "positions_written": len(positions),
        "input_tokens_processed": total_tokens,
        "full_forward_check_max_abs_error": smoke_max_error,
        "wall_seconds": elapsed,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(path.relative_to(ROOT)) for path in shard_files],
        "output_sha256": {path.name: sha256_file(path) for path in shard_files},
    }
    atomic_json(run_manifest, completed)
    print(
        f"MODEL_RUN_COMPLETE model={key} positions={len(positions)} tokens={total_tokens} "
        f"wall_seconds={elapsed:.1f} full_forward_max_abs_error={smoke_max_error}",
        flush=True,
    )
    del model, tokenizer
    gc.collect()
    cleanup_cuda(torch)
    if evict_after:
        _evict_snapshot(config, key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_target_dp.yaml")
    parser.add_argument("--model", required=True, help="Pinned target key from the config")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--position-limit", type=int, help="Run an initial subset for a GPU smoke test")
    parser.add_argument("--run-root", type=Path, help="Separate output directory, e.g. results/cross_target_dp/smoke")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evict-after", action="store_true", help="Remove only this revision's weight shards after a completed run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model not in config["models"]:
        parser.error(f"unknown target model key {args.model!r}")
    output_root = ROOT / config["paths"]["outputs"]
    positions_path = output_root / "evaluation_positions.parquet"
    positions, _ = _all_positions(config, positions_path)
    if args.position_limit is not None:
        if args.position_limit < 1:
            parser.error("--position-limit must be positive")
        positions = positions.head(args.position_limit).copy()
    run_model(
        config,
        args.model,
        positions,
        batch_size=args.batch_size or int(config["inference"]["batch_size"]),
        force=args.force,
        evict_after=args.evict_after,
        run_root=(ROOT / args.run_root if args.run_root and not args.run_root.is_absolute() else args.run_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
