#!/usr/bin/env python3
"""CLI for the reproducible Table 1 pipeline.

Examples:
  python scripts/table1_pipeline.py prepare-data
  python scripts/table1_pipeline.py audit-models --pair Q2 --device cuda \\
    --draft-model-path /models/qwen-draft --target-model-path /models/qwen-target
  python scripts/table1_pipeline.py run-sd --pair Q2 --shard-index 0 --num-shards 8 \\
    --draft-model-path /models/qwen-draft --target-model-path /models/qwen-target
  python scripts/table1_pipeline.py align-morphology --pair Q2 --shard-index 0 --num-shards 8
  python scripts/table1_pipeline.py build-table1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.table1_data import (
    freeze_prompt_pool,
    iter_hf_documents,
    load_prompt_pool,
    resolve_dataset_revision,
)
from src.table1_runner import (
    audit_pair_tokenizers,
    load_pair_records,
    resolve_model_revision,
    run_sd_shard,
    run_smoke_test,
    smoke_gate_mode,
)


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic base_config: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config.pop("base_config", None)
    if not base:
        return config

    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = (merge(result[key], value)
                           if isinstance(value, dict) and isinstance(result.get(key), dict)
                           else value)
        return result

    return merge(load_config(path.parent / base, _seen | {path}), config)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _production_config(args: argparse.Namespace) -> dict[str, Any]:
    """Route production artifacts without moving frozen input/model paths."""
    config = load_config(args.config)
    if not getattr(args, "output_dir", None):
        return config
    output = Path(args.output_dir).expanduser().resolve()
    return {**config, "paths": {**config["paths"],
            "runs": str(output / "runs"), "results": str(output / "results")}}


def _abs(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _write_environment_manifest(root: str | Path, output_dir: str | Path | None = None) -> Path:
    """Persist runtime versions and hardware identity beside Table 1 artifacts."""
    import torch

    root_path = Path(root).resolve()
    package_names = (
        "torch", "transformers", "datasets", "accelerate", "huggingface-hub",
        "mistral-common", "kiwipiepy", "pyarrow", "pandas", "PyYAML",
    )
    packages: dict[str, str | None] = {}
    for package in package_names:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(root_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    try:
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu_query = []
    path = (Path(output_dir).expanduser().resolve() if output_dir else root_path) / "metadata/environment.json"
    _write_json(path, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit,
        "packages": packages,
        "gpu_query": gpu_query,
        "torch_cuda_runtime": torch.version.cuda,
        # Never dump the entire environment: it may contain HF/Modal credentials.
        "runtime_options": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED",
        )},
    })
    return path


def _load_pair_meta(path: Path, pair_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model metadata; run audit-models first: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    try:
        return metadata["pairs"][pair_id]
    except KeyError as exc:
        raise RuntimeError(f"No audited metadata for pair {pair_id}") from exc


def _model_info_for_reference(model_reference: str, token: str | None) -> dict[str, Any]:
    """Create auditable metadata for either a local model directory or Hub ID."""
    candidate = Path(str(model_reference)).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Model path must be a directory: {candidate}")
        resolved_path = str(candidate.resolve())
        return {
            "id": str(model_reference),
            "local_path": resolved_path,
            "requested_revision": "local",
            "resolved_revision": None,
            "private": None,
            "gated": None,
        }
    return resolve_model_revision(str(model_reference), token=token)


def _effective_pair_spec(
    pair_spec: dict[str, Any],
    *,
    draft_model_path: str | None = None,
    target_model_path: str | None = None,
    model_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the pair config with the actual model references used by a run."""
    effective = dict(pair_spec)
    if draft_model_path:
        effective["draft"] = draft_model_path
    elif model_meta:
        draft = model_meta.get("draft", {})
        effective["draft"] = draft.get("local_path") or draft.get("id", effective["draft"])
    if target_model_path:
        effective["target"] = target_model_path
    elif model_meta:
        target = model_meta.get("target", {})
        effective["target"] = target.get("local_path") or target.get("id", effective["target"])
    return effective


def _normalise_model_reference(value: str) -> str:
    candidate = Path(str(value)).expanduser()
    return str(candidate.resolve()) if candidate.exists() else str(value)


def _assert_override_matches_audit(
    model_meta: dict[str, Any],
    *,
    draft_model_path: str | None = None,
    target_model_path: str | None = None,
) -> None:
    """Prevent accidentally running SD with a model different from its gate."""
    for side, override in (("draft", draft_model_path), ("target", target_model_path)):
        if not override:
            continue
        info = model_meta.get(side, {})
        audited = info.get("local_path") or info.get("id")
        if _normalise_model_reference(str(override)) != _normalise_model_reference(str(audited)):
            raise RuntimeError(
                f"{side} model override differs from the audited model for this pair. "
                "Run audit-models again with the same --" + side + "-model-path."
            )


def _add_model_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--draft-model-path",
        help="Local draft-model directory or Hugging Face model ID for this pair",
    )
    parser.add_argument(
        "--target-model-path",
        help="Local target-model directory or Hugging Face model ID for this pair",
    )


def command_prepare_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    _write_environment_manifest(root)
    dataset = cfg["dataset"]
    paths = cfg["paths"]
    output = _abs(root, paths["prompts"])
    metadata_path = _abs(root, paths["dataset_metadata"])
    if output.exists() and metadata_path.exists() and not args.force:
        print(f"DATA_ALREADY_FROZEN={output}")
        return 0

    token = _token()
    revision_info = resolve_dataset_revision(
        dataset["id"], requested_revision=dataset.get("revision"), token=token
    )
    scan_limit = dataset.get("scan_limit")
    raw_records = iter_hf_documents(
        dataset["id"], dataset["config"], dataset["split"],
        revision=revision_info["resolved_revision"], token=token,
    )
    if scan_limit is not None:
        raw_records = itertools.islice(raw_records, int(scan_limit))

    def records_with_text_field():
        text_field = dataset.get("text_field", "text")
        for row in raw_records:
            item = dict(row)
            if text_field != "text":
                item["text"] = item.get(text_field)
            yield item

    metadata = {
        **revision_info,
        "config": dataset["config"],
        "split": dataset["split"],
        "text_field": dataset.get("text_field", "text"),
        "scan_limit": scan_limit,
        "boilerplate_cleaner": dataset.get("boilerplate_cleaner", "none"),
    }
    selected = freeze_prompt_pool(
        records_with_text_field(),
        output_path=output,
        common_ids_path=_abs(root, paths["common_ids"]),
        extra_ids_path=_abs(root, paths["extra_ids"]),
        dataset_metadata_path=metadata_path,
        dataset_metadata=metadata,
        limit=int(dataset["pool_size"]),
        seed=int(cfg["seed"]),
        min_chars=int(dataset["min_chars"]),
        min_hangul_ratio=float(dataset["min_hangul_ratio"]),
        dataset_id=dataset["id"],
        dataset_config=dataset["config"],
        dataset_split=dataset["split"],
    )
    print(json.dumps({
        "status": "complete",
        "prompts": len(selected),
        "common": 20_000,
        "extra": len(selected) - 20_000,
        "dataset_revision": revision_info["resolved_revision"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


def _write_compatibility_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "family", "draft_model", "target_model", "draft_revision", "target_revision",
        "compatible", "vocab_equal", "model_vocab_size_equal", "special_tokens_equal",
        "backend_serialization_equal", "probe_count_requested", "probe_count_completed",
        "smoke_passed", "smoke_mode", "smoke_prompts_checked", "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            row = {key: report.get(key) for key in fields}
            row["reasons"] = ";".join(report.get("reasons", []))
            writer.writerow(row)


def command_audit_models(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    _write_environment_manifest(root)
    pairs = cfg["pairs"]
    if (args.draft_model_path or args.target_model_path) and not args.pair:
        raise ValueError(
            "--pair is required when using --draft-model-path/--target-model-path; "
            "do not apply one custom pair to all five audits"
        )
    pair_ids = [args.pair] if args.pair else list(pairs)
    pool_path = _abs(root, cfg["paths"]["prompts"])
    records = load_prompt_pool(pool_path)
    metadata_path = _abs(root, cfg["paths"]["model_metadata"])
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"pairs": {}}
    reports: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        if pair_id not in pairs:
            raise KeyError(f"Unknown pair: {pair_id}")
        pair_spec = _effective_pair_spec(
            pairs[pair_id],
            draft_model_path=args.draft_model_path,
            target_model_path=args.target_model_path,
        )
        pair_records = load_pair_records(pool_path, pair_spec)
        smoke_limit = 5 if args.quick_smoke else int(args.smoke_limit or cfg["inference"]["smoke_prompts"])
        smoke_max_new_tokens = (
            32 if args.quick_smoke
            else int(args.smoke_max_new_tokens or cfg["inference"]["max_new_tokens"])
        )
        quick_smoke_requested = (
            args.quick_smoke
            or smoke_limit != int(cfg["inference"]["smoke_prompts"])
            or smoke_max_new_tokens != int(cfg["inference"]["max_new_tokens"])
        )
        draft_info = _model_info_for_reference(pair_spec["draft"], token=_token())
        target_info = _model_info_for_reference(pair_spec["target"], token=_token())
        model_meta = {"draft": draft_info, "target": target_info}
        report = audit_pair_tokenizers(
            pair_id,
            pair_spec,
            model_meta,
            pair_records,
            probe_count=int(cfg["inference"]["compatibility_probe_prompts"]),
            token=_token(),
            require_offsets=not quick_smoke_requested,
        )
        if not report["compatible"]:
            report.update({"smoke_passed": False, "smoke_mode": "NOT_RUN", "smoke_prompts_checked": 0})
            _write_json(_abs(root, f"audit/tokenizer_compatibility_{pair_id}.json"), report)
            raise RuntimeError(f"Tokenizer compatibility audit failed for {pair_id}: {report['reasons']}")
        smoke_config = {
            **cfg["inference"],
            "smoke_prompts": smoke_limit,
            "max_new_tokens": smoke_max_new_tokens,
            "require_offsets": not quick_smoke_requested,
        }
        smoke = run_smoke_test(
            pair_id,
            pair_spec,
            model_meta,
            pair_records,
            device=args.device,
            config=smoke_config,
            token=_token(),
        )
        smoke_reference_rows = smoke.pop("reference_rows")
        smoke_event_rows = smoke.pop("event_rows")
        validation_dir = _abs(root, cfg["paths"].get("validation", "validation"))
        _write_parquet(validation_dir / f"{pair_id}_smoke.parquet", smoke_event_rows)
        _write_parquet(validation_dir / f"{pair_id}_smoke_references.parquet", smoke_reference_rows)
        smoke["smoke_mode"] = smoke_gate_mode(
            smoke_limit,
            smoke_max_new_tokens,
            full_prompts=int(cfg["inference"]["smoke_prompts"]),
            full_max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        )
        report.update({"smoke_passed": smoke["passed"], "smoke_prompts_checked": smoke["prompts_checked"]})
        report["smoke_mode"] = smoke["smoke_mode"]
        report["smoke"] = smoke
        report["smoke_artifacts"] = {
            "events": str(validation_dir / f"{pair_id}_smoke.parquet"),
            "references": str(validation_dir / f"{pair_id}_smoke_references.parquet"),
        }
        report["model"] = model_meta
        existing["pairs"][pair_id] = {**model_meta, "compatibility": report, "smoke": smoke}
        _write_json(_abs(root, f"audit/tokenizer_compatibility_{pair_id}.json"), report)
        _write_json(validation_dir / f"{pair_id}_smoke_report.json", report)
        reports.append(report)
        _write_json(metadata_path, existing)
        print(
            f"AUDIT_PASS pair={pair_id} probes={report['probe_count_completed']} "
            f"smoke={smoke['prompts_checked']} mode={smoke['smoke_mode']}",
            flush=True,
        )
    all_reports = [
        value["compatibility"]
        for value in existing["pairs"].values()
        if "compatibility" in value
    ]
    _write_compatibility_csv(_abs(root, cfg["paths"]["compatibility_csv"]), all_reports)
    _write_json(_abs(root, "metadata/tokenizer_hashes.json"), {
        "pairs": {
            pair_id: {
                "draft_model": value.get("draft", {}).get("id"),
                "target_model": value.get("target", {}).get("id"),
                "draft_revision": value.get("draft", {}).get("resolved_revision"),
                "target_revision": value.get("target", {}).get("resolved_revision"),
                "backend_sha256": value.get("compatibility", {}).get("backend_sha256"),
                "draft_class": value.get("compatibility", {}).get("draft_class"),
                "target_class": value.get("compatibility", {}).get("target_class"),
                "vocab_size_draft": value.get("compatibility", {}).get("model_vocab_size_draft"),
                "vocab_size_target": value.get("compatibility", {}).get("model_vocab_size_target"),
            }
            for pair_id, value in existing["pairs"].items()
            if "compatibility" in value
        }
    })
    return 0


def command_run_sd(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    pair_spec = cfg["pairs"][args.pair]
    model_meta = _load_pair_meta(_abs(root, cfg["paths"]["model_metadata"]), args.pair)
    _assert_override_matches_audit(
        model_meta,
        draft_model_path=args.draft_model_path,
        target_model_path=args.target_model_path,
    )
    pair_spec = _effective_pair_spec(
        pair_spec,
        draft_model_path=args.draft_model_path,
        target_model_path=args.target_model_path,
        model_meta=model_meta,
    )
    if getattr(args, "sd_batch_size", None) is not None:
        pair_spec["sd_batch_size"] = args.sd_batch_size
    compatibility = model_meta.get("compatibility", {})
    smoke = model_meta.get("smoke", {})
    smoke_mode = smoke.get("smoke_mode")
    if not smoke_mode:
        # Backward-compatible interpretation for a full smoke artifact created
        # before smoke_mode was added.  Quick smoke artifacts always carry the
        # explicit QUICK_ONLY marker.
        smoke_mode = smoke_gate_mode(
            int(smoke.get("prompts_checked", 0)),
            int(smoke.get("smoke_max_new_tokens", cfg["inference"]["max_new_tokens"])),
            full_prompts=int(cfg["inference"]["smoke_prompts"]),
            full_max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        )
    if not compatibility.get("compatible") or not smoke.get("passed") or smoke_mode != "FULL_200":
        raise RuntimeError(
            f"Pair {args.pair} has no full compatibility+smoke gate; "
            f"smoke_mode={smoke_mode}"
        )
    result = run_sd_shard(
        root=root,
        pair_id=args.pair,
        pair_spec=pair_spec,
        model_meta=model_meta,
        config={**cfg, **cfg["inference"], "paths": cfg["paths"]},
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
        token=_token(),
        progress_flush_every=args.progress_flush_every,
        progress_log_every=args.progress_log_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_run_table1(args: argparse.Namespace) -> int:
    """Run one pair's complete Table 1 data path in a fixed shard topology."""
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be at least 1")
    root = Path(args.root).resolve()
    cfg = load_config(args.config)
    prompt_path = _abs(root, cfg["paths"]["prompts"])
    dataset_metadata_path = _abs(root, cfg["paths"]["dataset_metadata"])
    if not prompt_path.exists() or not dataset_metadata_path.exists():
        command_prepare_data(argparse.Namespace(
            config=args.config,
            root=args.root,
            force=False,
        ))
    audit_args = argparse.Namespace(
        config=args.config,
        root=args.root,
        pair=args.pair,
        device=args.device,
        quick_smoke=False,
        smoke_limit=None,
        smoke_max_new_tokens=None,
        draft_model_path=args.draft_model_path,
        target_model_path=args.target_model_path,
    )
    command_audit_models(audit_args)
    stage_results: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        run_args = argparse.Namespace(
            config=args.config,
            root=args.root,
            pair=args.pair,
            shard_index=shard_index,
            num_shards=args.num_shards,
            device=args.device,
            draft_model_path=args.draft_model_path,
            target_model_path=args.target_model_path,
            progress_flush_every=args.progress_flush_every,
            progress_log_every=args.progress_log_every,
            sd_batch_size=getattr(args, "sd_batch_size", None),
        )
        command_run_sd(run_args)
        align_args = argparse.Namespace(
            config=args.config,
            root=args.root,
            pair=args.pair,
            shard_index=shard_index,
            num_shards=args.num_shards,
            device=args.align_device,
        )
        command_align(align_args)
        stage_results.append({"shard_index": shard_index, "status": "complete"})
    print(json.dumps({
        "pair": args.pair,
        "status": "complete",
        "shards": stage_results,
        "next": "run build-table1 after all five pairs finish",
    }, ensure_ascii=False, indent=2))
    return 0


def command_run_table1_main(args: argparse.Namespace) -> int:
    """Run production inference/alignment without audit or smoke gates.

    This intentionally bypasses the full audit and the 200-prompt smoke run.
    The worker checks tokenizer compatibility and prompt-split integrity.
    Strict mode checks independent target parity; microbatched mode records
    target-block verification without claiming independent scalar parity.
    """
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be at least 1")
    root = Path(args.root).resolve()
    cfg = _production_config(args)
    if getattr(args, "require_microbatched", False):
        # Check every pair before the first model load, not after Q1 finishes.
        for pair_id, settings in cfg["pairs"].items():
            decoder = settings.get("decoder", cfg["inference"].get("decoder", "cached"))
            verification = settings.get("target_verification", cfg["inference"].get("target_verification"))
            if decoder != "microbatched" or verification != "batched_native":
                raise ValueError(
                    f"{pair_id}: fast launcher requires decoder=microbatched and "
                    "target_verification=batched_native; scalar fallback is disabled"
                )
        print(f"FAST_ONLY config={Path(args.config).resolve()} scalar_fallback=disabled", flush=True)
    configured_paths = cfg.get("model_paths", {}).get(args.pair, {})
    draft_model_path = args.draft_model_path or configured_paths.get("draft")
    target_model_path = args.target_model_path or configured_paths.get("target")
    if not draft_model_path or not target_model_path:
        raise ValueError(
            "run-table1-main requires model paths either in config.model_paths "
            "or via --draft-model-path/--target-model-path"
        )
    prompt_path = _abs(root, cfg["paths"]["prompts"])
    dataset_metadata_path = _abs(root, cfg["paths"]["dataset_metadata"])
    if not prompt_path.exists() or not dataset_metadata_path.exists():
        command_prepare_data(argparse.Namespace(
            config=args.config,
            root=args.root,
            force=False,
        ))

    pair_spec = _effective_pair_spec(
        cfg["pairs"][args.pair],
        draft_model_path=draft_model_path,
        target_model_path=target_model_path,
    )
    if getattr(args, "sd_batch_size", None) is not None:
        pair_spec["sd_batch_size"] = args.sd_batch_size
    model_meta = {
        "draft": _model_info_for_reference(draft_model_path, token=_token()),
        "target": _model_info_for_reference(target_model_path, token=_token()),
    }
    environment_path = _write_environment_manifest(root, getattr(args, "output_dir", None))
    launch_manifest = {
        "pair_id": args.pair,
        "mode": "production_without_audit_or_smoke",
        "draft": model_meta["draft"],
        "target": model_meta["target"],
        "num_shards": int(args.num_shards),
        "config": str(Path(args.config).resolve()),
        "alignment_deferred": bool(args.skip_align),
        "require_microbatched": bool(getattr(args, "require_microbatched", False)),
        "resolved_config": cfg,
        "effective_pair_spec": pair_spec,
        "device": args.device,
        "align_device": args.align_device,
        "environment": json.loads(environment_path.read_text(encoding="utf-8")),
    }
    _write_json(environment_path.parent / f"table1_main_{args.pair}.json", launch_manifest)
    launch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    _write_json(_abs(root, cfg["paths"]["runs"]) / args.pair / "launches" / f"{launch_stamp}.json",
                launch_manifest)

    stage_results: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        result = run_sd_shard(
            root=root,
            pair_id=args.pair,
            pair_spec=pair_spec,
            model_meta=model_meta,
            config={**cfg, **cfg["inference"], "paths": cfg["paths"]},
            shard_index=shard_index,
            num_shards=args.num_shards,
            device=args.device,
            token=_token(),
            progress_flush_every=args.progress_flush_every,
            progress_log_every=args.progress_log_every,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.skip_align:
            print(
                f"ALIGNMENT_DEFERRED pair={args.pair} shard={shard_index}; "
                "run align-morphology after GPU inference",
                flush=True,
            )
            stage_results.append({"shard_index": shard_index, "status": "sd_complete", "alignment": "deferred"})
        else:
            align_args = argparse.Namespace(
                config=args.config,
                root=args.root,
                output_dir=getattr(args, "output_dir", None),
                pair=args.pair,
                shard_index=shard_index,
                num_shards=args.num_shards,
                device=args.align_device,
                target_model_path=target_model_path,
            )
            command_align(align_args)
            stage_results.append({"shard_index": shard_index, "status": "complete"})

    print(json.dumps({
        "pair": args.pair,
        "status": "sd_complete" if args.skip_align else "complete",
        "mode": "production_without_audit_or_smoke",
        "shards": stage_results,
        "next": (
            "run align-morphology for each shard, then build-table1"
            if args.skip_align else "run build-table1 after all five pairs finish"
        ),
    }, ensure_ascii=False, indent=2))
    return 0


def command_align(args: argparse.Namespace) -> int:
    from src.table1_morphology import align_pair_shard

    cfg = _production_config(args)
    result = align_pair_shard(
        root=Path(args.root).resolve(),
        config=cfg,
        pair_id=args.pair,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
        target_model_path=(getattr(args, "target_model_path", None)
                           or cfg.get("model_paths", {}).get(args.pair, {}).get("target")),
        token=_token(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_build(args: argparse.Namespace) -> int:
    from src.table1_morphology import build_table1_outputs

    cfg = _production_config(args)
    result = build_table1_outputs(root=Path(args.root).resolve(), config=cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/table1_pipeline.yaml"))
    parser.add_argument("--root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare_data)

    audit = sub.add_parser("audit-models")
    audit.add_argument("--pair")
    audit.add_argument("--device", default="cuda")
    audit.add_argument("--quick-smoke", action="store_true", help="Check 5 prompts with max_new_tokens=32")
    audit.add_argument("--smoke-limit", type=int)
    audit.add_argument("--smoke-max-new-tokens", type=int)
    _add_model_override_args(audit)
    audit.set_defaults(func=command_audit_models)

    run = sub.add_parser("run-sd")
    run.add_argument("--pair", required=True, choices=["Q1", "Q2", "Q3", "M1", "G1"])
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--device", default="cuda")
    run.add_argument("--progress-flush-every", type=int, default=64)
    run.add_argument("--progress-log-every", type=int, default=100)
    run.add_argument("--sd-batch-size", type=int, help="Override prompt microbatch size for decoder=microbatched")
    _add_model_override_args(run)
    run.set_defaults(func=command_run_sd)

    run_table1 = sub.add_parser(
        "run-table1",
        help="Audit, run every SD shard, and align one pair; build the final table separately",
    )
    run_table1.add_argument("--pair", required=True, choices=["Q1", "Q2", "Q3", "M1", "G1"])
    run_table1.add_argument("--num-shards", type=int, default=1)
    run_table1.add_argument("--device", default="cuda")
    run_table1.add_argument("--align-device", default="cpu")
    run_table1.add_argument("--progress-flush-every", type=int, default=64)
    run_table1.add_argument("--progress-log-every", type=int, default=100)
    run_table1.add_argument("--sd-batch-size", type=int)
    _add_model_override_args(run_table1)
    run_table1.set_defaults(func=command_run_table1)

    run_table1_main = sub.add_parser(
        "run-table1-main",
        help="Run full production inference/alignment without audit or smoke gates",
    )
    run_table1_main.add_argument("--pair", required=True, choices=["Q1", "Q2", "Q3", "M1", "G1"])
    run_table1_main.add_argument("--num-shards", type=int, default=1)
    run_table1_main.add_argument("--device", default="cuda")
    run_table1_main.add_argument("--align-device", default="cpu")
    run_table1_main.add_argument(
        "--skip-align", action="store_true",
        help="finish GPU SD only; run align-morphology separately to overlap CPU work",
    )
    run_table1_main.add_argument("--progress-flush-every", type=int, default=64)
    run_table1_main.add_argument("--progress-log-every", type=int, default=100)
    run_table1_main.add_argument("--sd-batch-size", type=int)
    run_table1_main.add_argument(
        "--require-microbatched", action="store_true",
        help="Fail before loading models unless every pair uses the no-fallback microbatched decoder",
    )
    _add_model_override_args(run_table1_main)
    run_table1_main.set_defaults(func=command_run_table1_main)

    align = sub.add_parser("align-morphology")
    align.add_argument("--pair", required=True, choices=["Q1", "Q2", "Q3", "M1", "G1"])
    align.add_argument("--shard-index", type=int, default=0)
    align.add_argument("--num-shards", type=int, default=1)
    align.add_argument("--device", default="cpu")
    align.add_argument("--target-model-path")
    align.set_defaults(func=command_align)

    build = sub.add_parser("build-table1")
    build.set_defaults(func=command_build)
    for command in (run_table1_main, align, build):
        command.add_argument(
            "--output-dir",
            help="Store production artifacts under PATH/{runs,results,metadata}; inputs still use --root",
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
