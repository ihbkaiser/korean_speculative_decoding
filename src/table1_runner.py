"""Model audit and resumable greedy speculative-decoding shards for Table 1."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

from tqdm.auto import tqdm

from .data import encode_prompt
from .batched_speculative import speculative_greedy_microbatch
from .models import (
    assert_tokenizer_compatible,
    compare_tokenizers,
    config_vocab_size,
    load_models,
    load_table1_tokenizer,
)
from .speculative_decoding import (
    greedy_generate,
    greedy_generate_batch,
    speculative_greedy,
    speculative_greedy_cached,
    verify_greedy_equivalence,
)
from .table1_data import load_prompt_pool


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


class ProgressWriter:
    """Write resumable progress with bounded shared-filesystem flushes."""

    def __init__(self, handle: TextIO, *, flush_every: int = 64):
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.handle = handle
        self.flush_every = flush_every
        self.pending = 0

    def write(self, item: dict[str, Any]) -> None:
        self.handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self.handle.flush()
        self.pending = 0


def _recover_microbatch_progress_tail(path: Path) -> None:
    """Preserve and remove only an incomplete final record after a hard kill."""
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n") or handle.read(1):
                    raise RuntimeError(f"Corrupt checkpoint record before EOF in {path}")
                backup = path.with_name(f"{path.name}.incomplete-tail-{time.time_ns()}")
                with backup.open("xb") as saved:
                    saved.write(line)
                handle.truncate(offset)
                print(f"Recovered incomplete final checkpoint record; saved bytes to {backup}", flush=True)
                return
            if not line.endswith(b"\n"):
                handle.write(b"\n")
                return


def _write_parquet_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def resolve_model_revision(
    model_id: str,
    *,
    requested_revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(model_id, revision=requested_revision)
    resolved = getattr(info, "sha", None)
    if not resolved:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {model_id}")
    return {
        "id": model_id,
        "requested_revision": requested_revision or "main",
        "resolved_revision": resolved,
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False)),
    }


def load_pair_records(pool_path: str | Path, pair_spec: dict[str, Any]) -> list[dict[str, Any]]:
    records = load_prompt_pool(pool_path)
    split = pair_spec["prompt_split"]
    if split == "common_20k":
        selected = records[:20_000]
    elif split == "all_40k":
        selected = records[:40_000]
    else:
        raise ValueError(f"Unknown prompt split: {split}")
    expected_count = int(pair_spec["prompt_count"])
    if len(selected) != expected_count:
        raise AssertionError(f"Expected {expected_count} prompt records for {split}, found {len(selected)}")
    if len({row["doc_id"] for row in selected}) != len(selected):
        raise AssertionError("pair prompt records contain duplicate doc_id")
    return selected


def prompt_ids_sha256(records: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["doc_id"]) for row in records).encode("utf-8")).hexdigest()


def resolve_model_reference(
    model_meta: dict[str, Any],
    side: str,
    *,
    override: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a model name/path and revision for Transformers loading.

    A local model directory must not receive a Hub commit SHA: the revision is
    meaningful for Hub identifiers only.  ``override`` is intentionally also
    allowed to be a Hub identifier so one CLI option can cover both use cases.
    """
    if override:
        candidate = Path(str(override)).expanduser()
        if candidate.exists():
            return str(candidate.resolve()), None
        return str(override), None

    info = model_meta[side]
    local_path = info.get("local_path")
    if local_path:
        candidate = Path(str(local_path)).expanduser()
        if candidate.exists():
            return str(candidate.resolve()), None
    resolved_revision = info.get("resolved_revision")
    return str(info["id"]), str(resolved_revision) if resolved_revision else None


def _model_kwargs(model_meta: dict[str, Any], side: str) -> tuple[str, str | None]:
    return resolve_model_reference(model_meta, side)


def build_run_identity(
    *,
    pair_id: str,
    pair_spec: dict[str, Any],
    model_meta: dict[str, Any],
    config: dict[str, Any],
    records: list[dict[str, Any]],
    shard_records: list[dict[str, Any]],
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    """Build the immutable identity used to validate resume artifacts."""
    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    target_name, target_revision = _model_kwargs(model_meta, "target")
    identity = {
        "pair_id": pair_id,
        "prompt_split": pair_spec.get("prompt_split"),
        "prompt_count": len(records),
        "prompt_ids_sha256": prompt_ids_sha256(records),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "shard_prompt_count": len(shard_records),
        "shard_prompt_ids_sha256": prompt_ids_sha256(shard_records),
        "draft_model": draft_name,
        "draft_revision": draft_revision,
        "target_model": target_name,
        "target_revision": target_revision,
        "max_prompt_tokens": int(config["max_prompt_tokens"]),
        "max_new_tokens": int(config["max_new_tokens"]),
        "speculative_k": int(config["speculative_k"]),
        "decoder": str(pair_spec.get("decoder", config.get("decoder", "cached"))),
        "dtype": _pair_inference_setting(pair_spec, config, "dtype", "float16"),
        "attention_backend": _pair_inference_setting(pair_spec, config, "attention_backend", "sdpa"),
        "reference_batch_size": int(
            pair_spec.get("reference_batch_size", config.get("reference_batch_size", 1))
        ),
        "target_verification": _pair_inference_setting(
            pair_spec, config, "target_verification", "sequential"
        ),
    }
    if identity["decoder"] == "microbatched":
        identity["sd_batch_size"] = int(pair_spec.get("sd_batch_size", config.get("sd_batch_size", 8)))
        if identity["sd_batch_size"] < 1:
            raise ValueError("sd_batch_size must be positive")
        identity["reference_policy"] = "block_target_verified_no_scalar_reference"
        if pair_spec.get("tokenizer_backend"):
            identity["tokenizer_backend"] = pair_spec["tokenizer_backend"]
            identity["fix_mistral_regex"] = bool(pair_spec.get("fix_mistral_regex", False))
    return identity


def _pair_inference_setting(
    pair_spec: dict[str, Any], config: dict[str, Any], key: str, default: str,
) -> str:
    """Resolve an optional pair-specific inference override."""
    return str(pair_spec.get(key, config.get(key, default)))


def select_speculative_decoder(pair_spec: dict[str, Any], config: dict[str, Any]):
    """Select a pinned decoder without silently changing audit semantics."""
    decoder_name = str(pair_spec.get("decoder", config.get("decoder", "cached"))).lower()
    decoders = {
        "legacy": speculative_greedy,
        "cached": speculative_greedy_cached,
    }
    try:
        return decoders[decoder_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown speculative decoder {decoder_name!r}; expected legacy or cached"
        ) from exc


def generate_reference_ids(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    target_model: Any,
    pair_spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, list[int]]:
    """Generate target references, using only parity-approved microbatches."""
    if not records:
        return {}
    prompt_ids_by_doc = {
        str(record["doc_id"]): encode_prompt(
            tokenizer, str(record["text"]), int(config["max_prompt_tokens"])
        )
        for record in records
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    batch_size = int(pair_spec.get("reference_batch_size", config.get("reference_batch_size", 1)))
    if batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if batch_size == 1:
        return {
            doc_id: greedy_generate(
                target_model,
                prompt_ids,
                int(config["max_new_tokens"]),
                eos_token_id,
            )
            for doc_id, prompt_ids in tqdm(
                prompt_ids_by_doc.items(), desc="Target references", unit="prompt",
                dynamic_ncols=True, mininterval=1,
            )
        }

    # Grouping by exact prompt length avoids padding/mask-dependent numerical
    # changes. Pair-level batch sizes are enabled only after B200 parity pilots.
    buckets: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for doc_id, prompt_ids in prompt_ids_by_doc.items():
        buckets[len(prompt_ids)].append((doc_id, prompt_ids))
    references: dict[str, list[int]] = {}
    with tqdm(total=len(records), desc="Target references", unit="prompt",
              dynamic_ncols=True, mininterval=1) as bar:
        for bucket in buckets.values():
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start:start + batch_size]
                outputs = greedy_generate_batch(
                    target_model,
                    [prompt_ids for _, prompt_ids in chunk],
                    int(config["max_new_tokens"]),
                    eos_token_id,
                )
                references.update({doc_id: output for (doc_id, _), output in zip(chunk, outputs)})
                bar.update(len(chunk))
    return references


def audit_pair_tokenizers(
    pair_id: str,
    pair_spec: dict[str, Any],
    model_meta: dict[str, Any],
    prompt_records: list[dict[str, Any]],
    *,
    probe_count: int = 1000,
    token: str | None = None,
    require_offsets: bool = True,
) -> dict[str, Any]:
    from transformers import AutoConfig

    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    target_name, target_revision = _model_kwargs(model_meta, "target")
    tokenizer_options = {"backend": pair_spec.get("tokenizer_backend", "auto"),
                         "fix_mistral_regex": pair_spec.get("fix_mistral_regex", False)}
    draft_tokenizer = load_table1_tokenizer(
        draft_name, revision=draft_revision, token=token, **tokenizer_options
    )
    target_tokenizer = load_table1_tokenizer(
        target_name, revision=target_revision, token=token, **tokenizer_options
    )
    probes = [str(row["text"]) for row in prompt_records[:probe_count]]
    compatibility = compare_tokenizers(
        draft_tokenizer,
        target_tokenizer,
        probes=probes,
        require_offsets=require_offsets,
    )
    draft_config = AutoConfig.from_pretrained(draft_name, revision=draft_revision, token=token)
    target_config = AutoConfig.from_pretrained(target_name, revision=target_revision, token=token)
    draft_vocab_size = config_vocab_size(draft_config)
    target_vocab_size = config_vocab_size(target_config)
    compatibility.update({
        "pair_id": pair_id,
        "family": pair_spec["family"],
        "draft_model": draft_name,
        "target_model": target_name,
        "draft_revision": draft_revision,
        "target_revision": target_revision,
        "model_vocab_size_draft": draft_vocab_size,
        "model_vocab_size_target": target_vocab_size,
        "model_vocab_size_equal": draft_vocab_size == target_vocab_size,
        "probe_count_requested": probe_count,
        "probe_count_completed": len(compatibility.get("probes", [])),
    })
    if draft_vocab_size != target_vocab_size:
        compatibility["compatible"] = False
        compatibility.setdefault("reasons", []).append("model config vocab sizes differ")
    return compatibility


def _event_row(
    event: dict[str, Any],
    *,
    pair_id: str,
    draft_model: str,
    target_model: str,
    tokenizer: Any,
    record: dict[str, Any],
    prompt_token_count: int,
    token_text_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    token_id = int(event["draft_token_id"])
    if token_text_cache is not None and token_id in token_text_cache:
        raw_token_text = token_text_cache[token_id]
    else:
        raw_token_text = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if token_text_cache is not None:
            token_text_cache[token_id] = raw_token_text
    return {
        "doc_id": record["doc_id"],
        "raw_text_hash": record["raw_text_hash"],
        "pair_id": pair_id,
        "draft_model": draft_model,
        "target_model": target_model,
        "prompt_token_count": int(prompt_token_count),
        "generated_token_position": int(event["output_token_position"]),
        "proposal_slot": int(event["proposal_position"]) - 1,
        "block_id": int(event["round_index"]),
        "token_id": token_id,
        "token_text": raw_token_text,
        "accepted": bool(event["accepted"]) if event["accepted"] is not None else None,
        "rejected": bool(event["rejected"]) if event["rejected"] is not None else None,
        "invalidated_by_earlier_rejection": bool(event["invalidated_after_first_rejection"]),
        "draft_entropy": event["draft_entropy"],
        "target_entropy": event["target_entropy"],
        "draft_logprob_of_proposed_token": event["draft_logprob"],
        "draft_top1_logprob": event.get("draft_top1_logprob"),
        "target_logprob_of_proposed_token": event["target_logprob"],
        "target_top1_token_id": int(event["target_greedy_token_id"]),
        "target_top1_logprob": event["target_top1_logprob"],
        "target_margin": float(event["target_top1_logprob"] - event["target_logprob"]),
        "target_rank_of_proposed_token": int(event["target_rank_of_proposed_token"]),
        "later_proposals_invalidated": bool(event["invalidated_after_first_rejection"]),
        "is_first_rejection": bool(event["is_first_rejection"]),
        "accepted_prefix_length": int(event["accepted_prefix_length"]),
        "first_rejection_output_position": event["first_rejection_output_position"],
        "pool_index": record.get("pool_index"),
        "pool_split": record.get("pool_split"),
    }


def _prompt_run(
    record: dict[str, Any],
    *,
    pair_id: str,
    pair_spec: dict[str, Any],
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    reference_ids: list[int] | None = None,
    strict_reference_check: bool = True,
    batch_target_verification: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_ids = encode_prompt(tokenizer, str(record["text"]), int(config["max_prompt_tokens"]))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if reference_ids is None:
        reference_ids = greedy_generate(
            target_model, prompt_ids, int(config["max_new_tokens"]), eos_token_id
        )
    decoder = select_speculative_decoder(pair_spec, config)
    decoder_kwargs: dict[str, Any] = {"prompt_id": record["doc_id"]}
    if decoder is speculative_greedy_cached:
        if batch_target_verification is None:
            verification_mode = _pair_inference_setting(
                pair_spec, config, "target_verification", "sequential"
            ).lower()
            batch_target_verification = verification_mode in {
                "batched", "batched_fallback", "block",
            }
        decoder_kwargs["batch_target_verification"] = bool(batch_target_verification)
    speculative_ids, raw_events = decoder(
        draft_model,
        target_model,
        prompt_ids,
        int(config["max_new_tokens"]),
        eos_token_id,
        int(config["speculative_k"]),
        **decoder_kwargs,
    )
    exact = reference_ids is None or reference_ids == speculative_ids
    if strict_reference_check and reference_ids is not None:
        verify_greedy_equivalence(reference_ids, speculative_ids, record["doc_id"])
    if reference_ids is None:
        reference_ids = speculative_ids
    reference_text = tokenizer.decode(
        reference_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    reference_row = {
        "doc_id": record["doc_id"],
        "raw_text_hash": record["raw_text_hash"],
        "pair_id": pair_id,
        "draft_model": pair_spec["draft"],
        "target_model": pair_spec["target"],
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_text": tokenizer.decode(
            prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
        "target_continuation_token_ids": reference_ids,
        "target_continuation_text": reference_text,
        "speculative_token_ids": speculative_ids,
        "exact_sd_target": bool(exact),
        "reference_source": "provided_target_reference" if exact else "provided_target_reference_mismatch",
        "completed": True,
        "pool_index": record.get("pool_index"),
        "pool_split": record.get("pool_split"),
        "source_index": record.get("source_index"),
    }
    event_rows = [
        _event_row(
            event,
            pair_id=pair_id,
            draft_model=pair_spec["draft"],
            target_model=pair_spec["target"],
            tokenizer=tokenizer,
            record=record,
            prompt_token_count=len(prompt_ids),
        )
        for event in raw_events
    ]
    return reference_row, event_rows


def _repair_reference_mismatch(
    record: dict[str, Any],
    reference: dict[str, Any],
    *,
    target_model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replace a rare batch-reference disagreement with a scalar target ref.

    B200 batch kernels are normally parity-stable, but a near-tied argmax can
    differ between a batch-64 prefill and the singleton cached path used by
    the strict verifier.  Regenerating only this prompt keeps the common case
    fast without weakening the exact token-ID gate.
    """
    prompt_ids = [int(token_id) for token_id in reference["prompt_token_ids"]]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    scalar_ids = greedy_generate(
        target_model,
        prompt_ids,
        int(config["max_new_tokens"]),
        eos_token_id,
    )
    speculative_ids = [int(token_id) for token_id in reference["speculative_token_ids"]]
    verify_greedy_equivalence(scalar_ids, speculative_ids, record["doc_id"])
    batch_ids = [int(token_id) for token_id in reference["target_continuation_token_ids"]]
    reference["batch_reference_token_ids"] = batch_ids
    reference["batch_reference_first_difference"] = next(
        (
            index
            for index, (left, right) in enumerate(zip(batch_ids, scalar_ids))
            if left != right
        ),
        min(len(batch_ids), len(scalar_ids))
        if len(batch_ids) != len(scalar_ids)
        else None,
    )
    reference["target_continuation_token_ids"] = scalar_ids
    reference["target_continuation_text"] = tokenizer.decode(
        scalar_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    reference["exact_sd_target"] = True
    reference["reference_source"] = "scalar_target_fallback_after_batch_mismatch"
    reference["scalar_parity_validated"] = True
    return reference


def run_smoke_test(
    pair_id: str,
    pair_spec: dict[str, Any],
    model_meta: dict[str, Any],
    prompt_records: list[dict[str, Any]],
    *,
    device: str,
    config: dict[str, Any],
    token: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    target_name, target_revision = _model_kwargs(model_meta, "target")
    tokenizer_options = {"backend": pair_spec.get("tokenizer_backend", "auto"),
                         "fix_mistral_regex": pair_spec.get("fix_mistral_regex", False)}
    draft_tokenizer = load_table1_tokenizer(draft_name, revision=draft_revision, token=token, **tokenizer_options)
    target_tokenizer = load_table1_tokenizer(target_name, revision=target_revision, token=token, **tokenizer_options)
    tokenizer_report = compare_tokenizers(
        draft_tokenizer,
        target_tokenizer,
        require_offsets=bool(config.get("require_offsets", True)),
    )
    if not tokenizer_report["compatible"]:
        raise RuntimeError("Smoke tokenizer compatibility failed: " + "; ".join(tokenizer_report["reasons"]))
    draft_model, target_model = load_models(
        draft_name,
        target_name,
        device=device,
        draft_revision=draft_revision,
        target_revision=target_revision,
        token=token,
        dtype=_pair_inference_setting(pair_spec, config, "dtype", "float16"),
        attention_backend=_pair_inference_setting(pair_spec, config, "attention_backend", "sdpa"),
    )
    smoke_limit = min(int(config.get("smoke_prompts", 200)), len(prompt_records))
    smoke_records = prompt_records[:smoke_limit]
    reference_ids_by_doc = generate_reference_ids(
        smoke_records,
        tokenizer=target_tokenizer,
        target_model=target_model,
        pair_spec=pair_spec,
        config=config,
    )
    started = time.time()
    checked: list[str] = []
    reference_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for record in smoke_records:
        reference, events = _prompt_run(
            record,
            pair_id=pair_id,
            pair_spec=pair_spec,
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=target_tokenizer,
            config=config,
            reference_ids=reference_ids_by_doc[str(record["doc_id"])],
            strict_reference_check=False,
            batch_target_verification=False,
        )
        if not reference["exact_sd_target"]:
            reference = _repair_reference_mismatch(
                record,
                reference,
                target_model=target_model,
                tokenizer=target_tokenizer,
                config=config,
            )
        checked.append(str(record["doc_id"]))
        reference_rows.append(reference)
        for event in events:
            for field in (
                "draft_entropy", "target_entropy", "draft_logprob",
                "target_logprob", "draft_top1_logprob", "target_top1_logprob",
            ):
                value = event.get(field)
                if value is None or not math.isfinite(float(value)):
                    raise AssertionError(f"smoke event has non-finite {field}: {value!r}")
            invalidated = bool(event["invalidated_after_first_rejection"])
            if invalidated:
                if event.get("rejected") is not None or event.get("accepted") is not False:
                    raise AssertionError("invalidated proposal has invalid acceptance accounting")
            elif not isinstance(event.get("rejected"), bool) or not isinstance(event.get("accepted"), bool):
                raise AssertionError("valid proposal has invalid acceptance accounting")
        event_rows.extend(events)
    return {
        "pair_id": pair_id,
        "passed": True,
        "prompts_checked": len(checked),
        "smoke_max_new_tokens": int(config["max_new_tokens"]),
        "prompt_ids_sha256": hashlib.sha256("\n".join(checked).encode("utf-8")).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "device": device,
        "reference_rows": reference_rows,
        "event_rows": event_rows,
    }


def smoke_gate_mode(
    smoke_prompts: int,
    smoke_max_new_tokens: int,
    *,
    full_prompts: int = 200,
    full_max_new_tokens: int = 128,
) -> str:
    """Label whether a smoke run is eligible for the expensive SD stage."""
    if smoke_prompts == full_prompts and smoke_max_new_tokens == full_max_new_tokens:
        return f"FULL_{full_prompts}"
    return "QUICK_ONLY"


def _run_microbatch_shard(
    *, pair_id, pair_spec, config, draft_model, target_model, tokenizer,
    shard_records, completed, progress_path, shard_dir, run_identity,
    progress_flush_every, progress_log_every,
):
    """Production block-verified SD; no pre-generation or scalar reruns."""
    batch_size = int(pair_spec.get("sd_batch_size", config.get("sd_batch_size", 8)))
    if batch_size < 1:
        raise ValueError("sd_batch_size must be positive")
    pending = [row for row in shard_records if str(row["doc_id"]) not in completed]
    token_text_cache: dict[int, str] = {}
    started = time.perf_counter()
    resumed_count = len(completed)
    next_log = len(completed) + max(1, progress_log_every)
    prefix = f"[{pair_id} microbatch {batch_size}]"
    print(f"{prefix} resumed={len(completed)} pending={len(pending)} "
          "verification=target_block scalar_fallback=disabled", flush=True)
    with tqdm(total=len(shard_records), initial=resumed_count,
              desc=f"{pair_id} SD (B={batch_size})", unit="prompt",
              dynamic_ncols=True, mininterval=1) as bar, \
            progress_path.open("a", encoding="utf-8") as progress:
        writer = ProgressWriter(progress, flush_every=progress_flush_every)
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            prompts = [encode_prompt(tokenizer, str(row["text"]), int(config["max_prompt_tokens"]))
                       for row in chunk]
            outputs, raw_events = speculative_greedy_microbatch(
                draft_model, target_model, prompts, int(config["max_new_tokens"]),
                getattr(tokenizer, "eos_token_id", None), int(config["speculative_k"]),
                prompt_ids=[str(row["doc_id"]) for row in chunk],
            )
            for record, prompt, output, raw in zip(chunk, prompts, outputs, raw_events):
                reference = {
                    "doc_id": record["doc_id"], "raw_text_hash": record["raw_text_hash"],
                    "pair_id": pair_id, "draft_model": pair_spec["draft"], "target_model": pair_spec["target"],
                    "prompt_token_count": len(prompt), "prompt_token_ids": prompt,
                    "prompt_text": tokenizer.decode(prompt, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False),
                    "target_continuation_token_ids": output,
                    "target_continuation_text": tokenizer.decode(output, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False),
                    "speculative_token_ids": output,
                    # Every emitted token is selected by target block logits.
                    # This is NOT an independent scalar equivalence check.
                    "exact_sd_target": True, "completed": True,
                    "reference_source": "block_target_verified_no_scalar_reference",
                    "scalar_parity_validated": False, "independent_target_reference": False,
                    "pool_index": record.get("pool_index"), "pool_split": record.get("pool_split"),
                    "source_index": record.get("source_index"),
                }
                events = [_event_row(event, pair_id=pair_id, draft_model=pair_spec["draft"],
                                     target_model=pair_spec["target"], tokenizer=tokenizer,
                                     record=record, prompt_token_count=len(prompt),
                                     token_text_cache=token_text_cache) for event in raw]
                writer.write({"reference": reference, "events": events, "run_identity": run_identity})
                completed[str(record["doc_id"])] = (reference, events)
            # A completed microbatch is the restart unit. Never retain whole
            # batches behind the slower legacy 64-prompt flush interval.
            writer.flush()
            bar.update(len(chunk))
            if (progress_log_every and len(completed) >= next_log) or len(completed) == len(shard_records):
                elapsed = time.perf_counter() - started
                rate = (len(completed) - resumed_count) / max(elapsed, 1e-9)
                eta = (len(shard_records) - len(completed)) / max(rate, 1e-9)
                bar.write(f"{prefix} {len(completed)}/{len(shard_records)} "
                          f"prompts/s={rate:.3f} elapsed={elapsed:.1f}s eta={eta:.1f}s")
                next_log = len(completed) + max(1, progress_log_every)
    references = [completed[str(row["doc_id"])][0] for row in shard_records]
    events = [event for row in shard_records for event in completed[str(row["doc_id"])][1]]
    _write_parquet_atomic(references, shard_dir / "references.parquet")
    _write_parquet_atomic(events, shard_dir / "sd_events.parquet")
    metadata = {
        **run_identity, "prompt_count": len(references),
        "all_outputs_target_verified": True,
        "all_target_sd_outputs_equal": None,
        "independent_target_reference": False, "scalar_parity_validated": False,
        "scalar_fallback_prompt_count": 0,
        "reference_sources": ["block_target_verified_no_scalar_reference"],
        "bonus_token_enabled": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(shard_dir / "run_metadata.json", metadata)
    (shard_dir / "COMPLETE").write_text(
        "All prompts completed target-block-verified greedy SD; scalar parity not asserted.\n",
        encoding="utf-8",
    )
    return {**metadata, "status": "complete", "path": str(shard_dir)}


def run_sd_shard(
    *,
    root: str | Path,
    pair_id: str,
    pair_spec: dict[str, Any],
    model_meta: dict[str, Any],
    config: dict[str, Any],
    shard_index: int = 0,
    num_shards: int = 1,
    device: str = "cuda",
    token: str | None = None,
    progress_flush_every: int = 64,
    progress_log_every: int = 100,
) -> dict[str, Any]:
    """Run one deterministic, resumable prompt shard."""
    from transformers import AutoTokenizer

    if shard_index < 0 or num_shards < 1 or shard_index >= num_shards:
        raise ValueError("invalid shard index/count")
    if progress_flush_every < 1:
        raise ValueError("progress_flush_every must be at least 1")
    if progress_log_every < 0:
        raise ValueError("progress_log_every must be non-negative")
    root_path = Path(root)
    records = load_pair_records(root_path / config["paths"]["prompts"], pair_spec)
    shard_records = records[shard_index::num_shards]
    shard_dir = root_path / config["paths"]["runs"] / pair_id / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_identity = build_run_identity(
        pair_id=pair_id,
        pair_spec=pair_spec,
        model_meta=model_meta,
        config=config,
        records=records,
        shard_records=shard_records,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    dataset_metadata_path = root_path / config["paths"].get("dataset_metadata", "")
    if dataset_metadata_path.exists():
        dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        run_identity["dataset_revision"] = dataset_metadata.get("resolved_revision")
    complete_marker = shard_dir / "COMPLETE"
    if complete_marker.exists():
        metadata_path = shard_dir / "run_metadata.json"
        previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        previous_identity = {key: previous.get(key) for key in run_identity}
        if previous_identity != run_identity:
            raise RuntimeError(
                f"Completed shard {shard_dir} belongs to a different model/configuration. "
                "Choose a fresh --root or remove that pair's old run artifact before rerunning."
            )
        return {"pair_id": pair_id, "shard_index": shard_index, "status": "already_complete", "path": str(shard_dir)}

    progress_path = shard_dir / "progress.jsonl"
    completed: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    if progress_path.exists():
        if run_identity["decoder"] == "microbatched":
            _recover_microbatch_progress_tail(progress_path)
        expected_records = {str(row["doc_id"]): row for row in shard_records}
        with progress_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                reference = item.get("reference", {})
                if run_identity["decoder"] == "microbatched" and item.get("run_identity") != run_identity:
                    raise RuntimeError("Cannot resume microbatch SD from a different configuration "
                                       "or legacy checkpoint; use a separate runs path")
                doc_id = str(reference.get("doc_id"))
                expected_record = expected_records.get(doc_id)
                if expected_record is None:
                    raise RuntimeError(f"progress contains a prompt outside this shard: {doc_id}")
                if reference.get("raw_text_hash") != expected_record.get("raw_text_hash"):
                    raise RuntimeError(f"progress raw_text_hash mismatch for doc_id={doc_id}")
                if not reference.get("completed") or not reference.get("exact_sd_target"):
                    raise RuntimeError(f"progress contains a parity-failed prompt: {doc_id}")
                completed[doc_id] = (reference, item.get("events", []))

    draft_name, draft_revision = _model_kwargs(model_meta, "draft")
    target_name, target_revision = _model_kwargs(model_meta, "target")
    tokenizer_options = {"backend": pair_spec.get("tokenizer_backend", "auto"),
                         "fix_mistral_regex": pair_spec.get("fix_mistral_regex", False)}
    draft_tokenizer = load_table1_tokenizer(draft_name, revision=draft_revision, token=token, **tokenizer_options)
    target_tokenizer = load_table1_tokenizer(target_name, revision=target_revision, token=token, **tokenizer_options)
    tokenizer_report = assert_tokenizer_compatible(draft_tokenizer, target_tokenizer)
    _write_json(shard_dir / "tokenizer_compatibility.json", tokenizer_report)
    draft_model, target_model = load_models(
        draft_name,
        target_name,
        device=device,
        draft_revision=draft_revision,
        target_revision=target_revision,
        token=token,
        dtype=_pair_inference_setting(pair_spec, config, "dtype", "float16"),
        attention_backend=_pair_inference_setting(pair_spec, config, "attention_backend", "sdpa"),
    )

    pending_records = [
        record for record in shard_records
        if str(record["doc_id"]) not in completed
    ]
    if run_identity["decoder"] == "microbatched":
        return _run_microbatch_shard(
            pair_id=pair_id, pair_spec=pair_spec, config=config,
            draft_model=draft_model, target_model=target_model, tokenizer=target_tokenizer,
            shard_records=shard_records, completed=completed, progress_path=progress_path,
            shard_dir=shard_dir, run_identity=run_identity,
            progress_flush_every=progress_flush_every, progress_log_every=progress_log_every,
        )
    reference_ids_by_doc = generate_reference_ids(
        pending_records,
        tokenizer=target_tokenizer,
        target_model=target_model,
        pair_spec=pair_spec,
        config=config,
    )
    started = time.time()
    with tqdm(total=len(shard_records), initial=len(completed),
              desc=f"{pair_id} SD", unit="prompt", dynamic_ncols=True,
              mininterval=1) as bar, progress_path.open("a", encoding="utf-8") as progress:
        progress_writer = ProgressWriter(progress, flush_every=progress_flush_every)
        for index, record in enumerate(shard_records):
            doc_id = str(record["doc_id"])
            if doc_id in completed:
                continue
            reference, events = _prompt_run(
                record,
                pair_id=pair_id,
                pair_spec=pair_spec,
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=target_tokenizer,
                config=config,
                reference_ids=reference_ids_by_doc[doc_id],
                strict_reference_check=False,
            )
            if not reference["exact_sd_target"]:
                verification_mode = _pair_inference_setting(
                    pair_spec, config, "target_verification", "sequential"
                ).lower()
                if verification_mode in {"batched", "batched_fallback", "block"}:
                    # Any batch-reference disagreement is a local slow-path:
                    # regenerate the scalar target IDs and rerun the decoder
                    # with singleton target verification so event rows are
                    # strict as well as the final continuation.
                    scalar_ids = greedy_generate(
                        target_model,
                        [int(token_id) for token_id in reference["prompt_token_ids"]],
                        int(config["max_new_tokens"]),
                        getattr(target_tokenizer, "eos_token_id", None),
                    )
                    batch_ids = list(reference["target_continuation_token_ids"])
                    reference, events = _prompt_run(
                        record,
                        pair_id=pair_id,
                        pair_spec=pair_spec,
                        draft_model=draft_model,
                        target_model=target_model,
                        tokenizer=target_tokenizer,
                        config=config,
                        reference_ids=scalar_ids,
                        strict_reference_check=True,
                        batch_target_verification=False,
                    )
                    reference["batch_reference_token_ids"] = batch_ids
                    reference["batch_reference_first_difference"] = next(
                        (
                            offset
                            for offset, (left, right) in enumerate(zip(batch_ids, scalar_ids))
                            if left != right
                        ),
                        min(len(batch_ids), len(scalar_ids))
                        if len(batch_ids) != len(scalar_ids)
                        else None,
                    )
                    reference["reference_source"] = (
                        "scalar_decoder_fallback_after_batched_reference_mismatch"
                    )
                    reference["scalar_parity_validated"] = True
                else:
                    reference = _repair_reference_mismatch(
                        record,
                        reference,
                        target_model=target_model,
                        tokenizer=target_tokenizer,
                        config=config,
                    )
                bar.write(
                    f"[{pair_id} shard {shard_index}/{num_shards}] "
                    f"scalar parity fallback doc_id={doc_id} "
                    f"batch_first_difference={reference.get('batch_reference_first_difference')}",
                )
            item = {"reference": reference, "events": events}
            progress_writer.write(item)
            completed[doc_id] = (reference, events)
            bar.update(1)
            processed = index + 1
            if progress_log_every and (
                processed % progress_log_every == 0 or processed == len(shard_records)
            ):
                bar.write(
                    f"[{pair_id} shard {shard_index}/{num_shards}] "
                    f"{processed}/{len(shard_records)} doc_id={doc_id} "
                    f"generated={len(reference['target_continuation_token_ids'])} "
                    f"checkpoint_pending={progress_writer.pending}",
                )
        progress_writer.flush()

    reference_rows = [completed[str(record["doc_id"])][0] for record in shard_records]
    event_rows = [event for record in shard_records for event in completed[str(record["doc_id"])][1]]
    if any(not row["exact_sd_target"] for row in reference_rows):
        raise AssertionError("completed shard contains an SD/target mismatch")
    _write_parquet_atomic(reference_rows, shard_dir / "references.parquet")
    _write_parquet_atomic(event_rows, shard_dir / "sd_events.parquet")
    metadata = {
        **run_identity,
        "pair_id": pair_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_count": len(reference_rows),
        "prompt_ids_sha256": prompt_ids_sha256(reference_rows),
        "all_target_sd_outputs_equal": True,
        "scalar_fallback_prompt_count": sum(
            str(row.get("reference_source", "")).startswith("scalar_")
            for row in reference_rows
        ),
        "reference_sources": sorted({str(row.get("reference_source", "unknown")) for row in reference_rows}),
        "target_verification": _pair_inference_setting(
            pair_spec, config, "target_verification", "sequential"
        ),
        "reference_batch_size": int(
            pair_spec.get("reference_batch_size", config.get("reference_batch_size", 1))
        ),
        "dtype": _pair_inference_setting(pair_spec, config, "dtype", "float16"),
        "attention_backend": _pair_inference_setting(pair_spec, config, "attention_backend", "sdpa"),
        "draft_model": draft_name,
        "draft_revision": draft_revision,
        "target_model": target_name,
        "target_revision": target_revision,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(shard_dir / "run_metadata.json", metadata)
    complete_marker.write_text("All prompts passed exact token-ID parity.\n", encoding="utf-8")
    return {**metadata, "status": "complete", "path": str(shard_dir)}
