"""Artifact integrity checks for the selected no-scalar-fallback workflow."""

import json
import shutil
from copy import deepcopy
from pathlib import Path


def snapshot_sources(root, destination):
    """Freeze an upload allowlist, including the actual root shell launcher."""
    root, destination = Path(root), Path(destination)
    for name in ("src", "scripts", "configs", "tests"):
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pytest.ini", "run_table1.sh"):
        shutil.copy2(root / name, destination / name)


def validate_trial_request(stage, prompts, max_new_tokens, batch_sizes):
    """Allow larger decode-only trials without opening a full-run route."""
    if stage not in {"inspect", "benchmark", "pilot", "smoke-all", "smoke-finish", "launcher-smoke"}:
        raise ValueError("Unknown bounded trial stage; no full-run stage exists")
    limit = 2048 if stage == "benchmark" else 512
    if not 1 <= prompts <= limit or not 1 <= max_new_tokens <= 128:
        raise ValueError(f"{stage} is limited to {limit} prompts and 128 output tokens")
    if not batch_sizes or any(not 1 <= size <= prompts for size in batch_sizes):
        raise ValueError("Batch sizes must be positive and fit at least one full prompt batch")


def make_launcher_smoke_config(config, prompts, batch_size, max_new_tokens, model_paths):
    """Build an isolated bounded fixture; never alter the production config."""
    validate_trial_request("launcher-smoke", prompts, max_new_tokens, (batch_size,))
    bounded = deepcopy(config)
    bounded["model_paths"] = deepcopy(model_paths)
    bounded["inference"]["max_new_tokens"] = max_new_tokens
    for pair in bounded["pairs"].values():
        pair.update(prompt_count=prompts, sd_batch_size=batch_size)
    return bounded


def validate_fast_shard(shard_dir, *, expected_prompts, max_new_tokens):
    import numpy as np
    import pandas as pd

    shard = Path(shard_dir)
    assert (shard / "COMPLETE").exists(), "SD completion marker missing"
    metadata = json.loads((shard / "run_metadata.json").read_text(encoding="utf-8"))
    refs = pd.read_parquet(shard / "references.parquet")
    events = pd.read_parquet(shard / "sd_events.parquet")
    assert len(refs) == expected_prompts and refs.doc_id.nunique() == expected_prompts, "prompt count mismatch"
    assert metadata["scalar_fallback_prompt_count"] == 0, "unexpected scalar fallback"
    assert metadata["decoder"] == "microbatched", "unexpected decoder"
    assert refs.completed.all() and refs.exact_sd_target.all(), "incomplete target verification"
    assert (refs.reference_source == "block_target_verified_no_scalar_reference").all(), "reference policy mismatch"
    assert not refs.scalar_parity_validated.any() and not refs.independent_target_reference.any(), "reference policy mismatch"
    lengths = refs.target_continuation_token_ids.map(len)
    assert lengths.between(1, max_new_tokens).all(), "generation budget violated"
    assert len(events) > 0 and set(events.doc_id).issubset(set(refs.doc_id)), "proposal events missing or outside pool"
    numeric = ["draft_entropy", "target_entropy", "draft_logprob_of_proposed_token",
               "draft_top1_logprob", "target_logprob_of_proposed_token", "target_top1_logprob",
               "target_margin", "target_rank_of_proposed_token"]
    assert np.isfinite(events[numeric].to_numpy(dtype=float)).all(), "non-finite audit statistics"
    accepted = events.accepted.fillna(False).astype(bool)
    assert (events.loc[accepted, "token_id"] == events.loc[accepted, "target_top1_token_id"]).all(), "accepted token differs from target"
    assert not (accepted & events.invalidated_by_earlier_rejection).any(), "invalidated proposal accepted"
    return {"prompt_count": len(refs), "event_count": len(events),
            "generated_tokens": int(lengths.sum()), "scalar_fallback_prompt_count": 0,
            "scalar_parity_validated": False, "artifact_integrity_passed": True}
