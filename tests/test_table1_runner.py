from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.table1_pipeline as table1_pipeline
from scripts.table1_pipeline import _write_environment_manifest, build_parser, command_run_table1

from src.speculative_decoding import speculative_greedy, speculative_greedy_cached
from src.table1_runner import (
    _event_row,
    build_run_identity,
    load_pair_records,
    prompt_ids_sha256,
    ProgressWriter,
    resolve_model_reference,
    select_speculative_decoder,
    smoke_gate_mode,
)


def _pool(tmp_path):
    rows = [
        {"doc_id": f"p{i}", "raw_text_hash": f"h{i}", "text": f"한국어 문서 {i}", "pool_index": i}
        for i in range(4)
    ]
    path = tmp_path / "prompts.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path, rows


def test_pair_prompt_selection_is_frozen_and_hashable(tmp_path):
    path, rows = _pool(tmp_path)
    spec = {"prompt_split": "common_20k", "prompt_count": 4}
    selected = load_pair_records(path, spec)
    assert [row["doc_id"] for row in selected] == ["p0", "p1", "p2", "p3"]
    assert prompt_ids_sha256(selected) == prompt_ids_sha256(rows)


def test_pair_prompt_selection_rejects_wrong_expected_count(tmp_path):
    path, _ = _pool(tmp_path)
    try:
        load_pair_records(path, {"prompt_split": "common_20k", "prompt_count": 20_000})
    except AssertionError as exc:
        assert "Expected 20000" in str(exc)
    else:
        raise AssertionError("expected prompt-count mismatch")


def test_smoke_gate_distinguishes_full_gate_from_quick_smoke():
    assert smoke_gate_mode(200, 128, full_prompts=200, full_max_new_tokens=128) == "FULL_200"
    assert smoke_gate_mode(5, 32, full_prompts=200, full_max_new_tokens=128) == "QUICK_ONLY"


def test_pair_decoder_override_wins_over_legacy_global_default():
    assert select_speculative_decoder({}, {"decoder": "legacy"}) is speculative_greedy
    assert select_speculative_decoder({"decoder": "cached"}, {"decoder": "legacy"}) is speculative_greedy_cached


def test_company_table1_config_defaults_all_pairs_to_cached():
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "table1_pipeline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["inference"]["decoder"] == "cached"
    assert {
        pair_spec.get("decoder", config["inference"]["decoder"])
        for pair_spec in config["pairs"].values()
    } == {"cached"}


def test_custom_model_path_override_disables_hub_revision(tmp_path):
    draft_path = tmp_path / "draft-model"
    draft_path.mkdir()
    metadata = {
        "draft": {"id": "old/draft", "resolved_revision": "old-sha"},
        "target": {"id": "old/target", "resolved_revision": "target-sha"},
    }

    model_name, revision = resolve_model_reference(
        metadata, "draft", override=str(draft_path)
    )

    assert model_name == str(draft_path)
    assert revision is None


def test_model_reference_uses_audited_local_path_without_revision(tmp_path):
    target_path = tmp_path / "target-model"
    target_path.mkdir()
    metadata = {
        "target": {
            "id": "old/target",
            "local_path": str(target_path),
            "resolved_revision": "target-sha",
        }
    }

    assert resolve_model_reference(metadata, "target") == (str(target_path), None)


def test_table1_cli_accepts_pair_specific_model_overrides():
    args = build_parser().parse_args([
        "audit-models",
        "--pair", "Q3",
        "--draft-model-path", "/models/q3-draft",
        "--target-model-path", "/models/q3-target",
    ])

    assert args.pair == "Q3"
    assert args.draft_model_path == "/models/q3-draft"
    assert args.target_model_path == "/models/q3-target"


def test_table1_cli_accepts_single_pair_end_to_end_command():
    args = build_parser().parse_args([
        "run-table1",
        "--pair", "M1",
        "--num-shards", "2",
        "--draft-model-path", "/models/m1-draft",
        "--target-model-path", "/models/m1-target",
    ])

    assert args.pair == "M1"
    assert args.num_shards == 2
    assert args.align_device == "cpu"


def test_table1_cli_accepts_buffered_progress_and_deferred_alignment_flags():
    args = build_parser().parse_args([
        "run-table1-main",
        "--pair", "Q1",
        "--skip-align",
        "--progress-flush-every", "7",
        "--progress-log-every", "11",
    ])

    assert args.skip_align is True
    assert args.progress_flush_every == 7
    assert args.progress_log_every == 11


def test_progress_writer_flushes_in_batches():
    class CountingBuffer(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_calls = 0

        def flush(self):
            self.flush_calls += 1
            super().flush()

    buffer = CountingBuffer()
    writer = ProgressWriter(buffer, flush_every=2)
    writer.write({"doc_id": "p1"})
    assert buffer.flush_calls == 0
    writer.write({"doc_id": "p2"})
    assert buffer.flush_calls == 1
    writer.write({"doc_id": "p3"})
    writer.flush()
    assert buffer.flush_calls == 2
    assert buffer.getvalue().count("\n") == 3


def test_company_launcher_overlaps_alignment_after_each_gpu_pair():
    launcher = (Path(__file__).resolve().parents[1] / "scripts" / "run_company_table1.sh").read_text(
        encoding="utf-8"
    )
    assert "--skip-align" in launcher
    assert 'run_pair_align "$pair" &' in launcher
    assert 'wait "$pid"' in launcher


def test_run_table1_rejects_empty_shard_topology(tmp_path):
    args = build_parser().parse_args([
        "--root", str(tmp_path),
        "run-table1",
        "--pair", "Q1",
        "--num-shards", "0",
    ])

    with pytest.raises(ValueError, match="at least 1"):
        command_run_table1(args)


@pytest.mark.parametrize("custom_output", [False, True])
def test_main_table1_command_skips_audit_and_smoke(monkeypatch, tmp_path, custom_output):
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    config = {
        "paths": {
            "prompts": "data/prompts.parquet",
            "dataset_metadata": "metadata/dataset_revision.json",
            "model_metadata": "metadata/model_revisions.json",
            "runs": "runs/table1",
        },
        "pairs": {
            "Q1": {
                "draft": "configured/draft",
                "target": "configured/target",
                "prompt_split": "common_20k",
                "prompt_count": 1,
            }
        },
        "model_paths": {
            "Q1": {"draft": str(draft), "target": str(target)},
        },
        "inference": {"max_new_tokens": 128},
    }
    monkeypatch.setattr(table1_pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(table1_pipeline, "command_prepare_data", lambda _: 0)
    monkeypatch.setattr(
        table1_pipeline,
        "command_audit_models",
        lambda _: (_ for _ in ()).throw(AssertionError("audit must not run")),
    )
    run_calls = []
    align_calls = []
    monkeypatch.setattr(table1_pipeline, "run_sd_shard", lambda **kwargs: run_calls.append(kwargs) or {"status": "complete"})
    monkeypatch.setattr(table1_pipeline, "command_align", lambda args: align_calls.append(args) or 0)

    output = tmp_path / "external output" if custom_output else tmp_path
    args = build_parser().parse_args([
        "--root", str(tmp_path),
        "run-table1-main",
        "--pair", "Q1",
    ] + (["--output-dir", str(output)] if custom_output else []))

    assert table1_pipeline.command_run_table1_main(args) == 0
    assert len(run_calls) == 1
    assert run_calls[0]["model_meta"]["draft"]["local_path"] == str(draft.resolve())
    assert align_calls[0].target_model_path == str(target)
    manifest = json.loads((output / "metadata/table1_main_Q1.json").read_text())
    expected_config = {**config, "paths": {**config["paths"],
                       "runs": str(output / "runs"), "results": str(output / "results")}} if custom_output else config
    assert manifest["resolved_config"] == expected_config
    assert run_calls[0]["config"]["paths"] == expected_config["paths"]
    assert manifest["effective_pair_spec"]["target"] == str(target)
    assert manifest["environment"]["packages"]
    assert "torch_cuda_runtime" in manifest["environment"]
    launches = list((output / ("runs" if custom_output else "runs/table1") / "Q1/launches").glob("*.json"))
    assert len(launches) == 1
    assert json.loads(launches[0].read_text()) == manifest
    if custom_output:
        assert align_calls[0].output_dir == str(output)
        assert (output / "metadata/environment.json").exists()
        assert not (tmp_path / "metadata").exists()
        assert config["paths"]["runs"] == "runs/table1", "do not mutate the input config"


@pytest.mark.parametrize("stage", ["align-morphology", "build-table1"])
def test_postprocessing_uses_custom_output_paths(monkeypatch, tmp_path, stage):
    from src import table1_morphology

    config = {"paths": {"runs": "runs/old", "results": "results/old", "prompts": "data/frozen.parquet"}}
    monkeypatch.setattr(table1_pipeline, "load_config", lambda _: config)
    captured = []
    operation = "align_pair_shard" if stage == "align-morphology" else "build_table1_outputs"
    monkeypatch.setattr(table1_morphology, operation, lambda **kwargs: captured.append(kwargs) or {})
    output = tmp_path / "external output"
    args = build_parser().parse_args([
        "--root", str(tmp_path), stage, "--output-dir", str(output),
    ] + (["--pair", "Q1"] if stage == "align-morphology" else []))
    assert args.func(args) == 0
    assert captured[0]["root"] == tmp_path
    assert captured[0]["config"]["paths"] == {
        "runs": str(output / "runs"), "results": str(output / "results"), "prompts": "data/frozen.parquet",
    }


def test_main_table1_can_defer_alignment(monkeypatch, tmp_path):
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    config = {
        "paths": {
            "prompts": "data/prompts.parquet",
            "dataset_metadata": "metadata/dataset_revision.json",
            "model_metadata": "metadata/model_revisions.json",
            "runs": "runs/table1",
        },
        "pairs": {
            "Q1": {
                "draft": "configured/draft",
                "target": "configured/target",
                "prompt_split": "common_20k",
                "prompt_count": 1,
            }
        },
        "model_paths": {"Q1": {"draft": str(draft), "target": str(target)}},
        "inference": {"max_new_tokens": 128},
    }
    monkeypatch.setattr(table1_pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(table1_pipeline, "command_prepare_data", lambda _: 0)
    run_calls = []
    align_calls = []
    monkeypatch.setattr(
        table1_pipeline,
        "run_sd_shard",
        lambda **kwargs: run_calls.append(kwargs) or {"status": "complete"},
    )
    monkeypatch.setattr(table1_pipeline, "command_align", lambda args: align_calls.append(args) or 0)

    args = build_parser().parse_args([
        "--root", str(tmp_path),
        "run-table1-main",
        "--pair", "Q1",
        "--skip-align",
        "--progress-flush-every", "7",
        "--progress-log-every", "11",
    ])

    assert table1_pipeline.command_run_table1_main(args) == 0
    assert len(run_calls) == 1
    assert run_calls[0]["progress_flush_every"] == 7
    assert run_calls[0]["progress_log_every"] == 11
    assert align_calls == []


def test_event_artifact_contains_margin_and_top1_fields():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "í•™"

    event = {
        "draft_token_id": 7,
        "output_token_position": 3,
        "proposal_position": 1,
        "round_index": 2,
        "accepted": False,
        "rejected": True,
        "invalidated_after_first_rejection": False,
        "draft_entropy": 0.1,
        "target_entropy": 0.2,
        "draft_logprob": -1.2,
        "target_logprob": -2.0,
        "draft_top1_logprob": -1.0,
        "target_top1_logprob": -0.5,
        "target_greedy_token_id": 8,
        "target_rank_of_proposed_token": 3,
        "is_first_rejection": True,
        "accepted_prefix_length": 0,
        "first_rejection_output_position": 3,
    }
    record = {"doc_id": "p1", "raw_text_hash": "h1", "pool_index": 4, "pool_split": "common_20k", "text": "í•™"}

    row = _event_row(
        event,
        pair_id="Q1",
        draft_model="draft",
        target_model="target",
        tokenizer=Tokenizer(),
        record=record,
        prompt_token_count=2,
    )

    assert row["draft_top1_logprob"] == -1.0
    assert row["target_margin"] == pytest.approx(1.5)
    assert row["pool_index"] == 4


def test_run_identity_covers_prompt_and_decoding_provenance():
    records = [
        {"doc_id": "p1"},
        {"doc_id": "p2"},
    ]
    pair_spec = {
        "draft": "draft",
        "target": "target",
        "prompt_split": "common_20k",
        "prompt_count": 2,
        "decoder": "cached",
        "dtype": "float32",
        "attention_backend": "eager",
        "reference_batch_size": 64,
    }
    model_meta = {
        "draft": {"id": "draft", "resolved_revision": "d-sha"},
        "target": {"id": "target", "resolved_revision": "t-sha"},
    }
    config = {
        "max_prompt_tokens": 128,
        "max_new_tokens": 128,
        "speculative_k": 4,
        "paths": {"dataset_metadata": "metadata/dataset_revision.json"},
    }

    identity = build_run_identity(
        pair_id="Q1",
        pair_spec=pair_spec,
        model_meta=model_meta,
        config=config,
        records=records,
        shard_records=records[:1],
        shard_index=0,
        num_shards=2,
    )

    assert identity["prompt_count"] == 2
    assert identity["shard_prompt_count"] == 1
    assert identity["speculative_k"] == 4
    assert identity["decoder"] == "cached"
    assert identity["draft_revision"] == "d-sha"
    assert identity["prompt_ids_sha256"] != identity["shard_prompt_ids_sha256"]


def test_environment_manifest_is_written_for_reproducibility(tmp_path):
    path = _write_environment_manifest(tmp_path)

    assert path == tmp_path / "metadata/environment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["python_version"]
    assert "packages" in payload
