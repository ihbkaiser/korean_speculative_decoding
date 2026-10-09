from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.table1_runner as runner
from scripts.table1_pipeline import load_config
from tests.test_batched_speculative import PrefixModel
from tests.test_pipeline import CharTokenizer


@pytest.fixture
def fast_run(monkeypatch, tmp_path):
    records = [{"doc_id": f"p{i}", "text": chr(i + 2) + chr(1), "raw_text_hash": f"h{i}"}
               for i in range(5)]
    config = {
        "max_prompt_tokens": 128, "max_new_tokens": 9, "speculative_k": 4,
        "decoder": "microbatched", "sd_batch_size": 2,
        "target_verification": "batched_native",
        "paths": {"prompts": "prompts.parquet", "runs": "runs/fast",
                  "dataset_metadata": "metadata/dataset_revision.json"},
    }
    pair = {"draft": "draft", "target": "target", "prompt_split": "common_20k", "prompt_count": 5}
    meta = {"draft": {"id": "draft"}, "target": {"id": "target"}}
    tokenizer = CharTokenizer()
    tokenizer.eos_token_id = None
    target, draft = PrefixModel(eos=None), PrefixModel(1, eos=None)
    monkeypatch.setattr(runner, "load_pair_records", lambda *_: records)
    monkeypatch.setattr(runner, "assert_tokenizer_compatible", lambda *_: {})
    monkeypatch.setattr(runner, "load_models", lambda *args, **kwargs: (draft, target))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(runner, "generate_reference_ids", lambda *args, **kwargs: pytest.fail("pre-reference pass called"))
    monkeypatch.setattr(runner, "greedy_generate", lambda *args, **kwargs: pytest.fail("scalar fallback called"))
    kwargs = dict(root=tmp_path, pair_id="Q1", pair_spec=pair, model_meta=meta, config=config,
                  device="cpu", progress_log_every=1)
    return kwargs, records, target


def test_fast_shard_writes_target_verified_artifacts_without_scalar_calls(fast_run):
    kwargs, records, target = fast_run
    result = runner.run_sd_shard(**kwargs)
    shard = Path(result["path"])
    assert result["all_outputs_target_verified"] is True
    assert result["all_target_sd_outputs_equal"] is None
    assert result["scalar_fallback_prompt_count"] == 0
    refs = pd.read_parquet(shard / "references.parquet")
    events = pd.read_parquet(shard / "sd_events.parquet")
    assert refs.doc_id.tolist() == [row["doc_id"] for row in records]
    assert not refs.scalar_parity_validated.any()
    assert events["invalidated_by_earlier_rejection"].any()
    assert {"doc_id", "pair_id", "draft_model", "target_model", "prompt_token_count",
            "generated_token_position", "proposal_slot", "block_id", "token_id", "token_text",
            "accepted", "rejected", "invalidated_by_earlier_rejection", "draft_entropy",
            "target_entropy", "draft_logprob_of_proposed_token", "target_logprob_of_proposed_token",
            "target_top1_token_id", "target_top1_logprob", "target_rank_of_proposed_token",
            "target_margin", "later_proposals_invalidated"}.issubset(events.columns)
    assert {"prompt_token_ids", "prompt_text", "target_continuation_token_ids",
            "target_continuation_text", "pool_index", "pool_split",
            "reference_source", "scalar_parity_validated"}.issubset(refs.columns)
    calls = target.calls
    assert runner.run_sd_shard(**kwargs)["status"] == "already_complete"
    assert target.calls == calls


@pytest.mark.parametrize("verification", ["batched", "sequential"])
def test_strict_mismatch_fails_without_scalar_repair_or_decoder_retry(fast_run, monkeypatch, verification):
    kwargs, records, _ = fast_run
    kwargs["config"] = {**kwargs["config"], "decoder": "cached", "target_verification": verification}
    monkeypatch.setattr(runner, "generate_reference_ids", lambda *args, **options: {
        row["doc_id"]: [99] for row in records
    })
    prompt_runs = []
    original = runner._prompt_run

    def track(*args, **options):
        prompt_runs.append(args[0]["doc_id"])
        return original(*args, **options)

    monkeypatch.setattr(runner, "_prompt_run", track)
    with pytest.raises(RuntimeError, match="scalar fallback is disabled"):
        runner.run_sd_shard(**kwargs)
    assert prompt_runs == ["p0"]
    shard = kwargs["root"] / "runs/fast/Q1/shards/shard-00000-of-00001"
    assert not (shard / "COMPLETE").exists()
    assert not (shard / "progress.jsonl").read_text().strip()


@pytest.mark.parametrize("broken_pair", ["Q1", "Q2", "Q3", "M1", "G1"])
def test_fast_launch_checks_all_pair_decoders_before_loading_models(monkeypatch, tmp_path, broken_pair):
    from scripts import table1_pipeline

    config = load_config(Path(__file__).resolve().parents[1] / "configs/table1_fast_b200.yaml")
    config["pairs"][broken_pair]["decoder"] = "cached"
    monkeypatch.setattr(table1_pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(table1_pipeline, "command_prepare_data", lambda *_: pytest.fail("preflight ran too late"))
    monkeypatch.setattr(table1_pipeline, "run_sd_shard", lambda **_: pytest.fail("models must not load"))
    args = table1_pipeline.build_parser().parse_args([
        "--root", str(tmp_path), "run-table1-main", "--pair", "Q1", "--require-microbatched",
    ])
    with pytest.raises(ValueError, match=f"{broken_pair}.*microbatched"):
        args.func(args)


def test_fast_decoder_error_propagates_without_retry(fast_run, monkeypatch):
    kwargs, _, _ = fast_run
    batches = []

    def oom(*args, **options):
        batches.append(options["prompt_ids"])
        raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(runner, "speculative_greedy_microbatch", oom)
    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        runner.run_sd_shard(**kwargs)
    assert batches == [["p0", "p1"]]


def test_smoke_mismatch_fails_without_scalar_repair(fast_run, monkeypatch):
    kwargs, records, _ = fast_run
    config = {**kwargs["config"], "decoder": "cached", "target_verification": "sequential"}
    monkeypatch.setattr(runner, "generate_reference_ids", lambda *args, **options: {
        row["doc_id"]: [99] for row in records
    })
    with pytest.raises(RuntimeError, match="scalar fallback is disabled"):
        runner.run_smoke_test("Q1", kwargs["pair_spec"], kwargs["model_meta"], records,
                              device="cpu", config=config)


def test_fast_shard_resumes_completed_microbatch_after_interruption(fast_run, monkeypatch):
    kwargs, records, target = fast_run
    original = runner.speculative_greedy_microbatch
    calls = []

    def interrupt(*args, **options):
        calls.append(options["prompt_ids"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(*args, **options)

    monkeypatch.setattr(runner, "speculative_greedy_microbatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_sd_shard(**kwargs)
    shard = kwargs["root"] / "runs/fast/Q1/shards/shard-00000-of-00001"
    saved = (shard / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(saved) == 2
    assert not (shard / "COMPLETE").exists()
    resumed = []

    def track(*args, **options):
        resumed.extend(options["prompt_ids"])
        return original(*args, **options)

    monkeypatch.setattr(runner, "speculative_greedy_microbatch", track)
    assert runner.run_sd_shard(**kwargs)["status"] == "complete"
    assert resumed == ["p2", "p3", "p4"]


def test_main_shard_dumps_tokenizer_fingerprints_without_a_separate_audit(fast_run, monkeypatch):
    import json
    from src.models import assert_tokenizer_compatible

    kwargs, _, _ = fast_run
    monkeypatch.setattr(runner, "assert_tokenizer_compatible", assert_tokenizer_compatible)
    shard = Path(runner.run_sd_shard(**kwargs)["path"])
    report_path = shard / "tokenizer_compatibility.json"
    assert report_path.exists(), "production tokenizer audit was not persisted"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["compatible"] and report["offsets_supported"]
    assert len(report["backend_sha256"]["target"]) == 64
    assert report["backend_sha256"]["draft"] == report["backend_sha256"]["target"]


def test_progress_counts_checkpointed_prompts_and_resumes(fast_run, monkeypatch):
    kwargs, _, _ = fast_run
    bars = []

    class ProgressSpy:
        def __init__(self, **options):
            self.options = options
            self.updates = []
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count):
            self.updates.append(count)

        def write(self, message):
            pass

    monkeypatch.setattr(runner, "tqdm", ProgressSpy)
    original = runner.speculative_greedy_microbatch
    calls = 0

    def interrupt(*args, **options):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **options)

    monkeypatch.setattr(runner, "speculative_greedy_microbatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_sd_shard(**kwargs)
    assert bars[0].options["total"] == 5
    assert bars[0].options["initial"] == 0
    assert bars[0].updates == [2]
    monkeypatch.setattr(runner, "speculative_greedy_microbatch", original)
    runner.run_sd_shard(**kwargs)
    assert bars[1].options["initial"] == 2
    assert bars[1].updates == [2, 1]


def test_partial_fast_checkpoint_refuses_numeric_change(fast_run, monkeypatch):
    kwargs, _, _ = fast_run
    original = runner.speculative_greedy_microbatch
    calls = 0

    def interrupt(*args, **options):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **options)

    monkeypatch.setattr(runner, "speculative_greedy_microbatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_sd_shard(**kwargs)
    kwargs["config"]["dtype"] = "bfloat16"
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.run_sd_shard(**kwargs)


@pytest.mark.parametrize("tail", [b'{"reference":', b'{"text":"\xe2\x82'])
def test_incomplete_final_checkpoint_record_is_preserved(tmp_path, tail):
    progress = tmp_path / "progress.jsonl"
    progress.write_bytes(b'{"completed":true}\n' + tail)
    runner._recover_microbatch_progress_tail(progress)
    assert progress.read_bytes() == b'{"completed":true}\n'
    backups = list(tmp_path.glob("*.incomplete-tail-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == tail


def test_corruption_before_final_record_is_not_ignored(tmp_path):
    progress = tmp_path / "progress.jsonl"
    original = b'{"broken":\n{"completed":true}\n'
    progress.write_bytes(original)
    with pytest.raises(RuntimeError, match="before EOF"):
        runner._recover_microbatch_progress_tail(progress)
    assert progress.read_bytes() == original


def test_fast_config_inherits_model_paths_and_isolates_output():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/table1_fast_b200.yaml")
    assert config["model_paths"]["Q1"]["target"].endswith("ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    assert config["paths"]["runs"] == "runs/table1_fast_b200"
    assert config["inference"]["sd_batch_size"] == 256
    assert config["pairs"]["Q1"]["sd_batch_size"] == 256
    assert "sd_batch_size" not in config["pairs"]["Q3"]
    assert config["pairs"]["M1"]["tokenizer_backend"] == "tokenizers"
    assert {pair["dtype"] for pair in config["pairs"].values()} == {"bfloat16"}
    assert {pair["decoder"] for pair in config["pairs"].values()} == {"microbatched"}


def test_cli_microbatch_override_is_recorded_in_identity(fast_run, monkeypatch):
    from scripts import table1_pipeline

    kwargs, records, _ = fast_run
    config = {"inference": kwargs["config"], "paths": kwargs["config"]["paths"],
              "pairs": {"Q1": kwargs["pair_spec"]},
              "model_paths": {"Q1": {"draft": str(kwargs["root"]), "target": str(kwargs["root"])}}}
    monkeypatch.setattr(table1_pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(table1_pipeline, "command_prepare_data", lambda _: None)
    captured = []
    monkeypatch.setattr(table1_pipeline, "run_sd_shard", lambda **options: captured.append(options) or {})
    args = table1_pipeline.build_parser().parse_args([
        "--root", str(kwargs["root"]), "run-table1-main", "--pair", "Q1",
        "--skip-align", "--sd-batch-size", "16", "--require-microbatched",
    ])
    table1_pipeline.command_run_table1_main(args)
    identity = runner.build_run_identity(
        pair_id="Q1", pair_spec=captured[0]["pair_spec"], model_meta=kwargs["model_meta"],
        config=kwargs["config"], records=records, shard_records=records, shard_index=0, num_shards=1,
    )
    assert identity["sd_batch_size"] == 16
