from pathlib import Path

import pandas as pd
import pytest

import src.table1_runner as runner
from tests.test_table1_fast_runner import fast_run


def test_benchmark_allows_one_actual_2048_prompt_batch():
    import src.table1_smoke as smoke

    assert hasattr(smoke, "validate_trial_request"), "bounded large-batch validator missing"
    smoke.validate_trial_request("benchmark", 2048, 128, (256, 2048))


def test_launcher_smoke_config_is_bounded_and_keeps_fast_contract():
    from scripts.table1_pipeline import load_config
    import src.table1_smoke as smoke

    original = load_config(Path(__file__).resolve().parents[1] / "configs/table1_fast_b200.yaml")
    paths = {pair: {side: f"/cache/{pair}/{side}" for side in ("draft", "target")}
             for pair in original["pairs"]}
    assert hasattr(smoke, "make_launcher_smoke_config"), "launcher smoke configuration helper missing"
    bounded = smoke.make_launcher_smoke_config(original, 256, 256, 128, paths)
    assert bounded["model_paths"] == paths
    assert {row["prompt_count"] for row in bounded["pairs"].values()} == {256}
    assert {row["decoder"] for row in bounded["pairs"].values()} == {"microbatched"}
    assert {row["sd_batch_size"] for row in bounded["pairs"].values()} == {256}
    assert original["pairs"]["Q2"]["prompt_count"] == 40000
    with pytest.raises(ValueError):
        smoke.make_launcher_smoke_config(original, 20000, 256, 128, paths)


def test_modal_source_snapshot_contains_root_launcher(tmp_path):
    import src.table1_smoke as smoke

    assert hasattr(smoke, "snapshot_sources"), "Modal source snapshot helper missing"
    smoke.snapshot_sources(Path(__file__).resolve().parents[1], tmp_path)
    assert (tmp_path / "run_table1.sh").is_file(), "Modal upload omitted root launcher"
    assert (tmp_path / "scripts/run_company_table1_fast.sh").is_file()
    assert not (tmp_path / "hf_token").exists()


@pytest.mark.parametrize("stage,prompts,tokens,sizes", [
    ("benchmark", 2049, 128, (2048,)),
    ("benchmark", 512, 128, (2048,)),
    ("benchmark", 2048, 129, (2048,)),
    ("benchmark", 2048, 128, (0,)),
    ("pilot", 2048, 128, (2048,)),
    ("smoke-all", 2048, 128, (2048,)),
    ("full", 20000, 128, (2048,)),
])
def test_trial_guard_rejects_full_runs_and_false_batch_labels(stage, prompts, tokens, sizes):
    import src.table1_smoke as smoke

    assert hasattr(smoke, "validate_trial_request"), "bounded large-batch validator missing"
    with pytest.raises(ValueError):
        smoke.validate_trial_request(stage, prompts, tokens, sizes)


def test_explicit_tokenizers_backend_keeps_offsets_despite_auto_class(tmp_path):
    import json
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from src.models import load_table1_tokenizer

    tiny = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        {"<unk>": 0, "한국어": 1}, unk_token="<unk>")), unk_token="<unk>",
        clean_up_tokenization_spaces=False)
    tiny.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "MistralCommonBackend"
    config_path.write_text(json.dumps(config))
    loaded = load_table1_tokenizer(tmp_path, backend="tokenizers", fix_mistral_regex=True,
                                   local_files_only=True)
    assert loaded.is_fast
    assert loaded("한국어", return_offsets_mapping=True)["offset_mapping"] == [(0, 3)]


def _validate(shard):
    from src.table1_smoke import validate_fast_shard

    return validate_fast_shard(shard, expected_prompts=5, max_new_tokens=9)


def test_smoke_validates_fast_artifacts_without_claiming_scalar_parity(fast_run):
    kwargs, _, _ = fast_run
    shard = Path(runner.run_sd_shard(**kwargs)["path"])
    result = _validate(shard)
    assert result["prompt_count"] == 5
    assert result["event_count"] > 0
    assert result["scalar_fallback_prompt_count"] == 0
    assert result["scalar_parity_validated"] is False


@pytest.mark.parametrize("mutation,match", [
    ("nan", "non-finite"),
    ("scalar", "reference policy"),
    ("missing_prompt", "prompt count"),
    ("bad_accept", "accepted token"),
])
def test_smoke_rejects_corrupt_or_wrong_contract_artifacts(fast_run, mutation, match):
    kwargs, _, _ = fast_run
    shard = Path(runner.run_sd_shard(**kwargs)["path"])
    refs = pd.read_parquet(shard / "references.parquet")
    events = pd.read_parquet(shard / "sd_events.parquet")
    if mutation == "nan":
        events.loc[0, "draft_entropy"] = float("nan")
    elif mutation == "scalar":
        refs.loc[0, "reference_source"] = "scalar_fallback"
    elif mutation == "missing_prompt":
        refs = refs.iloc[:-1]
    else:
        events.loc[0, "accepted"] = True
        events.loc[0, "target_top1_token_id"] = int(events.loc[0, "token_id"]) + 1
    refs.to_parquet(shard / "references.parquet", index=False)
    events.to_parquet(shard / "sd_events.parquet", index=False)
    with pytest.raises(AssertionError, match=match):
        _validate(shard)
