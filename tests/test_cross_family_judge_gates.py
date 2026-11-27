from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.rebuttal.score_cross_family_judge as scorer


def test_gate_b_accepts_synthetic_functioning_judge_output():
    rows = pd.DataFrame({
        "entropy": np.linspace(1.5, 2.5, 100),
        "top2_margin": np.linspace(0.5, 2.0, 100),
        "is_top1": [True] * 40 + [False] * 60,
    })

    result = scorer._gate_b(rows, vocabulary_size=32_768)

    assert result["passed"]
    assert result["checks"] == {
        "mean_entropy": True,
        "high_entropy_share": True,
        "top1_accuracy": True,
    }
    assert result["positions"] == 100
    assert result["entropy_quantiles"]["p50"] == pytest.approx(2.0)
    assert result["top2_margin_quantiles"]["p50"] == pytest.approx(1.25)


def test_gate_b_rejects_synthetic_nonfunctioning_judge_output():
    rows = pd.DataFrame({
        "entropy": [10.0] * 20,
        "top2_margin": [0.01] * 20,
        "is_top1": [False] * 20,
    })

    result = scorer._gate_b(rows, vocabulary_size=100)

    assert not result["passed"]
    assert result["checks"] == {
        "mean_entropy": False,
        "high_entropy_share": False,
        "top1_accuracy": False,
    }


def test_scorer_has_no_streaming_loader_and_refuses_incomplete_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "qwen8"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "model_key": "qwen8",
        "revision": "pinned-revision",
        "weights_evicted": False,
        "weight_files": [{"filename": "model-00001-of-00002.safetensors", "bytes": 100}],
    }
    (snapshot / "naacl_download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(scorer, "model_path", lambda config, judge: snapshot)

    def unexpected_load(*args, **kwargs):
        pytest.fail("load_model must not run for an incomplete snapshot")

    monkeypatch.setattr(scorer, "load_model", unexpected_load)

    assert not hasattr(scorer, "_load_streamed_model")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        scorer._load_standard_judge({"models": {"qwen8": {"revision": "pinned-revision"}}}, "qwen8")
