from __future__ import annotations

import pytest

from src.rebuttal.adaptive_sd import (
    EntropyStop,
    FixedK,
    MarginStop,
    Top1Stop,
    greedy_target_only,
    speculative_greedy_adaptive,
)
from src.speculative_decoding import greedy_generate
from tests.test_batched_speculative import PrefixModel


POLICIES = [FixedK(4), MarginStop(100.0), EntropyStop(-1.0), Top1Stop(1.1)]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("eos_token_id", [None, 16])
def test_adaptive_policies_match_target_greedy_and_obey_budget(policy, eos_token_id):
    prompt = [3, 2]
    max_new_tokens = 23
    expected = greedy_generate(PrefixModel(eos=eos_token_id), prompt, max_new_tokens, eos_token_id)
    target = PrefixModel(eos=eos_token_id)
    actual, counts = speculative_greedy_adaptive(
        PrefixModel(shift=1, eos=eos_token_id), target, prompt, max_new_tokens,
        eos_token_id, policy, max_block=8,
    )

    assert actual == expected
    assert len(actual) <= max_new_tokens
    assert counts["target_forwards"] == counts["blocks"] == target.calls
    if eos_token_id is not None and eos_token_id in actual:
        assert actual[-1] == eos_token_id
        assert actual.count(eos_token_id) == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_adaptive_policies_handle_short_and_empty_budgets(policy):
    prompt = [3, 2]
    for budget in (0, 1, 3):
        target = PrefixModel(eos=None)
        expected = greedy_generate(PrefixModel(eos=None), prompt, budget, None)
        actual, counts = speculative_greedy_adaptive(
            PrefixModel(shift=1, eos=None), target, prompt, budget, None, policy,
        )
        assert actual == expected
        assert counts["target_forwards"] == counts["blocks"] == target.calls
        if budget == 0:
            assert counts == {
                "target_forwards": 0,
                "draft_forwards": 0,
                "drafted_tokens": 0,
                "accepted_tokens": 0,
                "blocks": 0,
            }


def test_fixed_k4_audit_has_accepted_prefix_then_correction_or_bonus():
    prompt = [3, 2]
    target = PrefixModel(eos=None)
    expected = greedy_generate(PrefixModel(eos=None), prompt, 19, None)
    actual, counts, blocks = speculative_greedy_adaptive(
        PrefixModel(shift=1, eos=None), target, prompt, 19, None,
        FixedK(4), max_block=16, audit=True,
    )

    assert actual == expected
    assert counts["target_forwards"] == counts["blocks"] == target.calls
    assert blocks
    for block in blocks:
        proposals = block["proposed_ids"][:block["valid_proposal_count"]]
        target_ids = block["verified_target_ids"]
        rejection = block["first_rejection_index"]
        if rejection is None:
            assert proposals == target_ids
            expected_committed = proposals
            if block["bonus_token_id"] is not None:
                expected_committed.append(block["bonus_token_id"])
        else:
            assert proposals[:rejection] == target_ids[:rejection]
            assert proposals[rejection] != target_ids[rejection]
            expected_committed = proposals[:rejection] + [target_ids[rejection]]
            assert block["bonus_token_id"] is None
        assert block["committed_ids"] == expected_committed


@pytest.mark.parametrize("policy", [FixedK(3), MarginStop(1.0), EntropyStop(0.8), Top1Stop(0.8)])
def test_adaptive_decoder_with_transformers_dynamic_cache(policy):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(20261125)
    config = LlamaConfig(
        vocab_size=37,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    target = LlamaForCausalLM(config).eval()
    draft = LlamaForCausalLM(config).eval()
    prompt = [1, 3, 7]
    expected = greedy_generate(target, prompt, 17, None)
    actual, counts = speculative_greedy_adaptive(draft, target, prompt, 17, None, policy)

    assert actual == expected
    assert counts["target_forwards"] == counts["blocks"]
