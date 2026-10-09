from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.batched_speculative import (
    _compact_cache, distribution_stats_batch, speculative_greedy_microbatch,
)
from src.models import distribution_stats
from src.speculative_decoding import greedy_generate


class PrefixModel(torch.nn.Module):
    """Causal model sensitive to token history, padding, and cache rollback."""

    def __init__(self, shift=0, eos=16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.shift, self.eos = shift, eos
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True):
        self.calls += 1
        if past_key_values is None:
            previous = input_ids[:, :0]
        else:
            previous = past_key_values[0][0][:, 0, :, 0].long()
        full = torch.cat((previous, input_ids), -1)
        if attention_mask is None:
            attention_mask = torch.ones_like(full)
        logits = torch.full((*input_ids.shape, 17), -10.0, device=input_ids.device)
        for step in range(input_ids.shape[1]):
            end = previous.shape[1] + step + 1
            visible = full[:, :end] * attention_mask[:, :end]
            lengths = attention_mask[:, :end].sum(-1)
            shift = (lengths % 3 == 0).long() if self.shift == "selective" else self.shift
            token = (visible.sum(-1) + lengths + shift) % 16
            if self.eos is not None:
                first_token = full.gather(1, attention_mask.argmax(-1)[:, None]).squeeze(1)
                token = torch.where(lengths >= (first_token % 5 + 7), self.eos, token)
            logits[:, step].scatter_(1, token[:, None], 10.0)
        keys = full[:, None, :, None].float()
        return SimpleNamespace(logits=logits, past_key_values=((keys, keys.clone()),))


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("shift", [0, 1, 5, "selective"])
def test_microbatch_exact_against_independent_scalar_with_rejections(k, shift):
    prompts = [[3, 2], [3, 1, 5], [3, 7], [3]]
    target = PrefixModel(eos=None)
    expected = [greedy_generate(target, p, 13, None) for p in prompts]
    actual, events = speculative_greedy_microbatch(
        PrefixModel(shift, eos=None), target, prompts, 13, None, k,
    )
    assert actual == expected
    for row_events in events:
        for event in row_events:
            assert event["accepted"] == (
                event["draft_token_id"] == event["target_greedy_token_id"]
                and not event["invalidated_after_first_rejection"]
            )
            if event["invalidated_after_first_rejection"]:
                assert event["rejected"] is None


@pytest.mark.parametrize("shift", [0, 1, "selective"])
def test_eos_and_mixed_finished_rows(shift):
    prompts = [[3, 2], [3, 1, 5], [3, 7], [3]]
    target = PrefixModel()
    expected = [greedy_generate(target, p, 15, 16) for p in prompts]
    actual, events = speculative_greedy_microbatch(
        PrefixModel(shift), target, prompts, 15, 16, 4,
    )
    assert actual == expected
    assert all(row[-1] == 16 and row.count(16) == 1 for row in actual)
    assert all(event["output_token_position"] < 15 for row in events for event in row)


def test_one_target_call_per_round_and_bonus_token():
    target = PrefixModel(eos=None)
    output, events = speculative_greedy_microbatch(
        PrefixModel(eos=None), target, [[3, 2], [3, 7]], 20, None, 4,
    )
    assert all(len(row) == 20 for row in output)
    assert target.calls == 4  # K accepted proposals + one target bonus
    assert len(events[0]) == 16


def test_stats_batch_matches_scalar():
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 31)
    ids = torch.randint(31, (2, 4))
    greedy, values = distribution_stats_batch(logits, ids)
    for row in range(2):
        for slot in range(4):
            expected = distribution_stats(logits[row, slot], int(ids[row, slot]))
            assert greedy[row, slot] == expected["greedy_token_id"]
            assert values[row, slot].tolist() == pytest.approx([
                expected["logprob"], expected["entropy"],
                expected["top1_logprob"], expected["rank"],
            ], abs=2e-6)


def test_compaction_removes_padding_and_rejected_suffix_per_row():
    keys = torch.arange(12).reshape(2, 1, 6, 1).float()
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]])
    past, new_mask = _compact_cache(((keys, keys.clone()),), mask, [2, 4])
    assert new_mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert past[0][0][0, 0, -2:, 0].tolist() == [1, 2]
    assert past[0][0][1, 0, :, 0].tolist() == [6, 7, 8, 10]


def test_microbatch_real_transformers_dynamic_cache():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(22)
    cfg = LlamaConfig(vocab_size=37, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=None, pad_token_id=0)
    target = LlamaForCausalLM(cfg).eval()
    draft = LlamaForCausalLM(cfg).eval()
    prompts = [[1, 2, 3], [1, 4], [1, 7, 8, 9]]
    expected = [greedy_generate(target, p, 11, None) for p in prompts]
    actual, _ = speculative_greedy_microbatch(draft, target, prompts, 11, None, 4)
    assert actual == expected


@pytest.mark.parametrize("cache_api", ["v4", "v5"])
def test_mutable_dynamic_cache_apis(cache_api):
    class CachedPrefix(PrefixModel):
        def forward(self, input_ids, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=True):
            if past_key_values is not None:
                keys = (past_key_values.layers[0].keys if cache_api == "v5"
                        else past_key_values.key_cache[0])
                past_key_values = ((keys, keys.clone()),)
            result = super().forward(input_ids, attention_mask, position_ids, past_key_values, use_cache)
            keys, values = result.past_key_values[0]
            result.past_key_values = (SimpleNamespace(layers=[SimpleNamespace(
                keys=keys, values=values, cumulative_length=keys.shape[-2])]) if cache_api == "v5"
                else SimpleNamespace(key_cache=[keys], value_cache=[values], _seen_tokens=keys.shape[-2]))
            return result

    prompts = [[3, 2], [3], [3, 6, 9]]
    expected = [greedy_generate(PrefixModel(eos=None), p, 15, None) for p in prompts]
    actual, _ = speculative_greedy_microbatch(
        CachedPrefix("selective", eos=None), CachedPrefix(eos=None), prompts, 15, None, 4,
    )
    assert actual == expected


def test_empty_budget_never_calls_models():
    model = PrefixModel()
    assert speculative_greedy_microbatch(model, model, [[1]], 0, None) == ([[]], [[]])
    assert model.calls == 0
