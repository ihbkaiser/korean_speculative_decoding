import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.rebuttal.adaptive_sd import (
    FixedK,
    MarginStop,
    greedy_target_only,
    speculative_greedy_adaptive,
)
from src.rebuttal.static_compiled_sd import (
    greedy_target_only_static,
    speculative_greedy_static,
)


def _tiny_pair():
    config = Qwen3Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        eos_token_id=40,
    )
    torch.manual_seed(10)
    draft = Qwen3ForCausalLM(config).eval()
    torch.manual_seed(20)
    target = Qwen3ForCausalLM(config).eval()
    return draft, target


def test_static_cache_matches_dynamic_through_rejection_rollback():
    torch.set_num_threads(1)
    draft, target = _tiny_pair()
    prompt = [27, 38, 11, 22, 33]

    dynamic, dynamic_counts, dynamic_events = speculative_greedy_adaptive(
        draft, target, prompt, 8, 40, FixedK(4), audit=True
    )
    static, static_counts, static_events = speculative_greedy_static(
        draft, target, prompt, 8, 40, FixedK(4), audit=True
    )

    assert dynamic == static
    assert dynamic_events[0]["accepted_prefix_length"] == 1
    assert dynamic_events[0]["first_rejection_index"] == 1
    assert static_events[0]["accepted_prefix_length"] == 1
    assert static_events[0]["first_rejection_index"] == 1
    assert static_counts["target_forwards"] == static_counts["blocks"]
    assert dynamic_counts["target_forwards"] == dynamic_counts["blocks"]


def test_static_target_only_and_margin_match_dynamic():
    torch.set_num_threads(1)
    draft, target = _tiny_pair()
    prompt = [27, 38, 11, 22, 33]

    dynamic_target, _ = greedy_target_only(target, prompt, 8, 40)
    static_target, _ = greedy_target_only_static(target, prompt, 8, 40)
    assert dynamic_target == static_target

    dynamic, _, _ = speculative_greedy_adaptive(
        draft, target, prompt, 8, 40, MarginStop(0.0), audit=True
    )
    static, counts = speculative_greedy_static(
        draft, target, prompt, 8, 40, MarginStop(0.0)
    )
    assert dynamic == static
    assert counts["target_forwards"] == counts["blocks"]
