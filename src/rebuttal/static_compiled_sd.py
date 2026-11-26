"""Fixed-shape Qwen speculative decoding over Transformers StaticCache.

StaticCache in Transformers 5.19 updates its write position from each layer's
``cumulative_length``.  After verification or rollback we reset that scalar to
the committed prefix length.  The fixed-capacity attention mask hides stale
suffix entries and later forwards overwrite them at explicit ``position_ids``.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import StaticCache

from src.rebuttal.adaptive_sd import FixedK, MarginStop, Policy, _draft_choice

_COMPILED_MODELS: dict[tuple[int, str], Any] = {}


def clear_compiled_models() -> None:
    """Release compiled wrappers before loading the next pair."""
    _COMPILED_MODELS.clear()


def _set_logical_length(cache: StaticCache, length: int) -> None:
    for layer in cache.layers:
        current = getattr(layer, "cumulative_length", None)
        if isinstance(current, torch.Tensor):
            current.fill_(length)
        else:
            layer.cumulative_length = int(length)


def _attention_mask(length: int, capacity: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, capacity), dtype=torch.long, device=device)
    mask[:, :length] = 1
    return mask


def _cache_for(model: Any, capacity: int) -> StaticCache:
    return StaticCache(model.config, max_cache_len=capacity)


def _call(model: Any, cache: StaticCache, ids: torch.Tensor, position_ids: torch.Tensor,
          logical_length: int, valid_end: int, capacity: int, logits_to_keep: int):
    _set_logical_length(cache, logical_length)
    return model(
        input_ids=ids,
        attention_mask=_attention_mask(valid_end, capacity, ids.device),
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=logits_to_keep,
    )


def _compiled(model: Any, enabled: bool, mode: str):
    if not enabled:
        return model
    key = (id(model), mode)
    if key not in _COMPILED_MODELS:
        _COMPILED_MODELS[key] = torch.compile(model, mode=mode, dynamic=False)
    return _COMPILED_MODELS[key]


@torch.inference_mode()
def greedy_target_only_static(
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    *,
    compile_steps: bool = False,
    compile_mode: str = "reduce-overhead",
) -> tuple[list[int], dict[str, int]]:
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = next(target_model.parameters()).device
    capacity = len(prompt_ids) + max_new_tokens + 17
    cache = _cache_for(target_model, capacity)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_positions = torch.arange(len(prompt_ids), device=device, dtype=torch.long)[None, :]
    output = _call(target_model, cache, prompt, prompt_positions, 0, len(prompt_ids),
                   capacity, logits_to_keep=1)
    logits = output.logits[0, -1]
    logical = len(prompt_ids)
    generated: list[int] = []
    forwards = 1
    step_model = _compiled(target_model, compile_steps, compile_mode)
    for index in range(max_new_tokens):
        token = int(logits.argmax().item())
        generated.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        if index + 1 < max_new_tokens:
            ids = torch.tensor([[token]], dtype=torch.long, device=device)
            positions = torch.tensor([[logical]], dtype=torch.long, device=device)
            output = _call(step_model, cache, ids, positions, logical, logical + 1,
                           capacity, logits_to_keep=1)
            logical += 1
            logits = output.logits[0, -1]
            forwards += 1
    return generated, {"target_forwards": forwards, "blocks": forwards}


@torch.inference_mode()
def speculative_greedy_static(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    policy: Policy,
    max_block: int = 16,
    *,
    compile_steps: bool = False,
    compile_mode: str = "reduce-overhead",
    audit: bool = False,
):
    """Greedy speculative decoder using one-token draft and 17-token verifies.

    Prompt prefills are eager. Draft steps always have shape ``[1, 1]``.
    Target verification uses shape ``[1, 17]`` with suffix pads masked out.
    The first verification call carries up to 16 proposals after the prompt
    prefill; later calls carry the pending committed token and up to 16
    proposals.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_block < 1:
        raise ValueError("max_block must be positive")
    if not isinstance(policy, (FixedK, MarginStop)):
        raise TypeError("Static compiled path currently supports FixedK and MarginStop")
    if max_new_tokens == 0:
        counters = {"target_forwards": 0, "draft_forwards": 0, "drafted_tokens": 0,
                    "accepted_tokens": 0, "blocks": 0, "prefill_forwards": 0}
        return ([], counters, []) if audit else ([], counters)

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    capacity = len(prompt_ids) + max_new_tokens + 17
    draft_cache = _cache_for(draft_model, capacity)
    target_cache = _cache_for(target_model, capacity)
    prompt_d = torch.tensor([prompt_ids], dtype=torch.long, device=draft_device)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    prompt_positions_d = torch.arange(len(prompt_ids), device=draft_device, dtype=torch.long)[None, :]
    prompt_positions_t = torch.arange(len(prompt_ids), device=target_device, dtype=torch.long)[None, :]
    draft_out = _call(draft_model, draft_cache, prompt_d, prompt_positions_d, 0,
                      len(prompt_ids), capacity, logits_to_keep=1)
    target_out = _call(target_model, target_cache, prompt_t, prompt_positions_t, 0,
                       len(prompt_ids), capacity, logits_to_keep=1)
    draft_logits = draft_out.logits[0, -1]
    target_prefill_logits = target_out.logits[0, -1]
    draft_logical = target_logical = len(prompt_ids)
    draft_step_model = _compiled(draft_model, compile_steps, compile_mode)
    target_verify_model = _compiled(target_model, compile_steps, compile_mode)

    generated: list[int] = []
    target_pending: int | None = None
    draft_pending: list[int] | None = None
    first_target_block = True
    finished = False
    counters = {"target_forwards": 0, "draft_forwards": 1, "drafted_tokens": 0,
                "accepted_tokens": 0, "blocks": 0, "prefill_forwards": 2}
    events: list[dict[str, Any]] = []
    block_limit = min(max_block, policy.k) if isinstance(policy, FixedK) else max_block

    while len(generated) < max_new_tokens and not finished:
        total_before = len(prompt_ids) + len(generated)
        proposal_limit = min(block_limit, max_new_tokens - len(generated))

        if draft_pending:
            for pending_token in draft_pending:
                ids = torch.tensor([[pending_token]], dtype=torch.long, device=draft_device)
                positions = torch.tensor([[draft_logical]], dtype=torch.long, device=draft_device)
                draft_out = _call(draft_step_model, draft_cache, ids, positions, draft_logical,
                                  draft_logical + 1, capacity, logits_to_keep=1)
                draft_logical += 1
                draft_logits = draft_out.logits[0, -1]
                counters["draft_forwards"] += 1
            draft_pending = None

        proposals: list[int] = []
        for index in range(proposal_limit):
            proposal, stop_predicate = _draft_choice(draft_logits, policy)
            proposal_id = int(proposal.reshape(()).item())
            proposals.append(proposal_id)
            should_stop = False
            if stop_predicate is not None:
                should_stop = bool(stop_predicate.reshape(()).item())
            if eos_token_id is not None and isinstance(policy, MarginStop):
                should_stop = should_stop or proposal_id == eos_token_id
            if should_stop or index + 1 == proposal_limit:
                break
            ids = torch.tensor([[proposal_id]], dtype=torch.long, device=draft_device)
            positions = torch.tensor([[draft_logical]], dtype=torch.long, device=draft_device)
            draft_out = _call(draft_step_model, draft_cache, ids, positions, draft_logical,
                              draft_logical + 1, capacity, logits_to_keep=1)
            draft_logical += 1
            draft_logits = draft_out.logits[0, -1]
            counters["draft_forwards"] += 1
        counters["drafted_tokens"] += len(proposals)

        valid_count = len(proposals)
        if eos_token_id is not None:
            for index, token in enumerate(proposals):
                if token == eos_token_id:
                    valid_count = index + 1
                    break

        # Right pad the verifier input to exactly 17 tokens. For the first
        # block there is no pending token because both models were prefilled.
        if first_target_block:
            target_values = proposals + [int(eos_token_id or 0)] * (17 - len(proposals))
            consumed_valid = len(proposals)
        else:
            if target_pending is None:
                raise RuntimeError("Target cache has no pending committed token")
            target_values = [target_pending] + proposals + [int(eos_token_id or 0)] * (16 - len(proposals))
            consumed_valid = 1 + len(proposals)
        target_ids = torch.tensor([target_values], dtype=torch.long, device=target_device)
        target_positions = torch.arange(target_logical, target_logical + 17,
                                        dtype=torch.long, device=target_device)[None, :]
        verify_out = _call(target_verify_model, target_cache, target_ids, target_positions,
                           target_logical, target_logical + consumed_valid, capacity,
                           logits_to_keep=17)
        verify_logits = verify_out.logits[0].float()
        if first_target_block:
            target_choices = [int(target_prefill_logits.argmax().item())]
            target_choices.extend(map(int, verify_logits[:max(0, len(proposals) - 1)].argmax(dim=-1).tolist()))
            bonus_id = int(verify_logits[len(proposals) - 1].argmax().item())
        else:
            target_choices = list(map(int, verify_logits[:len(proposals)].argmax(dim=-1).tolist()))
            bonus_id = int(verify_logits[len(proposals)].argmax().item())
        counters["target_forwards"] += 1
        counters["blocks"] += 1
        first_target_block = False

        mismatch_index = None
        accepted = 0
        for index in range(valid_count):
            if proposals[index] != target_choices[index]:
                mismatch_index = index
                break
            accepted += 1
        output_start = len(generated)
        correction_id = None
        committed: list[int]
        committed_bonus = None
        if mismatch_index is not None:
            correction_id = target_choices[mismatch_index]
            committed = proposals[:mismatch_index] + [correction_id]
            generated.extend(committed)
            counters["accepted_tokens"] += mismatch_index
            keep_length = total_before + mismatch_index
            _set_logical_length(target_cache, keep_length)
            _set_logical_length(draft_cache, keep_length)
            target_logical = draft_logical = keep_length
            if eos_token_id is not None and correction_id == eos_token_id:
                finished = True
            else:
                target_pending = correction_id
                draft_pending = [correction_id]
        else:
            committed = proposals[:valid_count]
            generated.extend(committed)
            counters["accepted_tokens"] += valid_count
            reached_eos = bool(valid_count and eos_token_id is not None
                               and proposals[valid_count - 1] == eos_token_id)
            if reached_eos:
                finished = True
            elif len(generated) < max_new_tokens:
                committed_bonus = bonus_id
                generated.append(bonus_id)
                target_pending = bonus_id
                draft_pending = [proposals[valid_count - 1], bonus_id]
                if eos_token_id is not None and bonus_id == eos_token_id:
                    finished = True
            else:
                finished = True
            if not reached_eos:
                logical = total_before + valid_count
                _set_logical_length(target_cache, logical)
                target_logical = logical
                # Draft cache has processed all proposals except its final
                # proposal; pending ids are consumed at the next block.
                draft_logical = total_before + max(0, valid_count - 1)
                _set_logical_length(draft_cache, draft_logical)

        if audit:
            events.append({"block": counters["blocks"] - 1, "output_start": output_start,
                           "proposed_ids": proposals, "verified_target_ids": target_choices[:valid_count],
                           "valid_proposal_count": valid_count, "accepted_prefix_length": accepted,
                           "first_rejection_index": mismatch_index,
                           "correction_token_id": correction_id, "bonus_token_id": committed_bonus,
                           "committed_ids": committed + ([] if committed_bonus is None else [committed_bonus])})

    return (generated, counters, events) if audit else (generated, counters)
