"""Single-prompt greedy speculative decoding with draft-side stop rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from src.batched_speculative import _logits_keep_argument
from src.speculative_decoding import _crop_past_key_values


@dataclass(frozen=True)
class FixedK:
    """Draft at most ``k`` tokens in each block."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("FixedK.k must be positive")


@dataclass(frozen=True)
class MarginStop:
    """Stop after a proposal whose draft top-two logit margin is below ``tau``."""

    tau: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau):
            raise ValueError("MarginStop.tau must be finite")


@dataclass(frozen=True)
class EntropyStop:
    """Stop after a proposal whose draft entropy exceeds ``h``."""

    h: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.h):
            raise ValueError("EntropyStop.h must be finite")


@dataclass(frozen=True)
class Top1Stop:
    """Stop after a proposal whose draft top-one probability is below ``p``."""

    p: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.p):
            raise ValueError("Top1Stop.p must be finite")


Policy: TypeAlias = FixedK | MarginStop | EntropyStop | Top1Stop


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _call(model: Any, ids: torch.Tensor, past: Any | None, keep_logits: int):
    kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": True}
    if past is not None:
        kwargs["past_key_values"] = past
    keep_argument = _logits_keep_argument(type(model))
    if keep_argument is not None:
        kwargs[keep_argument] = keep_logits
    return model(**kwargs)


def _draft_choice(logits: torch.Tensor, policy: Policy) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a device-side token id and, for adaptive rules, its stop predicate."""
    values = logits.float()
    if isinstance(policy, MarginStop):
        top_values, top_ids = torch.topk(values, k=2, dim=-1)
        return top_ids[..., 0], top_values[..., 0] - top_values[..., 1] < policy.tau
    if isinstance(policy, EntropyStop):
        normalizer = torch.logsumexp(values, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        entropy = normalizer - (probabilities * values).sum(dim=-1)
        return values.argmax(dim=-1), entropy > policy.h
    if isinstance(policy, Top1Stop):
        top_value, token_id = values.max(dim=-1)
        top1_probability = torch.exp(top_value - torch.logsumexp(values, dim=-1))
        return token_id, top1_probability < policy.p
    return values.argmax(dim=-1), None


@torch.inference_mode()
def greedy_target_only(
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], dict[str, int]]:
    """Greedy target-only generation with one persistent KV cache."""
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated: list[int] = []
    forwards = 0
    if max_new_tokens:
        device = _device(target_model)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        result = _call(target_model, ids, None, keep_logits=1)
        forwards += 1
        past = result.past_key_values
        for index in range(max_new_tokens):
            token = int(result.logits[0, -1].argmax(dim=-1).item())
            generated.append(token)
            if token == eos_token_id:
                break
            if index + 1 < max_new_tokens:
                next_ids = torch.tensor([[token]], dtype=torch.long, device=device)
                result = _call(target_model, next_ids, past, keep_logits=1)
                forwards += 1
                past = result.past_key_values
    return generated, {"target_forwards": forwards}


@torch.inference_mode()
def speculative_greedy_adaptive(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    policy: Policy,
    max_block: int = 16,
    *,
    audit: bool = False,
) -> tuple[list[int], dict[str, int]] | tuple[list[int], dict[str, int], list[dict[str, Any]]]:
    """Greedy speculative decoding with persistent draft and target caches.

    Each block uses exactly one target forward. The target verifies proposals
    and supplies either the correction token or the all-accepted bonus token.
    For fixed-length drafting, proposal IDs remain on-device until block
    verification, avoiding a host synchronization per proposal. Adaptive
    policies read one stop predicate per drafted proposal, which is included
    in their measured cost. ``audit=True`` adds one compact record per block.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_block < 1:
        raise ValueError("max_block must be positive")
    if not isinstance(policy, (FixedK, MarginStop, EntropyStop, Top1Stop)):
        raise TypeError(f"Unsupported drafting policy: {type(policy).__name__}")

    generated: list[int] = []
    counters = {
        "target_forwards": 0,
        "draft_forwards": 0,
        "drafted_tokens": 0,
        "accepted_tokens": 0,
        "blocks": 0,
    }
    events: list[dict[str, Any]] = []
    if max_new_tokens == 0:
        return (generated, counters, events) if audit else (generated, counters)

    draft_device = _device(draft_model)
    target_device = _device(target_model)
    draft_past = None
    target_past = None
    # The first target call includes the full prompt. Later calls prepend the
    # one pending correction/bonus token to each proposal block.
    first_target_block = True
    target_pending: int | None = None
    # After an all-accepted block, the draft cache is two tokens behind: the
    # last proposal and target bonus are consumed together next round.
    draft_pending: list[int] | None = None
    block_limit = min(max_block, policy.k) if isinstance(policy, FixedK) else max_block
    finished = False

    while len(generated) < max_new_tokens and not finished:
        total_before = len(prompt_ids) + len(generated)
        proposal_limit = min(block_limit, max_new_tokens - len(generated))
        proposals: list[torch.Tensor] = []

        if draft_past is None:
            draft_ids = torch.tensor([prompt_ids], dtype=torch.long, device=draft_device)
            draft_result = _call(draft_model, draft_ids, None, keep_logits=1)
        else:
            if not draft_pending:
                raise RuntimeError("draft cache has no pending committed token")
            draft_ids = torch.tensor([draft_pending], dtype=torch.long, device=draft_device)
            draft_result = _call(draft_model, draft_ids, draft_past, keep_logits=1)
        counters["draft_forwards"] += 1
        draft_past = draft_result.past_key_values
        draft_logits = draft_result.logits[0, -1]

        for _ in range(proposal_limit):
            proposal, stop_predicate = _draft_choice(draft_logits, policy)
            proposal = proposal.reshape(())
            proposals.append(proposal)
            if stop_predicate is not None:
                should_stop = stop_predicate.reshape(())
                if eos_token_id is not None:
                    should_stop = should_stop | proposal.eq(eos_token_id)
                if bool(should_stop.item()):
                    break
            if len(proposals) < proposal_limit:
                draft_result = _call(
                    draft_model,
                    proposal.reshape(1, 1),
                    draft_past,
                    keep_logits=1,
                )
                counters["draft_forwards"] += 1
                draft_past = draft_result.past_key_values
                draft_logits = draft_result.logits[0, -1]

        proposal_tensor = torch.stack(proposals).to(dtype=torch.long)
        counters["drafted_tokens"] += len(proposals)

        if first_target_block:
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            target_input = torch.cat((prompt_tensor, proposal_tensor.to(target_device)[None, :]), dim=1)
        else:
            if target_pending is None:
                raise RuntimeError("target cache has no pending committed token")
            pending_tensor = torch.tensor([[target_pending]], dtype=torch.long, device=target_device)
            target_input = torch.cat((pending_tensor, proposal_tensor.to(target_device)[None, :]), dim=1)

        target_result = _call(
            target_model,
            target_input,
            target_past,
            keep_logits=len(proposals) + 1,
        )
        counters["target_forwards"] += 1
        counters["blocks"] += 1
        target_past = target_result.past_key_values
        first_target_block = False

        # The last K+1 logits correspond to predictions for each of the K
        # proposals and the all-accepted bonus, in both the first and later blocks.
        verify_logits = target_result.logits[0, -(len(proposals) + 1):].float()
        target_choices = verify_logits[:-1].argmax(dim=-1)
        bonus_choice = verify_logits[-1].argmax(dim=-1).reshape(1)
        packed = torch.cat((proposal_tensor.to(target_device), target_choices, bonus_choice))
        packed_ids = [int(token) for token in packed.detach().cpu().tolist()]
        proposal_ids = packed_ids[:len(proposals)]
        target_ids = packed_ids[len(proposals):2 * len(proposals)]
        bonus_id = packed_ids[-1]

        # Drafts after their first EOS are generated without per-token host
        # reads for FixedK. They are never verified or committed.
        valid_count = len(proposal_ids)
        if eos_token_id is not None:
            for position, token in enumerate(proposal_ids):
                if token == eos_token_id:
                    valid_count = position + 1
                    break

        accepted = 0
        mismatch_index: int | None = None
        for position in range(valid_count):
            if proposal_ids[position] != target_ids[position]:
                mismatch_index = position
                break
            accepted += 1

        output_start = len(generated)
        committed: list[int]
        correction_id: int | None = None
        committed_bonus: int | None = None
        if mismatch_index is not None:
            correction_id = target_ids[mismatch_index]
            committed = proposal_ids[:mismatch_index] + [correction_id]
            generated.extend(committed)
            counters["accepted_tokens"] += mismatch_index
            if eos_token_id is not None and correction_id == eos_token_id:
                finished = True
            else:
                keep_length = total_before + mismatch_index
                target_past = _crop_past_key_values(target_past, keep_length)
                draft_past = _crop_past_key_values(draft_past, keep_length)
                if target_past is None or draft_past is None:
                    raise RuntimeError("KV cache rollback failed after a rejected draft block")
                target_pending = correction_id
                draft_pending = [correction_id]
        else:
            committed = proposal_ids[:valid_count]
            generated.extend(committed)
            counters["accepted_tokens"] += valid_count
            reached_eos = bool(
                eos_token_id is not None and valid_count and proposal_ids[valid_count - 1] == eos_token_id
            )
            if reached_eos:
                finished = True
            elif len(generated) < max_new_tokens:
                committed_bonus = bonus_id
                generated.append(bonus_id)
                target_pending = bonus_id
                draft_pending = [proposal_ids[valid_count - 1], bonus_id]
                if eos_token_id is not None and bonus_id == eos_token_id:
                    finished = True
            else:
                finished = True

        if audit:
            events.append({
                "block": counters["blocks"] - 1,
                "output_start": output_start,
                "proposed_ids": proposal_ids,
                "verified_target_ids": target_ids[:valid_count],
                "valid_proposal_count": valid_count,
                "accepted_prefix_length": accepted,
                "first_rejection_index": mismatch_index,
                "correction_token_id": correction_id,
                "bonus_token_id": committed_bonus,
                "committed_ids": committed + ([] if committed_bonus is None else [committed_bonus]),
            })

    return (generated, counters, events) if audit else (generated, counters)
