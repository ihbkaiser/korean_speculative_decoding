"""Deterministic target and custom speculative decoding implementations."""

from __future__ import annotations

from typing import Any

import torch

from .models import distribution_stats


def _as_batch(ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.inference_mode()
def greedy_generate(model: Any, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None) -> list[int]:
    """Target-only greedy reference generation, returning generated IDs only."""
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    device = next(model.parameters()).device
    output: list[int] = []
    context = _as_batch(prompt_ids, device)
    past = None
    for _ in range(max_new_tokens):
        if past is None:
            result = model(input_ids=context, use_cache=True)
        else:
            result = model(input_ids=_as_batch([output[-1]], device), past_key_values=past, use_cache=True)
        past = result.past_key_values
        token_id = int(torch.argmax(result.logits[0, -1], dim=-1).item())
        output.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return output


@torch.inference_mode()
def greedy_generate_batch(
    model: Any,
    prompt_ids_batch: list[list[int]],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[list[int]]:
    """Greedy reference generation for a same-length prompt microbatch.

    This is intentionally restricted to equal-length prompts: it avoids
    padding/mask-dependent numerical changes in the pinned exact path.  Each
    row has an independent KV cache batch entry; finished rows continue with
    EOS input while unfinished rows keep decoding, and their outputs are
    ignored after termination.
    """
    if not prompt_ids_batch:
        raise ValueError("prompt_ids_batch must contain at least one prompt")
    if any(not prompt_ids for prompt_ids in prompt_ids_batch):
        raise ValueError("prompt_ids_batch must not contain empty prompts")
    prompt_length = len(prompt_ids_batch[0])
    if any(len(prompt_ids) != prompt_length for prompt_ids in prompt_ids_batch):
        raise ValueError("greedy_generate_batch requires equal-length prompts")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = next(model.parameters()).device
    context = torch.tensor(prompt_ids_batch, dtype=torch.long, device=device)
    result = model(input_ids=context, use_cache=True)
    past = result.past_key_values
    next_tokens = torch.argmax(result.logits[:, -1], dim=-1)
    outputs: list[list[int]] = [[] for _ in prompt_ids_batch]
    finished = [False] * len(prompt_ids_batch)

    for step in range(max_new_tokens):
        for index, token in enumerate(next_tokens.tolist()):
            if not finished[index]:
                token_id = int(token)
                outputs[index].append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    finished[index] = True
        if step + 1 >= max_new_tokens or all(finished):
            break
        result = model(
            input_ids=next_tokens[:, None],
            past_key_values=past,
            use_cache=True,
        )
        past = result.past_key_values
        next_tokens = torch.argmax(result.logits[:, -1], dim=-1)
    return outputs


@torch.inference_mode()
def speculative_greedy(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    k: int = 4,
    prompt_id: int | str = 0,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Custom greedy speculative decoding with one audit row per proposal.

    Target proposals are checked sequentially through the target KV cache, using
    the same one-token cached path as target-only greedy decoding. Later
    proposals following a mismatch are retained in the log but marked invalidated.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if k < 1 or max_new_tokens < 0:
        raise ValueError("k must be positive and max_new_tokens non-negative")
    target_device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device
    generated: list[int] = []
    events: list[dict[str, Any]] = []
    round_index = 0
    finished = False
    target_result = target_model(input_ids=_as_batch(prompt_ids, target_device), use_cache=True)
    target_past = target_result.past_key_values
    target_next_logits = target_result.logits[0, -1]

    while len(generated) < max_new_tokens and not finished:
        current = prompt_ids + generated
        remaining = max_new_tokens - len(generated)
        proposal_limit = min(k, remaining)

        # Build a short draft cache for this proposal block.
        draft_context = _as_batch(current, draft_device)
        draft_result = draft_model(input_ids=draft_context, use_cache=True)
        draft_past = draft_result.past_key_values
        draft_distributions = []
        proposals: list[int] = []
        for proposal_index in range(proposal_limit):
            logits = draft_result.logits[0, -1]
            proposal_id = int(torch.argmax(logits, dim=-1).item())
            stats = distribution_stats(logits, proposal_id)
            draft_distributions.append(stats)
            proposals.append(proposal_id)
            if eos_token_id is not None and proposal_id == eos_token_id:
                break
            draft_result = draft_model(
                input_ids=_as_batch([proposal_id], draft_device),
                past_key_values=draft_past,
                use_cache=True,
            )
            draft_past = draft_result.past_key_values

        target_distributions = []
        accepted_prefix_length = 0
        first_rejection_output_position: int | None = None
        mismatch_index: int | None = None
        for j, proposal_id in enumerate(proposals):
            # Match the target-only reference's cached, single-token forward
            # path exactly. Batched full-sequence verification can select a
            # different argmax for near-tied logits on some GPU kernels.
            stats = distribution_stats(target_next_logits, proposal_id)
            target_distributions.append(stats)
            target_greedy_id = int(stats["greedy_token_id"])
            if proposal_id != target_greedy_id:
                mismatch_index = j
                first_rejection_output_position = len(generated) + j
                break
            accepted_prefix_length += 1
            if eos_token_id is not None and proposal_id == eos_token_id:
                break
            target_result = target_model(
                input_ids=_as_batch([proposal_id], target_device),
                past_key_values=target_past,
                use_cache=True,
            )
            target_past = target_result.past_key_values
            target_next_logits = target_result.logits[0, -1]

        # These proposals were invalidated by an earlier mismatch. Preserve
        # their requested target statistics under the hypothetical draft path,
        # but do not use those values to decide acceptance or generation.
        if mismatch_index is not None:
            for j in range(mismatch_index + 1, len(proposals)):
                hypothetical_context = current + proposals[:j]
                hypothetical = target_model(
                    input_ids=_as_batch(hypothetical_context, target_device),
                    use_cache=False,
                )
                target_distributions.append(
                    distribution_stats(hypothetical.logits[0, -1], proposals[j])
                )

        round_events: list[dict[str, Any]] = []
        for j, proposal_id in enumerate(proposals):
            target_greedy_id = target_distributions[j]["greedy_token_id"]
            rejected = proposal_id != target_greedy_id
            invalidated = mismatch_index is not None and j > mismatch_index
            round_events.append({
                "prompt_id": prompt_id,
                "round_index": round_index,
                "draft_token_id": proposal_id,
                "target_greedy_token_id": target_greedy_id,
                "accepted": not rejected and not invalidated,
                # Invalidated proposals were never checked on the valid target
                # path, so they are not counted as actual rejections.
                "rejected": None if invalidated else rejected,
                "is_first_rejection": mismatch_index == j,
                "invalidated_after_first_rejection": invalidated,
                "proposal_position": j + 1,
                "output_token_position": len(generated) + j,
                "draft_logprob": draft_distributions[j]["logprob"],
                "target_logprob": target_distributions[j]["logprob"],
                "draft_entropy": draft_distributions[j]["entropy"],
                "target_entropy": target_distributions[j]["entropy"],
                "draft_top1_logprob": draft_distributions[j]["top1_logprob"],
                "target_top1_logprob": target_distributions[j]["top1_logprob"],
                "target_rank_of_proposed_token": target_distributions[j]["rank"],
                "accepted_prefix_length": None,
                "first_rejection_output_position": first_rejection_output_position,
            })

        if mismatch_index is not None:
            # Emit accepted draft prefix and the target correction token.
            generated.extend(proposals[:mismatch_index])
            correction = int(target_distributions[mismatch_index]["greedy_token_id"])
            generated.append(correction)
            if eos_token_id is not None and correction == eos_token_id:
                finished = True
            else:
                target_result = target_model(
                    input_ids=_as_batch([correction], target_device),
                    past_key_values=target_past,
                    use_cache=True,
                )
                target_past = target_result.past_key_values
                target_next_logits = target_result.logits[0, -1]
        else:
            generated.extend(proposals)
            if proposals and eos_token_id is not None and proposals[-1] == eos_token_id:
                finished = True

        for row in round_events:
            row["accepted_prefix_length"] = accepted_prefix_length
        events.extend(round_events)
        round_index += 1

        if len(generated) >= max_new_tokens:
            break
    return generated, events


def _crop_past_key_values(past_key_values: Any, target_length: int) -> Any | None:
    """Crop a Transformers cache to ``target_length`` when the format allows it.

    DynamicCache exposes ``crop`` directly.  Legacy tuple caches are handled
    for decoder-only models whose key/value tensors use sequence dimension
    ``-2``.  Returning ``None`` deliberately requests a correctness-preserving
    full draft re-prefill on the next decoding block.
    """
    if past_key_values is None:
        return None
    target_length = int(target_length)
    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        try:
            get_length = getattr(past_key_values, "get_seq_length", None)
            if callable(get_length):
                current_length = int(get_length())
                if target_length > current_length:
                    return None
                if target_length < current_length:
                    # Negative crop values remove a suffix and are the
                    # forward-compatible Transformers API.
                    crop(-(current_length - target_length))
            else:
                crop(target_length)
            if callable(get_length) and int(get_length()) != target_length:
                return None
            return past_key_values
        except Exception:
            return None
    if isinstance(past_key_values, tuple):
        try:
            return tuple(
                tuple(value[..., :target_length, :] for value in layer)
                for layer in past_key_values
            )
        except (TypeError, IndexError, AttributeError):
            return None
    return None


@torch.inference_mode()
def speculative_greedy_cached(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    k: int = 4,
    prompt_id: int | str = 0,
    batch_target_verification: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Exact greedy SD with a persistent draft KV cache.

    The original implementation re-prefills the draft model with the whole
    committed context at the start of every speculative block.  This variant
    keeps the draft cache across blocks.  After a rejection it crops the
    uncommitted proposal suffix and feeds the target correction token; if the
    cache format cannot be cropped, it falls back to a full draft prefill on
    the next block without changing output semantics.

    ``batch_target_verification`` verifies a complete proposal block with one
    target forward and rolls the target cache back after a rejection.  It is
    substantially faster on GPUs.  Because batched and singleton kernels can
    disagree on a numerically near-tied argmax, callers that require strict
    singleton parity must compare the returned IDs and retry with this option
    disabled when needed.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if k < 1 or max_new_tokens < 0:
        raise ValueError("k must be positive and max_new_tokens non-negative")
    target_device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device
    generated: list[int] = []
    events: list[dict[str, Any]] = []
    round_index = 0
    finished = False

    target_result = target_model(
        input_ids=_as_batch(prompt_ids, target_device), use_cache=True,
    )
    target_past = target_result.past_key_values
    target_next_logits = target_result.logits[0, -1]
    draft_past = None
    draft_next_logits = None

    while len(generated) < max_new_tokens and not finished:
        current = prompt_ids + generated
        base_position = len(generated)
        proposal_limit = min(k, max_new_tokens - base_position)

        if draft_past is None or draft_next_logits is None:
            draft_result = draft_model(
                input_ids=_as_batch(current, draft_device), use_cache=True,
            )
            draft_past = draft_result.past_key_values
            draft_next_logits = draft_result.logits[0, -1]

        draft_distributions: list[dict[str, Any]] = []
        proposals: list[int] = []
        for _ in range(proposal_limit):
            logits = draft_next_logits
            proposal_id = int(torch.argmax(logits, dim=-1).item())
            draft_distributions.append(distribution_stats(logits, proposal_id))
            proposals.append(proposal_id)
            if eos_token_id is not None and proposal_id == eos_token_id:
                break
            draft_result = draft_model(
                input_ids=_as_batch([proposal_id], draft_device),
                past_key_values=draft_past,
                use_cache=True,
            )
            draft_past = draft_result.past_key_values
            draft_next_logits = draft_result.logits[0, -1]

        target_distributions: list[dict[str, Any]] = []
        accepted_prefix_length = 0
        first_rejection_output_position: int | None = None
        mismatch_index: int | None = None
        target_eos_accepted = False
        batched_target_logits = None
        verification_cache_length: int | None = None
        use_batched_target_verification = bool(batch_target_verification and proposals)
        if use_batched_target_verification:
            if not hasattr(target_past, "crop") or not hasattr(target_past, "get_seq_length"):
                # Legacy tuple caches cannot be safely rolled back after a
                # vectorized verification call.  Fall back to the exact
                # singleton path instead of silently changing semantics.
                use_batched_target_verification = False
            else:
                verification_cache_length = int(target_past.get_seq_length())
                target_result = target_model(
                    input_ids=_as_batch(proposals, target_device),
                    past_key_values=target_past,
                    use_cache=True,
                )
                target_past = target_result.past_key_values
                batched_target_logits = target_result.logits[0]

        for j, proposal_id in enumerate(proposals):
            if use_batched_target_verification and j > 0:
                decision_logits = batched_target_logits[j - 1]
            else:
                decision_logits = target_next_logits
            stats = distribution_stats(decision_logits, proposal_id)
            target_distributions.append(stats)
            if mismatch_index is not None:
                # In batched mode the remaining logits are retained only for
                # audit rows; they must not affect the committed prefix.
                continue
            target_greedy_id = int(stats["greedy_token_id"])
            if proposal_id != target_greedy_id:
                mismatch_index = j
                first_rejection_output_position = base_position + j
                if not use_batched_target_verification:
                    break
                continue
            accepted_prefix_length += 1
            if eos_token_id is not None and proposal_id == eos_token_id:
                target_eos_accepted = True
                break
            if not use_batched_target_verification:
                target_result = target_model(
                    input_ids=_as_batch([proposal_id], target_device),
                    past_key_values=target_past,
                    use_cache=True,
                )
                target_past = target_result.past_key_values
                target_next_logits = target_result.logits[0, -1]

        if mismatch_index is not None and not use_batched_target_verification:
            # Preserve the original audit contract for invalidated suffixes.
            for j in range(mismatch_index + 1, len(proposals)):
                hypothetical_context = current + proposals[:j]
                hypothetical = target_model(
                    input_ids=_as_batch(hypothetical_context, target_device),
                    use_cache=False,
                )
                target_distributions.append(
                    distribution_stats(hypothetical.logits[0, -1], proposals[j])
                )

        round_events: list[dict[str, Any]] = []
        for j, proposal_id in enumerate(proposals):
            target_greedy_id = target_distributions[j]["greedy_token_id"]
            rejected = proposal_id != target_greedy_id
            invalidated = mismatch_index is not None and j > mismatch_index
            round_events.append({
                "prompt_id": prompt_id,
                "round_index": round_index,
                "draft_token_id": proposal_id,
                "target_greedy_token_id": target_greedy_id,
                "accepted": not rejected and not invalidated,
                "rejected": None if invalidated else rejected,
                "is_first_rejection": mismatch_index == j,
                "invalidated_after_first_rejection": invalidated,
                "proposal_position": j + 1,
                "output_token_position": base_position + j,
                "draft_logprob": draft_distributions[j]["logprob"],
                "target_logprob": target_distributions[j]["logprob"],
                "draft_entropy": draft_distributions[j]["entropy"],
                "target_entropy": target_distributions[j]["entropy"],
                "draft_top1_logprob": draft_distributions[j]["top1_logprob"],
                "target_top1_logprob": target_distributions[j]["top1_logprob"],
                "target_rank_of_proposed_token": target_distributions[j]["rank"],
                "accepted_prefix_length": None,
                "first_rejection_output_position": first_rejection_output_position,
            })

        if mismatch_index is not None:
            if use_batched_target_verification:
                if verification_cache_length is None:
                    raise RuntimeError("missing target cache length for batched verification")
                target_past = _crop_past_key_values(
                    target_past,
                    verification_cache_length + accepted_prefix_length,
                )
                if target_past is None:
                    raise RuntimeError("target cache rollback failed after batched verification")
            generated.extend(proposals[:mismatch_index])
            correction = int(target_distributions[mismatch_index]["greedy_token_id"])
            generated.append(correction)
            if eos_token_id is not None and correction == eos_token_id:
                finished = True
                draft_past = None
                draft_next_logits = None
            else:
                target_result = target_model(
                    input_ids=_as_batch([correction], target_device),
                    past_key_values=target_past,
                    use_cache=True,
                )
                target_past = target_result.past_key_values
                target_next_logits = target_result.logits[0, -1]

                # The draft cache contains the whole proposal block.  Remove
                # the rejecting proposal and invalidated suffix, then replay
                # the target correction so the next block starts in sync.
                committed_length = len(prompt_ids) + base_position + mismatch_index
                cropped = _crop_past_key_values(draft_past, committed_length)
                if cropped is None:
                    draft_past = None
                    draft_next_logits = None
                else:
                    draft_result = draft_model(
                        input_ids=_as_batch([correction], draft_device),
                        past_key_values=cropped,
                        use_cache=True,
                    )
                    draft_past = draft_result.past_key_values
                    draft_next_logits = draft_result.logits[0, -1]
        else:
            generated.extend(proposals)
            if use_batched_target_verification and proposals and not target_eos_accepted:
                target_next_logits = batched_target_logits[len(proposals) - 1]
            if proposals and (target_eos_accepted or (eos_token_id is not None and proposals[-1] == eos_token_id)):
                finished = True

        for row in round_events:
            row["accepted_prefix_length"] = accepted_prefix_length
        events.extend(round_events)
        round_index += 1

    return generated, events


def verify_greedy_equivalence(reference_ids: list[int], speculative_ids: list[int], prompt_id: int | str = 0) -> None:
    if reference_ids != speculative_ids:
        common = 0
        for left, right in zip(reference_ids, speculative_ids):
            if left != right:
                break
            common += 1
        raise AssertionError(
            f"Prompt {prompt_id}: target greedy and speculative IDs differ at generated position {common}; "
            f"reference length={len(reference_ids)}, speculative length={len(speculative_ids)}"
        )


@torch.inference_mode()
def teacher_forced_scores(
    draft_model: Any,
    target_model: Any,
    prompt_ids: list[int],
    reference_ids: list[int],
    prompt_id: int | str = 0,
) -> list[dict[str, Any]]:
    """Score each reference token under both models in one forward per model."""
    if not reference_ids:
        return []
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    prefix_length = len(prompt_ids)
    scoring_ids = prompt_ids + reference_ids[:-1]
    draft_logits = draft_model(input_ids=_as_batch(scoring_ids, draft_device), use_cache=False).logits[0]
    target_logits = target_model(input_ids=_as_batch(scoring_ids, target_device), use_cache=False).logits[0]
    rows: list[dict[str, Any]] = []
    for index, token_id in enumerate(reference_ids):
        position = prefix_length - 1 + index
        draft_stats = distribution_stats(draft_logits[position], token_id)
        target_stats = distribution_stats(target_logits[position], token_id)
        rows.append({
            "prompt_id": prompt_id,
            "output_token_position": index,
            "target_token_id": token_id,
            "draft_greedy_token_id": draft_stats["greedy_token_id"],
            "target_greedy_token_id": target_stats["greedy_token_id"],
            "draft_target_disagreement": draft_stats["greedy_token_id"] != target_stats["greedy_token_id"],
            "draft_matches_target_token": draft_stats["greedy_token_id"] == token_id,
            "target_matches_target_token": target_stats["greedy_token_id"] == token_id,
            "draft_logprob": draft_stats["logprob"],
            "target_logprob": target_stats["logprob"],
            "draft_entropy": draft_stats["entropy"],
            "target_entropy": target_stats["entropy"],
        })
    return rows
