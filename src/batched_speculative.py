"""Microbatched greedy SD with one target verification forward per round.

The target's block logits decide acceptance, correction and the bonus token.
There is no scalar-reference regeneration. Like other block verifiers this is
greedy-equivalent in exact arithmetic, not a promise of bitwise scalar parity.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import torch


def distribution_stats_batch(logits: torch.Tensor, token_ids: torch.Tensor | None = None):
    """Compute audit statistics on-device; callers transfer compact rows once."""
    values = logits.float()
    greedy = values.argmax(-1)
    selected = greedy if token_ids is None else token_ids
    log_probs = values.log_softmax(-1)
    selected_logits = values.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    stats = torch.stack((
        log_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1),
        -(log_probs.exp() * log_probs).sum(-1),
        log_probs.gather(-1, greedy.unsqueeze(-1)).squeeze(-1),
        (values > selected_logits.unsqueeze(-1)).sum(-1).float() + 1,
    ), dim=-1)
    return greedy, stats


def _padded(rows: list[list[int]], device: torch.device):
    width = max(map(len, rows))
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(rows):
        if tokens:
            ids[row, -len(tokens):] = torch.tensor(tokens, device=device)
            mask[row, -len(tokens):] = 1
    return ids, mask


@lru_cache(maxsize=64)
def _logits_keep_argument(model_class):
    names = inspect.signature(model_class.forward).parameters
    return next((name for name in ("logits_to_keep", "num_logits_to_keep") if name in names), None)


def _forward(model, ids, mask, past=None, *, logits_to_keep=0):
    positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
    kwargs = dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                  past_key_values=past, use_cache=True)
    # Transformers has used both names. Avoid projecting the full prefill
    # through the large LM head when the model explicitly supports slicing.
    keep_argument = _logits_keep_argument(type(model))
    if logits_to_keep and keep_argument:
        kwargs[keep_argument] = logits_to_keep
    return model(**kwargs)


def _compact_cache(past, mask: torch.Tensor, keep_lengths: list[int]):
    """Discard each row's rejected suffix and left-pad its committed KV prefix.

    Rows can accept different numbers of tokens. A single global cache.crop()
    cannot represent that: gather each row's valid prefix instead. Supports
    standard dense HF DynamicCache (4.x and 5.x) and legacy tuple KV caches.
    Sliding/linear caches with a different physical layout fail explicitly.
    """
    lengths = torch.tensor(keep_lengths, device=mask.device)
    width = max(keep_lengths)
    retain = mask.bool() & (mask.cumsum(-1) <= lengths[:, None])
    columns = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = torch.where(retain, columns, -1).sort(-1).values[:, -width:]
    new_mask = (indices >= 0).long()
    indices = indices.clamp_min(0)

    def gather(value):
        if value.ndim != 4 or value.shape[0] != mask.shape[0] or value.shape[-2] != mask.shape[1]:
            raise RuntimeError("Microbatch SD requires standard dense KV caches; "
                               "this model uses an unsupported sliding/linear cache layout")
        index = indices[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[-1])
        return value.gather(-2, index)

    if isinstance(past, tuple):
        past = tuple(tuple(gather(value) for value in layer) for layer in past)
    elif hasattr(past, "layers"):
        for layer in past.layers:
            layer.keys, layer.values = gather(layer.keys), gather(layer.values)
            # Dynamic sliding layers also track the physical sequence length
            # separately. At short contexts their KV layout is still dense.
            if hasattr(layer, "cumulative_length"):
                if not isinstance(layer.cumulative_length, int):
                    raise RuntimeError("Static/linear cache layers are not supported by microbatch SD")
                layer.cumulative_length = width
    elif hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for layer in range(len(past.key_cache)):
            past.key_cache[layer] = gather(past.key_cache[layer])
            past.value_cache[layer] = gather(past.value_cache[layer])
        if hasattr(past, "_seen_tokens"):
            past._seen_tokens = width
    else:
        raise RuntimeError(f"Unsupported microbatch KV cache: {type(past).__name__}")
    return past, new_mask


@torch.inference_mode()
def speculative_greedy_microbatch(
    draft_model: Any, target_model: Any, prompt_ids_batch: list[list[int]],
    max_new_tokens: int, eos_token_id: int | None, k: int = 4,
    prompt_ids: list[int | str] | None = None,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    """Draft B x K, verify B x (K+1), commit accepted prefixes plus correction.

    The last committed token is left uncached for the next verification call,
    fusing correction/bonus consumption into that call. Audit statistics stay
    on the GPU until a single compact transfer per round. Finished rows are
    masked, and no proposal past EOS or the token budget becomes an event.
    """
    if not prompt_ids_batch or any(not row for row in prompt_ids_batch):
        raise ValueError("prompt_ids_batch must contain nonempty prompts")
    if k < 1 or max_new_tokens < 0:
        raise ValueError("k must be positive and max_new_tokens non-negative")
    count = len(prompt_ids_batch)
    labels = list(range(count)) if prompt_ids is None else prompt_ids
    if len(labels) != count:
        raise ValueError("prompt_ids must have one label per prompt")
    outputs: list[list[int]] = [[] for _ in range(count)]
    events: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    if not max_new_tokens:
        return outputs, events
    device = next(target_model.parameters()).device
    if next(draft_model.parameters()).device != device:
        raise ValueError("Microbatch draft and target must share a device")
    initial, initial_mask = _padded(prompt_ids_batch, device)
    draft_result = _forward(draft_model, initial, initial_mask, logits_to_keep=1)
    draft_past, draft_mask = draft_result.past_key_values, initial_mask
    draft_logits = draft_result.logits[:, -1]
    target_past = None
    target_mask = torch.zeros((count, 0), dtype=torch.long, device=device)
    target_pending, pending_mask = initial, initial_mask
    contexts = [len(row) for row in prompt_ids_batch]
    active = [True] * count
    round_index = 0

    while any(active):
        block_k = min(k, max(max_new_tokens - len(row) for row in outputs))
        proposal_columns, draft_stats = [], []
        for slot in range(block_k):
            proposed, stats = distribution_stats_batch(draft_logits)
            proposal_columns.append(proposed)
            draft_stats.append(stats)
            if slot + 1 < block_k:
                draft_mask = torch.cat((draft_mask, torch.ones((count, 1), dtype=torch.long, device=device)), -1)
                draft_result = _forward(draft_model, proposed[:, None], draft_mask,
                                        draft_past, logits_to_keep=1)
                draft_past, draft_logits = draft_result.past_key_values, draft_result.logits[:, -1]
        proposals = torch.stack(proposal_columns, -1)
        target_input = torch.cat((target_pending, proposals), -1)
        target_mask = torch.cat((target_mask, pending_mask,
                                 torch.ones_like(proposals)), -1)
        target_result = _forward(target_model, target_input, target_mask,
                                 target_past, logits_to_keep=block_k + 1)
        target_past = target_result.past_key_values
        logits = target_result.logits[:, -(block_k + 1):]
        target_ids, target_stats = distribution_stats_batch(logits[:, :-1], proposals)
        bonus_ids = logits[:, -1].argmax(-1)
        slots = torch.arange(block_k, device=device)[None, :]
        budgets = torch.tensor([max_new_tokens - len(row) if live else 0
                                for row, live in zip(outputs, active)], device=device)
        valid = slots < budgets[:, None]
        if eos_token_id is not None:
            eos = proposals == eos_token_id
            valid &= (eos.cumsum(-1) - eos.long()) == 0
        accepted_counts = ((proposals == target_ids) & valid).long().cumprod(-1).sum(-1)
        limits = valid.sum(-1)
        # One transfer containing all IDs and audit summaries; no .item()
        # synchronization inside the vocabulary/proposal loop.
        packed = torch.cat((proposals[..., None].float(), target_ids[..., None].float(),
                            torch.stack(draft_stats, 1), target_stats,
                            bonus_ids[:, None, None].expand(-1, block_k, 1).float(),
                            accepted_counts[:, None, None].expand(-1, block_k, 1).float(),
                            limits[:, None, None].expand(-1, block_k, 1).float()), -1).cpu().tolist()

        new_contexts, draft_keep, target_keep, draft_suffixes, target_suffixes = [], [], [], [], []
        for row in range(count):
            old_length = contexts[row]
            base_position = len(outputs[row])
            if active[row]:
                accepted, limit = map(int, packed[row][0][-2:])
                eos_slot = (limit - 1 if int(packed[row][limit - 1][0]) == eos_token_id else None)
                rejection = accepted if accepted < limit else None
                committed = [int(packed[row][j][0]) for j in range(accepted)]
                if rejection is not None:
                    committed.append(int(packed[row][rejection][1]))
                elif (eos_slot is None and base_position + accepted < max_new_tokens):
                    committed.append(int(packed[row][0][-3]))
                if eos_token_id is not None and eos_token_id in committed:
                    committed = committed[:committed.index(eos_token_id) + 1]
                outputs[row].extend(committed)
                first_rejection = None if rejection is None else base_position + rejection
                for slot in range(limit):
                    p, t, dl, de, dt, dr, tl, te, tt, tr = packed[row][slot][:10]
                    invalidated = rejection is not None and slot > rejection
                    events[row].append({
                        "prompt_id": labels[row], "round_index": round_index,
                        "draft_token_id": int(p), "target_greedy_token_id": int(t),
                        "accepted": p == t and not invalidated,
                        "rejected": None if invalidated else p != t,
                        "is_first_rejection": slot == rejection,
                        "invalidated_after_first_rejection": invalidated,
                        "proposal_position": slot + 1, "output_token_position": base_position + slot,
                        "draft_logprob": dl, "draft_entropy": de, "draft_top1_logprob": dt,
                        "target_logprob": tl, "target_entropy": te, "target_top1_logprob": tt,
                        "target_rank_of_proposed_token": int(tr),
                        "accepted_prefix_length": accepted,
                        "first_rejection_output_position": first_rejection,
                    })
                active[row] = (len(outputs[row]) < max_new_tokens
                               and (not outputs[row] or outputs[row][-1] != eos_token_id))
            new_length = len(prompt_ids_batch[row]) + len(outputs[row])
            new_contexts.append(new_length)
            # Target has scored all proposals; draft has consumed K-1 of them.
            tkeep = new_length - 1
            dkeep = min(tkeep, old_length + block_k - 1)
            target_keep.append(tkeep)
            draft_keep.append(dkeep)
            full_ids = prompt_ids_batch[row] + outputs[row]
            target_suffixes.append(full_ids[tkeep:])
            draft_suffixes.append(full_ids[dkeep:])
        if not any(active):
            break
        target_past, target_mask = _compact_cache(target_past, target_mask, target_keep)
        draft_past, draft_mask = _compact_cache(draft_past, draft_mask, draft_keep)
        target_pending, pending_mask = _padded(target_suffixes, device)
        draft_input, suffix_mask = _padded(draft_suffixes, device)
        draft_mask = torch.cat((draft_mask, suffix_mask), -1)
        draft_result = _forward(draft_model, draft_input, draft_mask, draft_past, logits_to_keep=1)
        draft_past, draft_logits = draft_result.past_key_values, draft_result.logits[:, -1]
        contexts = new_contexts
        round_index += 1

    return outputs, events
