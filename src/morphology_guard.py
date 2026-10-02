"""Exact H2/H3-compatible morphology detector for speculative proposal blocks.

This module is intentionally CPU-only. It projects the decoder's committed
text plus candidate IDs through the same tokenizer-offset and H2 classifier
used by the saved analyses. Any incomplete alignment disables guarding for
that proposal block.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any

from .h2_analysis import _classify_token, _span_tokenizer_ids


INTERNAL_BOUNDARY_TYPES = {
    "NOMINAL_TO_PARTICLE",
    "PREDICATE_TO_ENDING",
    "ENDING_TO_ENDING",
    "LEXICAL_TO_LEXICAL",
    "OTHER",
}


def _nonwhite_span(text: str, start: int, end: int) -> tuple[int, int, bool]:
    visible = [i for i in range(max(0, start), min(len(text), end)) if not text[i].isspace()]
    if not visible:
        return start, end, True
    return visible[0], visible[-1] + 1, False


def project_candidate_block(
    tokenizer: Any,
    kiwi: Any,
    prompt_ids: list[int],
    committed_ids: list[int],
    proposal_ids: list[int],
    prompt_id: int,
    round_index: int,
    pair: str,
    allow_local_exact_spans: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify proposed target-token slots using exact decoded character spans.

    By default, the online guard requires the complete block to round-trip
    exactly. Offline measurement may set ``allow_local_exact_spans`` to accept
    an individually exact token span when another token elsewhere in the
    proposed block prevents whole-block retokenization from being exact.
    """
    started = time.perf_counter()
    from scripts.analyze_h3 import boundary_type

    full_ids = [int(x) for x in prompt_ids + committed_ids + proposal_ids]
    prompt_length = len(prompt_ids)
    alignment = _span_tokenizer_ids(tokenizer, full_ids, prompt_length)
    text = str(alignment["text"])
    prompt_spans = [
        tuple(alignment["fallback_offsets"][i])
        for i in range(prompt_length)
        if int(full_ids[i]) not in set(getattr(tokenizer, "all_special_ids", []) or [])
        and alignment["fallback_offsets"][i][1] > alignment["fallback_offsets"][i][0]
    ]
    continuation_origin = max((int(end) for _, end in prompt_spans), default=0)
    continuation = text[continuation_origin:]

    eojeols = [
        {"eojeol_id": idx, "text": match.group(0), "char_start": match.start(), "char_end": match.end()}
        for idx, match in enumerate(re.finditer(r"\S+", continuation, flags=re.UNICODE))
    ]
    global_morphemes: list[dict[str, Any]] = []
    bad_morphemes: set[int] = set()
    for morph in kiwi.tokenize(continuation):
        start = int(morph.start)
        end = start + int(morph.len)
        idx = len(global_morphemes)
        global_morphemes.append({"surface": str(morph.form), "pos": str(morph.tag), "start": start, "end": end})
        if start < 0 or end <= start or end > len(continuation):
            bad_morphemes.add(idx)

    local_sequences: dict[int, list[dict[str, Any]]] = {}
    for eo in eojeols:
        seq: list[dict[str, Any]] = []
        for idx, morph in enumerate(kiwi.tokenize(eo["text"])):
            start = int(morph.start)
            end = start + int(morph.len)
            seq.append({
                "surface": str(morph.form), "pos": str(morph.tag), "start": eo["char_start"] + start,
                "end": eo["char_start"] + end, "local_start": start, "local_end": end, "index": idx,
            })
        local_sequences[int(eo["eojeol_id"])] = seq

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    records: list[dict[str, Any]] = []
    for slot, token_id in enumerate(proposal_ids, start=1):
        full_position = prompt_length + len(committed_ids) + slot - 1
        absolute_start, absolute_end = alignment["spans"][full_position]
        span_exact = full_position in alignment["exact_ids"]
        crosses_prompt = absolute_start < continuation_origin and absolute_end > 0 and token_id not in special_ids
        local_start = max(0, int(absolute_start) - continuation_origin)
        local_end = max(0, int(absolute_end) - continuation_origin)
        if local_start > len(continuation) or local_end > len(continuation):
            span_exact = False
            local_start = min(local_start, len(continuation))
            local_end = min(local_end, len(continuation))
        if token_id in special_ids:
            token_surface = str(tokenizer.convert_ids_to_tokens(int(token_id)))
            token_start, token_end, whitespace_only = local_start, local_end, True
        else:
            token_surface = continuation[local_start:local_end]
            token_start, token_end, whitespace_only = _nonwhite_span(continuation, local_start, local_end)

        morph_class, eo_ids, morph_ids, reason = _classify_token(
            span_exact=span_exact and (alignment["roundtrip_exact"] or allow_local_exact_spans),
            crosses_prompt=crosses_prompt,
            whitespace_only=whitespace_only,
            token_start=token_start,
            token_end=token_end,
            eojeols=eojeols,
            morphemes=global_morphemes,
            bad_morpheme_indices=bad_morphemes,
        )
        if token_id in special_ids:
            morph_class, reason = "KIWI_COMPLEX", "special_token_without_visible_character_span"

        crossed: list[dict[str, Any]] = []
        eo_id = eo_ids[0] if len(eo_ids) == 1 else None
        if morph_class == "CROSS_MORPHEME" and eo_id is not None:
            seq = local_sequences.get(int(eo_id), [])
            eo = eojeols[int(eo_id)]
            # Ensure the whole H2-overlapped morpheme sequence agrees with
            # H3's eojeol-local Kiwi parse, including exact character spans.
            h2_overlaps = [global_morphemes[i] for i in morph_ids]
            h3_overlaps = [m for m in seq if max(0, min(token_end, m["end"]) - max(token_start, m["start"])) > 0]
            exact_alignment = len(h2_overlaps) == len(h3_overlaps) and all(
                (a["surface"], a["pos"], a["start"], a["end"])
                == (b["surface"], b["pos"], b["start"], b["end"])
                for a, b in zip(h2_overlaps, h3_overlaps)
            )
            if exact_alignment:
                for left, right in zip(seq, seq[1:]):
                    if left["local_end"] == right["local_start"] and token_start < left["end"] < token_end:
                        crossed.append({
                            "left_surface": left["surface"], "left_pos": left["pos"],
                            "right_surface": right["surface"], "right_pos": right["pos"],
                            "boundary_char": left["end"],
                            "environment": boundary_type(left["pos"], right["pos"]),
                        })
            else:
                reason = "h2_h3_morpheme_span_disagreement"
                morph_class = "KIWI_COMPLEX"

        if morph_class == "CROSS_MORPHEME" and eo_id is None:
            reason = "cross_morpheme_not_within_single_eojeol"
            morph_class = "CROSS_EOJEOL"

        if not alignment["roundtrip_exact"] and not allow_local_exact_spans:
            status = "INVALID_ROUNDTRIP"
        elif not span_exact:
            status = "INVALID_TOKEN_SPAN"
        elif morph_class in {"KIWI_COMPLEX", "CROSS_EOJEOL"}:
            status = "INVALID_MORPHOLOGY"
        elif morph_class == "CROSS_MORPHEME" and len(crossed) != len(morph_ids) - 1:
            status = "INVALID_BOUNDARY_ALIGNMENT"
        else:
            status = "VALID"

        records.append({
            "pair": pair, "prompt_id": int(prompt_id), "round_index": int(round_index),
            "proposal_slot": int(slot), "full_token_position": int(full_position),
            "token_id": int(token_id), "token_surface": token_surface,
            "token_char_start": int(token_start), "token_char_end": int(token_end),
            "eojeol_id": eo_id,
            "eojeol": eojeols[int(eo_id)]["text"] if eo_id is not None else None,
            "h2_morph_class": morph_class, "num_boundaries_crossed": len(crossed),
            "crossed_boundary_sequence": "|".join(x["environment"] for x in crossed),
            "crossed_fine_pos_sequence": "|".join(f"{x['left_pos']}→{x['right_pos']}" for x in crossed),
            "kiwi_sequence": [
                {"surface": m["surface"], "fine_pos": m["pos"], "start": m["start"], "end": m["end"]}
                for m in (local_sequences.get(int(eo_id), []) if eo_id is not None else [])
            ],
            "alignment_roundtrip_exact": bool(alignment["roundtrip_exact"]),
            "span_exact": bool(span_exact), "classification_reason": reason, "detector_status": status,
        })
    return records, {
        "projection_seconds": time.perf_counter() - started,
        "roundtrip_exact": bool(alignment["roundtrip_exact"]),
        "continuation_text": continuation,
    }


class MorphologyGuard:
    """One-prompt stateless guard controller used by the greedy decoder."""

    def __init__(
        self,
        variant: str,
        tokenizer: Any,
        kiwi: Any,
        pair: str,
        slot_distribution: dict[int, float] | None = None,
        entropy_threshold: float | None = None,
        entropy_slot_probabilities: dict[int, float] | None = None,
        seed: int = 20260930,
    ) -> None:
        self.variant = variant
        self.tokenizer = tokenizer
        self.kiwi = kiwi
        self.pair = pair
        self.slot_distribution = slot_distribution or {}
        self.entropy_threshold = entropy_threshold
        self.entropy_slot_probabilities = entropy_slot_probabilities or {}
        self.seed = int(seed)
        self.requires_entropy = variant == "ENTROPY_MATCHED_GUARD"
        if self.requires_entropy and entropy_threshold is None:
            raise ValueError("Entropy-matched guard requires a Wikipedia-calibrated threshold")

    def inspect(
        self,
        prompt_ids: list[int],
        committed_ids: list[int],
        proposal_ids: list[int],
        proposal_entropies: list[float] | None,
        prompt_id: int,
        round_index: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.variant == "ENTROPY_MATCHED_GUARD":
            values = proposal_entropies or []
            crossing: tuple[int, float] | None = None
            for slot, entropy in enumerate(values, start=1):
                if slot < len(proposal_ids) and entropy >= float(self.entropy_threshold):
                    crossing = (slot, float(entropy))
                    break
            if crossing is not None:
                slot, entropy = crossing
                keep_probability = float(self.entropy_slot_probabilities.get(slot, 0.0))
                material = f"entropy|{self.seed}|{self.pair}|{prompt_id}|{round_index}|{slot}".encode()
                random_value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
                selected = random_value < keep_probability
                reason = "draft_entropy_threshold" if selected else "entropy_threshold_thinned"
                return {
                    "triggered": selected, "guard_slot": slot if selected else None,
                    "verify_slots": slot - 1 if selected else len(proposal_ids),
                    "reason": reason, "candidate_records": [],
                    "activation_evidence": {"slot": slot, "draft_entropy": entropy, "threshold": self.entropy_threshold, "keep_probability": keep_probability, "deterministic_uniform": random_value},
                    "controller_seconds": time.perf_counter() - started,
                    "projection_seconds": 0.0, "invalid_candidates": 0,
                    "candidate_count": max(0, len(proposal_ids) - 1),
                }
            return {
                "triggered": False, "guard_slot": None, "verify_slots": len(proposal_ids),
                "reason": "no_entropy_threshold_crossing", "candidate_records": [],
                "controller_seconds": time.perf_counter() - started,
                "projection_seconds": 0.0, "invalid_candidates": 0, "candidate_count": 0,
            }

        records, projection = project_candidate_block(
            self.tokenizer, self.kiwi, prompt_ids, committed_ids, proposal_ids, prompt_id, round_index, self.pair,
        )
        internal = [r for r in records if int(r["proposal_slot"]) < len(proposal_ids)]
        valid = [r for r in internal if r["detector_status"] == "VALID"]
        if self.variant in {"NP_BOUNDARY_GUARD", "RANDOM_MATCHED_GUARD"}:
            matches = [
                r for r in valid if r["h2_morph_class"] == "CROSS_MORPHEME"
                and int(r["num_boundaries_crossed"]) == 1
                and r["crossed_boundary_sequence"] == "NOMINAL_TO_PARTICLE"
            ]
        elif self.variant == "ALL_MORPH_BOUNDARY_GUARD":
            matches = [
                r for r in valid if r["h2_morph_class"] == "CROSS_MORPHEME"
                and int(r["num_boundaries_crossed"]) == 1
            ]
        else:
            raise ValueError(f"Unknown guard variant: {self.variant}")
        chosen: dict[str, Any] | None = min(matches, key=lambda r: int(r["proposal_slot"])) if matches else None
        activation_evidence = chosen
        if chosen is not None and self.variant == "RANDOM_MATCHED_GUARD":
            allowed = [slot for slot in range(1, len(proposal_ids))]
            weights = [float(self.slot_distribution.get(slot, 0.0)) for slot in allowed]
            total = sum(weights)
            if not allowed:
                chosen = None
            else:
                if total <= 0:
                    weights = [1.0 / len(allowed)] * len(allowed)
                else:
                    weights = [weight / total for weight in weights]
                material = f"{self.seed}|{self.pair}|{prompt_id}|{round_index}".encode()
                rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                import random
                rng = random.Random(rng_seed)
                chosen_slot = rng.choices(allowed, weights=weights, k=1)[0]
                chosen = next((r for r in internal if int(r["proposal_slot"]) == chosen_slot), None)
        invalid = sum(r["detector_status"] != "VALID" for r in internal)
        triggered = chosen is not None
        return {
            "triggered": triggered,
            "guard_slot": int(chosen["proposal_slot"]) if triggered else None,
            "verify_slots": int(chosen["proposal_slot"]) - 1 if triggered else len(proposal_ids),
            "reason": "valid_guard" if triggered else ("no_valid_guard_point" if projection["roundtrip_exact"] else "ambiguous_target_projection"),
            "chosen_record": chosen,
            "activation_evidence": activation_evidence,
            "candidate_records": records,
            "controller_seconds": time.perf_counter() - started,
            "projection_seconds": float(projection["projection_seconds"]),
            "invalid_candidates": int(invalid), "candidate_count": len(internal),
        }
