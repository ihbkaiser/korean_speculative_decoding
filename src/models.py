"""Model loading, tokenizer compatibility, and distribution statistics."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


class TokenizerCompatibilityError(RuntimeError):
    """Raised when two models cannot safely share token IDs and offsets."""


def _token_string(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_token_string(item) for item in value]
    return getattr(value, "content", str(value))


def _special_signature(tokenizer: Any) -> dict[str, Any]:
    fields = (
        "bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
        "cls_token", "mask_token", "additional_special_tokens",
    )
    result: dict[str, Any] = {}
    for field in fields:
        result[field] = _token_string(getattr(tokenizer, field, None))
        result[f"{field}_id"] = getattr(tokenizer, f"{field}_id", None)
    return result


def compare_tokenizers(
    draft_tokenizer: Any,
    target_tokenizer: Any,
    probes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare tokenizer IDs, special tokens, and fast-tokenizer offsets."""
    probe_texts = list(probes or [
        "한국어 형태소 분석과 토큰 경계",
        "서울은 대한민국의 수도이다.",
        "English mixed with 한글 and numbers 3090.",
        "띄어쓰기\n여러 줄 테스트",
    ])
    reasons: list[str] = []
    if not getattr(draft_tokenizer, "is_fast", False) or not getattr(target_tokenizer, "is_fast", False):
        reasons.append("both tokenizers must be fast tokenizers to provide offsets")
    draft_class = f"{type(draft_tokenizer).__module__}.{type(draft_tokenizer).__name__}"
    target_class = f"{type(target_tokenizer).__module__}.{type(target_tokenizer).__name__}"
    if draft_class != target_class:
        reasons.append("tokenizer implementation classes differ")

    draft_backend = getattr(draft_tokenizer, "backend_tokenizer", None)
    target_backend = getattr(target_tokenizer, "backend_tokenizer", None)
    backend_hashes: dict[str, str | None] = {"draft": None, "target": None}
    backend_equal = False
    if draft_backend is None or target_backend is None:
        reasons.append("fast-tokenizer backend serialization is unavailable")
    else:
        draft_serialized = draft_backend.to_str()
        target_serialized = target_backend.to_str()
        backend_hashes = {
            "draft": hashlib.sha256(draft_serialized.encode("utf-8")).hexdigest(),
            "target": hashlib.sha256(target_serialized.encode("utf-8")).hexdigest(),
        }
        backend_equal = draft_serialized == target_serialized
        if not backend_equal:
            reasons.append("fast-tokenizer backend normalizer/model/pre-tokenizer configuration differs")

    draft_vocab = draft_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    vocab_equal = draft_vocab == target_vocab
    if not vocab_equal:
        reasons.append("token-to-ID vocabularies differ")
    draft_added = draft_tokenizer.get_added_vocab()
    target_added = target_tokenizer.get_added_vocab()
    added_equal = draft_added == target_added
    if not added_equal:
        reasons.append("added-token vocabularies differ")
    draft_special = _special_signature(draft_tokenizer)
    target_special = _special_signature(target_tokenizer)
    special_equal = draft_special == target_special
    if not special_equal:
        reasons.append("special-token strings or IDs differ")

    probe_results = []
    for text in probe_texts:
        try:
            left = draft_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            right = target_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids_equal = list(left["input_ids"]) == list(right["input_ids"])
            offsets_equal = [tuple(x) for x in left["offset_mapping"]] == [tuple(x) for x in right["offset_mapping"]]
        except Exception as exc:  # report the actual compatibility failure
            ids_equal = offsets_equal = False
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        probe_results.append({"text": text, "ids_equal": ids_equal, "offsets_equal": offsets_equal, "error": error})
        if not ids_equal or not offsets_equal:
            reasons.append("probe encoding or offsets differ")
            break

    return {
        "compatible": not reasons,
        "draft_class": draft_class,
        "target_class": target_class,
        "backend_serialization_equal": backend_equal,
        "backend_sha256": backend_hashes,
        "vocab_size_draft": len(draft_vocab),
        "vocab_size_target": len(target_vocab),
        "vocab_equal": vocab_equal,
        "added_vocab_equal": added_equal,
        "special_tokens_equal": special_equal,
        "draft_special_tokens": draft_special,
        "target_special_tokens": target_special,
        "probes": probe_results,
        "reasons": sorted(set(reasons)),
    }


def assert_tokenizer_compatible(draft_tokenizer: Any, target_tokenizer: Any) -> dict[str, Any]:
    report = compare_tokenizers(draft_tokenizer, target_tokenizer)
    if not report["compatible"]:
        raise TokenizerCompatibilityError("Incompatible draft/target tokenizers: " + "; ".join(report["reasons"]))
    return report


def load_compatible_tokenizers(draft_name: str, target_name: str, **kwargs: Any):
    from transformers import AutoTokenizer

    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name, use_fast=True, **kwargs)
    target_tokenizer = AutoTokenizer.from_pretrained(target_name, use_fast=True, **kwargs)
    report = assert_tokenizer_compatible(draft_tokenizer, target_tokenizer)
    return target_tokenizer, report


def load_models(draft_name: str, target_name: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    common = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    draft_model = AutoModelForCausalLM.from_pretrained(draft_name, **common).to(device).eval()
    target_model = AutoModelForCausalLM.from_pretrained(target_name, **common).to(device).eval()
    if draft_model.config.vocab_size != target_model.config.vocab_size:
        raise TokenizerCompatibilityError("Draft and target model vocabulary sizes differ")
    return draft_model, target_model


def distribution_stats(logits: Any, token_id: int | None = None) -> dict[str, float | int]:
    """Stable greedy token, log-probability, and entropy from one logits row."""
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    greedy_id = int(torch.argmax(logits, dim=-1).item())
    selected = greedy_id if token_id is None else int(token_id)
    return {
        "greedy_token_id": greedy_id,
        "logprob": float(log_probs[selected].item()),
        "entropy": float(entropy.item()),
    }
