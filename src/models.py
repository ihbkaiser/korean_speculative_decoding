"""Model loading, tokenizer compatibility, and distribution statistics."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


class TokenizerCompatibilityError(RuntimeError):
    """Raised when two models cannot safely share token IDs and offsets."""


def config_vocab_size(config: Any) -> int:
    """Read vocabulary size from plain or multimodal Transformers configs."""
    value = getattr(config, "vocab_size", None)
    if value is not None:
        return int(value)
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        value = text_config.get("vocab_size")
    else:
        value = getattr(text_config, "vocab_size", None)
    if value is None:
        raise AttributeError(f"No vocab_size found in {type(config).__name__}")
    return int(value)


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


def _added_vocab(tokenizer: Any) -> dict[str, Any] | None:
    """Return added-token vocabulary when the tokenizer API exposes it."""
    getter = getattr(tokenizer, "get_added_vocab", None)
    if callable(getter):
        return dict(getter())
    encoder = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(encoder, dict):
        return dict(encoder)
    # MistralCommonBackend keeps its special vocabulary in mistral-common and
    # intentionally does not implement the legacy Transformers method. Its
    # special-token signature and full vocab are compared separately.
    return None


def compare_tokenizers(
    draft_tokenizer: Any,
    target_tokenizer: Any,
    probes: Iterable[str] | None = None,
    *,
    require_offsets: bool = True,
) -> dict[str, Any]:
    """Compare tokenizer IDs, special tokens, and fast-tokenizer offsets."""
    probe_texts = list(probes or [
        "한국어 형태소 분석과 토큰 경계",
        "서울은 대한민국의 수도이다.",
        "English mixed with 한글 and numbers 3090.",
        "띄어쓰기\n여러 줄 테스트",
    ])
    reasons: list[str] = []
    offsets_supported = bool(
        getattr(draft_tokenizer, "is_fast", False)
        and getattr(target_tokenizer, "is_fast", False)
    )
    if require_offsets and not offsets_supported:
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
        if require_offsets:
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
    draft_added = _added_vocab(draft_tokenizer)
    target_added = _added_vocab(target_tokenizer)
    added_equal = (
        draft_added == target_added
        if draft_added is not None and target_added is not None
        else type(draft_tokenizer) is type(target_tokenizer)
    )
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
            if offsets_supported:
                left = draft_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                right = target_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                ids_equal = list(left["input_ids"]) == list(right["input_ids"])
                offsets_equal = [tuple(x) for x in left["offset_mapping"]] == [tuple(x) for x in right["offset_mapping"]]
            else:
                left = draft_tokenizer(text, add_special_tokens=False)
                right = target_tokenizer(text, add_special_tokens=False)
                ids_equal = list(left["input_ids"]) == list(right["input_ids"])
                offsets_equal = None
        except Exception as exc:  # report the actual compatibility failure
            ids_equal = offsets_equal = False
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        probe_results.append({"text": text, "ids_equal": ids_equal, "offsets_equal": offsets_equal, "error": error})
        if not ids_equal or (require_offsets and not offsets_equal):
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
        "offsets_supported": offsets_supported,
        "offsets_required": require_offsets,
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


def load_models(
    draft_name: str,
    target_name: str,
    device: str = "cuda",
    *,
    draft_revision: str | None = None,
    target_revision: str | None = None,
    token: str | None = None,
    dtype: str = "float16",
    attention_backend: str = "sdpa",
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported model dtype: {dtype}")
    if attention_backend not in {"sdpa", "flash_attention_2", "eager"}:
        raise ValueError(
            "Unsupported attention backend: "
            f"{attention_backend!r}; expected sdpa, flash_attention_2, or eager"
        )
    common = {"torch_dtype": dtype_map[dtype], "low_cpu_mem_usage": True}
    common["attn_implementation"] = attention_backend
    if token:
        common["token"] = token
    draft_kwargs = dict(common)
    target_kwargs = dict(common)
    if draft_revision:
        draft_kwargs["revision"] = draft_revision
    if target_revision:
        target_kwargs["revision"] = target_revision

    def load_one(model_name: str, kwargs: dict[str, Any]):
        # Ministral 3 is exposed by Transformers as a multimodal conditional
        # generation class, even for text-only prompts; AutoModelForCausalLM
        # does not register Mistral3Config.
        config_kwargs = {
            key: kwargs[key]
            for key in ("revision", "token")
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        if getattr(config, "model_type", None) == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            model_class = Mistral3ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        return model_class.from_pretrained(model_name, **kwargs).to(device).eval()

    draft_model = load_one(draft_name, draft_kwargs)
    target_model = load_one(target_name, target_kwargs)
    if config_vocab_size(draft_model.config) != config_vocab_size(target_model.config):
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
    selected_logit = logits.float()[selected]
    rank = int((logits.float() > selected_logit).sum().item()) + 1
    return {
        "greedy_token_id": greedy_id,
        "logprob": float(log_probs[selected].item()),
        "entropy": float(entropy.item()),
        "top1_logprob": float(log_probs[greedy_id].item()),
        "rank": rank,
    }
