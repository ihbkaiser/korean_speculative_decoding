from __future__ import annotations

import torch
import json

from src.alignment import boundary_f1, token_ids_to_character_offsets
from src.models import TokenizerCompatibilityError, assert_tokenizer_compatible
from src.speculative_decoding import greedy_generate, speculative_greedy, verify_greedy_equivalence


class CharTokenizer:
    is_fast = True
    bos_token = None
    eos_token = "<eos>"
    unk_token = None
    sep_token = None
    pad_token = None
    cls_token = None
    mask_token = None
    additional_special_tokens = []
    bos_token_id = None
    eos_token_id = 0
    unk_token_id = None
    sep_token_id = None
    pad_token_id = None
    cls_token_id = None
    mask_token_id = None
    all_special_ids = [0]

    def __init__(self, offset_shift: int = 0, wrong_offsets: bool = False):
        self.offset_shift = offset_shift
        self.wrong_offsets = wrong_offsets
        self.backend_tokenizer = type("Backend", (), {
            "to_str": lambda _self: json.dumps({"offset_shift": offset_shift, "wrong_offsets": wrong_offsets})
        })()

    def get_vocab(self):
        return {chr(i): i + 1 + self.offset_shift for i in range(128)}

    def get_added_vocab(self):
        return {"<eos>": 0}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False, max_length=None):
        ids = [ord(char) + 1 + self.offset_shift for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            text = text[:max_length]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = [(i, i + 1) for i in range(len(text))]
            if self.wrong_offsets and offsets:
                offsets[0] = (0, min(2, len(text)))
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        chars = []
        for token_id in ids:
            if token_id == 0 and skip_special_tokens:
                continue
            if token_id == 0:
                chars.append("<eos>")
            else:
                chars.append(chr(token_id - 1 - self.offset_shift))
        return "".join(chars)


class NonRoundtripCharTokenizer(CharTokenizer):
    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if result["input_ids"]:
            result["input_ids"][-1] += 1
        return result


class ScriptedModel(torch.nn.Module):
    """Tiny cache-aware causal model whose next token depends on output index."""

    def __init__(self, scripted_output, prompt_length=2, vocab_size=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.scripted_output = list(scripted_output)
        self.prompt_length = prompt_length
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        past_length = int(past_key_values or 0)
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -20.0, device=input_ids.device)
        for local_position in range(input_ids.shape[1]):
            absolute_position = past_length + local_position
            output_index = absolute_position - (self.prompt_length - 1)
            if output_index < 0:
                token_id = self.scripted_output[0]
            elif output_index < len(self.scripted_output):
                token_id = self.scripted_output[output_index]
            else:
                token_id = self.scripted_output[-1]
            logits[0, local_position, token_id] = 20.0
        new_cache = past_length + input_ids.shape[1]
        return type("ModelOutput", (), {"logits": logits, "past_key_values": new_cache if use_cache else None})()


def test_tokenizer_compatibility_pass_and_fail():
    assert assert_tokenizer_compatible(CharTokenizer(), CharTokenizer())["compatible"]
    try:
        assert_tokenizer_compatible(CharTokenizer(), CharTokenizer(offset_shift=1))
    except TokenizerCompatibilityError:
        pass
    else:
        raise AssertionError("Expected tokenizer mismatch to fail")
    try:
        assert_tokenizer_compatible(CharTokenizer(), CharTokenizer(wrong_offsets=True))
    except TokenizerCompatibilityError:
        pass
    else:
        raise AssertionError("Expected offset mismatch to fail")


def test_target_greedy_equals_speculative_output_and_first_rejection_logging():
    target = ScriptedModel([1, 2, 3, 4])
    draft = ScriptedModel([1, 0, 0, 0])
    prompt = [6, 6]
    reference = greedy_generate(target, prompt, max_new_tokens=8, eos_token_id=4)
    speculative, events = speculative_greedy(draft, target, prompt, max_new_tokens=8, eos_token_id=4, k=4)
    verify_greedy_equivalence(reference, speculative)
    assert reference == [1, 2, 3, 4]
    assert events[0]["accepted"] is True
    assert events[1]["is_first_rejection"] is True
    assert events[1]["accepted_prefix_length"] == 1
    assert events[1]["rejected"] is True
    assert events[2]["invalidated_after_first_rejection"] is True
    assert events[2]["rejected"] is None
    assert events[1]["first_rejection_output_position"] == 1


def test_boundary_alignment_calculation():
    assert boundary_f1([], []) == 1.0
    assert boundary_f1([1], []) == 0.0
    assert boundary_f1([1, 2], [1, 3]) == 0.5
    assert boundary_f1([1, 2], [1, 2]) == 1.0


def test_token_to_character_offset_roundtrip():
    tokenizer = CharTokenizer()
    text = "한국어 eojeol"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [0]
    alignment = token_ids_to_character_offsets(tokenizer, ids)
    assert alignment["text"] == text
    assert alignment["offsets"][:-1] == [(i, i + 1) for i in range(len(text))]
    assert alignment["offsets"][-1] == (0, 0)
    assert alignment["roundtrip_exact"] is True

    fallback = token_ids_to_character_offsets(NonRoundtripCharTokenizer(), ids[:-1])
    assert fallback["text"] == text
    assert fallback["roundtrip_exact"] is False
    assert fallback["offsets"] == [(i, i + 1) for i in range(len(text))]
