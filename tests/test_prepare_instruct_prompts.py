from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rebuttal/prepare_instruct_prompts.py"
SPEC = importlib.util.spec_from_file_location("prepare_instruct_prompts", SCRIPT)
PROMPT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROMPT_MODULE)


class TinyBaseTokenizer:
    def decode(self, token_ids, **kwargs):
        assert token_ids == [17, 18]
        assert kwargs["skip_special_tokens"] is False
        assert kwargs["clean_up_tokenization_spaces"] is False
        return "앞 문장입니다."


class TinyChatTokenizer:
    chat_template = "fixture"

    def __init__(self):
        self.enable_thinking_values = []

    def _render(self, messages):
        return (
            "<|im_start|>user\n"
            + messages[0]["content"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.enable_thinking_values.append(enable_thinking)
        assert add_generation_prompt is True
        assert enable_thinking is False
        rendered = self._render(messages)
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def test_renders_short_token_consistent_prompt_ending_in_assistant_header():
    chat_tokenizer = TinyChatTokenizer()
    rows = PROMPT_MODULE.render_prompt_rows(
        [
            {
                "doc_id": "doc-1",
                "source_index": 12,
                "pool_index": 0,
                "pool_split": "common_20k",
                "prompt_ids": [17, 18],
            }
        ],
        base_tokenizer=TinyBaseTokenizer(),
        chat_tokenizer=chat_tokenizer,
        max_prompt_tokens=512,
    )
    rendered = rows[0]["text"]
    token_ids = chat_tokenizer(rendered, add_special_tokens=False)["input_ids"]
    messages = [
        {
            "role": "user",
            "content": f"{PROMPT_MODULE.USER_INSTRUCTION}\n\n앞 문장입니다.",
        }
    ]
    assert len(token_ids) < 512
    assert token_ids == chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert rendered.endswith("<|im_start|>assistant\n")
    assert chat_tokenizer.enable_thinking_values and all(
        value is False for value in chat_tokenizer.enable_thinking_values
    )
