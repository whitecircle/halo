#!/usr/bin/env python3
"""Offline-GRPO text prep contracts.

1. Completion render: completions are rendered as ``template(prompt + completion)`` minus the
   rendered-prompt prefix — never standalone. Standalone rendering raises on strict templates
   (Qwen3.5 rejects assistant-only message lists) and injects a mid-sequence BOS on BOS-emitting
   templates.
2. The eval-generation prompt's null cap means no cap, not HF's ``model_max_length``.

Usage:
    python tests/cpu/data/test_offline_grpo_render.py
"""

from functools import partial

import pytest
from accelerate import PartialState
from datasets import Dataset

PartialState()

from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.data.pipeline.text_rows import prepare_generative_row
from halo.trainers.grpo.offline.offline_rules import reject_unsupported_reference_input
from scripts.training.grpo.offline import render_offline_grpo_row


class StrictTemplateTokenizer:
    """Qwen3.5-style STRICT template: raises on a conversation not opening with system/user."""

    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if not messages or messages[0]["role"] not in ("system", "user"):
            raise ValueError("Conversation must start with a system or user turn (strict template)")
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


class BosTemplateTokenizer:
    """Lenient BOS-emitting template: standalone completion renders gain a leading BOS."""

    bos_token = "<bos>"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self.bos_token + "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


ROW = {
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "completions": [
        [{"role": "assistant", "content": "4"}],
        [{"role": "assistant", "content": "five"}],
    ],
    "rewards": [1.0, 0.0],
}


def test_offline_grpo_render_survives_strict_template():
    """A bare ``apply_chat_template([assistant-msg])`` raises on strict templates."""
    out = render_offline_grpo_row(
        {**ROW, "completions": [list(c) for c in ROW["completions"]]}, StrictTemplateTokenizer(), None
    )
    assert out["prompt"] == "<|user|>What is 2+2?<|end|>"
    assert out["completions"] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]


def test_offline_grpo_rendered_supplied_references_are_refused_by_the_trainer():
    scores = [[-0.5, -0.75], [-1.25]]
    raw = Dataset.from_list([{**ROW, REF_PER_TOKEN_LOGPS_COLUMN: scores}])
    rendered = raw.map(partial(render_offline_grpo_row, tokenizer=StrictTemplateTokenizer(), tools_field=None))
    assert rendered["completions"][0] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]
    with pytest.raises(ValueError, match="Supplied ref_per_token_logps"):
        reject_unsupported_reference_input(rendered, None)


def test_offline_grpo_render_no_mid_sequence_bos():
    """A standalone completion render carries the template's leading BOS, which would put a BOS
    mid-sequence once prompt and completion are concatenated."""
    tok = BosTemplateTokenizer()
    out = render_offline_grpo_row({**ROW, "completions": [list(c) for c in ROW["completions"]]}, tok, None)
    for completion in out["completions"]:
        assert tok.bos_token not in completion, f"mid-sequence BOS in completion: {completion!r}"
        # The prefix-strip invariant: prompt + completion reconstructs the full rendered conversation.
    full = tok.apply_chat_template(ROW["prompt"] + ROW["completions"][0])
    assert out["prompt"] + out["completions"][0] == full


class HFTruncationTokenizer:
    """Models HF's real truncation contract: ``truncation=True`` with ``max_length=None`` resolves
    the bound against ``model_max_length`` rather than meaning "unbounded"."""

    model_max_length = 4
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        ids = [100 + i for i, _ in enumerate(text.split())]
        if truncation:
            ids = ids[: self.model_max_length if max_length is None else max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return " ".join(m["content"] for m in messages)

    def pad(self, encoded, **kwargs):
        return encoded


def test_generative_row_null_prompt_cap_is_no_cap():
    """``prepare_generative_row`` builds the eval-time generation prompts for offline GRPO / DPO /
    SMPO, where ``truncation=True`` with a nullable cap would cut an uncapped prompt at
    ``model_max_length`` — silently shortening the very prompts the eval generations are judged on."""
    row = {"prompt": [{"role": "user", "content": "one two three four five six"}]}
    out = prepare_generative_row(row, HFTruncationTokenizer(), max_length=None)
    assert len(out["input_ids"]) == 6, f"uncapped generation prompt truncated: {out['input_ids']}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
