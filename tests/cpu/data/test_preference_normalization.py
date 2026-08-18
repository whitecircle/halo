#!/usr/bin/env python3
"""normalize_preference_row + reward implicit-prompt: hub preference-shape adaptation.

Covers the three hub shapes (string prompt + full-conversation chosen/rejected — ultrafeedback/tulu-3;
already-contract continuation rows; implicit prompt — Skywork-Reward), the fail-loud cases, and the
prompt-less branch of build_reward_preprocess_fn.

Usage:
    python tests/cpu/data/test_preference_normalization.py
"""

import copy
import sys

import pytest

from src.data.pipeline.preferences import (
    apply_chat_template_to_preference_data,
    build_reward_preprocess_fn,
    normalize_preference_row,
)


class StubTokenizer:
    """Chat-template stub: renders `<role>text</role>` per message; tokenizes by whitespace."""

    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert isinstance(messages, list), f"apply_chat_template got {type(messages).__name__}, not messages"
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=True):
        ids = list(range(len(text.split())))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _tulu3_row():
    return {
        "prompt": "what is 2+2?",
        "chosen": [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "5"},
        ],
    }


def test_string_prompt_wrapped_and_prefix_stripped():
    row = normalize_preference_row(_tulu3_row())
    assert row["prompt"] == [{"role": "user", "content": "what is 2+2?"}]
    assert row["chosen"] == [{"role": "assistant", "content": "4"}]
    assert row["rejected"] == [{"role": "assistant", "content": "5"}]


def test_contract_shape_passes_through():
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}],
    }
    snapshot = copy.deepcopy(row)
    assert normalize_preference_row(dict(row)) == snapshot


def test_implicit_prompt_extracted_from_shared_prefix():
    row = {
        "chosen": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad"},
        ],
    }
    out = normalize_preference_row(row)
    assert [m["role"] for m in out["prompt"]] == ["system", "user"]
    assert out["chosen"] == [{"role": "assistant", "content": "good"}]
    assert out["rejected"] == [{"role": "assistant", "content": "bad"}]


def test_multiturn_continuation_keeps_post_prompt_turns():
    row = {
        "prompt": "q1",
        "chosen": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        "rejected": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "b1"},
        ],
    }
    out = normalize_preference_row(row)
    assert len(out["chosen"]) == 3 and out["chosen"][-1]["content"] == "a2"


def test_no_shared_prefix_and_no_prompt_raises():
    row = {
        "chosen": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "a"}],
        "rejected": [{"role": "user", "content": "y"}, {"role": "assistant", "content": "b"}],
    }
    with pytest.raises(ValueError, match="share no leading messages"):
        normalize_preference_row(row)


def test_continuation_not_ending_on_assistant_raises():
    row = {
        "prompt": "q",
        "chosen": [{"role": "user", "content": "q"}],  # nothing left after the prompt strip
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
    }
    with pytest.raises(ValueError, match="assistant continuation"):
        normalize_preference_row(row)


def test_apply_chat_template_renders_prompt_once():
    row = apply_chat_template_to_preference_data(_tulu3_row(), StubTokenizer())
    assert row["prompt"] == "<user>what is 2+2?</user>"
    assert row["chosen"] == "<assistant>4</assistant>", "the user turn must not be rendered into chosen"
    assert (row["prompt"] + row["chosen"]).count("<user>") == 1


def test_reward_preprocess_without_prompt_column():
    fn = build_reward_preprocess_fn(StubTokenizer(), max_length=64)
    examples = {
        "chosen": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "good answer here"}]],
        "rejected": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]],
    }
    out = fn(examples)
    assert len(out["input_ids_chosen"]) == 1
    assert len(out["input_ids_chosen"][0]) > len(out["input_ids_rejected"][0]), (
        "full conversations must be templated when no prompt column exists"
    )


def test_reward_preprocess_templates_the_prompt_into_both_encodings():
    """A Bradley-Terry reward model scores prompt+answer. Dropping the ``prompt +`` concatenation
    trains it to score answers with no question — and leaves every row shape intact, so it is
    invisible to any length or count assertion. Asserted on what reaches the chat template."""

    class _Recording(StubTokenizer):
        def __init__(self):
            self.rendered: list[list[str]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.rendered.append([m["role"] for m in messages])
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = _Recording()
    fn = build_reward_preprocess_fn(tokenizer, max_length=64)
    fn(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "chosen": [[{"role": "assistant", "content": "a"}]],
            "rejected": [[{"role": "assistant", "content": "b"}]],
        }
    )
    assert tokenizer.rendered == [["user", "assistant"], ["user", "assistant"]], (
        f"both encodings must template prompt+completion, got {tokenizer.rendered}"
    )


def test_generative_prep_normalizes_hub_shapes_before_templating(monkeypatch):
    """The eval-generation prep receives RAW hub rows; a string prompt (tulu-3 mixtures) must go
    through the same contract normalizer as the training path — the chat template dies on a bare
    string. Asserted at the seam: the row that reaches ``prepare_generative_row`` is normalized."""
    from src.data.pipeline import preferences as prefs

    seen = {}

    def fake_prepare(row, tokenizer, max_length, tools_field=None):
        seen["prompt"] = row["prompt"]
        return {"input_ids": [1], "attention_mask": [1]}

    monkeypatch.setattr(prefs, "prepare_generative_row", fake_prepare)
    out = prefs._normalized_generative_row(_tulu3_row(), StubTokenizer(), max_length=8)
    assert seen["prompt"] == [{"role": "user", "content": "what is 2+2?"}]
    assert out == {"input_ids": [1], "attention_mask": [1]}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
