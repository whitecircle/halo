#!/usr/bin/env python
"""Preference tokenization must not append a second turn terminator.

Chat templates close the final turn with ``<terminator>\\n``, so the last token of a rendered
completion is the NEWLINE. Testing only ``input_ids[-1]`` therefore reads an already-terminated row as
unterminated and appends a duplicate ender the policy never emits — measured on Qwen3-4B-Instruct as
``Four.<|im_end|>\\n<|im_end|>``. That token lands inside the mean log-prob that IS the SMPO margin,
for chosen and rejected alike, and no shipped preference config pins a chat template that would avoid
it. The SFT render path already walks back over template whitespace; this pins the same rule here.

Run: pytest tests/cpu/trainers/test_preference_terminator.py
"""

import pytest

from src.data.spans import ends_with_terminator

TERMINATOR = 100
NEWLINE = 7
CONTENT = 42
ROLE_MARKER = 200  # a config-declared ender (GLM-4 style), absent from tokenizer.eos_token_id


class _StubTokenizer:
    """Decodes only what the whitespace walk needs: content vs template whitespace."""

    eos_token_id = TERMINATOR

    def decode(self, ids):
        return {TERMINATOR: "<eot>", NEWLINE: "\n", CONTENT: "hi", ROLE_MARKER: "<|user|>"}[ids[0]]


@pytest.fixture
def tokenizer():
    return _StubTokenizer()


def test_terminator_followed_by_template_newline_counts_as_terminated(tokenizer):
    """The real shape: ``... <terminator> \\n``. Appending here is the bug."""
    assert ends_with_terminator([CONTENT, TERMINATOR, NEWLINE], tokenizer, {TERMINATOR})


def test_bare_terminator_counts_as_terminated(tokenizer):
    assert ends_with_terminator([CONTENT, TERMINATOR], tokenizer, {TERMINATOR})


def test_content_tail_is_not_terminated(tokenizer):
    """A genuinely unterminated completion must still receive one — the walk stops at content."""
    assert not ends_with_terminator([TERMINATOR, CONTENT], tokenizer, {TERMINATOR})


def test_whitespace_only_sequence_is_not_terminated(tokenizer):
    assert not ends_with_terminator([NEWLINE, NEWLINE], tokenizer, {TERMINATOR})


def test_empty_sequence_is_not_terminated(tokenizer):
    assert not ends_with_terminator([], tokenizer, {TERMINATOR})


def test_config_declared_role_marker_counts(tokenizer):
    """GLM-4/Gemma close turns with a marker carried on the config, not tokenizer.eos_token_id."""
    assert ends_with_terminator([CONTENT, ROLE_MARKER, NEWLINE], tokenizer, {TERMINATOR, ROLE_MARKER})
    assert not ends_with_terminator([CONTENT, ROLE_MARKER, NEWLINE], tokenizer, {TERMINATOR})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
