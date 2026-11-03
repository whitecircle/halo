#!/usr/bin/env python
"""Preference tokenization must not append a second turn terminator.
Templates end a turn with a terminator and a newline; a duplicate ender would land in the SMPO margin's log-prob."""

import pytest

from halo.data.turn_markers import ends_with_terminator

TERMINATOR = 100
NEWLINE = 7
CONTENT = 42
ROLE_MARKER = 200  # a config-declared ender (GLM-4 style), absent from tokenizer.eos_token_id


class _StubTokenizer:
    eos_token_id = TERMINATOR

    def decode(self, ids):
        return {TERMINATOR: "<eot>", NEWLINE: "\n", CONTENT: "hi", ROLE_MARKER: "<|user|>"}[ids[0]]


@pytest.fixture
def tokenizer():
    return _StubTokenizer()


def test_terminator_followed_by_template_newline_counts_as_terminated(tokenizer):
    assert ends_with_terminator([CONTENT, TERMINATOR, NEWLINE], tokenizer, {TERMINATOR})


def test_bare_terminator_counts_as_terminated(tokenizer):
    assert ends_with_terminator([CONTENT, TERMINATOR], tokenizer, {TERMINATOR})


def test_content_tail_is_not_terminated(tokenizer):
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
