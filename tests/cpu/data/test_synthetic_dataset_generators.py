#!/usr/bin/env python
"""The shared synthetic generators in ``tests/common/datasets.py`` are a fixed data contract.

GPU suites assert ``loss_decreased`` on real checkpoints over this data, so a generator that
changes what it emits moves every one of those trajectories at once. Two properties are pinned:
the single-turn generator emits exactly one user/assistant pair per row (the multi-turn variant
lives in ``create_sft_dataset``), and a given ``(seed, templates)`` pair reproduces the same rows.

Run: python tests/cpu/data/test_synthetic_dataset_generators.py
"""

import pytest

from tests.common.datasets import (
    SINGLE_TURN_TEMPLATES,
    VERBOSE_MATH_TEMPLATES,
    create_sft_dataset,
    create_single_turn_sft_dataset,
)


class _RoleTokenizer:
    """Templating stand-in: renders each message as ``<role>: <content>`` so turns are countable."""

    chat_template = "role-prefixed"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@pytest.mark.parametrize("templates", [SINGLE_TURN_TEMPLATES, VERBOSE_MATH_TEMPLATES])
def test_single_turn_generator_emits_one_exchange_per_row(templates):
    rows = create_single_turn_sft_dataset(32, _RoleTokenizer(), seed=42, templates=templates)["text"]
    assert len(rows) == 32
    for text in rows:
        assert text.count("user:") == 1, f"expected one user turn, got: {text!r}"
        assert text.count("assistant:") == 1, f"expected one assistant turn, got: {text!r}"


def test_single_turn_generator_is_seed_reproducible():
    first = create_single_turn_sft_dataset(16, _RoleTokenizer(), seed=7)["text"]
    again = create_single_turn_sft_dataset(16, _RoleTokenizer(), seed=7)["text"]
    other_seed = create_single_turn_sft_dataset(16, _RoleTokenizer(), seed=8)["text"]
    assert first == again
    assert first != other_seed, "a seed that changes nothing would hide a broken generator"


def test_the_multi_turn_generator_is_a_different_data_contract():
    """``create_sft_dataset`` is not a drop-in for the single-turn one: its follow-up turns change
    the token count, hence the loss trajectory of any suite pinned on the flat rows."""
    multi = create_sft_dataset(64, _RoleTokenizer(), seed=42)["text"]
    assert any(text.count("user:") > 1 for text in multi), "multi_turn_ratio=0.3 must produce follow-ups"
    flat = create_single_turn_sft_dataset(64, _RoleTokenizer(), seed=42)["text"]
    assert all(text.count("user:") == 1 for text in flat)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
