#!/usr/bin/env python
"""An assistant turn carrying CoT renders its reasoning on every roster family. Harmony reads only ``thinking``,
most read ``reasoning_content``; handed the wrong spelling a template renders an empty block, training the family
to skip reasoning. transformers fills both spellings before its own render, a rollout server renders the shipped
jinja with its own stack, so every family is rendered both ways. Real tokenizers on purpose, since the spelling
lives in the shipped jinja."""

import functools
import os

import pytest
from transformers.utils import chat_template_utils

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from halo.environments.episodes.messages import REASONING_KEYS, Message
from tests.common.models import (
    BAILING_MOE_RING_MINI,
    GEMMA3_4B_IT,
    GEMMA4_31B_QAT,
    GLM4_FLASH,
    GPT_OSS_20B_OPENAI,
    LFM2_24B_A2B,
    MISTRAL3_119B_MOE,
    QWEN3_0_6B,
    QWEN3_5_2B,
    QWEN3_6_MOE_35B,
    ZAYA_8B,
)
from tests.common.tokenizers import try_cached_tokenizer

# One tokenizer per reasoning family on the roster, plus two non-reasoning controls whose templates
# drop CoT under every spelling (asserted below, not assumed).
ROSTER = (
    QWEN3_0_6B,
    QWEN3_5_2B,
    QWEN3_6_MOE_35B,
    ZAYA_8B,
    GPT_OSS_20B_OPENAI,
    GLM4_FLASH,
    GEMMA4_31B_QAT,
    MISTRAL3_119B_MOE,
    BAILING_MOE_RING_MINI,
    LFM2_24B_A2B,
    GEMMA3_4B_IT,
)
# A cache miss must not quietly empty the matrix; below this the file proves nothing.
MIN_REASONING_FAMILIES = 5
# The one family whose template reads ``thinking`` and nothing else.
HARMONY = GPT_OSS_20B_OPENAI
# ``transformers``: the trainer's render, which fills both spellings. ``template``: the shipped jinja on the message
# as emitted, which is what a server without the fill renders.
RENDERS = ("transformers", "template")
# Held apart from ``REASONING_KEYS`` so the controls read the templates, not the emission.
SPELLINGS = ("thinking", "reasoning_content")

COT = "Twenty one plus twenty one is forty two."
ANSWER = "The answer is 42."
USER = "What is 21 plus 21?"


@functools.cache
def _load(model):
    """Tokenizer, or ``None`` when it is not in the local cache (cached: every test probes the roster)."""
    return try_cached_tokenizer(model, trust_remote_code=True)


@pytest.fixture
def template_alone(monkeypatch):
    """The render reads the message as emitted; a transformers without the fill fails here, not silently."""
    monkeypatch.setattr(chat_template_utils, "_alias_reasoning_fields", lambda chat: chat)


@pytest.fixture(params=RENDERS)
def render(request):
    if request.param == "template":
        request.getfixturevalue("template_alone")
    return request.param


def _single_key(key):
    return {"role": "assistant", "content": ANSWER, key: COT}


def _reasoning_tokenizers():
    """``{model: tokenizer}`` of the cached reasoning families; skips when none is cached."""
    cached = {model: _load(model) for model in ROSTER if _reads_reasoning(model)}
    if not cached:
        pytest.skip("no reasoning tokenizer in the matrix is cached")
    return cached


def _render(tok, assistant_dict):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": USER}, assistant_dict], tokenize=False, add_generation_prompt=False
        )
    except Exception:  # a template may reject a shape or ship none at all
        return None


def _case(model):
    """``(tokenizer, render carrying CoT, render with no CoT at all)``, skipping when unusable."""
    tok = _load(model)
    if tok is None:
        pytest.skip(f"{model} tokenizer unavailable")
    with_cot = _render(tok, Message.assistant(ANSWER, thinking=COT).to_dict(include_thinking=True))
    bare = _render(tok, Message.assistant(ANSWER).to_dict(include_thinking=True))
    if with_cot is None or bare is None:
        pytest.skip(f"{model}'s template does not render this conversation")
    return tok, with_cot, bare


@functools.cache
def _reads_reasoning(model):
    """True when the template renders CoT under some spelling; probed per key, so a third spelling fails the
    assertions instead of being skipped."""
    tok = _load(model)
    if tok is None:
        return False
    for key in (*REASONING_KEYS, "reasoning"):
        rendered = _render(tok, _single_key(key))
        if rendered is not None and COT in rendered:
            return True
    return False


def _diff_region(bare, with_cot):
    """The differing substrings once the shared prefix and suffix are cut, isolating the reasoning slot."""
    head = 0
    while head < min(len(bare), len(with_cot)) and bare[head] == with_cot[head]:
        head += 1
    tail = 0
    while tail < min(len(bare), len(with_cot)) - head and bare[-1 - tail] == with_cot[-1 - tail]:
        tail += 1
    return bare[head : len(bare) - tail], with_cot[head : len(with_cot) - tail]


@pytest.mark.parametrize("model", ROSTER)
def test_reasoning_survives_the_render_for_every_reasoning_family(model, render):
    """Exactly once: every template gets both key spellings, so one reading both would duplicate the reasoning."""
    if not _reads_reasoning(model):
        pytest.skip(f"{model}'s template renders no CoT under any spelling")
    _tok, with_cot, _bare = _case(model)

    assert COT in with_cot, f"{model} renders no reasoning: the emitted key is not the one it reads"
    assert with_cot.count(COT) == 1, f"{model} renders the reasoning {with_cot.count(COT)} times"


@pytest.mark.parametrize("model", ROSTER)
def test_no_family_renders_an_empty_reasoning_block(model, render):
    """The slot is located by diffing against the no-CoT render. Under a single-key emission the template alone
    renders the two identically for all but gpt-oss."""
    if not _reads_reasoning(model):
        pytest.skip(f"{model}'s template renders no CoT under any spelling")
    _tok, with_cot, bare = _case(model)

    assert with_cot != bare, f"{model} renders reasoning-carrying and reasoning-free turns identically"
    empty_slot, filled_slot = _diff_region(bare, with_cot)
    assert COT in filled_slot, f"{model}'s reasoning slot does not hold the CoT: {filled_slot!r}"
    assert not empty_slot.strip(), f"{model} keeps text of its own in the slot the CoT fills: {empty_slot!r}"
    assert ANSWER in with_cot, f"{model} lost the answer when reasoning was added"


@pytest.mark.parametrize("model", ROSTER)
def test_non_reasoning_family_renders_the_same_turn_with_or_without_cot(model, render):
    """A template with no reasoning slot must be unaffected: emitting extra keys cannot leak the CoT
    into its rendered turn as stray text."""
    if _reads_reasoning(model):
        pytest.skip(f"{model} is a reasoning family")
    _tok, with_cot, bare = _case(model)

    assert with_cot == bare
    assert COT not in with_cot


def test_matrix_covers_the_reasoning_roster():
    reasoning = [model for model in ROSTER if _reads_reasoning(model)]
    assert len(reasoning) >= MIN_REASONING_FAMILIES, f"only {reasoning} reachable; the matrix proves too little"


def test_template_alone_loses_a_single_key_emission_on_most_of_the_roster(template_alone):
    """The control for the ``template`` renders above: without both keys they fail on several families. A permissive
    template (LFM-2) reading either spelling is fine, so only the count and harmony are pinned."""
    lost = {key: [] for key in SPELLINGS}
    for model, tok in _reasoning_tokenizers().items():
        for key in SPELLINGS:
            rendered = _render(tok, _single_key(key))
            if rendered is not None and COT not in rendered:
                lost[key].append(model)

    assert len(lost["thinking"]) >= MIN_REASONING_FAMILIES - 1, f"'thinking' alone loses the CoT only on {lost}"
    if _reads_reasoning(HARMONY):
        assert HARMONY not in lost["thinking"], f"{HARMONY}'s template reads 'thinking'"
        assert HARMONY in lost["reasoning_content"], f"{HARMONY}'s template reads 'reasoning_content' too"


def test_transformers_renders_a_single_key_emission_on_every_reasoning_family():
    """The fill the ``transformers`` renders above rest on: either spelling alone reaches the slot, once."""
    for model, tok in _reasoning_tokenizers().items():
        for key in SPELLINGS:
            rendered = _render(tok, _single_key(key))
            assert rendered is not None, f"{model}'s template rejects a turn carrying {key!r}"
            assert rendered.count(COT) == 1, f"{model} renders {key!r} alone {rendered.count(COT)} times"


def test_reasoning_keys_stay_out_of_the_default_dict():
    """The engine request carries reasoning only where the env opts in, and an unknown field can 400 a request."""
    plain = Message.assistant(ANSWER, thinking=COT).to_dict()

    assert not set(plain) & set(REASONING_KEYS)
    assert plain == {"role": "assistant", "content": ANSWER}


def test_dict_round_trip_preserves_reasoning_under_either_spelling():
    """``from_dict`` also accepts a server-shaped message, whose CoT arrives as ``reasoning_content``."""
    original = Message.assistant(ANSWER, thinking=COT)

    assert Message.from_dict(original.to_dict(include_thinking=True)).thinking == COT
    for key in REASONING_KEYS:
        assert Message.from_dict({"role": "assistant", "content": ANSWER, key: COT}).thinking == COT
    assert Message.from_dict({"role": "assistant", "content": ANSWER}).thinking is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
