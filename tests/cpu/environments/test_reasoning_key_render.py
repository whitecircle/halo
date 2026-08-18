#!/usr/bin/env python
"""Pinning tests: an assistant turn carrying CoT must render its reasoning on EVERY roster family.

``Message.to_dict(include_thinking=True)`` must emit every key spelling the roster reads. Harmony
(gpt-oss) reads only ``thinking``; Qwen3/3.5/3.6, Zaya, GLM-4, Gemma 4, Mistral4 and Bailing read
``reasoning_content``; a permissive template (LFM-2) accepts either. Handed the spelling it does not
know, a template renders the reasoning scaffolding with NOTHING in it — an empty
``<think>\\n\\n</think>`` — so every reasoning turn of an env-GRPO trajectory trains those families
to skip reasoning entirely.

Real tokenizers on purpose: the key spelling lives in the shipped jinja, so a hand-built template
would agree with whatever the code emits and could never catch this. ``_single_key_to_dict``
holds a single-key emission verbatim, and
:func:`test_single_key_emission_loses_reasoning_on_most_of_the_roster` asserts it still loses the CoT
on several cached families — so the assertions here are demonstrably non-vacuous.

    python tests/cpu/environments/test_reasoning_key_render.py
"""

import functools
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.environments.base import REASONING_KEYS, Message
from tests.common.models import ZAYA_8B
from tests.common.tokenizers import try_cached_tokenizer

# One tokenizer per reasoning family on the roster, plus two non-reasoning controls whose templates
# drop CoT under every spelling (asserted below, not assumed).
ROSTER = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.6-35B-A3B",
    ZAYA_8B,
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.7-Flash",
    "google/gemma-4-31B-it-qat-w4a16-ct",
    "mistralai/Mistral-Small-4-119B-2603",
    "inclusionAI/Ring-mini-linear-2.0",
    "LiquidAI/LFM2-24B-A2B",
    "google/gemma-3-4b-it",
)
# A cache miss must not quietly empty the matrix; below this the file proves nothing.
MIN_REASONING_FAMILIES = 5
# The one family whose template reads ``thinking`` and nothing else, so a single-key emission has to
# keep working for it.
HARMONY = "openai/gpt-oss-20b"

COT = "Twenty one plus twenty one is forty two."
ANSWER = "The answer is 42."
USER = "What is 21 plus 21?"


@functools.cache
def _load(model):
    """Tokenizer, or ``None`` when it is not in the local cache (cached: every test probes the roster)."""
    return try_cached_tokenizer(model, trust_remote_code=True)


def _single_key_to_dict(message):
    """The single-key emission: assistant CoT under ``thinking`` alone.

    Held verbatim, independent of ``to_dict``, so the assertions in this file are shown to reject a
    real regression rather than restating whatever ``to_dict`` currently does.
    """
    d = {"role": message.role, "content": message.content}
    if message.thinking:
        d["thinking"] = message.thinking
    return d


def _render(tok, assistant_dict):
    """Render user -> assistant through the shipped chat template, or ``None`` if it is rejected."""
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
    """True when the family's template renders CoT under SOME spelling — i.e. it is a reasoning family.

    Probed per key rather than declared, so a family that reads a third spelling counts as a reasoning
    family here and then fails the assertions below instead of being skipped.
    """
    tok = _load(model)
    if tok is None:
        return False
    for key in (*REASONING_KEYS, "reasoning"):
        rendered = _render(tok, {"role": "assistant", "content": ANSWER, key: COT})
        if rendered is not None and COT in rendered:
            return True
    return False


def _diff_region(bare, with_cot):
    """The substrings of ``bare`` and ``with_cot`` that differ, once the shared prefix/suffix is cut.

    Isolates the reasoning slot: ``bare``'s region is the empty block, ``with_cot``'s region the
    same block carrying the CoT.
    """
    head = 0
    while head < min(len(bare), len(with_cot)) and bare[head] == with_cot[head]:
        head += 1
    tail = 0
    while tail < min(len(bare), len(with_cot)) - head and bare[-1 - tail] == with_cot[-1 - tail]:
        tail += 1
    return bare[head : len(bare) - tail], with_cot[head : len(with_cot) - tail]


@pytest.mark.parametrize("model", ROSTER)
def test_reasoning_survives_the_render_for_every_reasoning_family(model):
    """A reasoning family must see the CoT in its render, exactly once.

    Exactly once is the other half: the emission hands every template both key spellings, so a
    template that reads both would duplicate the reasoning into the trained row.
    """
    if not _reads_reasoning(model):
        pytest.skip(f"{model}'s template renders no CoT under any spelling")
    _tok, with_cot, _bare = _case(model)

    assert COT in with_cot, f"{model} renders no reasoning: the emitted key is not the one it reads"
    assert with_cot.count(COT) == 1, f"{model} renders the reasoning {with_cot.count(COT)} times"


@pytest.mark.parametrize("model", ROSTER)
def test_no_family_renders_an_empty_reasoning_block(model):
    """The specific harm: reasoning present, yet the render's reasoning slot holds nothing.

    The slot is located by diffing the CoT render against the no-CoT render, so this makes no
    assumption about a family's markers. Under a single-key emission the two renders are IDENTICAL
    for every family except gpt-oss — an empty ``<think>`` block is the training target.
    """
    if not _reads_reasoning(model):
        pytest.skip(f"{model}'s template renders no CoT under any spelling")
    _tok, with_cot, bare = _case(model)

    assert with_cot != bare, f"{model} renders reasoning-carrying and reasoning-free turns identically"
    empty_slot, filled_slot = _diff_region(bare, with_cot)
    assert COT in filled_slot, f"{model}'s reasoning slot does not hold the CoT: {filled_slot!r}"
    assert not empty_slot.strip(), f"{model} keeps text of its own in the slot the CoT fills: {empty_slot!r}"
    assert ANSWER in with_cot, f"{model} lost the answer when reasoning was added"


@pytest.mark.parametrize("model", ROSTER)
def test_non_reasoning_family_renders_the_same_turn_with_or_without_cot(model):
    """A template with no reasoning slot must be unaffected: emitting extra keys cannot leak the CoT
    into its rendered turn as stray text."""
    if _reads_reasoning(model):
        pytest.skip(f"{model} is a reasoning family")
    _tok, with_cot, bare = _case(model)

    assert with_cot == bare
    assert COT not in with_cot


def test_matrix_covers_the_reasoning_roster():
    """Anti-vacuity control: enough reasoning families must be cached for the assertions to bite."""
    reasoning = [model for model in ROSTER if _reads_reasoning(model)]
    assert len(reasoning) >= MIN_REASONING_FAMILIES, f"only {reasoning} reachable; the matrix proves too little"


def test_single_key_emission_loses_reasoning_on_most_of_the_roster():
    """Anti-vacuity control for the emission itself: the single-key dict must still lose the CoT.

    Asserting that ``_single_key_to_dict`` fails on several reasoning families is what makes the two
    tests above non-tautological. Harmony must survive it (gpt-oss reads ``thinking``), which pins
    that serving the rest of the roster does not cost harmony its CoT. Only harmony's survival is
    pinned: which *other* families survive is a property of their shipped jinja, and one permissive
    enough to accept every spelling (LFM-2 reads ``thinking``, ``reasoning_content`` and
    ``reasoning``) is not a regression.
    """
    message = Message.assistant(ANSWER, thinking=COT)
    lost, kept = [], []
    for model in ROSTER:
        if not _reads_reasoning(model):
            continue
        tok = _load(model)
        rendered = _render(tok, _single_key_to_dict(message))
        if rendered is None:
            continue
        (kept if COT in rendered else lost).append(model)

    if not lost and not kept:
        pytest.skip("no reasoning tokenizer in the matrix is cached")
    assert len(lost) >= MIN_REASONING_FAMILIES - 1, f"single-key emission no longer loses the CoT on {lost}"
    if HARMONY in (*lost, *kept):  # skipped entirely when harmony is not cached
        assert HARMONY in kept, f"{HARMONY} reads 'thinking'; single-key emission must not lose its CoT"


def test_reasoning_keys_stay_out_of_the_vllm_facing_dict():
    """The generation-request serialization must carry NO reasoning key under any spelling.

    ``Trajectory.get_conversation`` (the rollout ``observation`` handed to vLLM) uses the
    ``include_thinking=False`` default; an unknown field can 400 a request, so widening the training
    emission must not widen this one.
    """
    plain = Message.assistant(ANSWER, thinking=COT).to_dict()

    assert not set(plain) & set(REASONING_KEYS)
    assert plain == {"role": "assistant", "content": ANSWER}


def test_dict_round_trip_preserves_reasoning_under_either_spelling():
    """``from_dict`` is the inverse of ``to_dict`` and must also accept a server-shaped message, whose
    CoT arrives as ``reasoning_content`` (what ``ray_actors`` parses responses from)."""
    original = Message.assistant(ANSWER, thinking=COT)

    assert Message.from_dict(original.to_dict(include_thinking=True)).thinking == COT
    for key in REASONING_KEYS:
        assert Message.from_dict({"role": "assistant", "content": ANSWER, key: COT}).thinking == COT
    assert Message.from_dict({"role": "assistant", "content": ANSWER}).thinking is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
