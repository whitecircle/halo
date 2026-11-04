#!/usr/bin/env python
"""Text-only inference seams refuse an image-carrying conversation: the reward model is a text classifier, so a chat
template expands an image into placeholder tokens with no pixels behind them. The stub past each guard raises
``_PastTheGuard``, which a clean text conversation must reach."""

import asyncio
import json
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from halo.data.chat_render import render_generation_prompt
from halo.data.conversation import conversation_carries_images, reject_image_content
from halo.entrypoints import rejection_sampling, reward_scoring

_TEXT_TURNS = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
_IMAGE_TURNS = [
    {"role": "user", "content": [{"type": "image", "image": "b64"}, {"type": "text", "text": "what is this?"}]},
    {"role": "assistant", "content": "a cat"},
]


class _PastTheGuard(Exception):
    """Raised by the first stub past the guard: the conversation was accepted."""


def _rm_tokenizer():
    """Reward-model tokenizer stand-in raising from the chat template, the call the guard precedes."""

    def _reached_the_template(*_args, **_kwargs):
        raise _PastTheGuard

    return types.SimpleNamespace(apply_chat_template=_reached_the_template)


def _score_offloaded(*args):
    """The awaited seam both RM scripts call: the batched scorer, run on a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return asyncio.run(reward_scoring.score_conversations_offloaded(executor, *args))


# A guard the thread offload path did not carry would leave the scripts unprotected.
_RM_SCORERS = pytest.mark.parametrize(
    "score",
    [reward_scoring.score_conversations, _score_offloaded],
    ids=["score_conversations", "score_conversations_offloaded"],
)


@_RM_SCORERS
@pytest.mark.parametrize("correct_answer", [None, "4"])
def test_score_refuses_an_image_conversation(score, correct_answer):
    """The answer-context branch must not bypass the guard."""
    with pytest.raises(ValueError, match="carries an image content part"):
        score(_rm_tokenizer(), None, "cpu", [_IMAGE_TURNS], correct_answer, 8)


@_RM_SCORERS
def test_a_text_conversation_still_reaches_the_reward_model(score):
    with pytest.raises(_PastTheGuard):
        score(_rm_tokenizer(), None, "cpu", [_TEXT_TURNS], "4", 8)


_NON_TEXT_PARTS = [
    {"type": "image_url", "image_url": {"url": "x"}},
    {"type": "input_image", "image_url": "x"},
    {"type": "video", "video": "x"},
    {"type": "audio", "audio": "x"},
    {"type": "input_audio", "input_audio": {"data": "x"}},
    {"type": "input_text", "text": "x"},
]


def _ids(part):
    return part["type"]


class _RendersAnything:
    """A template that takes any part: what it does with one is exactly what the refusal must not leave to chance."""

    chat_template = "stub"

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}


@pytest.mark.parametrize("part", _NON_TEXT_PARTS, ids=_ids)
def test_the_generation_prompt_refuses_every_media_part(part):
    """Online GRPO's only gate on its rows: Qwen3.6 renders a placeholder for these and Gemma 4's recipe template drops
    them, so a run would train blind to them with no error."""
    messages = [{"role": "user", "content": [part, {"type": "text", "text": "what is this?"}]}]
    with pytest.raises(ValueError, match=part["type"]):
        render_generation_prompt(_RendersAnything(), messages)


@pytest.mark.parametrize("part", _NON_TEXT_PARTS, ids=_ids)
def test_a_json_string_conversation_is_refused_too(part):
    conversation = json.dumps([{"role": "user", "content": [part]}])
    with pytest.raises(ValueError, match=part["type"]):
        reject_image_content(conversation, "field")


def test_text_parts_plain_strings_and_a_part_with_no_type_pass():
    reject_image_content([{"role": "user", "content": [{"type": "text", "text": "x"}, {"text": "y"}]}], "field")
    reject_image_content([{"role": "user", "content": "x"}], "field")
    reject_image_content(json.dumps([{"role": "user", "content": [{"type": "text", "text": "x"}]}]), "field")
    reject_image_content("not json", "field")


def test_a_thinking_part_stays_on_the_text_path():
    """Mistral Small 4's and Command A+'s templates write it as their reasoning block, so a refusal would turn away their
    native assistant turn."""
    turn = {"role": "assistant", "content": [{"type": "thinking", "thinking": "2+2"}, {"type": "text", "text": "4"}]}
    reject_image_content([_TEXT_TURNS[0], turn], "field")


def test_an_image_part_keeps_its_vlm_remedy():
    with pytest.raises(ValueError, match="carries an image content part"):
        reject_image_content(_IMAGE_TURNS, "field")


def test_run_routing_still_reads_image_parts_alone():
    """The VLM path extracts ``image`` parts only, so another kind must not route there and lose its data."""
    assert conversation_carries_images(_IMAGE_TURNS)
    for part in _NON_TEXT_PARTS:
        assert not conversation_carries_images([{"role": "user", "content": [part]}])


def test_rejection_sampling_scores_through_the_guarded_seam():
    """A run that grew its own scoring loop would score images with no guard in front of it."""
    script = rejection_sampling
    assert script.score_conversations_offloaded is reward_scoring.score_conversations_offloaded
    assert not hasattr(script, "score_conversations"), (
        f"{script.__name__} holds its own score_conversations — the image guard covers only the "
        f"shared one, so a second scoring path scores conversations nothing checked"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
