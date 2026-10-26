#!/usr/bin/env python
"""Text-only inference seams must REFUSE an image-carrying conversation, not render it.

Reward-model scoring takes a conversation straight from the user's data. It has no vision path:
the reward model is a text sequence classifier, so a chat template expands an image part into
placeholder tokens with no pixels behind them and the returned "reward" is a number computed over
holes — which `rejection_sample` then writes into a DPO/SMPO or offline-GRPO training file.

Driven through the functions the scripts actually call, so the guard is proven to sit ON the path:
the stub past each guard raises `_PastTheGuard`, which is what a clean text conversation must reach.

Run: pytest tests/cpu/inference/test_image_content_refusal.py
"""

import asyncio
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from halo.entrypoints import rejection_sampling, reward_scoring

_TEXT_TURNS = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
_IMAGE_TURNS = [
    {"role": "user", "content": [{"type": "image", "image": "b64"}, {"type": "text", "text": "what is this?"}]},
    {"role": "assistant", "content": "a cat"},
]


class _PastTheGuard(Exception):
    """Raised by the first stub past the guard — the conversation was accepted."""


def _rm_tokenizer():
    """Reward-model tokenizer stand-in raising from the chat template, the call the guard precedes."""

    def _reached_the_template(*_args, **_kwargs):
        raise _PastTheGuard

    return types.SimpleNamespace(apply_chat_template=_reached_the_template)


def _score_offloaded(*args):
    """The awaited seam both RM scripts call: the batched scorer, run on a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return asyncio.run(reward_scoring.score_conversations_offloaded(executor, *args))


# Both entry points into the reward model: the batched scorer and the thread offload the scripts
# await. A guard that the offload path did not carry would leave the scripts unprotected.
_RM_SCORERS = pytest.mark.parametrize(
    "score",
    [reward_scoring.score_conversations, _score_offloaded],
    ids=["score_conversations", "score_conversations_offloaded"],
)


@_RM_SCORERS
@pytest.mark.parametrize("correct_answer", [None, "4"])
def test_score_refuses_an_image_conversation(score, correct_answer):
    """Every scored conversation feeds a preference / offline-GRPO training file; the answer-context
    branch must not bypass the guard."""
    with pytest.raises(ValueError, match="carries an image content part"):
        score(_rm_tokenizer(), None, "cpu", [_IMAGE_TURNS], correct_answer, 8)


@_RM_SCORERS
def test_a_text_conversation_still_reaches_the_reward_model(score):
    """Anti-vacuity: the guard rejects image parts, not conversations."""
    with pytest.raises(_PastTheGuard):
        score(_rm_tokenizer(), None, "cpu", [_TEXT_TURNS], "4", 8)


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
