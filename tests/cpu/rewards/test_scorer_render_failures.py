#!/usr/bin/env python
"""CPU tests: a scorer that cannot build its request books a verdict-less result, it does not raise.

:class:`src.rewards.scoring.Scorer` promises that ``score`` never raises for a sample — "one bad call
cannot take a rollout batch down with it". The request itself was already guarded; what was not is the
work BEFORE it: the reward model's chat-template render (under ``transcript: full`` the messages carry
tool turns and ``tool_calls``, which a reward model's template may refuse) and the judge's prompt
build (which serializes the row's reference). Either raise escapes ``RewardComposer.score`` — it
gathers without ``return_exceptions`` — then ``settle_async`` and the episode dispatcher, and lands in
the Ray actor's catch-all, which masks the WHOLE episode. For a template mismatch that is every
episode of the run, with the launch probe green: its sample is a plain user/assistant pair.

Run: python tests/cpu/rewards/test_scorer_render_failures.py  (or pytest)
"""

import asyncio

import pytest

from src.rewards.judge import GenerativeJudge
from src.rewards.reward_model import ServedRewardModel
from src.rewards.samples import ScoringSample
from src.rewards.spec import JudgeTerm, Requirement, RewardModelTerm

REQUIREMENTS = (Requirement(name="correctness", description="The answer is right."),)


class _RefusingTemplate:
    """A reward model whose chat template refuses the conversation it is handed — the shape a
    tool-calling transcript takes against a template with no tool branch."""

    def apply_chat_template(self, messages, tokenize):
        raise ValueError("template does not support the 'tool' role")


class _Unserializable:
    """A reference answer ``json.dumps`` cannot render."""


def _sample(reference=None) -> ScoringSample:
    return ScoringSample(
        prompt=[{"role": "user", "content": "task"}],
        completion=[{"role": "assistant", "content": "answer"}],
        reference=reference,
    )


def test_a_reward_model_render_failure_scores_the_batch_with_errors():
    term = RewardModelTerm(name="pref", url="http://rm:8100", model="org/rm", transcript="full")
    scorer = ServedRewardModel(term)
    scorer._tokenizer = _RefusingTemplate()
    samples = [_sample(), _sample()]

    results = asyncio.run(scorer.score(samples))

    assert len(results) == len(samples), "every sample must come back with a verdict slot"
    for result in results:
        assert result.score is None
        assert result.error and "tool" in result.error, result.error
    asyncio.run(scorer.aclose())


def test_a_judge_prompt_build_failure_scores_the_sample_with_an_error():
    term = JudgeTerm(name="quality", requirements=REQUIREMENTS, include_reference=True)
    scorer = GenerativeJudge(term)
    # A client the test never reaches: the failure happens while building the request.
    scorer._client = object()

    results = asyncio.run(scorer.score([_sample(reference=_Unserializable())]))

    assert len(results) == 1
    assert results[0].score is None
    assert results[0].error and "TypeError" in results[0].error, results[0].error


def test_a_scorable_sample_is_unaffected_by_the_guards():
    """The guards must not swallow a working render: the same judge on a serializable reference gets
    past the prompt build and fails only at the (absent) request."""
    term = JudgeTerm(name="quality", requirements=REQUIREMENTS)
    scorer = GenerativeJudge(term)
    scorer._client = object()

    results = asyncio.run(scorer.score([_sample(reference={"answer": 42})]))

    assert results[0].score is None
    assert results[0].error.startswith("request failed:"), results[0].error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
