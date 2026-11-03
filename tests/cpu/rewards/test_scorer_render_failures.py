#!/usr/bin/env python
"""A scorer that cannot build its request books a verdict-less result.
A raise would escape the composer's gather into the Ray actor's catch-all, which masks the whole episode."""

import asyncio

import pytest

from halo.rewards.samples import ScoringSample
from halo.rewards.scorers.judge import GenerativeJudge
from halo.rewards.scorers.reward_model import ServedRewardModel
from halo.rewards.terms import JudgeTerm, Requirement, RewardModelTerm

REQUIREMENTS = (Requirement(name="correctness", description="The answer is right."),)


class _RefusingTemplate:
    """A chat template with no tool branch refuses a tool-calling transcript."""

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
    term = RewardModelTerm(name="pref", url="http://rm:8100", model="org/rm", view="full")
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
    scorer._client = object()

    results = asyncio.run(scorer.score([_sample(reference=_Unserializable())]))

    assert len(results) == 1
    assert results[0].score is None
    assert results[0].error.startswith("prompt build failed: TypeError"), results[0].error


def test_a_scorable_sample_is_unaffected_by_the_guards():
    term = JudgeTerm(name="quality", requirements=REQUIREMENTS)
    scorer = GenerativeJudge(term)
    scorer._client = object()

    results = asyncio.run(scorer.score([_sample(reference={"answer": 42})]))

    assert results[0].score is None
    assert results[0].error.startswith("request failed:"), results[0].error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
