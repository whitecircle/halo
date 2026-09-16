#!/usr/bin/env python
"""CPU tests: reward terms as TRL reward functions, and the composer that prices them.

Run: python tests/cpu/rewards/test_reward_functions.py  (or pytest)
"""

import asyncio
import inspect

import pytest

from src.rewards import composer as composer_module
from src.rewards.composer import RewardComposer, build_scorer
from src.rewards.functions import ScorerRewardFunction, TermRewardFunction, reward_functions
from src.rewards.samples import ScoringSample, samples_from_completions
from src.rewards.scoring import Scorer, ScoreResult
from src.rewards.spec import EnvironmentTerm, JudgeTerm, Requirement, RewardModelTerm
from src.rewards.verifiable import RLVR_GRADERS, AccuracyTerm, FormatTerm

JUDGE = JudgeTerm(name="quality", weight=0.3, exponent=2.0, requirements=(Requirement(name="r", description="d"),))


class _FakeScorer(Scorer):
    """Scores by the length of the final assistant text; ``fail`` texts score ``None``."""

    def __init__(self, term):
        super().__init__(term)
        self.seen: list[ScoringSample] = []
        self.verified = 0
        self.closed = False

    @property
    def metric_keys(self):
        return ("judge/quality/r", "judge/quality/completion_tokens")

    async def score_one(self, sample):
        self.seen.append(sample)
        text = sample.completion[-1]["content"]
        if text == "fail":
            return ScoreResult(None, error="fail")
        return ScoreResult(min(len(text) / 10, 1.0), {"judge/quality/r": len(text) / 10})

    async def verify(self):
        self.verified += 1

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake_scorers(monkeypatch):
    monkeypatch.setitem(composer_module.SCORERS, JudgeTerm, _FakeScorer)


def test_reward_functions_are_named_after_their_terms_with_their_weights(fake_scorers):
    terms = (AccuracyTerm(weight=1.0), FormatTerm(weight=0.5, pattern="ok"), JUDGE)
    functions, weights = reward_functions(terms, RLVR_GRADERS, reference_column="answer")
    assert [f.__name__ for f in functions] == ["accuracy", "format", "quality"]
    assert weights == [1.0, 0.5, 0.3]
    assert [inspect.iscoroutinefunction(f) for f in functions] == [False, False, True]
    assert isinstance(functions[2], ScorerRewardFunction) and isinstance(functions[2].scorer, _FakeScorer)


def test_graders_run_through_the_term_shape():
    accuracy, fmt = reward_functions(
        (AccuracyTerm(exponent=2.0), FormatTerm(pattern=r"<answer>.*</answer>")),
        RLVR_GRADERS,
        reference_column="answer",
    )[0]
    completions = [r"\boxed{4}", r"\boxed{5}", [{"role": "assistant", "content": "<answer>4</answer>"}]]
    assert accuracy(prompts=["q"] * 3, completions=completions, answer=["4", "4", "4"]) == [1.0, 0.0, 0.0]
    assert fmt(prompts=["q"] * 3, completions=completions) == [0.0, 0.0, 1.0]
    halves = TermRewardFunction(AccuracyTerm(exponent=2.0), lambda prompts, completions, **kw: [0.5, None])
    assert halves(prompts=["a", "b"], completions=["x", "y"]) == [0.25, None]


def test_scorer_function_forwards_reference_shapes_scores_and_logs_metrics(fake_scorers):
    (function,), _ = reward_functions((JUDGE,), {}, reference_column="answer")
    logged = {}
    prompts = [[{"role": "user", "content": "q"}], "plain prompt", "p3"]
    completions = [[{"role": "assistant", "content": "12345"}], "fail", "1234567890"]
    scores = asyncio.run(
        function(
            prompts=prompts,
            completions=completions,
            answer=["a1", "a2", "a3"],
            log_metric=lambda k, v: logged.__setitem__(k, v),
        )
    )
    assert scores == [pytest.approx(0.25), None, 1.0]
    scorer = function.scorer
    assert [s.reference for s in scorer.seen] == ["a1", "a2", "a3"]
    assert scorer.seen[1].prompt == [{"role": "user", "content": "plain prompt"}]
    assert scorer.seen[1].completion == [{"role": "assistant", "content": "fail"}]
    # Every fixed key is logged on every call (a key no sample carried logs 0), so the per-key gathers
    # TRL runs across ranks see the same key set on a rank whose samples all failed.
    assert logged == {
        "judge/quality/r": pytest.approx((0.5 + 1.0) / 2),
        "judge/quality/completion_tokens": 0.0,
        "quality/scored_frac": pytest.approx(2 / 3),
    }
    failed_only = {}
    asyncio.run(
        function(
            prompts=["p"], completions=["fail"], answer=["a"], log_metric=lambda k, v: failed_only.__setitem__(k, v)
        )
    )
    assert failed_only == {"judge/quality/r": 0.0, "judge/quality/completion_tokens": 0.0, "quality/scored_frac": 0.0}


def test_unknown_term_type_is_refused():
    with pytest.raises(TypeError, match="neither a scorer nor a grader"):
        reward_functions((EnvironmentTerm(),), RLVR_GRADERS, reference_column=None)
    with pytest.raises(TypeError, match="no external scorer"):
        build_scorer(EnvironmentTerm())


def test_samples_from_completions_requires_aligned_lengths():
    with pytest.raises(ValueError):
        samples_from_completions(["a", "b"], ["x"])


def test_composer_prices_every_term_and_refuses_a_missing_score():
    composer = RewardComposer((EnvironmentTerm(weight=2.0, exponent=2.0), JUDGE))
    assert composer.components({"objective": 0.5, "quality": 1.0}) == {"reward/objective": 0.5, "reward/quality": 0.3}
    with pytest.raises(ValueError, match="no score for reward term"):
        composer.components({"objective": 1.0})
    with pytest.raises(ValueError, match="at most one environment term"):
        RewardComposer((EnvironmentTerm(), EnvironmentTerm()))


def test_composer_scores_external_terms_per_sample(fake_scorers):
    second = RewardModelTerm(name="pref", url="http://rm", model="m")
    monkeypatched = RewardComposer((EnvironmentTerm(), JUDGE, second))
    monkeypatched._scorers = {"quality": _FakeScorer(JUDGE), "pref": _FakeScorer(second)}
    samples = [
        ScoringSample(prompt=[], completion=[{"role": "assistant", "content": "12345"}]),
        ScoringSample(prompt=[], completion=[{"role": "assistant", "content": "fail"}]),
    ]
    verdicts = asyncio.run(monkeypatched.score(samples))
    assert [sorted(v) for v in verdicts] == [["pref", "quality"], ["pref", "quality"]]
    assert verdicts[0]["quality"].score == 0.5 and verdicts[1]["pref"].score is None
    asyncio.run(monkeypatched.verify())
    assert all(scorer.verified == 1 for scorer in monkeypatched.scorers.values())
    scorers = list(monkeypatched.scorers.values())
    asyncio.run(monkeypatched.aclose())
    assert all(scorer.closed for scorer in scorers) and monkeypatched._scorers is None


def test_composer_without_external_terms_builds_no_scorer():
    composer = RewardComposer((EnvironmentTerm(),))
    assert asyncio.run(composer.score([ScoringSample(prompt=[], completion=[])])) == [{}]
    assert composer.scorers == {} and composer.external_terms == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
