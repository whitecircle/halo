#!/usr/bin/env python
"""CPU tests: reward terms as TRL reward functions, the scorer catalog, and the composer that scores
the external terms and settles their verdicts into one reward.

Run: python tests/cpu/rewards/test_reward_functions.py  (or pytest)
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import ClassVar

import pytest

from src.rewards.composer import RewardComposer, Settlement
from src.rewards.functions import ScorerRewardFunction, TermRewardFunction, reward_functions
from src.rewards.graders.verifiable import RLVR_GRADERS, AccuracyTerm, FormatTerm
from src.rewards.samples import ScoringSample
from src.rewards.scorers import catalog as catalog_module
from src.rewards.scorers.base import Scorer, ScoreResult, scored_metric_key
from src.rewards.scorers.catalog import SCORERS, build_scorer
from src.rewards.scorers.judge import GenerativeJudge
from src.rewards.scorers.reward_model import ServedRewardModel
from src.rewards.terms import (
    Check,
    EnvironmentTerm,
    JudgeTerm,
    Requirement,
    RewardModelTerm,
    RewardTerm,
)

JUDGE = JudgeTerm(name="quality", weight=0.3, exponent=2.0, requirements=(Requirement(name="r", description="d"),))
LENIENT = JudgeTerm(name="lenient", on_error="neutral", requirements=(Requirement(name="r", description="d"),))
PREF = RewardModelTerm(name="pref", url="http://rm", model="m")
GATE = JudgeTerm(
    name="gate",
    weight=0.0,
    view="full",
    checks=(Check(name="cheat", description="d", veto=True), Check(name="sloppy", description="d")),
)


@dataclass(frozen=True, kw_only=True)
class _LocalTerm(RewardTerm):
    """A term neither the scorer catalog nor any grader table knows."""

    source: ClassVar[str] = "local"


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
    monkeypatch.setitem(catalog_module.SCORERS, JudgeTerm, _FakeScorer)


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
    assert scorer.seen[2].final_answer == "1234567890"
    # Every fixed key is logged on every call (a key no sample carried logs 0), so the per-key gathers
    # TRL runs across ranks see the same key set on a rank whose samples all failed.
    assert logged == {
        "judge/quality/r": pytest.approx((0.5 + 1.0) / 2),
        "judge/quality/completion_tokens": 0.0,
        "judge/quality/scored": pytest.approx(2 / 3),
    }
    failed_only = {}
    asyncio.run(
        function(
            prompts=["p"], completions=["fail"], answer=["a"], log_metric=lambda k, v: failed_only.__setitem__(k, v)
        )
    )
    assert failed_only == {"judge/quality/r": 0.0, "judge/quality/completion_tokens": 0.0, "judge/quality/scored": 0.0}


def test_a_neutral_term_returns_zero_for_a_missing_verdict(fake_scorers):
    """TRL leaves a ``None`` out of the row's reward; a neutral term instead prices the failure at 0."""
    (strict,), _ = reward_functions((JUDGE,), {}, reference_column=None)
    (lenient,), _ = reward_functions((LENIENT,), {}, reference_column=None)
    logged = {}
    assert asyncio.run(strict(prompts=["p", "q"], completions=["fail", "12345"])) == [None, pytest.approx(0.25)]
    assert asyncio.run(
        lenient(prompts=["p", "q"], completions=["fail", "12345"], log_metric=lambda k, v: logged.__setitem__(k, v))
    ) == [0.0, 0.5]
    assert logged["judge/lenient/scored"] == 0.5


def test_scored_metric_key_names_the_source_and_the_term():
    assert scored_metric_key(JUDGE) == "judge/quality/scored"
    assert scored_metric_key(PREF) == "reward_model/pref/scored"
    assert scored_metric_key(EnvironmentTerm()) == "environment/objective/scored"


def test_reward_functions_refuse_a_veto_judge():
    with pytest.raises(ValueError, match="list 'requirements' instead"):
        reward_functions((AccuracyTerm(), GATE), RLVR_GRADERS, reference_column=None)


def test_scorer_function_reads_the_conversation_column_over_the_rendered_prompt(fake_scorers):
    """TRL hands the rendered template text as the prompt; the scorers must read the conversation it
    was rendered from, which the online script keeps in its own column."""
    (function,), _ = reward_functions((JUDGE,), {}, prompt_column="conversation", reference_column="answer")
    conversation = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "q"}]
    asyncio.run(
        function(
            prompts=["<|im_start|>system\nBe terse.<|im_end|>..."],
            completions=["12345"],
            conversation=[conversation],
            answer=["a1"],
        )
    )
    (sample,) = function.scorer.seen
    assert sample.prompt == conversation and sample.reference == "a1"


def test_unknown_term_type_is_refused():
    with pytest.raises(TypeError, match="neither a scorer nor a grader"):
        reward_functions((EnvironmentTerm(),), RLVR_GRADERS, reference_column=None)
    with pytest.raises(TypeError, match="'local' has neither a scorer nor a grader"):
        reward_functions((_LocalTerm(name="local"),), RLVR_GRADERS, reference_column=None)


def test_the_catalog_is_derived_from_each_scorers_term_type():
    assert {JudgeTerm: GenerativeJudge, RewardModelTerm: ServedRewardModel} == SCORERS
    assert GenerativeJudge.term_type is JudgeTerm and ServedRewardModel.term_type is RewardModelTerm
    scorer = build_scorer(PREF)
    assert isinstance(scorer, ServedRewardModel) and scorer.term is PREF
    assert isinstance(build_scorer(JUDGE), GenerativeJudge)
    with pytest.raises(TypeError, match="reward source 'environment' has no external scorer"):
        build_scorer(EnvironmentTerm())


def test_composer_scores_external_terms_per_sample(fake_scorers):
    monkeypatched = RewardComposer((EnvironmentTerm(), JUDGE, PREF))
    monkeypatched._scorers = {"quality": _FakeScorer(JUDGE), "pref": _FakeScorer(PREF)}
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


def test_composer_construction_refuses_an_ungated_veto_judge_and_an_unscored_term():
    with pytest.raises(ValueError, match=r"veto judge term\(s\) \['gate'\] gate the objective"):
        RewardComposer((GATE,))
    with pytest.raises(TypeError, match=r"reward source\(s\) \['local'\] have no external scorer"):
        RewardComposer((EnvironmentTerm(), _LocalTerm(name="local")))
    composer = RewardComposer((EnvironmentTerm(), GATE, PREF))
    assert composer.external_terms == (GATE, PREF)


def test_settle_prices_the_verdicts_and_merges_their_metrics():
    composer = RewardComposer((EnvironmentTerm(), JUDGE, PREF))
    settlement = composer.settle(
        {"reward/objective": 1.0},
        {
            "quality": ScoreResult(0.5, {"judge/quality/r": 0.5}, detail="Half right."),
            "pref": ScoreResult(1.0, {"reward_model/pref/logit": 3.0}),
        },
    )
    assert settlement.components == {
        "reward/objective": 1.0,
        "reward/quality": pytest.approx(0.3 * 0.25),
        "reward/pref": 1.0,
    }
    assert settlement.reward == pytest.approx(2.075)
    assert settlement.metrics == {
        "judge/quality/r": 0.5,
        "judge/quality/scored": 1.0,
        "reward_model/pref/logit": 3.0,
        "reward_model/pref/scored": 1.0,
    }
    assert settlement.details == {"quality": "Half right."} and settlement.errors == {}
    assert settlement.invalid_reason is None


def test_a_missing_verdict_voids_the_episode_unless_the_term_is_neutral():
    strict = RewardComposer((EnvironmentTerm(), JUDGE))
    settlement = strict.settle({"reward/objective": 1.0}, {"quality": ScoreResult(None, error="request failed: boom")})
    assert settlement.invalid_reason == "reward term 'quality' scored nothing: request failed: boom"
    assert settlement.components == {"reward/objective": 1.0, "reward/quality": 0.0}
    assert settlement.errors == {"quality": "request failed: boom"}
    assert settlement.metrics == {"judge/quality/scored": 0.0}

    neutral = RewardComposer((EnvironmentTerm(), LENIENT))
    settlement = neutral.settle({"reward/objective": 1.0}, {"lenient": ScoreResult(None, error="boom")})
    assert settlement.invalid_reason is None
    assert settlement.components == {"reward/objective": 1.0, "reward/lenient": 0.0} and settlement.reward == 1.0
    assert settlement.errors == {"lenient": "boom"} and settlement.metrics == {"judge/lenient/scored": 0.0}


def test_the_first_failing_non_neutral_term_names_the_invalid_reason():
    composer = RewardComposer((EnvironmentTerm(), LENIENT, JUDGE, PREF))
    down = ScoreResult(None, error="down")
    settlement = composer.settle({"reward/objective": 1.0}, {"lenient": down, "quality": down, "pref": down})
    assert settlement.invalid_reason == "reward term 'quality' scored nothing: down"
    assert settlement.errors == {"lenient": "down", "quality": "down", "pref": "down"}


def test_a_veto_zeroes_the_objective():
    composer = RewardComposer((EnvironmentTerm(weight=2.0), GATE))
    vetoed = composer.settle(
        {"reward/objective": 2.0}, {"gate": ScoreResult(0.5, {"judge/gate/veto": 1.0}, veto=True)}
    )
    assert vetoed.components == {"reward/objective": 0.0, "reward/gate": 0.0} and vetoed.reward == 0.0
    assert vetoed.invalid_reason is None
    # Every credit goes with the objective — a bonus the hack collected on the way too — the penalties stand.
    shaped = composer.settle(
        {"reward/objective": 2.0, "reward/submission": 0.1, "reward/resubmission": -0.2},
        {"gate": ScoreResult(0.5, {}, veto=True)},
    )
    assert shaped.components == {
        "reward/objective": 0.0,
        "reward/submission": 0.0,
        "reward/resubmission": -0.2,
        "reward/gate": 0.0,
    }
    assert vetoed.metrics == {"judge/gate/veto": 1.0, "judge/gate/scored": 1.0}
    clean = composer.settle({"reward/objective": 2.0}, {"gate": ScoreResult(0.5, {"judge/gate/veto": 0.0})})
    assert clean.components == {"reward/objective": 2.0, "reward/gate": 0.0}
    # A priced veto judge charges its fired process flags on top of the zeroed objective.
    priced = RewardComposer((EnvironmentTerm(), JudgeTerm(name="gate", weight=-0.2, checks=GATE.checks)))
    settlement = priced.settle({"reward/objective": 1.0}, {"gate": ScoreResult(1.0, veto=True)})
    assert settlement.components == {"reward/objective": 0.0, "reward/gate": pytest.approx(-0.2)}
    assert settlement.reward == pytest.approx(-0.2)


def test_settlement_reward_sums_the_components():
    assert Settlement(
        {"reward/objective": 0.5, "reward/quality": 0.3, "reward/penalty": -0.2}
    ).reward == pytest.approx(0.6)
    assert Settlement({}).reward == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
