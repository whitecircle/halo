#!/usr/bin/env python
"""CPU tests: reward terms parsed from a ``rewards:`` list and priced as ``weight * score ** exponent``.

Run: python tests/cpu/rewards/test_reward_spec.py  (or pytest)
"""

import math

import pytest

from src.configs.environment_config import EnvironmentConfig
from src.rewards.composer import RewardComposer
from src.rewards.spec import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    EnvironmentTerm,
    JudgeTerm,
    Requirement,
    RewardModelTerm,
    parse_reward_terms,
    sources_of,
)
from src.rewards.verifiable import AccuracyTerm, FormatTerm

SOURCES = sources_of(EnvironmentTerm, JudgeTerm, RewardModelTerm, AccuracyTerm, FormatTerm)
REQUIREMENTS = [{"name": "correctness", "description": "The answer is right."}]


def test_terms_parse_with_their_defaults():
    terms = parse_reward_terms(
        [
            {"source": "environment"},
            {"source": "judge", "name": "quality", "weight": 0.3, "requirements": REQUIREMENTS},
            {"source": "reward_model", "name": "pref", "url": "http://rm:8100/", "model": "org/rm"},
            {"source": "accuracy"},
            {"source": "format", "weight": 0.5},
        ],
        SOURCES,
    )
    environment, judge, reward_model, accuracy, fmt = terms
    assert (
        isinstance(environment, EnvironmentTerm)
        and environment.name == "objective"
        and environment.key == "reward/objective"
    )
    assert isinstance(judge, JudgeTerm)
    assert (judge.model, judge.base_url, judge.reasoning_effort) == (
        DEFAULT_JUDGE_MODEL,
        DEFAULT_OPENROUTER_BASE_URL,
        "medium",
    )
    assert judge.temperature is None and judge.scale == 10 and judge.transcript == "final"
    assert judge.requirements == (Requirement(name="correctness", description="The answer is right."),)
    assert isinstance(reward_model, RewardModelTerm) and reward_model.backend == "vllm"
    assert reward_model.server_root == "http://rm:8100"
    assert (accuracy.name, accuracy.weight) == ("accuracy", 1.0)
    assert (fmt.name, fmt.weight) == ("format", 0.5)


def test_price_is_weight_times_shaped_score():
    term = EnvironmentTerm(weight=2.0, exponent=2.0)
    assert term.shape(0.5) == 0.25
    assert term.price(0.5) == 0.5
    assert term.price(1.0) == 2.0
    penalty = JudgeTerm(name="verbosity", weight=-0.2, requirements=(Requirement(name="v", description="d"),))
    assert penalty.price(1.0) == -0.2


@pytest.mark.parametrize("score", [-0.1, 1.5, float("nan"), True, "0.5"])
def test_scores_outside_the_unit_interval_are_refused(score):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EnvironmentTerm().shape(score)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"source": "environment", "name": "pass_rate"}, "always named 'objective'"),
        ({"source": "accuracy", "exponent": 0}, "exponent must be > 0"),
        ({"source": "accuracy", "weight": float("inf")}, "finite"),
        ({"source": "accuracy", "name": "a/b"}, "without '/'"),
        ({"source": "accuracy", "bogus": 1}, "unknown option"),
        ({"name": "x"}, "'source' is required"),
        ({"source": "mystery"}, "unknown reward source 'mystery'"),
        ({"source": "judge", "name": "q"}, "requirements"),
        ({"source": "judge", "name": "q", "requirements": []}, "at least one requirement"),
        (
            {"source": "judge", "name": "q", "requirements": REQUIREMENTS, "reasoning_effort": "ultra"},
            "reasoning_effort",
        ),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS, "scale": 0}, "scale"),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS, "transcript": "tail"}, "transcript"),
        ({"source": "judge", "name": "q", "requirements": [{"name": "r", "description": " "}]}, "description"),
        (
            {"source": "judge", "name": "q", "requirements": [{"name": "r", "description": "d", "weight": 0}]},
            "weight must be > 0",
        ),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS + REQUIREMENTS}, "unique"),
        ({"source": "judge", "name": "q", "requirements": "correctness"}, "must be a list"),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS, "structured_output": "no"}, "true or false"),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS, "include_reference": "off"}, "true or false"),
        ({"source": "reward_model", "name": "p", "url": "http://x", "model": "m", "backend": "tgi"}, "backend"),
        ({"source": "reward_model", "name": "p", "url": "http://x", "model": "m", "logit_scale": 0}, "logit_scale"),
        ({"source": "reward_model", "name": "p", "url": "http://x", "model": "m", "label_index": -1}, "label_index"),
        ({"source": "format", "pattern": "("}, "invalid pattern"),
    ],
)
def test_invalid_terms_are_refused_with_their_location(spec, message):
    with pytest.raises(ValueError, match=message) as info:
        parse_reward_terms([spec], SOURCES)
    assert "rewards[0]" in str(info.value) or "source" in str(info.value)


def test_duplicate_term_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        parse_reward_terms([{"source": "accuracy"}, {"source": "format", "name": "accuracy"}], SOURCES)


def test_typed_terms_pass_through_only_when_their_source_is_admitted():
    term = AccuracyTerm()
    assert parse_reward_terms([term], SOURCES) == (term,)
    with pytest.raises(ValueError, match="not available here"):
        parse_reward_terms([EnvironmentTerm()], sources_of(AccuracyTerm))
    with pytest.raises(ValueError, match="list of reward terms"):
        parse_reward_terms({"source": "accuracy"}, SOURCES)


def test_judge_score_is_the_weighted_fraction_of_the_scale_with_clamping():
    term = JudgeTerm(
        name="q",
        scale=10,
        requirements=(Requirement(name="a", description="d"), Requirement(name="b", description="d", weight=3.0)),
    )
    assert term.score_from({"a": 10, "b": 10}) == 1.0
    assert term.score_from({"a": 10, "b": 0}) == pytest.approx(0.25)
    # Out-of-scale replies are clamped, never extrapolated into a score above 1 or below 0.
    assert term.score_from({"a": 25, "b": -4}) == pytest.approx(0.25)
    # Fractional weights round a full score past 1 by an ulp; the term must still price it.
    fractional = JudgeTerm(
        name="f",
        scale=5,
        requirements=(
            Requirement(name="a", description="d", weight=0.1),
            Requirement(name="b", description="d", weight=0.7),
        ),
    )
    assert fractional.score_from({"a": 5, "b": 5}) == 1.0
    assert fractional.price(fractional.score_from({"a": 5, "b": 5})) == 1.0


def test_reward_model_normalization_is_a_shifted_scaled_logistic():
    term = RewardModelTerm(name="p", url="http://x", model="m", logit_shift=2.0, logit_scale=4.0)
    assert term.normalize(2.0) == 0.5
    assert term.normalize(6.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert term.normalize(-1000.0) == pytest.approx(0.0) and term.normalize(1000.0) == pytest.approx(1.0)
    assert term.normalize(3.0) + term.normalize(1.0) == pytest.approx(1.0)


def test_a_reward_with_no_terms_is_refused():
    """``rewards: []`` leaves the episode reward as turn shaping alone — a run with no objective at
    all, which trains on shaping and reports nothing wrong. The composer refuses it wherever it is
    built (including inside a Ray actor), and the config refuses it before the cluster comes up."""
    with pytest.raises(ValueError, match="at least one term"):
        RewardComposer([])
    with pytest.raises(ValueError, match="at least one reward term"):
        EnvironmentConfig(rewards=[])
    # The upper bound and the lower bound are the same guard's two sides.
    with pytest.raises(ValueError, match="at most one environment term"):
        RewardComposer([EnvironmentTerm(), EnvironmentTerm()])
    assert len(RewardComposer([EnvironmentTerm()]).terms) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
