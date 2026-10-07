#!/usr/bin/env python
"""CPU tests: reward terms parsed from a ``rewards:`` list and priced as ``weight * score ** exponent``.

Run: python tests/cpu/rewards/test_reward_terms.py  (or pytest)
"""

import math

import pytest

from src.configs.environment_config import EnvironmentConfig
from src.inference.endpoints import DEFAULT_OPENROUTER_BASE_URL
from src.rewards.composer import RewardComposer
from src.rewards.graders.verifiable import AccuracyTerm, FormatTerm
from src.rewards.terms import (
    Check,
    EnvironmentTerm,
    JudgeTerm,
    OnError,
    ReasoningEffort,
    Requirement,
    RewardModelBackend,
    RewardModelTerm,
    View,
    parse_reward_terms,
    sources_of,
)

SOURCES = sources_of(EnvironmentTerm, JudgeTerm, RewardModelTerm, AccuracyTerm, FormatTerm)
REQUIREMENTS = [{"name": "correctness", "description": "The answer is right."}]
CHECKS = [
    {"name": "cheat", "description": "The policy read the answer key.", "veto": True},
    {"name": "sloppy", "description": "The policy skipped a required step."},
    {"name": "rude", "description": "The policy insulted the user."},
]
JUDGE_SPEC = {"source": "judge", "name": "q", "requirements": REQUIREMENTS}
RM_SPEC = {"source": "reward_model", "name": "p", "url": "http://x", "model": "m"}


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
    assert isinstance(environment, EnvironmentTerm) and environment.name == "objective"
    assert isinstance(judge, JudgeTerm) and not judge.is_veto
    assert (judge.model, judge.base_url) == ("openai/gpt-5.6-luna", DEFAULT_OPENROUTER_BASE_URL)
    assert judge.reasoning_effort is ReasoningEffort.MEDIUM
    assert judge.view is View.FINAL and judge.on_error is OnError.INVALID
    assert judge.temperature is None and judge.scale == 10 and judge.max_view_chars == 60_000
    assert judge.requirements == (Requirement(name="correctness", description="The answer is right."),)
    assert isinstance(reward_model, RewardModelTerm) and reward_model.backend is RewardModelBackend.VLLM
    assert reward_model.view is View.FINAL and reward_model.url == "http://rm:8100/"
    assert (accuracy.name, accuracy.weight) == ("accuracy", 1.0)
    assert (fmt.name, fmt.weight) == ("format", 0.5)


def test_enum_fields_are_coerced_from_config_strings():
    judge, reward_model = parse_reward_terms(
        [
            {**JUDGE_SPEC, "view": "digest", "on_error": "neutral", "reasoning_effort": "high"},
            {**RM_SPEC, "backend": "sglang", "view": "full"},
        ],
        SOURCES,
    )
    assert judge.view is View.DIGEST and judge.on_error is OnError.NEUTRAL
    assert judge.reasoning_effort is ReasoningEffort.HIGH
    assert reward_model.backend is RewardModelBackend.SGLANG and reward_model.view is View.FULL
    # Direct construction coerces too: the renderers dispatch on the view by identity.
    assert JudgeTerm(name="q", requirements=judge.requirements, view="full").view is View.FULL
    assert JudgeTerm(name="q", requirements=judge.requirements, reasoning_effort=None).reasoning_effort is None


@pytest.mark.parametrize(
    ("spec", "field", "admitted"),
    [
        ({**JUDGE_SPEC, "view": "tail"}, "view", "('final', 'full', 'digest')"),
        ({**JUDGE_SPEC, "on_error": "maybe"}, "on_error", "('invalid', 'neutral')"),
        (
            {**JUDGE_SPEC, "reasoning_effort": "ultra"},
            "reasoning_effort",
            "('none', 'minimal', 'low', 'medium', 'high', 'xhigh')",
        ),
        ({**RM_SPEC, "backend": "tgi"}, "backend", "('vllm', 'sglang')"),
        ({**RM_SPEC, "view": "digest"}, "view", "('final', 'full')"),
    ],
)
def test_an_unknown_enum_spelling_names_the_admitted_values(spec, field, admitted):
    with pytest.raises(ValueError, match=f"{field} must be one of") as info:
        parse_reward_terms([spec], SOURCES)
    assert admitted in str(info.value)


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
        ({"source": "judge", "name": "q"}, "either 'requirements'"),
        ({"source": "judge", "name": "q", "requirements": []}, "either 'requirements'"),
        ({**JUDGE_SPEC, "checks": CHECKS, "weight": 0.0}, "either 'requirements'"),
        ({**JUDGE_SPEC, "scale": 0}, "scale"),
        ({**JUDGE_SPEC, "max_view_chars": 0}, "max_view_chars"),
        ({**JUDGE_SPEC, "bogus": 1}, "unknown option"),
        ({**JUDGE_SPEC, "requirements": [{"name": "scored", "description": "d"}]}, "own metric keys"),
        ({"source": "judge", "name": "q", "requirements": [{"name": "r", "description": " "}]}, "description"),
        (
            {"source": "judge", "name": "q", "requirements": [{"name": "r", "description": "d", "weight": 0}]},
            "weight must be > 0",
        ),
        ({"source": "judge", "name": "q", "requirements": REQUIREMENTS + REQUIREMENTS}, "unique"),
        ({"source": "judge", "name": "q", "requirements": "correctness"}, "must be a list"),
        ({**JUDGE_SPEC, "structured_output": "no"}, "true or false"),
        ({**JUDGE_SPEC, "include_reference": "off"}, "true or false"),
        ({**JUDGE_SPEC, "include_reasoning": "off"}, "true or false"),
        ({"source": "judge", "name": "q", "checks": CHECKS, "weight": 0.5}, "never adds reward"),
        ({"source": "judge", "name": "q", "checks": CHECKS + CHECKS}, "unique"),
        ({"source": "judge", "name": "q", "checks": [{"name": "a/b", "description": "d"}]}, "without '/'"),
        (
            {"source": "judge", "name": "q", "checks": [{"name": "a", "description": "d", "veto": "yes"}]},
            "true or false",
        ),
        (
            {"source": "judge", "name": "q", "checks": [{"name": "a", "description": "d", "weight": 1}]},
            "unknown option",
        ),
        ({**RM_SPEC, "logit_scale": 0}, "logit_scale"),
        ({**RM_SPEC, "label_index": -1}, "label_index"),
        ({"source": "format", "pattern": "("}, "invalid pattern"),
    ],
)
def test_invalid_terms_are_refused_with_their_location(spec, message):
    with pytest.raises(ValueError, match=message) as info:
        parse_reward_terms([spec], SOURCES)
    assert "rewards[0]" in str(info.value)


def test_a_judge_lists_exactly_one_of_requirements_or_checks():
    requirements = (Requirement(name="r", description="d"),)
    checks = (Check(name="c", description="d"),)
    assert not JudgeTerm(name="q", requirements=requirements).is_veto
    assert JudgeTerm(name="q", weight=0.0, checks=checks).is_veto
    with pytest.raises(ValueError, match="either 'requirements'"):
        JudgeTerm(name="q", weight=0.0, requirements=requirements, checks=checks)
    with pytest.raises(ValueError, match="either 'requirements'"):
        JudgeTerm(name="q")


def test_a_checks_term_from_config_is_unpriced_and_neutral_by_default():
    (term,) = parse_reward_terms([{"source": "judge", "name": "gate", "checks": CHECKS}], SOURCES)
    assert term.is_veto and term.weight == 0.0 and term.on_error is OnError.NEUTRAL
    assert term.checks == (
        Check(name="cheat", description="The policy read the answer key.", veto=True),
        Check(name="sloppy", description="The policy skipped a required step."),
        Check(name="rude", description="The policy insulted the user."),
    )
    # A negative weight prices the fired process flags as a penalty; the config may also keep on_error strict.
    (penalty,) = parse_reward_terms(
        [{"source": "judge", "name": "gate", "checks": CHECKS, "weight": -0.2, "on_error": "invalid"}], SOURCES
    )
    assert penalty.weight == -0.2 and penalty.on_error is OnError.INVALID and penalty.price(0.5) == -0.1
    # The constructor's own weight default is a scoring judge's 1.0: a direct veto build must say 0.
    with pytest.raises(ValueError, match="never adds reward"):
        JudgeTerm(name="gate", checks=penalty.checks)


def test_an_empty_checks_list_beside_requirements_leaves_a_scoring_judge_as_it_is():
    """``checks: []`` is no veto judge: the veto defaults (weight 0, on_error neutral) must not apply."""
    (term,) = parse_reward_terms([{**JUDGE_SPEC, "checks": []}], SOURCES)
    assert not term.is_veto and term.weight == 1.0 and term.on_error is OnError.INVALID


def test_flag_fraction_counts_the_process_checks_only():
    term = JudgeTerm(
        name="gate",
        weight=0.0,
        checks=(
            Check(name="cheat", description="d", veto=True),
            Check(name="sloppy", description="d"),
            Check(name="rude", description="d"),
        ),
    )
    assert term.flag_fraction({"cheat": True, "sloppy": True, "rude": False}) == 0.5
    assert term.flag_fraction({"cheat": True}) == 0.0
    assert term.flag_fraction({"sloppy": True, "rude": True}) == 1.0
    veto_only = JudgeTerm(name="gate", weight=0.0, checks=(Check(name="cheat", description="d", veto=True),))
    assert veto_only.flag_fraction({"cheat": True}) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "a/b", "description": "d"}, "without '/'"),
        ({"name": " a", "description": "d"}, "without '/'"),
        ({"name": "", "description": "d"}, "non-blank"),
        ({"name": "a", "description": " "}, "description must be a non-blank string"),
        ({"name": "a", "description": "d", "veto": "yes"}, "veto must be true or false"),
        ({"name": "a", "description": "d", "veto": 1}, "veto must be true or false"),
    ],
)
def test_check_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Check(**kwargs)


def test_reward_model_views_are_final_and_full():
    assert RewardModelTerm.views == (View.FINAL, View.FULL)
    assert RewardModelTerm(name="p", url="http://x", model="m", view="full").view is View.FULL
    with pytest.raises(ValueError, match=r"view must be one of \('final', 'full'\)"):
        RewardModelTerm(name="p", url="http://x", model="m", view="digest")


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
    assert term.score_from(term.requirement_fractions({"a": 10, "b": 10})) == 1.0
    assert term.score_from(term.requirement_fractions({"a": 10, "b": 0})) == pytest.approx(0.25)
    # Out-of-scale replies are clamped, never extrapolated into a score above 1 or below 0.
    assert term.requirement_fractions({"a": 25, "b": -4}) == {"a": 1.0, "b": 0.0}
    assert term.score_from(term.requirement_fractions({"a": 25, "b": -4})) == pytest.approx(0.25)
    # Fractional weights round a full score past 1 by an ulp; the term must still price it.
    fractional = JudgeTerm(
        name="f",
        scale=5,
        requirements=(
            Requirement(name="a", description="d", weight=0.1),
            Requirement(name="b", description="d", weight=0.7),
        ),
    )
    full = fractional.requirement_fractions({"a": 5, "b": 5})
    assert fractional.score_from(full) == 1.0 and fractional.price(fractional.score_from(full)) == 1.0


def test_reward_model_normalization_is_a_shifted_scaled_logistic():
    term = RewardModelTerm(name="p", url="http://x", model="m", logit_shift=2.0, logit_scale=4.0)
    assert term.normalize(2.0) == 0.5
    assert term.normalize(6.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert term.normalize(-1000.0) == pytest.approx(0.0) and term.normalize(1000.0) == pytest.approx(1.0)
    assert term.normalize(3.0) + term.normalize(1.0) == pytest.approx(1.0)
    # Past exp's range on either side (|z| > ~710 at scale 1): no overflow, the saturated value.
    unit = RewardModelTerm(name="p", url="http://x", model="m")
    assert unit.normalize(-1e4) == 0.0 and unit.normalize(1e4) == 1.0


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
