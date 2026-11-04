#!/usr/bin/env python
"""Reward terms parsed from a ``rewards:`` list and priced as ``weight * score ** exponent``."""

import math

import pytest

from halo.configs.grpo_environmental import EnvironmentConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.inference.endpoints import DEFAULT_OPENROUTER_BASE_URL
from halo.rewards.composer import RewardComposer
from halo.rewards.graders.verifiable import AccuracyTerm, FormatTerm
from halo.rewards.terms import (
    Check,
    EnvironmentTerm,
    JudgeReasoningEffort,
    JudgeTerm,
    OnError,
    Requirement,
    RewardModelBackend,
    RewardModelTerm,
    View,
    parse_reward_terms,
    resolve_environment_checks,
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
    assert judge.reasoning_effort is JudgeReasoningEffort.MEDIUM
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
    assert judge.reasoning_effort is JudgeReasoningEffort.HIGH
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
    with pytest.raises(ValueError, match=r"score must be (in \[0, 1\]|a finite number), got"):
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
            {"source": "judge", "name": "q", "checks": [{"name": "a", "description": "d", "price": 1}]},
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
    # A negative weight prices the fired process flags as a penalty.
    (penalty,) = parse_reward_terms(
        [{"source": "judge", "name": "gate", "checks": CHECKS, "weight": -0.2, "on_error": "invalid"}], SOURCES
    )
    assert penalty.weight == -0.2 and penalty.on_error is OnError.INVALID and penalty.price(0.5) == -0.1
    # The constructor's weight default is a scoring judge's 1.0, so a direct veto build must say 0.
    with pytest.raises(ValueError, match="never adds reward"):
        JudgeTerm(name="gate", checks=penalty.checks)


def test_an_empty_checks_list_beside_requirements_leaves_a_scoring_judge_as_it_is():
    """The veto defaults (weight 0, on_error neutral) must not apply."""
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


def test_flag_fraction_weighs_each_process_check():
    term = JudgeTerm(
        name="gate",
        weight=-0.2,
        checks=(
            Check(name="cheat", description="d", veto=True),
            Check(name="logged", description="d", weight=0.0),
            Check(name="wasted", description="d"),
            Check(name="sloppy", description="d", weight=3.0),
        ),
    )
    # A zero-weight flag is logged, never priced.
    assert term.flag_fraction({"logged": True, "cheat": True}) == 0.0
    assert term.flag_fraction({"wasted": True}) == 0.25
    assert term.flag_fraction({"wasted": True, "sloppy": True, "logged": True}) == 1.0
    assert term.price(term.flag_fraction({"wasted": True})) == -0.05
    only_logged = JudgeTerm(name="gate", weight=-0.2, checks=(Check(name="logged", description="d", weight=0.0),))
    assert only_logged.flag_fraction({"logged": True}) == 0.0


def test_a_veto_check_given_a_weight_also_prices():
    """Each fired check costs its own share of the term weight on top of its gate, so the prices add."""
    term = JudgeTerm(
        name="gate",
        weight=-0.7,
        checks=(
            Check(name="guess", description="d", veto=True, weight=3.0),
            Check(name="probe", description="d", veto=True, weight=2.0),
            Check(name="gate_only", description="d", veto=True),
            Check(name="wasted", description="d"),
            Check(name="untested", description="d"),
        ),
    )
    assert [check.share for check in term.checks] == [3.0, 2.0, 0.0, 1.0, 1.0]
    assert term.price(term.flag_fraction({"guess": True})) == pytest.approx(-0.3)
    assert term.price(term.flag_fraction({"guess": True, "untested": True})) == pytest.approx(-0.4)
    assert term.flag_fraction({"gate_only": True}) == 0.0


def test_a_check_weight_parses_from_config():
    checks = [*CHECKS, {"name": "logged", "description": "d", "weight": 0}]
    (term,) = parse_reward_terms([{"source": "judge", "name": "gate", "checks": checks, "weight": -0.1}], SOURCES)
    assert term.checks[-1] == Check(name="logged", description="d", weight=0.0)
    assert [check.share for check in term.checks] == [0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "a/b", "description": "d"}, "without '/'"),
        ({"name": " a", "description": "d"}, "without '/'"),
        ({"name": "", "description": "d"}, "non-blank"),
        ({"name": "a", "description": " "}, "description must be a non-blank string"),
        ({"name": "a", "description": "d", "veto": "yes"}, "veto must be true or false"),
        ({"name": "a", "description": "d", "veto": 1}, "veto must be true or false"),
        ({"name": "a", "description": "d", "weight": -1.0}, "weight must be >= 0"),
        ({"name": "a", "description": "d", "weight": float("nan")}, "weight must be a finite number"),
        ({"name": "a", "description": "d", "weight": True}, "weight must be a finite number"),
        ({"name": "a", "description": "d", "veto": True, "weight": -1.0}, "weight must be >= 0"),
        ({"name": "a", "description": "d", "unless": " "}, "unless must be a non-blank string"),
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
    """With no terms a run would train on turn shaping alone and report nothing wrong."""
    with pytest.raises(ValueError, match="at least one term"):
        RewardComposer([])
    with pytest.raises(ValueError, match="at least one reward term"):
        parse_reward_terms([], SOURCES)
    with pytest.raises(ValueError, match="at least one reward term"):
        EnvironmentConfig(rewards=[])
    with pytest.raises(ValueError, match="at least one reward term"):
        OnlineGRPOScriptArguments(rewards=[])
    with pytest.raises(ValueError, match="at most one environment term"):
        RewardComposer([EnvironmentTerm(), EnvironmentTerm()])
    assert len(RewardComposer([EnvironmentTerm()]).terms) == 1


def test_the_rlvr_arm_admits_its_graders_and_the_external_scorers_only():
    """The RLVR arm has no environment whose grade an environment term would read."""
    args = OnlineGRPOScriptArguments(rewards=[{"source": "accuracy"}, {"source": "format"}, JUDGE_SPEC, RM_SPEC])
    assert [type(term) for term in args.reward_terms] == [AccuracyTerm, FormatTerm, JudgeTerm, RewardModelTerm]
    with pytest.raises(ValueError, match="unknown reward source 'environment'"):
        OnlineGRPOScriptArguments(rewards=[{"source": "environment"}])


def test_environment_checks_defer_a_veto_judge_s_checks_to_its_environment():
    """Refused where no environment's objective gates it or it stays unbound."""
    spec = {"source": "judge", "name": "audit", "checks": "environment"}
    term, judge = parse_reward_terms([spec, JUDGE_SPEC], SOURCES)
    assert term.is_veto and term.checks == "environment"
    assert (term.weight, term.on_error) == (0.0, OnError.NEUTRAL)
    checks = tuple(Check(**check) for check in CHECKS)
    bound, kept = resolve_environment_checks([term, judge], checks)
    assert bound.checks == checks and bound.name == "audit" and bound.weight == 0.0
    assert kept is judge
    assert resolve_environment_checks([term], ()) == (term,)
    with pytest.raises(ValueError, match=r"\['audit'\] list 'checks'"):
        OnlineGRPOScriptArguments(rewards=[spec])
    with pytest.raises(ValueError, match="declares none"):
        RewardComposer([EnvironmentTerm(), term])


def test_a_judge_context_file_is_read_into_its_context_when_a_config_parses(tmp_path):
    """The config forwarded to the actors holds the text, never a path another node cannot resolve."""
    setting = tmp_path / "setting.txt"
    setting.write_text("The policy has a scratchpad.\n", encoding="utf-8")
    spec = {**JUDGE_SPEC, "context_file": str(setting)}
    env_config = EnvironmentConfig(rewards=[{"source": "environment"}, spec])
    forwarded = env_config.to_env_config()["reward_terms"][1]
    assert forwarded["context"] == "The policy has a scratchpad.\n" and "context_file" not in forwarded
    assert env_config.reward_terms[1].context == "The policy has a scratchpad.\n"
    online = OnlineGRPOScriptArguments(rewards=[{"source": "accuracy"}, spec])
    assert online.reward_terms[1].context == "The policy has a scratchpad.\n"
    with pytest.raises(ValueError, match="not both"):
        EnvironmentConfig(rewards=[{"source": "environment"}, {**spec, "context": "inline"}])
    with pytest.raises(ValueError, match="is not a file"):
        EnvironmentConfig(rewards=[{"source": "environment"}, {**spec, "context_file": str(tmp_path / "missing.txt")}])
    with pytest.raises(ValueError, match=r"unknown option\(s\) \['context_file'\]"):
        EnvironmentConfig(rewards=[{"source": "environment"}, {**RM_SPEC, "context_file": str(setting)}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
