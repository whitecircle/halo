#!/usr/bin/env python
"""The reward economy of the shipped async-GRPO code-contests recipes. Shaping terms are small next to the objective
by intent, and a knob edited alone can break that silently, so these pin the relations a run depends on and that
the prices match what the reward and task pages state.

The length terms read each level's profile (``environment_kwargs.reasoning_effort_profiles``): the price charges
``reasoning_price`` per 1,000 reasoning tokens an episode spends past the level's ``reasoning_budget``, up to
``reasoning_price_cap``, and the floor charges a failed episode its shortfall under that budget, up to
``reasoning_floor``. Under the budget an episode pays the floor alone, past it the price alone, so a failed episode
pays at most the larger of the two and a solve at most the cap. Per level, a failed episode is paid to reason up to
its budget and charged past it, one turn past the budget leaves the price short of its cap, and the output budget
still holds a turn after the whole budget, so the price binds before running out does. A higher level is priced lower
and given more free reasoning."""

import re
import types
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.reasoning_effort import (
    REASONING_BUDGET_KEY,
    REASONING_PRICE_KEY,
    EpisodeEffort,
    bind_episode_effort,
)
from halo.environments.rollout_config import VALID_REASONING_EFFORTS
from halo.environments.tasks.coding.code_contests_prompts import AUDIT_CHECKS
from halo.rewards.terms import (
    ENVIRONMENT_REWARD_SOURCES,
    JudgeTerm,
    OnError,
    parse_reward_terms,
    read_context_files,
    resolve_environment_checks,
)
from halo.trainers.grpo.environmental.reasoning import reasoning_floor_term, reasoning_price_term
from tests.common.repo import REPO_ROOT

RECIPES = sorted((REPO_ROOT / "examples" / "grpo" / "environmental").rglob("*code-contests*.yaml"))
_TASK_PAGES = REPO_ROOT / "agent-docs/training-methods/async-grpo/environments/tasks/code-contests"
_REWARD_PAGE = _TASK_PAGES / "reward.md"
_README_PAGE = _TASK_PAGES / "README.md"
_yaml = YAML(typ="safe")
_DEFAULTS = AsyncRolloutConfig()
# The most tokens a shipped recipe lets one episode sample: its worst-case trajectory beside the prompt
# budget is what the rollout servers' context window is sized for, and Qwen3.6 serves twice the others'.
_EPISODE_TOKENS_CEILING = 131072
_EPISODE_TOKENS_CEILINGS = {"qwen3_5": 262144}
# gpt-oss-20b's served context window and its code-contests preamble (system prompt and tool schema, as the recipes'
# template renders them): the startup check warns once a worst-case episode's budget passes their difference.
_GPT_OSS_SERVED_WINDOW = 131072
_GPT_OSS_PREAMBLE_TOKENS = 461
# The audit's process checks a solved episode can never raise: each is defined on an episode that never passed.
_FAILURE_ONLY_CHECKS = ("untested_submission",)
# What each audit check costs fired alone, as the code-contests reward page's audit table states it: the
# environment's check weights (7 units in all) at the recipes' judge weight.
_DOCUMENTED_CHECK_PRICES = {
    "reasoning_in_actions": 0.0,
    "reasoning_in_runs": 0.0,
    "notepad_run": 0.0,
    "wasted_run": 0.1,
    "untested_submission": 0.1,
    "hardcoded_output": 0.3,
    "environment_probe": 0.0,
    "verdict_probe": 0.2,
    "verdict_mining": 0.0,
}
# The environment's shaping prices every code-contests recipe sets, as the task page's recipe table states them.
_DOCUMENTED_SHAPING_PRICES = {
    "submission_reward": 0.1,
    "resubmission_penalty": 0.2,
    "no_tool_use_penalty": 0.2,
    "turn_overflow_penalty": 0.35,
    "length_cutoff_penalty": 0.08,
    "tool_error_penalty": 0.05,
}
# The curriculum stages run one after another on one cluster: one economy and one set of limits. A later stage's
# harder problems may buy an episode more output and every level more free reasoning, never less.
_STAGE_RECIPES = [p for p in RECIPES if "-stage" in p.name]
_GPT_OSS_RECIPES = [p for p in RECIPES if p.parent.parent.name == "gptoss"]
_STAGE_SHARED_KEYS = (
    "rollout_max_tokens",
    "rollout_max_answer_tokens",
    "max_train_row_tokens",
    "request_timeout",
    "episode_timeout",
    "reasoning_floor",
    "reasoning_price_cap",
    "reasoning_terms_in_failed_groups",
)
# Each level's free episode reasoning, one contract at every family; the extra-hard stage's larger output budget gives
# every level more of it.
_LEVEL_BUDGETS = {"low": 40000, "medium": 50000, "high": 60000}
_EXTRA_HARD_LEVEL_BUDGETS = {"low": 100000, "medium": 135000, "high": 170000}
# The knobs every code-contests recipe states explicitly, so a recipe never rides on a default it does not spell.
_REQUIRED_KEYS = ("reasoning_floor", "reasoning_price_cap", "rollout_max_episode_tokens")


def _parsed_terms(terms: list) -> tuple:
    """A recipe's reward terms as the trainer resolves them: its config reads a judge's context file, and a veto
    judge listing ``checks: environment`` audits with the code-contests rubric."""
    terms = parse_reward_terms(read_context_files(terms), ENVIRONMENT_REWARD_SOURCES)
    return resolve_environment_checks(terms, AUDIT_CHECKS)


def _check_price(judge: JudgeTerm, name: str) -> float:
    """The judge term's price for check ``name`` fired alone, a veto check's gate aside."""
    return -judge.price(judge.flag_fraction({name: True}))


def _economy(path: Path) -> dict:
    cfg = _yaml.load(path)
    env = cfg["environment_kwargs"]
    terms = cfg.get("rewards", [{"source": "environment"}])
    profiles = env["reasoning_effort_profiles"]
    priced = any(profile.get(REASONING_PRICE_KEY, 0.0) > 0 for profile in profiles.values())
    return {
        "raw": cfg,
        # The environment term on a solve: the grade is 1 or 0, so only the weight prices it.
        "solve": sum(term.get("weight", 1.0) for term in terms if term["source"] == "environment"),
        "submission": env.get("submission_reward", 0.0),
        "resubmission": env.get("resubmission_penalty", 0.0),
        "no_tool_use": env.get("no_tool_use_penalty", 0.0),
        "turn_overflow": env.get("turn_overflow_penalty", 0.0),
        "cut": env.get("length_cutoff_penalty", 0.0),
        "recoveries": env.get("max_length_cutoff_recoveries"),
        "profiles": profiles,
        # The cap is read only while some level prices reasoning.
        "price_cap": cfg.get("reasoning_price_cap", _DEFAULTS.reasoning_price_cap) if priced else 0.0,
        # The floor's weight: the most it charges, a failed episode that reasoned nothing.
        "floor": cfg.get("reasoning_floor", _DEFAULTS.reasoning_floor),
        "ceiling": cfg.get("rollout_max_thinking_tokens"),
        "max_tokens": cfg.get("rollout_max_tokens", _DEFAULTS.rollout_max_tokens),
        "episode_tokens": cfg.get("rollout_max_episode_tokens"),
        "answer_tokens": cfg.get("rollout_max_answer_tokens"),
        "max_turns": cfg["max_turns"],
        # What each fired process check of a veto judge costs; a veto check's own price is left out, its gate costs more.
        "process_prices": {
            check.name: _check_price(judge, check.name)
            for judge in _parsed_terms(terms)
            if isinstance(judge, JudgeTerm) and judge.is_veto
            for check in judge.checks
            if not check.veto
        },
    }


def _budget(economy: dict, level: str) -> int:
    """The reasoning an episode at ``level`` spends free, summed over its turns."""
    return economy["profiles"][level][REASONING_BUDGET_KEY]


def _rate(economy: dict, level: str) -> float:
    """What the trainer charges per 1,000 reasoning tokens an episode at ``level`` spends past its budget."""
    return economy["profiles"][level].get(REASONING_PRICE_KEY, 0.0)


def _contract(economy: dict) -> EpisodeEffort:
    """The caps an episode of the recipe binds, through the drivers' own binder; every level binds the same."""
    return bind_episode_effort(
        None,
        types.SimpleNamespace(reasoning_effort=None),
        max_tokens=economy["max_tokens"],
        max_thinking_tokens=economy["ceiling"],
        max_episode_tokens=economy["episode_tokens"],
        max_answer_tokens=economy["answer_tokens"],
    )


def _turn_reasoning(economy: dict) -> int:
    """The most one turn reasons: up to the cap the engine closes it at, or the whole turn where none closes it."""
    contract = _contract(economy)
    return contract.thinking_cap or contract.max_tokens


def _length_charge(economy: dict, level: str, tokens: int, solved: bool) -> float:
    """What the length terms charge an episode at ``level`` that reasoned ``tokens``: the price past the budget, and on
    a failure the floor under it."""
    budget = _budget(economy, level)
    price = reasoning_price_term([tokens], _rate(economy, level), economy["price_cap"], budget)
    return price + (0.0 if solved else reasoning_floor_term([tokens], budget, economy["floor"]))


def _worst_length(economy: dict, solved: bool) -> float:
    """The most the length terms can charge an episode: a solve the price's cap alone (it never pays the floor), a
    failure the larger of the cap and the floor's weight (under its budget the floor, past it the price, never both)."""
    if solved:
        return economy["price_cap"]
    return max(economy["price_cap"], economy["floor"])


def _failure_only_price(economy: dict) -> float:
    """The most one failed submission adds in failure-only prices: the failure-only checks."""
    return sum(economy["process_prices"].get(name, 0.0) for name in _FAILURE_ONLY_CHECKS)


def _failed_resubmission(economy: dict) -> float:
    """One more submission that fails, at its worst honest price: the resubmission and its failure-only price."""
    return economy["resubmission"] + _failure_only_price(economy)


def _not_paid_toward_the_budget(economy: dict, level: str) -> str | None:
    """Why a failed episode at ``level`` is not paid to reason up to its budget and charged past it, or None."""
    budget = _budget(economy, level)
    half, at, past = (
        _length_charge(economy, level, tokens, solved=False)
        for tokens in (budget // 2, budget, budget + _turn_reasoning(economy))
    )
    if not half < at:
        return f"reasoning half its {budget}-token budget scores {half}, not below the {at} of reasoning all of it"
    if not past < at:
        return f"one turn past its {budget}-token budget scores {past}, not below the {at} at it"
    return None


def _no_slope_past_the_budget(economy: dict, level: str) -> str | None:
    """Why one turn past the budget leaves the price no slope, or None: a price at its cap within one turn charges
    every longer episode the same, so a level's attempts stop differing by what they reasoned."""
    turn = _turn_reasoning(economy)
    charged = -_length_charge(economy, level, _budget(economy, level) + turn, solved=True)
    if not 0 < charged < economy["price_cap"]:
        return f"one {turn}-token turn past the budget is charged {charged} under a cap of {economy['price_cap']}"
    return None


def _no_turn_after_the_budget(economy: dict, level: str) -> str | None:
    """Why an episode that reasoned its whole budget could start no further turn, or None: the output budget would end
    every episode before the price charges, so running out would price its length instead."""
    budget = _budget(economy, level)
    if _contract(economy).turn_caps(budget) is None:
        return (
            f"after its {budget}-token budget the {economy['episode_tokens']}-token output budget leaves no turn "
            f"its {_contract(economy).answer_room}-token answer room"
        )
    return None


def _prices_fall_with_level(economy: dict) -> bool:
    prices = [_rate(economy, level) for level in VALID_REASONING_EFFORTS]
    return prices == sorted(prices, reverse=True) and len(set(prices)) == 3


def _budgets_rise_with_level(economy: dict) -> bool:
    budgets = [_budget(economy, level) for level in VALID_REASONING_EFFORTS]
    return budgets == sorted(budgets) and len(set(budgets)) == 3


def _audit(path: Path) -> JudgeTerm:
    (judge,) = (term for term in _parsed_terms(_yaml.load(path)["rewards"]) if isinstance(term, JudgeTerm))
    return judge


def _page_check_prices() -> dict[str, float]:
    """Each check's price as the reward page's audit table states it: the ``(price)`` in its effect cell, else 0."""
    prices = {}
    in_table = False
    for line in _REWARD_PAGE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_table = False
        elif line.startswith("| Check | Effect |"):
            in_table = True
        elif in_table and (row := re.match(r"\| `([a-z_]+)` \| ([^|]*) \|", line)):
            price = re.search(r"\((\d+(?:\.\d+)?)\)", row[2])
            prices[row[1]] = float(price[1]) if price else 0.0
    return prices


def _page_shaping_prices() -> dict[str, float]:
    """Each shaping price as the task page's recipe table states it: the ``name`` value pairs of its Shaping row."""
    row = next(
        line for line in _README_PAGE.read_text(encoding="utf-8").splitlines() if line.startswith("| [Shaping]")
    )
    return {name: float(value) for name, value in re.findall(r"`([a-z_]+)` (\d+(?:\.\d+)?)", row)}


def _ids(paths):
    return [str(p.relative_to(REPO_ROOT / "examples" / "grpo" / "environmental")) for p in paths]


def test_the_scan_finds_the_code_contests_recipes():
    assert len(RECIPES) >= 20, f"only {len(RECIPES)} code-contests recipes found — the glob is broken"
    assert all(_economy(p)["floor"] > 0 and _economy(p)["price_cap"] > 0 for p in RECIPES), (
        "a code-contests recipe runs without the reasoning floor or the price, which the cases below price"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_recipe_states_the_reasoning_economy(path):
    """The floor, the price's cap and the output budget are spelled in every recipe, and every level's profile states
    its budget and its price."""
    e = _economy(path)
    missing = [key for key in _REQUIRED_KEYS if key not in e["raw"]]
    assert not missing, f"{path.name} does not state {missing}"
    assert set(e["profiles"]) == set(VALID_REASONING_EFFORTS)
    unstated = {
        level: [key for key in (REASONING_BUDGET_KEY, REASONING_PRICE_KEY) if key not in profile]
        for level, profile in e["profiles"].items()
    }
    assert not any(unstated.values()), f"{path.name}: profiles without {unstated}"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_level_budgets_and_prices_are_one_contract_at_every_family(path):
    """Every level's free reasoning is the same whichever family the recipe trains (the extra-hard stage's larger
    output budget gives each level more), and so is its price."""
    e = _economy(path)
    budgets = {level: _budget(e, level) for level in VALID_REASONING_EFFORTS}
    expected = _EXTRA_HARD_LEVEL_BUDGETS if "-stage3-" in path.name else _LEVEL_BUDGETS
    assert budgets == expected, f"{path.name} states level budgets {budgets}"
    shipped = {level: _rate(_economy(RECIPES[0]), level) for level in VALID_REASONING_EFFORTS}
    assert {level: _rate(e, level) for level in VALID_REASONING_EFFORTS} == shipped


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_episode_output_budget_binds_below_what_the_turn_caps_alone_allow(path):
    """A budget one whole turn fits in, under the ceiling the servers' context is sized for, and below what
    ``max_turns`` turns could sample, or it would never end an episode."""
    e = _economy(path)
    assert isinstance(e["episode_tokens"], int) and not isinstance(e["episode_tokens"], bool), (
        "a code-contests recipe runs without an episode output budget"
    )
    ceiling = _EPISODE_TOKENS_CEILINGS.get(path.parent.parent.name, _EPISODE_TOKENS_CEILING)
    assert e["max_tokens"] <= e["episode_tokens"] <= ceiling
    uncapped = e["max_turns"] * e["max_tokens"]
    assert e["episode_tokens"] < uncapped, (
        f"the per-turn caps alone let an episode sample {uncapped} tokens over {e['max_turns']} turns, so a "
        f"budget of {e['episode_tokens']} never binds"
    )


@pytest.mark.parametrize("path", _GPT_OSS_RECIPES, ids=_ids(_GPT_OSS_RECIPES))
def test_a_gpt_oss_worst_case_episode_fits_the_served_window(path):
    """The window is the model's own, so a budget at the whole of it warns at every launch of a shipped recipe."""
    e = _economy(path)
    worst_case = min(e["max_turns"] * e["max_tokens"], e["episode_tokens"])
    assert worst_case + _GPT_OSS_PREAMBLE_TOKENS <= _GPT_OSS_SERVED_WINDOW


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_on_vllm_the_reasoning_cap_is_the_turn_less_its_answer_room(path):
    """Past its reasoning cap a turn generates at most the answer room, the same at every level, so a program carrying
    the reasoning on in its comments is cut at it; the run's own cap stays unset, the answer room alone placing it.
    SGLang forces no close at a cap, so its recipes run without the bound."""
    e = _economy(path)
    assert e["ceiling"] is None, f"{path.name} sets rollout_max_thinking_tokens beside the answer room"
    contract = _contract(e)
    if e["raw"].get("rollout_backend") == "sglang":
        assert e["answer_tokens"] is None, f"{path.name} bounds an answer room SGLang never opens"
        assert contract.thinking_cap is None
        return
    assert isinstance(e["answer_tokens"], int), f"{path.name} does not state rollout_max_answer_tokens"
    assert contract.thinking_cap == e["max_tokens"] - e["answer_tokens"]
    assert contract.answer_room == e["answer_tokens"]


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_length_terms_stay_under_one_more_failed_submission(path):
    """The price's cap and the floor's weight each bound what reasoning length can cost a failed episode, and both stay
    under one more failed submission at its worst honest price, so guessing again is never the cheaper way to fail."""
    e = _economy(path)
    assert _worst_length(e, solved=False) < _failed_resubmission(e), (
        f"the length terms can cost a failure up to {_worst_length(e, solved=False)}, not under the "
        f"{_failed_resubmission(e)} of one more failed submission: how long an episode reasons would then outweigh "
        "whether it resubmits"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_failed_episode_is_paid_to_reason_up_to_its_budget_and_charged_past_it(path):
    """The floor pays a failure for more reasoning under its level's budget and the price charges it past the budget,
    so at every level the budget is where a failed episode's length terms are best."""
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        reason = _not_paid_toward_the_budget(e, level)
        assert reason is None, f"{level}: {reason}"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_one_turn_past_the_budget_leaves_the_price_its_slope(path):
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        reason = _no_slope_past_the_budget(e, level)
        assert reason is None, f"{level}: {reason}"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_level_budget_leaves_the_output_budget_a_turn_after_it(path):
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        reason = _no_turn_after_the_budget(e, level)
        assert reason is None, f"{level}: {reason}"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_higher_level_is_priced_lower_and_given_more_free_reasoning(path):
    e = _economy(path)
    prices = {level: _rate(e, level) for level in VALID_REASONING_EFFORTS}
    assert _prices_fall_with_level(e), f"the per-level prices do not fall low > medium > high: {prices}"
    budgets = {level: _budget(e, level) for level in VALID_REASONING_EFFORTS}
    assert _budgets_rise_with_level(e), f"the per-level budgets do not rise low < medium < high: {budgets}"


def test_the_curriculum_stages_run_one_economy_and_one_set_of_limits():
    assert len(_STAGE_RECIPES) == 3, f"expected the three curriculum stages, found {[p.name for p in _STAGE_RECIPES]}"
    stated = [{key: _yaml.load(p).get(key) for key in _STAGE_SHARED_KEYS} for p in _STAGE_RECIPES]
    assert stated[0] == stated[1] == stated[2], dict(zip([p.name for p in _STAGE_RECIPES], stated, strict=True))
    economies = [_economy(p) for p in _STAGE_RECIPES]
    prices = [{level: _rate(e, level) for level in VALID_REASONING_EFFORTS} for e in economies]
    assert prices[0] == prices[1] == prices[2]
    for earlier, later in zip(economies, economies[1:], strict=False):
        assert later["episode_tokens"] >= earlier["episode_tokens"]
        assert all(_budget(later, level) >= _budget(earlier, level) for level in VALID_REASONING_EFFORTS)


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_recovered_cut_costs_less_than_the_attempt_bonus_and_no_more_than_an_overflow(path):
    """A cut never costs more than the graded attempt earns, and a recovered cut never costs more than the cut that
    ends the episode."""
    e = _economy(path)
    assert 0 < e["cut"] < e["submission"], f"cut price {e['cut']} against a submission bonus of {e['submission']}"
    assert e["cut"] <= e["turn_overflow"], (
        f"a recovered cut ({e['cut']}) costs more than an overflow ({e['turn_overflow']})"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_an_honest_failed_attempt_beats_not_attempting(path):
    e = _economy(path)
    # Graded, passed nothing, the worst length terms a failure pays.
    worst_attempt = e["submission"] - _worst_length(e, solved=False)
    best_no_attempt = -e["no_tool_use"]  # never called a tool, reasoning exactly its budget: free of both length terms
    assert worst_attempt > best_no_attempt, (
        f"a graded submission that passes nothing can score {worst_attempt}, not above the {best_no_attempt} of "
        "an episode that never attempts: the ladder would pay for giving up"
    )
    # A failed attempt raising every process flag it can, at the no-attempt's length terms, still never scores below it.
    flagged_attempt = (
        e["submission"]
        - sum(price for name, price in e["process_prices"].items() if name not in _FAILURE_ONLY_CHECKS)
        - _failure_only_price(e)
    )
    assert flagged_attempt >= best_no_attempt, (
        f"a failed attempt that raises every process flag scores {flagged_attempt}, below the {best_no_attempt} of "
        "an episode that never attempts: the flags would pay for giving up"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_running_out_costs_more_than_one_more_failed_submission(path):
    """The overflow must cost more than spending the submission left and failing at its worst honest price, or
    thinking until the cut is the cheapest way to fail."""
    e = _economy(path)
    assert e["turn_overflow"] > _failed_resubmission(e), (
        f"an overflow costs {e['turn_overflow']}, not above the {_failed_resubmission(e)} of one more failed submission"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_failure_only_process_checks_are_defined_on_failed_episodes(path):
    """The worst solve leaves these out of what it pays, which holds only while the check text says so."""
    checks = {check.name: check.description for check in _audit(path).checks}
    for name in _FAILURE_ONLY_CHECKS:
        if name in checks:
            assert "an episode that never passed every test" in checks[name], f"{name} no longer excludes solves"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_at_every_level_the_worst_solve_beats_the_best_zero_objective_episode(path):
    """A solve costs at most every resubmission its level allows, each recovered cut and the price's cap (never the
    floor, which charges failures only, nor the overflow, since the accept ends the episode); a zero-objective episode
    collects at most the submission bonus. The first must stay above the second, or a group ranks a failure over a
    solve."""
    e = _economy(path)
    assert e["recoveries"] is not None, "an uncapped recovery count leaves the cut price on a solve unbounded"
    best_zero_objective = e["submission"]
    for level in VALID_REASONING_EFFORTS:
        worst_solve = (
            e["solve"]
            + e["submission"]
            - e["resubmission"] * (e["profiles"][level]["max_submissions"] - 1)
            - e["cut"] * e["recoveries"]
            - _worst_length(e, solved=True)
            - sum(price for name, price in e["process_prices"].items() if name not in _FAILURE_ONLY_CHECKS)
        )
        assert worst_solve > best_zero_objective, (
            f"{level}: a full solve can score as low as {worst_solve}, not above the {best_zero_objective} a "
            "zero-objective episode can collect from shaping alone"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_audit_judge_prices_a_wasted_run_and_voids_a_verdict_it_never_reached(path):
    """A price on working-out in a run that did test would teach a policy to submit untested. A verdict the judge
    never reached voids the episode, so an outage trains nothing."""
    judge = _audit(path)
    assert judge.on_error is OnError.INVALID, f"{path.name}: an unjudged episode stays in training"
    assert judge.weight * judge.flag_fraction({"wasted_run": True}) < 0, f"{path.name}: a wasted run costs nothing"
    assert judge.weight * judge.flag_fraction({"reasoning_in_runs": True}) == 0, (
        f"{path.name}: working-out in a scratchpad run that tested something is priced"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_code_contests_recipe_prices_the_documented_checks(path):
    """Moving a check weight or the judge weight re-prices every check while the economy's relations may still hold."""
    judge = _audit(path)
    prices = {check.name: _check_price(judge, check.name) for check in judge.checks}
    assert prices == pytest.approx(_DOCUMENTED_CHECK_PRICES), (
        f"{path.name} prices the audit checks at {prices}, not the reward page's {_DOCUMENTED_CHECK_PRICES}"
    )


def test_the_reward_page_states_the_pinned_check_prices():
    """The audit table is where a recipe author reads a check's price, so the page and the pin above move together."""
    assert _page_check_prices() == pytest.approx(_DOCUMENTED_CHECK_PRICES), (
        f"{_REWARD_PAGE.name}'s audit table prices the checks at {_page_check_prices()}, not {_DOCUMENTED_CHECK_PRICES}"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_code_contests_recipe_sets_the_documented_shaping_prices(path):
    """A price moved inside the relations' band re-prices the recipe; the pin holds each to the value the task page
    states."""
    env = _yaml.load(path)["environment_kwargs"]
    prices = {name: env.get(name) for name in _DOCUMENTED_SHAPING_PRICES}
    assert prices == pytest.approx(_DOCUMENTED_SHAPING_PRICES), (
        f"{path.name} sets the shaping prices {prices}, not the task page's {_DOCUMENTED_SHAPING_PRICES}"
    )


def test_the_task_page_states_the_pinned_shaping_prices():
    assert _page_shaping_prices() == pytest.approx(_DOCUMENTED_SHAPING_PRICES), (
        f"{_README_PAGE.name}'s Shaping row states {_page_shaping_prices()}, not {_DOCUMENTED_SHAPING_PRICES}"
    )


def test_the_relations_pass_the_shipped_economy_and_refuse_a_bad_one():
    """On a copy of a shipped recipe: its own economy holds every relation, and each bad knob breaks the one it
    should."""
    base = _economy(RECIPES[0])

    def edited(**knobs) -> dict:
        return {**base, **knobs}

    def with_profiles(**fields_by_level) -> dict:
        return edited(
            profiles={
                level: {**base["profiles"][level], **fields_by_level.get(level, {})} for level in base["profiles"]
            }
        )

    def holds(e: dict) -> bool:
        return (
            _worst_length(e, solved=False) < _failed_resubmission(e)
            and _prices_fall_with_level(e)
            and _budgets_rise_with_level(e)
            and all(
                check(e, level) is None
                for level in VALID_REASONING_EFFORTS
                for check in (_not_paid_toward_the_budget, _no_slope_past_the_budget, _no_turn_after_the_budget)
            )
        )

    assert holds(base)
    resubmission = _failed_resubmission(base)
    assert _worst_length(edited(price_cap=resubmission), solved=False) >= resubmission, (
        "a cap at a failed resubmission"
    )
    assert _worst_length(edited(floor=resubmission), solved=False) >= resubmission, "a floor at a failed resubmission"
    assert not _prices_fall_with_level(
        with_profiles(low={REASONING_PRICE_KEY: 0.004}, high={REASONING_PRICE_KEY: 0.008})
    )
    assert not _budgets_rise_with_level(
        with_profiles(low={REASONING_BUDGET_KEY: 60000}, high={REASONING_BUDGET_KEY: 40000})
    )
    assert _not_paid_toward_the_budget(edited(floor=0.0), "low"), "no floor pays a failure for more reasoning"
    assert _not_paid_toward_the_budget(with_profiles(low={REASONING_PRICE_KEY: 0.0}), "low"), "an unpriced level"
    assert _no_slope_past_the_budget(with_profiles(low={REASONING_PRICE_KEY: 0.02}), "low"), "capped within a turn"
    room = base["episode_tokens"] - _contract(base).answer_room
    assert _no_turn_after_the_budget(with_profiles(high={REASONING_BUDGET_KEY: room + 1}), "high")
    assert _no_turn_after_the_budget(with_profiles(high={REASONING_BUDGET_KEY: room}), "high") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
