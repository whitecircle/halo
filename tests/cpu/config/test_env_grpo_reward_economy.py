#!/usr/bin/env python
"""The reward economy of the shipped async-GRPO code-contests recipes. Shaping terms are small next to the objective
by intent, and a knob edited alone can break that silently, so these pin the relations a run depends on and that
the prices match what the reward and task pages state."""

import re
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from halo.configs.grpo_environmental import AsyncRolloutConfig
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
from halo.trainers.grpo.environmental.reasoning import (
    REASONING_TARGET_SHARE,
    reasoning_floor_term,
    reasoning_price_term,
)
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
_EPISODE_TOKENS_CEILINGS = {"qwen3_5": 160000}
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
# The curriculum stages run one after another on one cluster: one economy and one set of limits.
_STAGE_RECIPES = [p for p in RECIPES if "-stage" in p.name]
_STAGE_SHARED_KEYS = (
    "rollout_max_tokens",
    "rollout_max_answer_tokens",
    "rollout_max_episode_tokens",
    "max_train_row_tokens",
    "request_timeout",
    "episode_timeout",
    "reasoning_floor",
    "reasoning_price",
    "reasoning_price_cap",
    "reasoning_terms_in_failed_groups",
)
# The per-turn thinking budget of each level, one contract at every family.
_PER_TURN_BUDGETS = {"low": 8192, "medium": 12288, "high": 16384}
# The knobs every code-contests recipe states explicitly, so a recipe never rides on a default it does
# not spell. The price is not among them: a recipe that runs none omits it and its cap.
_REQUIRED_KEYS = ("reasoning_floor", "rollout_max_episode_tokens")


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
    return {
        "raw": cfg,
        # The environment term on a solve: the grade is 1 or 0, so only the weight prices it.
        "solve": sum(term.get("weight", 1.0) for term in terms if term["source"] == "environment"),
        "submission": env.get("submission_reward", 0.0),
        "resubmission": env.get("resubmission_penalty", 0.0),
        "unrun": env.get("unrun_submission_penalty", 0.0),
        "no_tool_use": env.get("no_tool_use_penalty", 0.0),
        "turn_overflow": env.get("turn_overflow_penalty", 0.0),
        "cut": env.get("length_cutoff_penalty", 0.0),
        "recoveries": env.get("max_length_cutoff_recoveries"),
        "profiles": env["reasoning_effort_profiles"],
        "price": cfg.get("reasoning_price"),
        "price_cap": (
            cfg.get("reasoning_price_cap", _DEFAULTS.reasoning_price_cap)
            if cfg.get("reasoning_price") is not None
            else 0.0
        ),
        # The floor's weight: the most it charges, an episode that reasoned nothing.
        "floor": cfg.get("reasoning_floor", _DEFAULTS.reasoning_floor),
        "past_budget": cfg.get("reasoning_price_past_budget", _DEFAULTS.reasoning_price_past_budget),
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
    """The per-turn thinking budget of ``level``: what the template states and the floor reads."""
    return economy["profiles"][level]["thinking_tokens"]


def _soft(economy: dict, level: str) -> bool:
    return economy["profiles"][level].get("enforce_thinking_cap") is False


def _cap(economy: dict, level: str) -> int | None:
    """The per-turn reasoning cap the engine enforces at ``level``: the level's budget, or under a soft budget the
    ceiling the answer room leaves of the turn. None on SGLang, which runs no answer bound and caps no reasoning."""
    if not _soft(economy, level):
        return _budget(economy, level)
    return None if economy["answer_tokens"] is None else economy["max_tokens"] - economy["answer_tokens"]


def _floor_tokens(economy: dict, level: str) -> int:
    """The reasoning the floor asks of an episode at ``level``: its share of the level's per-turn budget."""
    return round(REASONING_TARGET_SHARE * _budget(economy, level))


def _worst_length(economy: dict) -> float:
    """The most the length terms can cost an episode: the price's cap (0 with the price off) and the floor's whole
    weight, or, priced past a budget, the larger of the two."""
    if economy["past_budget"]:
        return max(economy["price_cap"], economy["floor"])
    return economy["price_cap"] + economy["floor"]


def _failure_only_price(economy: dict) -> float:
    """The most one failed submission adds in failure-only prices: the environment's price for a program never
    run as submitted and the failure-only checks, which also cover a program no earlier run exercised, so both
    can fire on one submission."""
    return economy["unrun"] + sum(economy["process_prices"].get(name, 0.0) for name in _FAILURE_ONLY_CHECKS)


def _failed_resubmission(economy: dict) -> float:
    """One more submission that fails, at its worst honest price: the resubmission and its failure-only price."""
    return economy["resubmission"] + _failure_only_price(economy)


def _price(economy: dict, level: str, tokens: int) -> float:
    if economy["price"] is None:
        return 0.0
    budget = _budget(economy, level) if economy["past_budget"] else None
    return reasoning_price_term([tokens], economy["price"][level], economy["price_cap"], budget)


def _not_paid_to_reason_more(economy: dict, level: str) -> str | None:
    """Why ``level`` is not paid to reason more below its floor, or None: reasoning the floor's tokens
    must score above reasoning half of them, and the price must not be capped by then, which would
    leave it no slope above the floor."""
    full = _floor_tokens(economy, level)
    half = full // 2
    at_half = _price(economy, level, half) + reasoning_floor_term([half], _budget(economy, level), economy["floor"])
    at_full = _price(economy, level, full) + reasoning_floor_term([full], _budget(economy, level), economy["floor"])
    if at_full <= at_half:
        return f"reasoning {full} tokens scores {at_full}, not above the {at_half} of reasoning {half}"
    if economy["price"] is not None and _price(economy, level, full) <= -economy["price_cap"]:
        return f"the price is already capped at the floor ({full} tokens), so it has no slope above it"
    return None


def _prices_fall_with_level(economy: dict) -> bool:
    prices = [economy["price"][level] for level in VALID_REASONING_EFFORTS]
    return prices == sorted(prices, reverse=True) and len(set(prices)) == 3


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
    assert all(_economy(p)["floor"] > 0 for p in RECIPES), (
        "a code-contests recipe runs without the reasoning floor, which the cases below price"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_recipe_states_the_reasoning_economy(path):
    """A price maps exactly the effort levels and brings its cap, and a recipe without one states no cap."""
    e = _economy(path)
    missing = [key for key in _REQUIRED_KEYS if key not in e["raw"]]
    assert not missing, f"{path.name} does not state {missing}"
    if e["price"] is None:
        assert "reasoning_price_cap" not in e["raw"], f"{path.name} states a cap for a price it does not run"
    else:
        assert "reasoning_price_cap" in e["raw"] and set(e["price"]) == set(VALID_REASONING_EFFORTS)


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_per_turn_thinking_budgets_are_one_contract_at_every_family(path):
    """Every level's budget is the same whichever family the recipe trains and sits below the turn cap. A level set
    is all enforced or all soft, and soft budgets are priced past them (their only ceiling below the engine's)."""
    e = _economy(path)
    budgets = {level: _budget(e, level) for level in VALID_REASONING_EFFORTS}
    assert budgets == _PER_TURN_BUDGETS, f"{path.name} states per-turn budgets {budgets}"
    assert max(budgets.values()) < e["max_tokens"]
    assert e["ceiling"] is None
    soft = {_soft(e, level) for level in VALID_REASONING_EFFORTS}
    assert len(soft) == 1, f"{path.name} mixes enforced and soft budgets"
    assert (e["price"] is not None and e["past_budget"]) == (soft == {True}), (
        f"{path.name}: soft budgets are priced past them, and only soft budgets are"
    )
    if soft == {True}:
        assert all(_cap(e, level) is None or budget < _cap(e, level) for level, budget in budgets.items()), (
            f"{path.name}: a soft budget at or past its ceiling is no softer than an enforced one"
        )


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


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_answer_room_binds_below_the_turn_ceiling_at_every_level(path):
    """A level whose cap plus the room reached ``rollout_max_tokens`` would leave the turn room to carry the
    reasoning on in a program's comments. A soft level's ceiling is where the room is left, so it binds by
    construction. SGLang forces no close at the cap, so its recipes run without the bound."""
    e = _economy(path)
    if e["raw"].get("rollout_backend") == "sglang":
        assert e["answer_tokens"] is None, f"{path.name} bounds an answer room SGLang never opens"
        return
    assert isinstance(e["answer_tokens"], int), f"{path.name} does not state rollout_max_answer_tokens"
    for level in VALID_REASONING_EFFORTS:
        if _soft(e, level):
            continue
        assert _cap(e, level) + e["answer_tokens"] < e["max_tokens"], (
            f"at {level}, a {_cap(e, level)}-token cap plus a {e['answer_tokens']}-token answer room does not "
            f"bind below the {e['max_tokens']}-token turn ceiling"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_length_terms_stay_under_one_more_failed_submission(path):
    """The length terms stay under one more failed submission at its worst honest price, so guessing again is never
    the cheaper way to fail."""
    e = _economy(path)
    assert _worst_length(e) < _failed_resubmission(e), (
        f"the length terms can cost up to {_worst_length(e)}, not under the {_failed_resubmission(e)} of one more "
        "failed submission: how long an episode reasons would then outweigh whether it resubmits"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_one_turn_at_the_ceiling_leaves_the_price_its_slope(path):
    """A price that reaches its cap inside one turn charges every longer episode the same, so more reasoning is
    free. A soft turn reasons to its ceiling (:func:`_cap`), or through the whole turn where the engine closes
    none."""
    e = _economy(path)
    if not e["past_budget"]:
        return
    for level in VALID_REASONING_EFFORTS:
        ceiling = _cap(e, level) or e["max_tokens"]
        charged = -_price(e, level, ceiling)
        assert 0 < charged < e["price_cap"], (
            f"{level}: one turn reasoning to the {ceiling}-token ceiling is charged {charged} under a cap of "
            f"{e['price_cap']}"
        )


def test_the_curriculum_stages_run_one_economy_and_one_set_of_limits():
    assert len(_STAGE_RECIPES) == 3, f"expected the three curriculum stages, found {[p.name for p in _STAGE_RECIPES]}"
    stated = [{key: _yaml.load(p).get(key) for key in _STAGE_SHARED_KEYS} for p in _STAGE_RECIPES]
    assert stated[0] == stated[1] == stated[2], dict(zip([p.name for p in _STAGE_RECIPES], stated, strict=True))


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
    # Graded, passed nothing, the worst length terms.
    worst_attempt = e["submission"] - _worst_length(e)
    best_no_attempt = -e["no_tool_use"]  # never called a tool, free of every length term
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
    """A solve costs at most every resubmission, each recovered cut and the worst length terms; a zero-objective
    episode collects at most the submission bonus. The first must stay above the second, or a group ranks a
    failure over a solve."""
    e = _economy(path)
    assert e["recoveries"] is not None, "an uncapped recovery count leaves the cut price on a solve unbounded"
    best_zero_objective = e["submission"]
    for level in VALID_REASONING_EFFORTS:
        worst_solve = (
            e["solve"]
            + e["submission"]
            - e["resubmission"] * (e["profiles"][level]["max_submissions"] - 1)
            - e["cut"] * e["recoveries"]
            - _worst_length(e)
            - sum(price for name, price in e["process_prices"].items() if name not in _FAILURE_ONLY_CHECKS)
        )
        assert worst_solve > best_zero_objective, (
            f"{level}: a full solve can score as low as {worst_solve}, not above the {best_zero_objective} a "
            "zero-objective episode can collect from shaping alone"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_below_its_floor_every_level_is_paid_to_reason_more(path):
    """The price pays for fewer tokens and the floor for more; where both act, the floor must win."""
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        reason = _not_paid_to_reason_more(e, level)
        assert reason is None, f"{level}: {reason} — below its floor the level is not paid to reason more"


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_higher_level_is_priced_lower_and_asked_for_no_less(path):
    e = _economy(path)
    if e["price"] is not None:
        assert _prices_fall_with_level(e), f"the per-level prices do not fall low > medium > high: {e['price']}"
    floors = [_floor_tokens(e, level) for level in VALID_REASONING_EFFORTS]
    assert floors == sorted(floors), (
        f"a higher level is asked for less reasoning: {dict(zip(VALID_REASONING_EFFORTS, floors, strict=True))}"
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


def test_the_price_relations_pass_a_sound_price_and_refuse_a_bad_one():
    """On a copy of a shipped recipe pricing every token (floor 0.05), the templates' example price holds every
    relation and each bad price breaks the one it should."""
    base = _economy(RECIPES[0])

    def priced(price: dict[str, float], cap: float) -> dict:
        return {**base, "price": price, "price_cap": cap, "floor": 0.05, "past_budget": False}

    sound = priced({"low": 0.0037, "medium": 0.0013, "high": 0.00018}, cap=0.05)
    assert _worst_length(sound) < _failed_resubmission(sound) and _prices_fall_with_level(sound)
    assert all(_not_paid_to_reason_more(sound, level) is None for level in VALID_REASONING_EFFORTS)
    # A cap that, with the floor, reaches one more failed submission.
    assert _worst_length(priced(sound["price"], cap=_failed_resubmission(base) - 0.05)) >= _failed_resubmission(base)
    assert not _prices_fall_with_level(priced({"low": 0.00018, "medium": 0.0013, "high": 0.0037}, cap=0.05))
    # 0.01 per 1k at low out-slopes a 0.05 floor over its 6144-token target (0.0081 per 1k).
    assert _not_paid_to_reason_more(priced({"low": 0.01, "medium": 0.005, "high": 0.001}, cap=0.08), "low")
    # A cap the low level's price reaches before its floor.
    assert _not_paid_to_reason_more(priced({"low": 0.005, "medium": 0.002, "high": 0.0005}, cap=0.02), "low")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
