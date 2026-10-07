#!/usr/bin/env python
"""CPU tests: the reward economy of the shipped async-GRPO code-contests recipes.

Every shaping term is small next to the objective by intent, but "small" is a relation between the
recipe's own numbers, and a knob edited alone can break it silently. These pin the relations a run
depends on: the length terms stay under the resubmission price, an honest failed attempt still beats
not attempting, at every effort level a solve paying every per-episode price beats the best
zero-objective episode, and — per effort level — the under-use floor out-slopes the reasoning price, so
below its floor a level is paid to reason more and never less.

The length terms are the price, the floor and the per-turn overlong charge. What they can cost an
episode is the larger of the price's cap plus the floor's weight, and — where a turn can be charged —
the price's cap, the whole overlong penalty and what the floor still charges an episode that reasoned
at least the charged turn's ramp start.

The episode output budget (``rollout_max_episode_tokens``) is the other relation: it has to bind below
what ``max_turns`` turns of ``rollout_max_tokens`` could sample, or it bounds nothing. And the per-turn
thinking budgets are one contract at every family, so a recipe states the same reasoning economy
whichever model it trains.

Run: python tests/cpu/config/test_env_grpo_reward_economy.py  (or pytest)
"""

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import VALID_REASONING_EFFORTS, Message, Trajectory
from src.trainers.grpo.reasoning_terms import (
    REASONING_TARGET_SHARE,
    reasoning_floor_term,
    reasoning_price_term,
    turn_overlong_term,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES = sorted((REPO_ROOT / "examples" / "grpo" / "environmental").rglob("*code-contests*.yaml"))
_yaml = YAML(typ="safe")
_DEFAULTS = AsyncTrainingConfig()
# The most tokens a shipped recipe lets one episode sample: its worst-case trajectory beside the prompt
# budget is what the rollout servers' context window is sized for.
_EPISODE_TOKENS_CEILING = 131072
# The per-turn thinking budget of each level, one contract at every family.
_PER_TURN_BUDGETS = {"low": 8192, "medium": 12288, "high": 16384}
# The knobs every code-contests recipe states explicitly, so a recipe never rides on a default it does
# not spell.
_REQUIRED_KEYS = ("reasoning_price", "reasoning_price_cap", "reasoning_floor", "rollout_max_episode_tokens")


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
        "no_tool_use": env.get("no_tool_use_penalty", 0.0),
        "turn_overflow": env.get("turn_overflow_penalty", 0.0),
        "cut": env.get("length_cutoff_penalty", 0.0),
        "recoveries": env.get("max_length_cutoff_recoveries"),
        "profiles": env["reasoning_effort_profiles"],
        "price": cfg.get("reasoning_price"),
        "price_cap": cfg.get("reasoning_price_cap", _DEFAULTS.reasoning_price_cap),
        "floor": cfg.get("reasoning_floor", _DEFAULTS.reasoning_floor),
        "overlong": cfg.get("turn_overlong_penalty", _DEFAULTS.turn_overlong_penalty),
        "ceiling": cfg.get("rollout_max_thinking_tokens"),
        "max_tokens": cfg.get("rollout_max_tokens", _DEFAULTS.rollout_max_tokens),
        "episode_tokens": cfg.get("rollout_max_episode_tokens"),
        "max_turns": cfg["max_turns"],
    }


def _cap(economy: dict, level: str) -> int:
    """The per-turn thinking cap a turn at ``level`` runs under: the level's budget (every recipe leaves
    the run-wide cap unset, which another case pins)."""
    return economy["profiles"][level]["thinking_tokens"]


def _floor_tokens(economy: dict, level: str) -> int:
    return round(REASONING_TARGET_SHARE * _cap(economy, level))


def _charge(reasoning: int, cap: int) -> float:
    """What one turn of ``reasoning`` tokens under ``cap`` pays of a unit overlong penalty."""
    turn = Message.assistant("step", reasoning_tokens=reasoning, thinking_cap=cap)
    return turn_overlong_term(Trajectory(messages=[Message.user("task"), turn]), penalty=1.0)[0]


def _ramp_start(cap: int) -> int:
    """The least reasoning a charged turn has done: the last count under ``cap`` the charge leaves free."""
    start = min(round(cap * REASONING_TARGET_SHARE), cap - 1)
    assert _charge(start, cap) == 0.0 and _charge(start + 1, cap) < 0.0
    return start


def _worst_length(economy: dict, level: str) -> float:
    """The most the length terms can cost an episode at ``level``. With no turn charged, the price's cap and
    the floor's weight. With one, the price's cap, the whole overlong penalty, and the floor on the least
    reasoning a charged episode has done, the charged turn's ramp start: the floor and the charge price
    opposite ends of the reasoning, so the floor's full weight never lands on a charged episode."""
    worst = economy["price_cap"] + economy["floor"]
    if not economy["overlong"]:
        return worst
    cap = _cap(economy, level)
    floor_left = -reasoning_floor_term([_ramp_start(cap)], cap, economy["floor"])
    return max(worst, economy["price_cap"] + economy["overlong"] + floor_left)


def _price(economy: dict, level: str, tokens: int) -> float:
    return reasoning_price_term([tokens], economy["price"][level], economy["price_cap"])


def _ids(paths):
    return [str(p.relative_to(REPO_ROOT / "examples" / "grpo" / "environmental")) for p in paths]


def test_the_scan_finds_the_code_contests_recipes():
    """Guards the scan: an empty roster would make every case below vacuously pass."""
    assert len(RECIPES) >= 20, f"only {len(RECIPES)} code-contests recipes found — the glob is broken"
    assert all(_economy(p)["price"] is not None and _economy(p)["floor"] > 0 for p in RECIPES), (
        "a code-contests recipe runs without the reasoning terms, which the per-level cases below read"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_every_recipe_states_the_reasoning_economy(path):
    """The price, its cap, the floor and the episode budget are spelled in every recipe, and the price
    maps exactly the effort levels."""
    e = _economy(path)
    missing = [key for key in _REQUIRED_KEYS if key not in e["raw"]]
    assert not missing, f"{path.name} does not state {missing}"
    assert set(e["price"]) == set(VALID_REASONING_EFFORTS)


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_per_turn_thinking_budgets_are_one_contract_at_every_family(path):
    """Every level's ``thinking_tokens`` is the same per-turn budget whichever family the recipe trains,
    and every one sits below the turn cap, so the turn keeps its answer room; the run's own cap is unset,
    so the level's budget is the one the engine enforces."""
    e = _economy(path)
    budgets = {level: e["profiles"][level]["thinking_tokens"] for level in VALID_REASONING_EFFORTS}
    assert budgets == _PER_TURN_BUDGETS, f"{path.name} states per-turn budgets {budgets}"
    assert max(budgets.values()) < e["max_tokens"]
    assert e["ceiling"] is None


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_episode_output_budget_binds_below_what_the_turn_caps_alone_allow(path):
    """Every recipe bounds the episode, at a budget one whole turn fits in and under the ceiling the
    servers' context is sized for; and below what ``max_turns`` turns of ``rollout_max_tokens`` could
    sample, or it would never end an episode."""
    e = _economy(path)
    assert isinstance(e["episode_tokens"], int) and not isinstance(e["episode_tokens"], bool), (
        "a code-contests recipe runs without an episode output budget"
    )
    assert e["max_tokens"] <= e["episode_tokens"] <= _EPISODE_TOKENS_CEILING
    uncapped = e["max_turns"] * e["max_tokens"]
    assert e["episode_tokens"] < uncapped, (
        f"the per-turn caps alone let an episode sample {uncapped} tokens over {e['max_turns']} turns, so a "
        f"budget of {e['episode_tokens']} never binds"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_length_terms_stay_under_the_resubmission_price(path):
    """Resubmitting is the decision this ladder prices hardest; the length terms together bound what
    reasoning length can ever cost, and that bound stays under it at every level."""
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        length = _worst_length(e, level)
        assert length < e["resubmission"], (
            f"{level}: the length terms can cost up to {length}, not under the {e['resubmission']} a resubmission "
            "costs: how long an episode reasons would then outweigh whether it resubmits"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_recovered_cut_costs_less_than_the_attempt_bonus_and_no_more_than_an_overflow(path):
    """The cut price makes the per-turn budget bind without outweighing the loop it protects: a cut
    never costs more than the graded attempt earns, and a recovered cut never costs more than the
    cut that ends the episode."""
    e = _economy(path)
    assert 0 < e["cut"] < e["submission"], f"cut price {e['cut']} against a submission bonus of {e['submission']}"
    assert e["cut"] <= e["turn_overflow"], (
        f"a recovered cut ({e['cut']}) costs more than an overflow ({e['turn_overflow']})"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_an_honest_failed_attempt_beats_not_attempting(path):
    e = _economy(path)
    # Graded, passed nothing, the worst length terms of any level.
    worst_attempt = e["submission"] - max(_worst_length(e, level) for level in VALID_REASONING_EFFORTS)
    best_no_attempt = -e["no_tool_use"]  # never called a tool, free of every length term
    assert worst_attempt > best_no_attempt, (
        f"a graded submission that passes nothing can score {worst_attempt}, not above the {best_no_attempt} of "
        "an episode that never attempts: the ladder would pay for giving up"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_at_every_level_the_worst_solve_beats_the_best_zero_objective_episode(path):
    """A solve costs at most every resubmission its level allows, the overflow, each recovered cut the
    cap admits and the worst length terms; a zero-objective episode collects at most the submission
    bonus. The first must stay above the second at every level, or a group ranks a failure over a solve."""
    e = _economy(path)
    assert e["recoveries"] is not None, "an uncapped recovery count leaves the cut price on a solve unbounded"
    best_zero_objective = e["submission"]
    for level in VALID_REASONING_EFFORTS:
        worst_solve = (
            e["solve"]
            + e["submission"]
            - e["resubmission"] * (e["profiles"][level]["max_submissions"] - 1)
            - e["turn_overflow"]
            - e["cut"] * e["recoveries"]
            - _worst_length(e, level)
        )
        assert worst_solve > best_zero_objective, (
            f"{level}: a full solve can score as low as {worst_solve}, not above the {best_zero_objective} a "
            "zero-objective episode can collect from shaping alone"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_below_its_floor_every_level_is_paid_to_reason_more(path):
    """The price pays for fewer tokens and the floor for more; where both act, the floor must win, or a
    level short of its floor has no reason — or a reason not — to think."""
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        minimum = _floor_tokens(e, level)
        half, full = minimum // 2, minimum
        at_half = _price(e, level, half) + reasoning_floor_term([half], _cap(e, level), e["floor"])
        at_full = _price(e, level, full) + reasoning_floor_term([full], _cap(e, level), e["floor"])
        assert at_full > at_half, (
            f"{level}: reasoning {full} tokens scores {at_full}, not above the {at_half} of reasoning {half} — "
            "below its floor the level is not paid to reason more"
        )
        assert _price(e, level, full) > -e["price_cap"], (
            f"{level}: the price is already capped at the floor ({minimum} tokens), so it has no slope above it"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_higher_level_is_priced_lower_and_asked_for_no_less(path):
    e = _economy(path)
    prices = [e["price"][level] for level in VALID_REASONING_EFFORTS]
    assert prices == sorted(prices, reverse=True) and len(set(prices)) == 3, (
        f"the per-level prices do not fall low > medium > high: {e['price']}"
    )
    floors = [_floor_tokens(e, level) for level in VALID_REASONING_EFFORTS]
    assert floors == sorted(floors), (
        f"a higher level is asked for less reasoning: {dict(zip(VALID_REASONING_EFFORTS, floors, strict=True))}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
