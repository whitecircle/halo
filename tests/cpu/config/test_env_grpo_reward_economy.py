#!/usr/bin/env python
"""CPU tests: the reward economy of the shipped async-GRPO code-contests recipes.

Every shaping term is small next to the objective by intent, but "small" is a relation between the
recipe's own numbers, and a knob edited alone can break it silently. These pin the relations a run
depends on: the length terms stay under the resubmission price, a fix costs less than a re-roll yet a
rescue still scores under a first-try solve, an honest failed attempt still beats not attempting, the
worst solve beats the best zero-objective episode, and — per effort level — the under-use floor
out-slopes the length price, so below its floor a level is paid to reason more and never less.

Run: python tests/cpu/config/test_env_grpo_reward_economy.py  (or pytest)
"""

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import VALID_REASONING_EFFORTS
from src.environments.episode import effort_length_floor, effort_length_penalty

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES = sorted((REPO_ROOT / "examples" / "grpo" / "environmental").rglob("*code-contests*.yaml"))
OBJECTIVE_MAX = 1.0  # the environment term: a pass fraction in [0, 1], whatever its exponent

_yaml = YAML(typ="safe")
_DEFAULTS = AsyncTrainingConfig()


def _economy(path: Path) -> dict:
    cfg = _yaml.load(path)
    env = cfg["environment_kwargs"]
    profiles = env["reasoning_effort_profiles"]
    return {
        "submission": env.get("submission_reward", 0.0),
        "execution": env.get("execution_progress_reward", 0.0),
        "resubmission": env.get("resubmission_penalty", 0.0),
        "refund": env.get("improved_resubmission_refund", 0.0),
        "no_tool_use": env.get("no_tool_use_penalty", 0.0),
        "turn_overflow": env.get("turn_overflow_penalty", 0.0),
        "cut": env.get("length_cutoff_penalty", 0.0),
        "max_submissions": max(profile["max_submissions"] for profile in profiles.values()),
        "profiles": profiles,
        "k0": cfg.get("effort_length_penalty_k0"),
        "tau": cfg.get("effort_length_penalty_tau", _DEFAULTS.effort_length_penalty_tau),
        "c_max": cfg.get("effort_length_penalty_c_max", _DEFAULTS.effort_length_penalty_c_max),
        "l_norm": cfg.get("effort_length_penalty_l_norm", _DEFAULTS.effort_length_penalty_l_norm),
        "levels": cfg.get("effort_length_penalty_levels", _DEFAULTS.effort_length_penalty_levels),
        "floor": cfg.get("effort_length_floor_weight", 0.0),
        "floor_budgets": cfg.get("effort_length_floor_budgets", _DEFAULTS.effort_length_floor_budgets),
    }


def _floor_tokens(economy: dict, level: str) -> int:
    return round(economy["floor_budgets"] * economy["profiles"][level]["thinking_tokens"])


def _price(economy: dict, level: str, tokens: int) -> float:
    levels = economy["levels"]
    return effort_length_penalty(
        [tokens],
        levels[level],
        min(levels.values()),
        economy["k0"],
        economy["tau"],
        economy["c_max"],
        economy["l_norm"],
    )


def _ids(paths):
    return [str(p.relative_to(REPO_ROOT / "examples" / "grpo" / "environmental")) for p in paths]


def test_the_scan_finds_the_code_contests_recipes():
    """Guards the scan: an empty roster would make every case below vacuously pass."""
    assert len(RECIPES) >= 20, f"only {len(RECIPES)} code-contests recipes found — the glob is broken"
    assert all(_economy(p)["k0"] is not None and _economy(p)["floor"] > 0 for p in RECIPES), (
        "a code-contests recipe runs without the effort length terms, which the per-level cases below read"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_length_terms_stay_under_the_resubmission_price(path):
    """Resubmitting is the decision this ladder prices hardest; the cap and the floor weight together
    bound what reasoning length can ever cost, and that bound stays under it."""
    e = _economy(path)
    length = e["c_max"] + e["floor"]
    assert length < e["resubmission"], (
        f"the length terms can cost up to {length}, not under the {e['resubmission']} a resubmission costs: how "
        "long an episode reasons would then outweigh whether it resubmits"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_fix_is_cheaper_than_a_re_roll_and_a_rescue_still_trails_a_first_try_solve(path):
    """The refund is what separates the two resubmissions a group contains. It stays partial: a free
    rescue would score level with a first-try solve and take the pressure off the first submission."""
    e = _economy(path)
    improved = e["resubmission"] * (1 - e["refund"])
    assert 0 < improved < e["resubmission"], (
        f"an improving resubmission costs {improved} against {e['resubmission']} for one that does not: the "
        "price must separate a fix from a re-roll, and a fix must still cost something"
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
    worst_attempt = e["submission"] - (e["c_max"] + e["floor"])  # graded, passed nothing, worst length terms
    best_no_attempt = -e["no_tool_use"]  # never called a tool, free of every length term
    assert worst_attempt > best_no_attempt, (
        f"a graded submission that passes nothing can score {worst_attempt}, not above the {best_no_attempt} of "
        "an episode that never attempts: the ladder would pay for giving up"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_the_worst_solve_beats_the_best_zero_objective_episode(path):
    e = _economy(path)
    shaping = e["submission"] + e["execution"]
    worst_solve = (
        OBJECTIVE_MAX
        + shaping
        - e["resubmission"] * (e["max_submissions"] - 1 - e["refund"])  # the solving one is the only fix
        - e["turn_overflow"]
        - (e["c_max"] + e["floor"])
    )
    assert worst_solve > shaping, (
        f"a full solve can score as low as {worst_solve}, not above the {shaping} a zero-objective episode can "
        "collect from shaping alone"
    )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_below_its_floor_every_level_is_paid_to_reason_more(path):
    """The price pays for fewer tokens and the floor for more; where both act, the floor must win, or a
    level short of its floor has no reason — or a reason not — to think."""
    e = _economy(path)
    for level in VALID_REASONING_EFFORTS:
        minimum = _floor_tokens(e, level)
        half, full = minimum // 2, minimum
        at_half = _price(e, level, half) + effort_length_floor([half], minimum, e["floor"])
        at_full = _price(e, level, full) + effort_length_floor([full], minimum, e["floor"])
        assert at_full > at_half, (
            f"{level}: reasoning {full} tokens scores {at_full}, not above the {at_half} of reasoning {half} — "
            "below its floor the level is not paid to reason more"
        )
        assert _price(e, level, full) > -e["c_max"], (
            f"{level}: the price is already capped at the floor ({minimum} tokens), so it has no slope above it"
        )


@pytest.mark.parametrize("path", RECIPES, ids=_ids(RECIPES))
def test_a_higher_level_is_priced_lower_and_asked_for_no_less(path):
    e = _economy(path)
    order = sorted(VALID_REASONING_EFFORTS, key=lambda level: e["levels"][level])
    assert order == list(VALID_REASONING_EFFORTS), f"the level scalars do not order low < medium < high: {e['levels']}"
    floors = [_floor_tokens(e, level) for level in order]
    assert floors == sorted(floors), (
        f"a higher level is asked for less reasoning: {dict(zip(order, floors, strict=False))}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
