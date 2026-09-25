#!/usr/bin/env python
"""CPU tests for the code-contests evaluation protocols (``eval_protocol``).

``leaderboard`` grades one program per sample with no scratchpad run. A leaderboard-labelled run that
kept a second submission or a scratchpad would report attempts-until-accept as pass@k, so the protocol
pins those budgets at every effort level: a config written for this run that contradicts it is
refused, while a training config's budgets, written under another protocol, give way. ``harness``
pins nothing. The eval script's ``--reasoning_effort none`` evaluates a non-thinking model at no level.

The environments run against a stub sandbox that echoes a canned result (no subprocesses, no network).

Run: python tests/cpu/environments/test_code_contests_protocols.py  (or pytest)
"""

import sys
from types import SimpleNamespace

import pytest

import src.environments.eval_runner as eval_runner
from scripts.environments._common import rollout_config_from_args
from scripts.environments.inference.run_code_contests import (
    FLAG_OWNED_ENV_KWARGS,
    contest_meta,
    default_max_tokens,
    parse_args,
    refuse_flag_owned_env_kwargs,
    resolve_env_config,
    resolve_eval_protocol,
    run_trajectory_path,
)
from src.environments.base import EPISODE_TOOL_BUDGETS_KEY, TOOL_CALL_COUNTS_KEY
from src.environments.envs.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    EVAL_PROTOCOL_KNOB_DEFAULTS,
    EVAL_PROTOCOLS,
    CodeContestsEnvironment,
)
from src.environments.envs.tasks.coding.datasets import ContestSelection
from src.environments.envs.tasks.coding.grading import VERDICT_DETAIL_FULL
from src.environments.registry import resolve_environment
from tests.common.code_contests import SINGLE_TEST_ANSWER, StubSandbox, call_tool, reset_episode

# The shipped recipes' ladder: effort buys submissions and scratchpad runs as well as thinking.
_RECIPE_PROFILES = {
    "low": {"thinking_tokens": 8192, "max_submissions": 1, "max_test_calls": 2},
    "medium": {"thinking_tokens": 12288, "max_submissions": 2, "max_test_calls": 4},
    "high": {"thinking_tokens": 16384, "max_submissions": 3, "max_test_calls": 6},
}


def _env(**kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(language="python", sandbox=StubSandbox(), **kwargs)


def _budgets(env) -> tuple[int, int]:
    return env.max_submissions, env.max_test_calls


def test_leaderboard_pins_one_submission_and_no_scratchpad():
    env = _env(eval_protocol="leaderboard", verdict_detail=VERDICT_DETAIL_FULL)
    assert env.eval_protocol == "leaderboard"
    assert _budgets(env) == (1, 0)
    assert env.grading_spec.verdict_detail == VERDICT_DETAIL_FULL, (
        "the verdict detail is the config's, not the protocol's"
    )
    schemas = {tool["function"]["name"]: tool["function"]["description"] for tool in env.get_tools_schema()}
    assert "This tool is disabled for this task." in schemas["python_repl"]
    assert "This is your only graded submission" in schemas["submit_solution"]


def test_the_harness_is_the_default_and_pins_nothing():
    default = _env()
    assert default.eval_protocol == DEFAULT_EVAL_PROTOCOL == "harness"
    assert _budgets(default) == (2, 5)
    assert _budgets(_env(eval_protocol="harness", max_submissions=3, max_test_calls=0)) == (3, 0)


@pytest.mark.parametrize("contradiction", [{"max_submissions": 3}, {"max_test_calls": 2}])
def test_a_config_contradicting_a_pin_is_refused(contradiction):
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(eval_protocol="leaderboard", **contradiction)


def test_a_config_agreeing_with_the_pins_is_accepted():
    assert _budgets(_env(eval_protocol="leaderboard", max_submissions=1, max_test_calls=0)) == (1, 0)


def test_an_unknown_protocol_is_refused():
    with pytest.raises(ValueError, match="eval_protocol must be one of"):
        _env(eval_protocol="pass_at_k")


def test_every_protocol_pins_only_knobs_the_environment_resolves():
    """A pin on any other knob would be dropped without a word, and the run labelled with a protocol
    it does not follow."""
    for name, pins in EVAL_PROTOCOLS.items():
        assert set(pins) <= set(EVAL_PROTOCOL_KNOB_DEFAULTS), name


def test_the_leaderboard_pins_hold_at_every_effort_level():
    """A training config's ladder binds three submissions at ``high``; under the leaderboard the
    episode still gets one and no scratchpad, the level keeps its thinking budget, and the task
    message states the budgets where the trained ladder put them."""
    harness = _env(reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = reset_episode(harness, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 6, "submit_solution": 3}

    leaderboard = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    for level in _RECIPE_PROFILES:
        traj = reset_episode(leaderboard, {"reasoning_effort": level, **SINGLE_TEST_ANSWER})
        assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 0, "submit_solution": 1}, level
        assert leaderboard.thinking_budget_for_effort(level) == _RECIPE_PROFILES[level]["thinking_tokens"]
        user = next(m for m in reversed(traj.messages) if m.role == "user")
        assert "Budgets for this task: 1 graded submission, 0 scratchpad runs." in user.content, level


def test_a_pinned_profile_key_is_still_validated():
    """The pins supersede a profile's budgets; they do not excuse an invalid one."""
    with pytest.raises(ValueError, match="max_submissions for effort 'high' must be >= 1"):
        _env(eval_protocol="leaderboard", reasoning_effort_profiles={"high": {"max_submissions": 0}})


def test_a_leaderboard_episode_grades_one_program_and_refuses_the_scratchpad():
    env = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})

    assert "Test limit reached (0)" in call_tool(env, traj, "python_repl")
    call_tool(env, traj, "submit_solution")
    assert "Submission limit reached (1)" in call_tool(env, traj, "submit_solution")

    assert traj.info[TOOL_CALL_COUNTS_KEY].get("python_repl", 0) == 0
    assert traj.info[TOOL_CALL_COUNTS_KEY]["submit_solution"] == 1
    assert (traj.info["tests_passed"], traj.info["tests_total"]) == (1, 1)
    assert traj.info["tested_before_submission"] is False


def test_the_registry_presets_take_the_protocol():
    env = resolve_environment("codeforces", {"eval_protocol": "leaderboard", "sandbox": StubSandbox()})
    assert env.grading_spec.comparison == "tokens"
    assert (env.eval_protocol, *_budgets(env)) == ("leaderboard", 1, 0)


# --- The eval script's side: the protocol flag over a training config ---


def test_a_training_config_s_budgets_give_way_to_the_flag_s_protocol():
    """Top-level budgets are the trained contract as much as the profile ones: both give way."""
    trained = {"max_submissions": 3, "max_test_calls": 6, "reasoning_effort_profiles": _RECIPE_PROFILES}
    eval_protocol, contract = resolve_eval_protocol("leaderboard", trained)
    assert eval_protocol == "leaderboard"
    assert "max_submissions" not in contract and "max_test_calls" not in contract
    assert _budgets(_env(**contract, eval_protocol=eval_protocol)) == (1, 0)
    # Without the flag the trained contract stands whole under the harness.
    assert resolve_eval_protocol(None, trained) == ("harness", trained)


def test_the_meta_records_the_config_the_environment_was_built_from():
    """Every flag that overrides the training config is what the meta line records, the env built
    from the same config."""
    trained = {"eval_protocol": "harness", "language": "python", "reasoning_effort": "low", "max_turns": 9}
    flags = SimpleNamespace(eval_protocol="leaderboard", language="cpp", reasoning_effort="high", max_turns=4)
    env_config = resolve_env_config(flags, trained, {"timeout_per_test": 3})
    env = CodeContestsEnvironment(sandbox=StubSandbox(), **env_config)
    recorded = contest_meta("livecodebench", ContestSelection(), env, env_config["reasoning_effort"], env_config)
    assert recorded["env_kwargs"] == {
        "eval_protocol": "leaderboard",
        "language": "cpp",
        "reasoning_effort": "high",
        "max_turns": 4,
        "timeout_per_test": 3,
    }
    assert recorded["eval_protocol"] == env.eval_protocol == "leaderboard"


def test_a_contradiction_stated_for_this_run_is_refused():
    """A config that names the protocol itself, or ``--env_kwargs`` laid over it, contradicts the pin."""
    own = {"eval_protocol": "leaderboard", "max_submissions": 3}
    eval_protocol, contract = resolve_eval_protocol(None, own)
    assert (eval_protocol, contract) == ("leaderboard", own)
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(**contract)
    _, contract = resolve_eval_protocol("leaderboard", {"max_submissions": 3})
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(**{**contract, "eval_protocol": "leaderboard", "max_submissions": 3})


def test_the_eval_script_takes_the_protocol_from_its_flag_only():
    with pytest.raises(SystemExit, match="--eval_protocol, not --env_kwargs"):
        refuse_flag_owned_env_kwargs({"eval_protocol": "leaderboard"})


def test_every_env_option_a_flag_sets_is_refused_in_env_kwargs():
    """``--env_kwargs`` is laid over the flags, so an option it shares with a flag silently beats it:
    ``{"max_turns": 3}`` would cap a ``--max_turns 16`` run at three turns. Read off the resolver, so a
    flag added to it without a refusal fails here."""
    flags = SimpleNamespace(max_turns=16, language="python", eval_protocol="harness", reasoning_effort="high")
    flag_set = set(resolve_env_config(flags, {}, {}))
    assert flag_set <= set(FLAG_OWNED_ENV_KWARGS), sorted(flag_set - set(FLAG_OWNED_ENV_KWARGS))
    for key in sorted(flag_set):
        with pytest.raises(SystemExit, match=f"--{key}, not --env_kwargs"):
            refuse_flag_owned_env_kwargs({key: 3})
    refuse_flag_owned_env_kwargs({"timeout_per_test": 3})


def test_a_default_run_keeps_its_trajectory_file_name():
    """The protocol and the selection name the file only where they depart from the defaults, so an
    existing harness run lands where it always did."""
    args = SimpleNamespace(
        save_trajectories=None, trajectory_dir="/runs", model="org/m", adapter="livecodebench", split="test"
    )
    assert run_trajectory_path(args, _env(), ContestSelection()) == "/runs/org-m__livecodebench__test__python.jsonl"
    window = ContestSelection.parse("2025-01-04", "2025-04-06", ["atcoder"])
    assert run_trajectory_path(args, _env(eval_protocol="leaderboard"), window) == (
        "/runs/org-m__livecodebench__test__python__leaderboard__2025-01-04..2025-04-06_atcoder.jsonl"
    )
    assert run_trajectory_path(args, _env(), window) == (
        "/runs/org-m__livecodebench__test__python__2025-01-04..2025-04-06_atcoder.jsonl"
    )


# --- No effort level: a non-thinking model ---


def _parse_flags(monkeypatch, *flags: str):
    """The eval script's own parser over ``flags`` plus the two it requires."""
    monkeypatch.setattr(sys, "argv", ["run_code_contests.py", "--dataset", "d", "--model", "m", *flags])
    return parse_args()


def test_the_effort_flag_spells_no_level_as_none(monkeypatch):
    """``--env_kwargs`` may not set an option its flag owns, so the flag itself must reach no level:
    ``none`` is ``reasoning_effort=None``, over a training config's level too."""
    args = _parse_flags(monkeypatch, "--reasoning_effort", "none")
    assert resolve_env_config(args, {}, {})["reasoning_effort"] is None
    assert resolve_env_config(args, {"reasoning_effort": "high"}, {})["reasoning_effort"] is None


@pytest.mark.parametrize(("flag", "max_tokens"), [(None, 12288), ("low", 8192), ("high", 20480), ("none", 32768)])
def test_the_default_generation_budget_follows_the_flag_s_level(flag, max_tokens):
    """A level's thinking budget plus the solution headroom; no level has no budget to size it from and
    takes the training rollout's default instead of looking a profile up."""
    assert default_max_tokens(flag) == max_tokens


async def test_a_no_level_episode_sends_no_level_and_records_none(monkeypatch):
    """``--reasoning_effort none`` end to end: the episode's requests carry no level, no thinking budget
    and no template variable, so a server without a reasoning parser serves them, and the trajectory and
    the meta line record no level."""
    args = _parse_flags(monkeypatch, "--reasoning_effort", "none")
    env_config = resolve_env_config(args, {}, {})
    env = CodeContestsEnvironment(sandbox=StubSandbox(), **env_config)
    rollout = rollout_config_from_args(
        args, None, default_temperature=0.2, default_max_tokens=default_max_tokens(args.reasoning_effort)
    )
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            answer="done", finish_reason="stop", completion_tokens=3, tool_calls=None, reasoning=None, token_ids=None
        )

    monkeypatch.setattr(eval_runner, "generate_openai_response", fake_generate)
    traj = await eval_runner.run_episode(env, "solve it", dict(SINGLE_TEST_ANSWER), None, rollout=rollout)

    assert env.reasoning_effort is None
    assert calls and all(call["extra_body"] == {} for call in calls)
    assert calls[0]["max_tokens"] == 32768
    assert eval_runner.serialize_trajectory(traj)["reasoning_effort"] is None
    meta = contest_meta(args.adapter, ContestSelection(), env, env_config["reasoning_effort"], env_config)
    assert meta["reasoning_effort"] is None and meta["env_kwargs"]["reasoning_effort"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
