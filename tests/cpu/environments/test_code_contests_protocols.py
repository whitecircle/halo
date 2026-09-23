#!/usr/bin/env python
"""CPU tests for the code-contests evaluation protocols (``eval_protocol``).

``leaderboard`` runs a benchmark's own contract inside the harness: one graded program, no scratchpad
run, the verdict alone. A leaderboard-labelled run that kept a second submission or a scratchpad would
report attempts-until-accept as pass@k, so the protocol pins those knobs at every effort level and
refuses a config that contradicts it. ``harness`` pins nothing.

The environments run against a stub sandbox that echoes a canned result (no subprocesses, no network).

Run: python tests/cpu/environments/test_code_contests_protocols.py  (or pytest)
"""

import pytest

from scripts.environments.inference import regrade_trajectories
from scripts.environments.inference.run_code_contests import refuse_flag_owned_env_kwargs
from src.environments.base import EPISODE_TOOL_BUDGETS_KEY, TOOL_CALL_COUNTS_KEY
from src.environments.envs.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    EVAL_PROTOCOL_KNOB_DEFAULTS,
    EVAL_PROTOCOLS,
    CodeContestsEnvironment,
)
from src.environments.registry import resolve_environment
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall


class _EchoSandbox(SandboxExecutor):
    """Every run prints ``X`` and exits cleanly."""

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return SandboxResult(stdout="X\n", returncode=0)


# The shipped recipes' ladder: effort buys submissions and scratchpad runs as well as thinking.
_RECIPE_PROFILES = {
    "low": {"thinking_tokens": 8192, "max_submissions": 1, "max_test_calls": 2},
    "medium": {"thinking_tokens": 12288, "max_submissions": 2, "max_test_calls": 4},
    "high": {"thinking_tokens": 16384, "max_submissions": 3, "max_test_calls": 6},
}
_TESTS = {"answer": {"tests": [{"input": "", "output": "X"}]}}


def _env(**kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(language="python", sandbox=_EchoSandbox(), **kwargs)


def _reset(env, context):
    ids, _ = env.reset(["solve it"], [context])
    return env.get_trajectories(ids)[0]


def _call(env, traj, name):
    results, _ = env._execute_tool_calls([NativeToolCall(id="c", name=name, arguments={"code": "print('X')"})], traj)
    return results[0].content


def _knobs(env) -> tuple:
    return env.max_submissions, env.max_test_calls, env.grading_spec.verdict_detail


def test_leaderboard_pins_one_submission_no_scratchpad_and_outcome_verdicts():
    env = _env(eval_protocol="leaderboard")
    assert env.eval_protocol == "leaderboard"
    assert _knobs(env) == (1, 0, "outcome")
    schemas = {tool["function"]["name"]: tool["function"]["description"] for tool in env.get_tools_schema()}
    assert "This tool is disabled for this task." in schemas["python_repl"]
    assert "This is your only graded submission" in schemas["submit_solution"]


def test_the_harness_is_the_default_and_pins_nothing():
    default = _env()
    assert default.eval_protocol == DEFAULT_EVAL_PROTOCOL == "harness"
    assert _knobs(default) == (2, 5, "full")
    configured = _env(eval_protocol="harness", max_submissions=3, max_test_calls=0, verdict_detail="outcome")
    assert _knobs(configured) == (3, 0, "outcome")


@pytest.mark.parametrize("contradiction", [{"max_submissions": 3}, {"max_test_calls": 2}, {"verdict_detail": "full"}])
def test_a_config_contradicting_a_pin_is_refused(contradiction):
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(eval_protocol="leaderboard", **contradiction)


def test_a_config_agreeing_with_the_pins_is_accepted():
    env = _env(eval_protocol="leaderboard", max_submissions=1, max_test_calls=0, verdict_detail="outcome")
    assert _knobs(env) == (1, 0, "outcome")


def test_an_unknown_protocol_is_refused():
    with pytest.raises(ValueError, match="eval_protocol must be one of"):
        _env(eval_protocol="pass_at_k")


def test_every_protocol_pins_only_knobs_the_environment_resolves():
    """A pin on any other knob would be dropped without a word, and the run labelled with a protocol
    it does not follow."""
    for name, pins in EVAL_PROTOCOLS.items():
        assert set(pins) <= set(EVAL_PROTOCOL_KNOB_DEFAULTS), name
        resolved = dict(zip(EVAL_PROTOCOL_KNOB_DEFAULTS, _knobs(_env(eval_protocol=name)), strict=True))
        assert resolved == {**EVAL_PROTOCOL_KNOB_DEFAULTS, **pins}, name


def test_the_leaderboard_pins_hold_at_every_effort_level():
    """A training config's ladder binds three submissions at ``high``; under the leaderboard the
    episode still gets one and no scratchpad, while the level keeps its thinking budget."""
    harness = _env(reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = _reset(harness, {"reasoning_effort": "high", **_TESTS})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 6, "submit_solution": 3}

    leaderboard = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    for level in _RECIPE_PROFILES:
        traj = _reset(leaderboard, {"reasoning_effort": level, **_TESTS})
        assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 0, "submit_solution": 1}, level
        assert leaderboard.thinking_budget_for_effort(level) == _RECIPE_PROFILES[level]["thinking_tokens"]
        # Uniform budgets need no per-episode contract in the task message.
        assert "Budgets for this task" not in traj.messages[-1].content


def test_a_pinned_profile_key_is_still_validated():
    """The pins supersede a profile's budgets; they do not excuse an invalid one."""
    with pytest.raises(ValueError, match="max_submissions for effort 'high' must be >= 1"):
        _env(eval_protocol="leaderboard", reasoning_effort_profiles={"high": {"max_submissions": 0}})


def test_a_leaderboard_episode_grades_one_program_and_refuses_the_scratchpad():
    env = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = _reset(env, {"reasoning_effort": "high", **_TESTS})

    assert "Test limit reached (0)" in _call(env, traj, "python_repl")
    _call(env, traj, "submit_solution")
    assert "Submission limit reached (1)" in _call(env, traj, "submit_solution")

    assert traj.info[TOOL_CALL_COUNTS_KEY].get("python_repl", 0) == 0
    assert traj.info[TOOL_CALL_COUNTS_KEY]["submit_solution"] == 1
    assert (traj.info["tests_passed"], traj.info["tests_total"]) == (1, 1)
    assert traj.info["tested_before_submission"] is False


def test_the_registry_presets_take_the_protocol():
    env = resolve_environment("codeforces", {"eval_protocol": "leaderboard", "sandbox": _EchoSandbox()})
    assert env.grading_spec.comparison == "tokens"
    assert (env.eval_protocol, env.max_submissions, env.max_test_calls) == ("leaderboard", 1, 0)


def test_the_regrader_rebuilds_the_run_s_protocol():
    """The re-grade counts submissions up to the rebuilt env's budget where an episode stamped none."""
    meta = {
        "env_type": "codeforces",
        "language": "python",
        "env_kwargs": {"reasoning_effort_profiles": _RECIPE_PROFILES},
    }
    rebuilt = regrade_trajectories.rebuild_environment({**meta, "eval_protocol": "leaderboard"})
    assert (rebuilt.eval_protocol, rebuilt.max_submissions) == ("leaderboard", 1)
    # A meta line naming no protocol ran the harness.
    assert regrade_trajectories.rebuild_environment(meta).eval_protocol == "harness"


def test_the_eval_script_takes_the_protocol_from_its_flag_only():
    with pytest.raises(SystemExit, match="--eval_protocol, not --env_kwargs"):
        refuse_flag_owned_env_kwargs({"eval_protocol": "leaderboard"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
