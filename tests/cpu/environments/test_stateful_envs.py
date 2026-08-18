#!/usr/bin/env python
"""
Integration tests for stateful, multi-turn execution environments.

These drive the environments through the real reset/step API with tool calls, asserting the
behaviour that makes SWE / code-contest RL work:

- SweEnvironment: a file written on one turn is readable by code run on a LATER turn (persistent
  per-episode workspace), and two concurrent episodes never see each other's files.
- CodeContestsEnvironment: submit_solution grades against hidden tests through the sandbox, for
  Python and (when g++ is present) C++ — including the language-aware test REPL tool name.

Run: python tests/cpu/environments/test_stateful_envs.py
"""

import json
import shutil

import pytest

from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.swe import SweEnvironment

_HAS_GPP = shutil.which("g++") is not None


def _skip(reason: str):
    """Register a genuine skip (not a silent pass): ``unittest.SkipTest`` is pytest's skip signal."""
    import unittest

    raise unittest.SkipTest(reason)


def _tool_call(name: str, **arguments) -> dict:
    """Build one OpenAI-format tool call for the env step context."""
    return {"id": f"call_{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _last_tool_result(trajectory) -> str:
    """Return the content of the most recent tool-result message in the trajectory."""
    for msg in reversed(trajectory.messages):
        if msg.role == "tool":
            return msg.content
    return ""


# SweEnvironment — persistent workspace across turns


def test_code_env_state_persists_across_turns():
    env = SweEnvironment(max_turns=10)
    try:
        episode_ids, _ = env.reset(["Build a helper and use it."])

        env.step(
            episode_ids,
            ["writing helper"],
            [
                {
                    "tool_calls": [
                        _tool_call("write_file", path="helper.py", content="def triple(x):\n    return x * 3\n")
                    ]
                }
            ],
        )

        # Turn 2 imports turn 1's file: only reachable if the session persisted.
        env.step(
            episode_ids,
            ["running"],
            [{"tool_calls": [_tool_call("run_code", code="import helper\nprint(helper.triple(14))")]}],
        )

        traj = env.get_trajectories(episode_ids)[0]
        assert _last_tool_result(traj).strip() == "42", f"run_code did not see the persisted file: {traj.info}"
    finally:
        env.close()


def test_code_env_read_file_sees_prior_write():
    env = SweEnvironment(max_turns=10)
    try:
        episode_ids, _ = env.reset(["task"])
        env.step(
            episode_ids,
            ["w"],
            [{"tool_calls": [_tool_call("write_file", path="notes.txt", content="remember me")]}],
        )
        env.step(episode_ids, ["r"], [{"tool_calls": [_tool_call("read_file", path="notes.txt")]}])
        traj = env.get_trajectories(episode_ids)[0]
        assert _last_tool_result(traj).strip() == "remember me"
    finally:
        env.close()


def test_code_env_episodes_are_isolated():
    """Two episodes on one env instance must not share a workspace."""
    env = SweEnvironment(max_turns=10)
    try:
        ids, _ = env.reset(["task A", "task B"])
        a, b = [ids[0]], [ids[1]]
        env.step(a, ["w"], [{"tool_calls": [_tool_call("write_file", path="who.txt", content="A")]}])
        env.step(b, ["w"], [{"tool_calls": [_tool_call("write_file", path="who.txt", content="B")]}])

        env.step(a, ["r"], [{"tool_calls": [_tool_call("read_file", path="who.txt")]}])
        env.step(b, ["r"], [{"tool_calls": [_tool_call("read_file", path="who.txt")]}])
        assert _last_tool_result(env.get_trajectories(a)[0]).strip() == "A"
        assert _last_tool_result(env.get_trajectories(b)[0]).strip() == "B"
    finally:
        env.close()


def test_sandbox_backend_outage_is_a_failed_tool_call():
    """A sandbox BACKEND failure must be recorded as a FAILED tool call, not a successful observation.

    Rendering the outage as an ordinary ``"Error: ..."`` string made the native protocol mark
    ``success=True`` and pay ``tool_success_reward`` for a run that never happened — infrastructure
    noise entering the reward. A program-level failure (non-zero exit) stays a successful call.
    """
    from src.environments.sandbox.base import SandboxResult

    env = SweEnvironment(max_turns=5, tool_success_reward=0.05, tool_error_penalty=0.1)
    try:
        episode_ids, _ = env.reset(["task"])
        eid = episode_ids[0]
        traj = env.get_trajectories([eid])[0]
        session = env._session_for(traj)
        session.run = lambda *a, **kw: SandboxResult(error="remote sandbox error: 503")

        env.step(episode_ids, ["x"], [{"tool_calls": [_tool_call("run_code", code="print(1)")]}])
        traj = env.get_trajectories([eid])[0]

        result = traj.info["tool_results"][-1]
        assert result["success"] is False, "a backend outage must not score as a successful tool call"
        assert "503" in result["content"]
        assert traj.total_reward < 0, f"the failed call must be penalized, got {traj.total_reward}"
    finally:
        env.close()


def test_code_env_cleanup_closes_sessions():
    env = SweEnvironment(max_turns=5)
    episode_ids, _ = env.reset(["task"])
    env.step(episode_ids, ["w"], [{"tool_calls": [_tool_call("write_file", path="f.txt", content="x")]}])
    eid = episode_ids[0]
    workdir = env._sessions[eid].workdir
    env.step(episode_ids, ["done"], [None])
    env.cleanup(episode_ids)
    import os

    assert eid not in env._sessions
    assert not os.path.exists(workdir), "cleanup must close the session and delete its workdir"
    env.close()


# CodeContestsEnvironment — sandbox grading (python + cpp)


def test_code_contests_python_grading():
    env = CodeContestsEnvironment(max_turns=5, language="python")
    context = {"answer": {"test_cases": [{"input": "3", "output": "6"}, {"input": "10", "output": "20"}]}}
    episode_ids, _ = env.reset(["Read n, print 2n."], [context])

    solution = "n = int(input())\nprint(n * 2)"
    env.step(episode_ids, ["submit"], [{"tool_calls": [_tool_call("submit_solution", code=solution)]}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["tests_passed"] == 2
    assert traj.info["tests_total"] == 2


def test_code_contests_python_test_tool_is_python_repl():
    env = CodeContestsEnvironment(max_turns=5, language="python")
    names = set(env.registry.names())
    assert "python_repl" in names and "submit_solution" in names


def test_code_contests_cpp_grading():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    env = CodeContestsEnvironment(max_turns=5, language="cpp")
    context = {"answer": {"test_cases": [{"input": "21", "output": "42"}, {"input": "0", "output": "0"}]}}
    episode_ids, _ = env.reset(["Read n, print 2n."], [context])

    solution = "#include <iostream>\nint main(){ long n; std::cin>>n; std::cout<<n*2; }"
    env.step(episode_ids, ["submit"], [{"tool_calls": [_tool_call("submit_solution", code=solution)]}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["tests_passed"] == 2, f"cpp grading failed: {traj.info.get('submission_result')}"
    assert traj.info["tests_total"] == 2


def test_code_contests_episodes_isolated_via_active_trajectory():
    """Two episodes on ONE env instance must each grade submit_solution against THEIR OWN test
    cases — the _ACTIVE_TRAJECTORY ContextVar must route per-episode even when episodes interleave
    (the Ray-actor pattern: one env serving many concurrent episodes). If the ContextVar leaked
    across episodes, A's doubler would be graded against B's tripler tests and fail."""
    env = CodeContestsEnvironment(max_turns=5, language="python")
    ctx_double = {"answer": {"test_cases": [{"input": "3", "output": "6"}]}}
    ctx_triple = {"answer": {"test_cases": [{"input": "3", "output": "9"}]}}
    ids, _ = env.reset(["double n", "triple n"], [ctx_double, ctx_triple])
    a, b = [ids[0]], [ids[1]]

    env.step(a, ["s"], [{"tool_calls": [_tool_call("submit_solution", code="print(int(input()) * 2)")]}])
    env.step(b, ["s"], [{"tool_calls": [_tool_call("submit_solution", code="print(int(input()) * 3)")]}])

    ta = env.get_trajectories(a)[0]
    tb = env.get_trajectories(b)[0]
    assert (ta.info["tests_passed"], ta.info["tests_total"]) == (1, 1), "episode A graded wrong trajectory"
    assert (tb.info["tests_passed"], tb.info["tests_total"]) == (1, 1), "episode B graded wrong trajectory"


def test_code_contests_cpp_uses_run_code_test_tool():
    """A C++ contest exposes a compile-and-run test tool (not the python REPL)."""
    env = CodeContestsEnvironment(max_turns=5, language="cpp")
    names = set(env.registry.names())
    assert "run_code" in names and "python_repl" not in names
    assert env.language == "cpp"


def test_code_contests_cpp_compile_error_fails_tests():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    env = CodeContestsEnvironment(max_turns=5, language="cpp")
    context = {"answer": {"test_cases": [{"input": "1", "output": "1"}]}}
    episode_ids, _ = env.reset(["x"], [context])
    env.step(
        episode_ids,
        ["submit"],
        [{"tool_calls": [_tool_call("submit_solution", code="int main(){ broken }")]}],
    )
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["tests_passed"] == 0, "a non-compiling submission must pass zero tests"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
