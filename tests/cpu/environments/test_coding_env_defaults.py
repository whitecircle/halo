#!/usr/bin/env python
"""CPU tests: the coding environments' defaults must not pay for an exploit.

- ``swe`` has no completion fallback: without a ``test_function`` it grades against the row's answer
  (``requires_answer`` on), refuses ``requires_answer=False`` at construction, and raises on an episode
  it has nothing to grade against — one successful tool call earns nothing.
- ``code_contests`` shows a wrong answer as its verdict alone (``verdict_detail="outcome"``) unless a
  run opts into ``full``: the expected output would turn every resubmission into a test oracle.
- A coding environment on a sandbox that does not confine the program warns, once per process and
  backend class: ``local``, ``bubblewrap`` with network, and an executor that declares nothing.

    python tests/cpu/environments/test_coding_env_defaults.py
"""

import json
import logging
import sys

import pytest

from scripts.environments.inference import run_env
from src.environments.base import OBJECTIVE_REWARD_KEY, REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import VERDICT_DETAIL_FULL, VERDICT_DETAIL_OUTCOME, GradingSpec
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.sandbox import resolve as resolve_module
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox

_JUDGE = {"source": "judge", "name": "quality", "requirements": [{"name": "done", "description": "Done."}]}


class _FixedSandbox(SandboxExecutor):
    """Every run returns one canned result; sessions are unsupported, so grading runs one-shot."""

    isolated = True

    def __init__(self, result: SandboxResult):
        self._result = result

    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return self._result


class _Undeclared(SandboxExecutor):
    """An executor that says nothing about confinement: treated as unconfined."""

    def open_session(self):
        raise NotImplementedError


def _bubblewrap(allow_network: bool) -> BubblewrapSandbox:
    """A bubblewrap executor without its construction probe, which needs namespace rights."""
    sandbox = object.__new__(BubblewrapSandbox)
    sandbox.allow_network = allow_network
    return sandbox


def _tool_call(name: str, **arguments) -> dict:
    return {"id": f"call_{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _swe(**kwargs) -> SweEnvironment:
    return SweEnvironment(sandbox_backend="local", **kwargs)


def _run_swe_episode(env: SweEnvironment, context: dict, answer: str):
    """One successful tool call, then a plain-text answer."""
    ids, _ = env.reset(["fix the bug"], [context])
    env.step(ids, ["writing"], [{**context, "tool_calls": [_tool_call("write_file", path="a.py", content="x")]}])
    return env.step(ids, [answer], [context])[0].trajectory


# swe grades nothing it has no grader for


def test_swe_without_a_test_function_grades_against_the_answer_column():
    assert _swe().requires_answer is True
    assert _swe(test_function=lambda trajectory: True).requires_answer is False
    # A reward that prices no environment grade (a judge alone) needs no answer.
    assert _swe(reward_terms=[_JUDGE]).requires_answer is False


def test_swe_refuses_to_construct_with_nothing_to_grade_against():
    with pytest.raises(ValueError, match="nothing to grade against"):
        _swe(requires_answer=False)
    # Each of the graders the refusal names clears it.
    assert _swe(requires_answer=False, test_function=lambda trajectory: True).requires_answer is False
    assert _swe(requires_answer=False, reward_terms=[_JUDGE]).requires_answer is False


def test_an_ungraded_swe_row_raises_instead_of_paying_its_tool_call():
    env = _swe()
    try:
        with pytest.raises(ValueError, match="nothing to grade this episode against"):
            _run_swe_episode(env, {}, "done")
    finally:
        env.close()


def test_a_judge_priced_swe_episode_needs_no_answer():
    """With no environment term the reward prices no grade of the env's own, so an answer-less row is
    not a fault: the episode completes and awaits its judge."""
    env = _swe(reward_terms=[_JUDGE])
    try:
        traj = _run_swe_episode(env, {}, "done")
    finally:
        env.close()
    assert traj.done and traj.info["completed"]
    assert OBJECTIVE_REWARD_KEY not in traj.info[REWARD_COMPONENTS_KEY]


def test_the_eval_driver_refuses_an_answerless_dataset_before_generating(monkeypatch):
    """The trainer's dataset gate, held by the generic eval driver too: an answer-graded environment
    over a dataset without the column fails before any episode runs, not after each one."""
    monkeypatch.setattr(run_env, "load_hf_split", lambda *args: [{"prompt": "fix the bug"}])
    monkeypatch.setattr(run_env, "create_openai_client", lambda **kwargs: pytest.fail("generation must not start"))
    argv = [
        "run_env.py",
        "--env_type",
        "swe",
        "--dataset",
        "tasks",
        "--model",
        "m",
        "--env_kwargs",
        '{"sandbox_backend": "local"}',
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="requires_answer"):
        run_env.main()


def test_swe_pays_the_answer_not_the_tool_call():
    env = _swe()
    try:
        wrong = _run_swe_episode(env, {"answer": "42"}, "41")
        right = _run_swe_episode(env, {"answer": "42"}, "42")
    finally:
        env.close()
    assert wrong.info["successful_tool_calls"] == right.info["successful_tool_calls"] == 1
    assert wrong.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 0.0
    assert right.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 1.0


def test_a_non_callable_validator_is_no_grader():
    """Only a callable validator grades; any other value with no answer must not reach the native
    protocol's completion payout."""
    env = _swe()
    try:
        with pytest.raises(ValueError, match="nothing to grade this episode against"):
            _run_swe_episode(env, {"validator": "exact"}, "done")
    finally:
        env.close()


def test_swe_null_answer_leaves_the_baseline_instead_of_paying():
    env = _swe()
    try:
        traj = _run_swe_episode(env, {"answer": None}, "done")
    finally:
        env.close()
    assert traj.episode_invalid
    assert traj.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 0.0


# verdict_detail defaults to outcome


def test_wrong_answer_verdict_hides_the_expected_output_by_default():
    assert GradingSpec(sandbox=_FixedSandbox(SandboxResult())).verdict_detail == VERDICT_DETAIL_OUTCOME
    env = CodeContestsEnvironment(sandbox=_FixedSandbox(SandboxResult(stdout="7\n", returncode=0)))
    assert env.grading_spec.verdict_detail == VERDICT_DETAIL_OUTCOME
    ids, _ = env.reset(["print 42"], [{"answer": {"tests": [{"input": "", "output": "SECRET42\n"}]}}])
    traj = env.step(ids, [""], [{"tool_calls": [_tool_call("submit_solution", code="print(7)")]}])[0].trajectory
    assert "Test 1: FAIL" in traj.info["submission_result"]
    assert "SECRET42" not in traj.info["submission_result"]
    assert "SECRET42" not in "".join(m.content for m in traj.messages if m.role == "tool")


def test_full_verdict_detail_stays_an_explicit_opt_in():
    env = CodeContestsEnvironment(
        sandbox=_FixedSandbox(SandboxResult(stdout="7\n", returncode=0)), verdict_detail=VERDICT_DETAIL_FULL
    )
    ids, _ = env.reset(["print 42"], [{"answer": {"tests": [{"input": "", "output": "SECRET42\n"}]}}])
    traj = env.step(ids, [""], [{"tool_calls": [_tool_call("submit_solution", code="print(7)")]}])[0].trajectory
    assert "Expected: SECRET42" in traj.info["submission_result"]


# a sandbox that does not confine the program warns once


def _unconfined_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "does not confine it" in record.getMessage()]


@pytest.fixture
def fresh_warnings(monkeypatch, caplog):
    monkeypatch.setattr(resolve_module, "_UNISOLATED_WARNED", set())
    caplog.set_level(logging.WARNING, logger=resolve_module.__name__)
    return caplog


def test_coding_envs_warn_once_per_process_on_the_local_backend(fresh_warnings):
    CodeContestsEnvironment(sandbox=_FixedSandbox(SandboxResult()))
    SweEnvironment(sandbox=_FixedSandbox(SandboxResult()))
    SweEnvironment(sandbox=RemoteSandbox("http://sandbox:8080"))
    assert not _unconfined_warnings(fresh_warnings), "a confining backend must not warn"

    CodeContestsEnvironment(sandbox_backend="local")
    SweEnvironment(sandbox_backend="local")
    SweEnvironment(sandbox=LocalSubprocessSandbox())
    warnings = _unconfined_warnings(fresh_warnings)
    assert len(warnings) == 1, warnings
    assert "CodeContestsEnvironment" in warnings[0] and "LocalSubprocessSandbox" in warnings[0]


def test_an_executor_that_declares_nothing_is_treated_as_unconfined(fresh_warnings):
    SweEnvironment(sandbox=_Undeclared())
    assert len(_unconfined_warnings(fresh_warnings)) == 1, "a backend that declares nothing must not pass as confined"


def test_bubblewrap_confines_only_without_network(fresh_warnings):
    SweEnvironment(sandbox=_bubblewrap(allow_network=False))
    assert not _unconfined_warnings(fresh_warnings)
    SweEnvironment(sandbox=_bubblewrap(allow_network=True))
    assert len(_unconfined_warnings(fresh_warnings)) == 1, "--share-net gives the program the host's network"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
