#!/usr/bin/env python
"""CPU tests: the coding environments' defaults must not pay for an exploit.

- ``swe`` has no completion fallback: without a ``test_function`` it grades against the row's answer
  (``requires_answer`` on), refuses ``requires_answer=False`` at construction, and raises on an episode
  it has nothing to grade against — one successful tool call earns nothing.
- ``code_contests`` shows a failed test as its verdict class alone (``verdict_detail="outcome"``)
  unless a run opts into ``full``: the expected output would turn every resubmission into a test
  oracle, and the program's stderr, exit code or output size would carry a hidden input back. A
  compiler's message shows under ``outcome`` only where no program can make a rebuild quote an input.
- A coding environment on a sandbox that does not confine the program warns, once per process and
  backend class: ``local``, ``bubblewrap`` with network, and an executor that declares nothing.

    python tests/cpu/environments/test_coding_env_defaults.py
"""

import json
import logging
import pathlib
import shutil
import sys

import pytest
from datasets import Dataset

from scripts.environments.inference import run_env
from src.environments.base import OBJECTIVE_REWARD_KEY, REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import (
    VERDICT_DETAIL_FULL,
    VERDICT_DETAIL_OUTCOME,
    GradingSpec,
    grade_solution,
    run_solution_against_tests,
)
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.eval_runner import require_answers
from src.environments.registry import resolve_environment
from src.environments.sandbox import resolve as resolve_module
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox
from tests.common.code_contests import RecordingSandboxSession, StubSandbox

_JUDGE = {"source": "judge", "name": "quality", "requirements": [{"name": "done", "description": "Done."}]}


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
    monkeypatch.setattr(run_env, "load_hf_split", lambda *args: Dataset.from_list([{"prompt": "fix the bug"}]))
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


@pytest.mark.parametrize(
    "build",
    [
        lambda: resolve_environment("native_math", {}),
        lambda: resolve_environment("native_coding", {}),
        lambda: resolve_environment("native_combined", {}),
        lambda: resolve_environment("mcp", {}),
        lambda: _swe(reward_terms=[_JUDGE]),
    ],
    ids=["native_math", "native_coding", "native_combined", "mcp", "judge-priced-swe"],
)
def test_the_eval_gate_passes_an_environment_that_grades_without_an_answer(build):
    env = build()
    try:
        require_answers(env, [{"context": {}}], "the rows")
    finally:
        env.close()


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


def _raising_test_function(trajectory):
    raise RuntimeError("grader bug")


@pytest.mark.parametrize(
    ("kwargs", "context"),
    [({}, {"answer": None}), ({"test_function": _raising_test_function}, {})],
    ids=["null-answer", "raising-test-function"],
)
def test_a_judge_priced_swe_episode_is_never_voided_by_an_unpriced_grade(kwargs, context):
    """With no environment term nothing reads the env's own grade, so a grader that would void it
    must not run: the episode stays in the baseline for its judge."""
    env = _swe(reward_terms=[_JUDGE], **kwargs)
    try:
        traj = _run_swe_episode(env, context, "done")
    finally:
        env.close()
    assert traj.done and not traj.episode_invalid


# verdict_detail defaults to outcome


def test_wrong_answer_verdict_hides_the_expected_output_by_default():
    assert GradingSpec(sandbox=StubSandbox(SandboxResult())).verdict_detail == VERDICT_DETAIL_OUTCOME
    env = CodeContestsEnvironment(sandbox=StubSandbox(SandboxResult(stdout="7\n", returncode=0)))
    assert env.grading_spec.verdict_detail == VERDICT_DETAIL_OUTCOME
    ids, _ = env.reset(["print 42"], [{"answer": {"tests": [{"input": "", "output": "SECRET42\n"}]}}])
    traj = env.step(ids, [""], [{"tool_calls": [_tool_call("submit_solution", code="print(7)")]}])[0].trajectory
    assert "Test 1: FAIL" in traj.info["submission_result"]
    assert "SECRET42" not in traj.info["submission_result"]
    assert "SECRET42" not in "".join(m.content for m in traj.messages if m.role == "tool")


def test_full_verdict_detail_stays_an_explicit_opt_in():
    env = CodeContestsEnvironment(
        sandbox=StubSandbox(SandboxResult(stdout="7\n", returncode=0)), verdict_detail=VERDICT_DETAIL_FULL
    )
    ids, _ = env.reset(["print 42"], [{"answer": {"tests": [{"input": "", "output": "SECRET42\n"}]}}])
    traj = env.step(ids, [""], [{"tool_calls": [_tool_call("submit_solution", code="print(7)")]}])[0].trajectory
    assert "Expected: SECRET42" in traj.info["submission_result"]


# Each program writes the hidden input it read into a channel of its own.
_ECHO_TO_STDERR_WRONG = "import sys\nsys.stderr.write(sys.stdin.read())\nprint('wrong')\n"
_ECHO_TO_STDERR_CRASH = "import sys\nsys.stderr.write(sys.stdin.read())\nsys.exit(1)\n"
_INPUT_AS_EXIT_CODE = "import sys\nsys.exit(ord(sys.stdin.read()[0]))\n"
_INPUT_AS_OUTPUT_SIZE = "import sys\nprint('x' * (10 + ord(sys.stdin.read()[0])))\n"


@pytest.mark.parametrize(
    ("program", "hidden_input", "leak", "verdict"),
    [
        (_ECHO_TO_STDERR_WRONG, "HIDDEN-4217", "HIDDEN-4217", "Test 1: FAIL"),
        (_ECHO_TO_STDERR_CRASH, "HIDDEN-4217", "HIDDEN-4217", "Test 1: RUNTIME ERROR"),
        (_INPUT_AS_EXIT_CODE, "S", "exit 83", "Test 1: RUNTIME ERROR"),
        (_INPUT_AS_OUTPUT_SIZE, "S", "94", "Test 1: OUTPUT LIMIT EXCEEDED (> 10 bytes)"),
    ],
    ids=["stderr-on-a-wrong-answer", "stderr-on-a-crash", "exit-code", "output-size"],
)
def test_a_graded_program_cannot_read_a_hidden_input_back_by_default(program, hidden_input, leak, verdict):
    """Under ``outcome`` a submission learns its verdict class, never what it wrote itself: stderr,
    an exit code and an output size each carry the input it read, and ``full`` shows them."""
    tests = [{"input": hidden_input, "output": "right"}]
    sandbox = LocalSubprocessSandbox()
    full = run_solution_against_tests(program, tests, sandbox=sandbox, max_output_size=10, verdict_detail="full")
    outcome = run_solution_against_tests(program, tests, sandbox=sandbox, max_output_size=10)
    assert leak in full.details, full.details
    assert outcome.details.splitlines()[1:] == [verdict], outcome.details


@pytest.mark.parametrize(
    "body",
    [
        {"status": "SandboxError", "message": "HIDDEN-4217"},
        {"status": "Failed", "compile_result": {"status": "Error", "stderr": "HIDDEN-4217"}},
    ],
    ids=["service-message", "unfinished-compile-step"],
)
def test_a_remote_error_shows_its_class_alone_by_default(body, caplog):
    """A service's error text can quote what the program wrote: the verdict shows the class, the log
    the text."""
    remote = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(body))
    tests = [{"input": "", "output": "right"}]
    outcome = run_solution_against_tests("code", tests, sandbox=remote, language="cpp")
    assert outcome.details.splitlines()[1:] == ["Test 1: ERROR -- grading infrastructure failure"], outcome.details
    assert "HIDDEN-4217" in caplog.text
    full = run_solution_against_tests("code", tests, sandbox=remote, language="cpp", verdict_detail="full")
    assert "HIDDEN-4217" in full.details


class _CheckerOutage(SandboxExecutor):
    """Runs the submission cleanly and loses each checker run to a backend whose error quotes the
    checker's files, the reference output among them."""

    isolated = True

    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        if files and "checker.py" in files:
            return SandboxResult(error=f"could not stage {files['correct_output.txt']!r}")
        return SandboxResult(stdout="7\n", returncode=0)


def test_a_checker_outage_shows_its_class_alone_by_default(caplog):
    spec = GradingSpec(sandbox=_CheckerOutage())
    tests = [{"input": "", "output": "HIDDEN-4217"}]
    outcome = grade_solution("print(7)", tests, spec, checker="# judge")
    assert outcome.details.splitlines()[1:] == ["Test 1: ERROR -- grading infrastructure failure"], outcome.details
    assert outcome.infra_errors == 1 and "HIDDEN-4217" in caplog.text
    full = grade_solution(
        "print(7)", tests, GradingSpec(sandbox=_CheckerOutage(), verdict_detail="full"), checker="# judge"
    )
    assert "HIDDEN-4217" in full.details


_REMOTE_COMPILE_ERROR = {
    "status": "Failed",
    "compile_result": {"status": "Finished", "return_code": 1, "stderr": "main.cpp:1:1: error: HIDDEN-4217"},
}


def test_a_remote_compile_message_shows_under_full_only():
    """A remote build shares each test's request with its stdin, so its message may quote it."""
    remote = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(_REMOTE_COMPILE_ERROR))
    tests = [{"input": "HIDDEN-4217", "output": "right"}]
    outcome = run_solution_against_tests("code", tests, sandbox=remote, language="cpp")
    assert outcome.details.splitlines()[1:] == ["COMPILATION ERROR (every test fails)"], outcome.details
    full = run_solution_against_tests("code", tests, sandbox=remote, language="cpp", verdict_detail="full")
    assert "HIDDEN-4217" in full.details


class _BuildsApart(StubSandbox):
    compiles_without_test_input = True


def test_a_compile_message_shows_under_outcome_only_where_the_backend_builds_apart():
    rejected = SandboxResult(stderr="main.cpp:1:14: error: undeclared_name", returncode=1, compile_failed=True)
    tests = [{"input": "", "output": ""}]
    shown = run_solution_against_tests("code", tests, sandbox=_BuildsApart(rejected), language="cpp")
    hidden = run_solution_against_tests("code", tests, sandbox=StubSandbox(rejected), language="cpp")
    assert "undeclared_name" in shown.details and "undeclared_name" not in hidden.details, hidden.details
    assert BubblewrapSandbox.compiles_without_test_input
    assert not LocalSubprocessSandbox.compiles_without_test_input and not RemoteSandbox.compiles_without_test_input


def _plant_input_and_force_a_rebuild(header: pathlib.Path) -> str:
    """A C++ program that writes the input it read into ``header`` as an ``#error`` and removes its
    working directory, so the rebuild the next test forces includes it into the compiler's message."""
    return (
        f'#if __has_include("{header}")\n#include "{header}"\n#endif\n'
        "#include <filesystem>\n#include <fstream>\n#include <iostream>\n#include <string>\n"
        "int main() {\n  std::string input;\n  std::getline(std::cin, input);\n"
        f'  std::ofstream("{header}") << "#error " << input << "\\n";\n'
        "  std::filesystem::remove_all(std::filesystem::current_path());\n"
        '  std::cout << "wrong";\n}\n'
    )


def test_a_local_rebuild_cannot_carry_a_hidden_input_into_the_verdict(tmp_path):
    """On ``local`` a program can plant a test's input in a host file and force a rebuild that includes
    it: the route is real (``full`` shows it), so ``outcome`` shows the compile error's class alone."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    header = tmp_path / "leak.h"
    program = _plant_input_and_force_a_rebuild(header)
    tests = [{"input": "HIDDEN-4217", "output": "right"}, {"input": "", "output": "right"}]
    full = run_solution_against_tests(
        program, tests, sandbox=LocalSubprocessSandbox(), language="cpp", verdict_detail="full"
    )
    assert "HIDDEN-4217" in full.details, full.details
    header.unlink()
    outcome = run_solution_against_tests(program, tests, sandbox=LocalSubprocessSandbox(), language="cpp")
    assert "COMPILATION ERROR" in outcome.details and "HIDDEN-4217" not in outcome.details, outcome.details


def test_a_bubblewrap_compile_message_shows_under_outcome():
    """The jailed build runs once, before any test and without stdin, and nothing can force another."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    try:
        sandbox = BubblewrapSandbox()
    except RuntimeError as exc:
        pytest.skip(f"bubblewrap cannot sandbox here: {exc}")
    grade = run_solution_against_tests(
        "int main() { return undeclared_name; }", [{"input": "", "output": ""}], sandbox=sandbox, language="cpp"
    )
    assert "COMPILATION ERROR" in grade.details and "undeclared_name" in grade.details, grade.details


# a sandbox that does not confine the program warns once


def _unconfined_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "does not confine it" in record.getMessage()]


@pytest.fixture
def fresh_warnings(monkeypatch, caplog):
    monkeypatch.setattr(resolve_module, "_UNISOLATED_WARNED", set())
    caplog.set_level(logging.WARNING, logger=resolve_module.__name__)
    return caplog


def test_coding_envs_warn_once_per_process_on_the_local_backend(fresh_warnings):
    CodeContestsEnvironment(sandbox=StubSandbox(SandboxResult()))
    SweEnvironment(sandbox=StubSandbox(SandboxResult()))
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
