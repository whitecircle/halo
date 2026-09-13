#!/usr/bin/env python
"""A code-contests run that lists several languages lets the model pick one per program.

Both tools then take a required ``language`` argument (the set's canonical names), every program runs
and is graded in the language its call named, the episode records the language it settled on as a
metric slice, and a call without a valid language is refused before it spends the episode's budget.
A run naming one language keeps the fixed-language contract (``python_repl``, no argument).

Run: python tests/cpu/environments/test_code_contests_languages.py  (or pytest)
"""

import json
import shutil

import pytest

from scripts.environments.inference.run_code_contests import parse_language_flag, refuse_env_kwargs_language
from src.environments.base import EPISODE_SLICES_KEY, TOOL_CALL_COUNTS_KEY
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import GradingSpec, grade_solution
from src.environments.sandbox.base import SandboxExecutor, SandboxResult

_ADD = {"answer": {"tests": [{"input": "2 3\n", "output": "5\n"}], "time_limit": 1.0}}
_PY_ADD = "a, b = map(int, input().split()); print(a + b)"
_CPP_ADD = (
    '#include <cstdio>\nint main(){long a,b; if(scanf("%ld %ld",&a,&b)!=2) return 1; printf("%ld\\n",a+b); return 0;}'
)
_HAS_GPP = shutil.which("g++") is not None


class _RecordingSandbox(SandboxExecutor):
    """One-shot runs only; records the language and timeout of every run and answers ``5``."""

    def __init__(self):
        self.runs: list[tuple[str, float]] = []

    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.runs.append((language, timeout))
        return SandboxResult(stdout="5\n", returncode=0)


def _call(cid, name, **arguments):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _env(**kwargs):
    kwargs.setdefault("output_comparison", "tokens")
    kwargs.setdefault("max_turns", 6)
    return CodeContestsEnvironment(language=["python", "cpp"], **kwargs)


def test_a_language_set_exposes_one_scratchpad_and_a_required_language_argument():
    env = _env(sandbox=_RecordingSandbox(), timeout_per_test=5)
    assert env.languages == ("python", "cpp") and env.language == "python" and env.chooses_language
    assert env.registry.names() == ["run_code", "submit_solution"]
    for tool in env.registry.list_tools():
        param = {p.name: p for p in tool.parameters}["language"]
        assert param.enum == ["python", "cpp"] and param.required
        assert "python or cpp program" in tool.description
    assert "python or cpp solution" in env.system_prompt
    assert "language argument (python, cpp)" in env.system_prompt
    assert (
        "A python solution gets at least 5 s per test; a compiled one runs at the problem's stated"
        in env.system_prompt
    )
    # Two compiled languages carry no floor clause: neither is floored.
    assert (
        "gets at least"
        not in CodeContestsEnvironment(language=["cpp", "c"], sandbox=_RecordingSandbox()).system_prompt
    )


def test_one_language_keeps_the_fixed_contract():
    env = CodeContestsEnvironment(language="python", sandbox=_RecordingSandbox())
    assert not env.chooses_language and env.languages == ("python",)
    assert env.registry.names() == ["python_repl", "submit_solution"]
    assert all(p.name != "language" for tool in env.registry.list_tools() for p in tool.parameters)
    assert "language argument" not in env.system_prompt and "python solution" in env.system_prompt


def test_each_program_runs_and_is_graded_in_the_language_its_call_named():
    sandbox = _RecordingSandbox()
    env = _env(sandbox=sandbox, timeout_per_test=5)
    ids, _ = env.reset(["a+b", "a+b"], [_ADD, _ADD])
    py, cpp = [ids[0]], [ids[1]]
    env.step(py, [""], [{"tool_calls": [_call("t", "run_code", code=_PY_ADD, language="python")]}])
    env.step(py, [""], [{"tool_calls": [_call("p", "submit_solution", code=_PY_ADD, language="python")]}])
    env.step(cpp, [""], [{"tool_calls": [_call("c", "submit_solution", code=_CPP_ADD, language="cpp")]}])

    # scratchpad (python, repl timeout), python grade (floored at 5 s), cpp grade (the stated 1 s limit)
    assert sandbox.runs == [("python", 15.0), ("python", 5.0), ("cpp", 1.0)]
    tp, tc = env.get_trajectories(py)[0], env.get_trajectories(cpp)[0]
    assert tp.info["submission_language"] == "python" and tc.info["submission_language"] == "cpp"
    assert tp.info[EPISODE_SLICES_KEY] == {"language": "python"}
    assert tc.info[EPISODE_SLICES_KEY] == {"language": "cpp"}
    assert env.rollout_metrics(tp)["episode/language_switches"] == 0.0


def test_a_missing_or_foreign_language_is_refused_before_the_budget_is_spent():
    env = _env(sandbox=_RecordingSandbox(), max_submissions=1)
    ids, _ = env.reset(["a+b"], [_ADD])
    calls = [
        _call("a", "submit_solution", code=_PY_ADD),
        _call("b", "submit_solution", code=_PY_ADD, language="java"),
        _call("c", "submit_solution", code=_PY_ADD, language="python"),
    ]
    env.step(ids, [""], [{"tool_calls": calls}])
    traj = env.get_trajectories(ids)[0]
    observations = [m.content for m in traj.messages if m.role == "tool"]
    assert "submit_solution: missing a required argument: 'language'" in observations[0]
    assert "submit_solution: language must be one of python, cpp, got 'java'" in observations[1]
    assert "Passed 1/1" in observations[2], "the two refusals left the one-submission budget intact"
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"submit_solution": 1}
    assert traj.done, "the admitted submission spent the cap and ended the episode"


def test_switching_languages_is_counted_and_the_graded_language_is_the_slice():
    env = _env(sandbox=_RecordingSandbox(), max_test_calls=4, max_submissions=2)
    ids, _ = env.reset(["a+b"], [_ADD])
    env.step(ids, [""], [{"tool_calls": [_call("1", "run_code", code="x", language="python")]}])
    env.step(ids, [""], [{"tool_calls": [_call("2", "run_code", code="x", language="cpp")]}])
    env.step(ids, [""], [{"tool_calls": [_call("3", "submit_solution", code="x", language="cpp")]}])
    env.step(ids, [""], [{"tool_calls": [_call("4", "submit_solution", code="x", language="python")]}])
    traj = env.get_trajectories(ids)[0]
    assert traj.info["language_switches"] == 2
    assert traj.info[EPISODE_SLICES_KEY] == {"language": "python"}
    assert traj.info["submission_language"] == "python"
    assert env.rollout_metrics(traj)["episode/language_switches"] == 2.0
    single = CodeContestsEnvironment(language="python", sandbox=_RecordingSandbox())
    ids, _ = single.reset(["a+b"], [_ADD])
    single.step(ids, [""], [{"tool_calls": [_call("s", "submit_solution", code="x")]}])
    metrics = single.rollout_metrics(single.get_trajectories(ids)[0])
    assert "episode/language_switches" not in metrics
    assert EPISODE_SLICES_KEY not in single.get_trajectories(ids)[0].info


@pytest.mark.skipif(not _HAS_GPP, reason="g++ not installed")
def test_real_grading_honors_the_named_language():
    """The local sandbox grades a C++ program as C++ and a Python program submitted as C++ as a
    compile failure, so the language is the call's, never guessed from the source."""
    env = _env()
    ids, _ = env.reset(["a+b", "a+b"], [_ADD, _ADD])
    good, bad = [ids[0]], [ids[1]]
    env.step(good, [""], [{"tool_calls": [_call("g", "submit_solution", code=_CPP_ADD, language="cpp")]}])
    env.step(bad, [""], [{"tool_calls": [_call("b", "submit_solution", code=_PY_ADD, language="cpp")]}])
    tg, tb = env.get_trajectories(good)[0], env.get_trajectories(bad)[0]
    assert tg.info["tests_passed"] == 1
    assert tb.info["tests_passed"] == 0 and "COMPILATION ERROR" in tb.info["submission_result"]


def test_the_eval_flag_normalizes_one_name_and_splits_a_list():
    assert parse_language_flag("python") == "python"
    assert parse_language_flag(" python, ") == "python"
    assert parse_language_flag("python,cpp") == ["python", "cpp"]
    with pytest.raises(SystemExit, match="names no language"):
        parse_language_flag(" , ")
    with pytest.raises(SystemExit, match="not --env_kwargs"):
        refuse_env_kwargs_language({"language": "cpp"})
    refuse_env_kwargs_language({"max_turns": 3})


def test_language_set_validation():
    with pytest.raises(ValueError, match="at least one language"):
        CodeContestsEnvironment(language=[], sandbox=_RecordingSandbox())
    with pytest.raises(ValueError, match="lists 'python' twice"):
        CodeContestsEnvironment(language=["python", "py"], sandbox=_RecordingSandbox())
    with pytest.raises(ValueError, match="unsupported language 'java'"):
        CodeContestsEnvironment(language=["python", "java"], sandbox=_RecordingSandbox())


def test_grade_solution_floors_only_interpreted_languages_per_call():
    sandbox = _RecordingSandbox()
    spec = GradingSpec(sandbox=sandbox, default_timeout=5.0, language="python")
    tests = [{"input": "2 3\n", "output": "5\n"}]
    grade_solution("x", tests, spec, time_limit=1.0)
    grade_solution("x", tests, spec, time_limit=1.0, language="cpp")
    grade_solution("x", tests, spec, time_limit=9.0, language="cpp")
    assert sandbox.runs == [("python", 5.0), ("cpp", 1.0), ("cpp", 9.0)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
