#!/usr/bin/env python
"""A code-contests run that lists several languages lets the model pick one per program.

Both tools then take a required ``language`` argument (the set's canonical names), every program runs
and is graded in the language its call named, the episode records the language it settled on as a
metric slice, and a call without a valid language, or with code evidently in another of the set's
languages, is refused before it spends the episode's budget. A run naming one language keeps the
fixed-language contract (``python_repl``, no argument).

Run: python tests/cpu/environments/test_code_contests_languages.py  (or pytest)
"""

import json
import platform
import shutil
import sys
from dataclasses import replace

import pytest

from scripts.environments.inference.run_code_contests import parse_language_flag, refuse_flag_owned_env_kwargs
from src.environments.base import EPISODE_SLICES_KEY, REWARD_COMPONENTS_KEY, TOOL_CALL_COUNTS_KEY
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment, evident_language
from src.environments.envs.tasks.coding.grading import GradingSpec, grade_solution
from src.environments.sandbox.base import LANGUAGES, SandboxExecutor, SandboxResult
from src.environments.sandbox.local import LocalSubprocessSandbox

_ADD = {"answer": {"tests": [{"input": "2 3\n", "output": "5\n"}], "time_limit": 1.0}}
# Every recording run answers 5, so this answer fails each submission and none ends the episode early.
_FAILING_ADD = {"answer": {"tests": [{"input": "2 3\n", "output": "6\n"}], "time_limit": 1.0}}
_PY_ADD = "a, b = map(int, input().split()); print(a + b)"
_CPP_ADD = (
    '#include <cstdio>\nint main(){long a,b; if(scanf("%ld %ld",&a,&b)!=2) return 1; printf("%ld\\n",a+b); return 0;}'
)
_CPP_SOURCE = "#include <bits/stdc++.h>\nusing namespace std;\nint main() { long a, b; cin >> a >> b; cout << a + b; }"
_C_SOURCE = '#include <stdio.h>\nint main(void) { long a, b; scanf("%ld %ld", &a, &b); printf("%ld", a + b); }'
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
    env.step(cpp, [""], [{"tool_calls": [_call("u", "run_code", code=_CPP_ADD, language="cpp")]}])
    env.step(cpp, [""], [{"tool_calls": [_call("c", "submit_solution", code=_CPP_ADD, language="cpp")]}])

    # Each scratchpad run is held to the limit its language is graded at: python floored at 5 s, cpp
    # at the stated 1 s.
    assert sandbox.runs == [("python", 5.0), ("python", 5.0), ("cpp", 1.0), ("cpp", 1.0)]
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
    ids, _ = env.reset(["a+b"], [_FAILING_ADD])
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
    """The local sandbox grades a C++ program as C++, and a program that is not evidently in another
    language (valid Python, but with no Python mark) submitted as C++ as a compile failure: past the
    label check, the language is the call's."""
    env = _env()
    ids, _ = env.reset(["a+b", "a+b"], [_ADD, _ADD])
    good, bad = [ids[0]], [ids[1]]
    unmarked = "a, b = map(int, input().split())\nsys.stdout.write(str(a + b))"
    env.step(good, [""], [{"tool_calls": [_call("g", "submit_solution", code=_CPP_ADD, language="cpp")]}])
    env.step(bad, [""], [{"tool_calls": [_call("b", "submit_solution", code=unmarked, language="cpp")]}])
    tg, tb = env.get_trajectories(good)[0], env.get_trajectories(bad)[0]
    assert tg.info["tests_passed"] == 1
    assert tb.info["tests_passed"] == 0 and "COMPILATION ERROR" in tb.info["submission_result"]


def test_the_eval_flag_normalizes_one_name_and_splits_a_list():
    assert parse_language_flag("python") == "python"
    assert parse_language_flag(" python, ") == "python"
    assert parse_language_flag("python,cpp") == ["python", "cpp"]
    with pytest.raises(SystemExit, match="names no language"):
        parse_language_flag(" , ")
    with pytest.raises(SystemExit, match="--language, not --env_kwargs"):
        refuse_flag_owned_env_kwargs({"language": "cpp"})
    refuse_flag_owned_env_kwargs({"timeout_per_test": 3})


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


def test_compiled_time_limit_scale_multiplies_only_the_compiled_limit():
    """The scale pays a compiled solution for shared grading cores; the interpreted floor is unscaled
    and the ``max_time_limit`` clamp still applies after it."""
    sandbox = _RecordingSandbox()
    spec = GradingSpec(sandbox=sandbox, default_timeout=5.0, max_time_limit=15.0, compiled_time_limit_scale=2.0)
    tests = [{"input": "2 3\n", "output": "5\n"}]
    grade_solution("x", tests, spec, time_limit=1.0, language="cpp")
    grade_solution("x", tests, spec, language="cpp")
    grade_solution("x", tests, spec, time_limit=1.0, language="python")
    grade_solution("x", tests, spec, time_limit=6.0, language="python")
    grade_solution("x", tests, spec, time_limit=9.0, language="cpp")
    # cpp 1 s x 2; cpp with no stated limit: the 5 s default x 2; python floored at 5 s, then 6 s unscaled;
    # cpp 9 s x 2 clamped to the 15 s cap.
    assert sandbox.runs == [("cpp", 2.0), ("cpp", 10.0), ("python", 5.0), ("python", 6.0), ("cpp", 15.0)]


def test_the_environment_grades_compiled_submissions_at_the_scaled_limit_and_dumps_the_scale():
    sandbox = _RecordingSandbox()
    env = _env(sandbox=sandbox, timeout_per_test=5, compiled_time_limit_scale=2.0)
    ids, _ = env.reset(["a+b", "a+b"], [_ADD, _ADD])
    py, cpp = [ids[0]], [ids[1]]
    env.step(py, [""], [{"tool_calls": [_call("p", "submit_solution", code=_PY_ADD, language="python")]}])
    env.step(cpp, [""], [{"tool_calls": [_call("c", "submit_solution", code=_CPP_ADD, language="cpp")]}])
    assert sandbox.runs == [("python", 5.0), ("cpp", 2.0)]
    # The re-grader takes the scale back off the trajectory meta, so it reproduces the run's contract.
    meta = env.grading_spec.to_meta()
    assert meta["compiled_time_limit_scale"] == 2.0
    unscaled = _env(sandbox=_RecordingSandbox()).grading_spec
    assert unscaled.compiled_time_limit_scale == 1.0
    assert unscaled.with_meta(meta).compiled_time_limit_scale == 2.0


def test_the_prompt_states_the_compiled_multiplier_only_when_it_is_not_one_and_the_clamp():
    scaled = _env(sandbox=_RecordingSandbox(), timeout_per_test=5, compiled_time_limit_scale=2.5, max_time_limit=12)
    assert "a compiled one runs at 2.5x the problem's stated time limit, at most 12 s." in scaled.system_prompt
    plain = _env(sandbox=_RecordingSandbox(), timeout_per_test=5)
    assert "a compiled one runs at the problem's stated time limit, at most 15 s." in plain.system_prompt
    assert "x the problem's" not in plain.system_prompt


def test_a_program_sent_with_the_wrong_language_is_refused_without_spending_the_call():
    """C++ sent as python, and Python sent as cpp, are refused before they run and returned to the
    budget unpriced: neither scratchpad run nor submission counts, the submission cap and its price
    stay whole, nothing is recorded as graded, and the language slice is untouched."""
    sandbox = _RecordingSandbox()
    env = _env(
        sandbox=sandbox,
        max_submissions=2,
        tool_success_reward=0.05,
        tool_error_penalty=0.5,
        submission_reward=0.1,
        resubmission_penalty=0.2,
    )
    ids, _ = env.reset(["a+b"], [_FAILING_ADD])
    traj = env.get_trajectories(ids)[0]
    calls = [
        _call("1", "run_code", code=_CPP_SOURCE, language="python"),
        _call("2", "submit_solution", code=_CPP_SOURCE, language="python"),
        _call("3", "run_code", code=_PY_ADD, language="cpp"),
        _call("4", "submit_solution", code=_PY_ADD, language="cpp"),
    ]
    (step,) = env.step(ids, [""], [{"tool_calls": calls}])
    to_cpp = 'this looks like cpp code sent with language "python"; send it again with language "cpp".'
    to_py = 'this looks like python code sent with language "cpp"; send it again with language "python".'
    assert [m.content for m in traj.messages if m.role == "tool"] == [
        f"Not run: {to_cpp} No scratchpad run was spent.",
        f"Not graded: {to_cpp} No submission was spent.",
        f"Not run: {to_py} No scratchpad run was spent.",
        f"Not graded: {to_py} No submission was spent.",
    ]
    assert sandbox.runs == [], "a refused program never reaches the sandbox"
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"run_code": 0, "submit_solution": 0}
    assert step.reward == 0.0 and not step.done, "neither paid as a success nor charged as an error"
    assert (traj.info["total_tool_calls"], traj.info["successful_tool_calls"]) == (4, 0)
    assert EPISODE_SLICES_KEY not in traj.info and "language_switches" not in traj.info
    for key in ("submission_result", "submission_language", "_submitted_code", "tested_before_submission"):
        assert key not in traj.info, key

    for cid in ("5", "6"):
        env.step(ids, [""], [{"tool_calls": [_call(cid, "submit_solution", code=_PY_ADD, language="python")]}])
    assert len(sandbox.runs) == 2 and traj.done, "both graded submissions of the cap were still there"
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components["reward/resubmission"] == pytest.approx(-0.2), "only the one real resubmission is priced"
    assert components["reward/turn_shaping"] == pytest.approx(0.1), "the two graded submissions alone are paid"
    assert traj.info["tested_before_submission"] is False


def test_a_program_quoting_the_other_language_keeps_its_label():
    """Valid Python carrying C++ source in a string, and C++ with Python in a comment, are not refused:
    each reading needs the parse to agree with the marks."""
    sandbox = _RecordingSandbox()
    env = _env(sandbox=sandbox, max_test_calls=4)
    ids, _ = env.reset(["a+b"], [_ADD])
    embeds = f"SRC = {_CPP_SOURCE!r}\nprint(len(SRC))"
    commented = '#include <cstdio>\n// def main(): print(int(input()) + 1)\nint main() { std::puts("1"); }'
    calls = [
        _call("1", "run_code", code=embeds, language="python"),
        _call("2", "run_code", code=commented, language="cpp"),
    ]
    env.step(ids, [""], [{"tool_calls": calls}])
    assert [language for language, _ in sandbox.runs] == ["python", "cpp"]


@pytest.mark.parametrize(
    ("code", "language", "offered", "evident"),
    [
        (_CPP_SOURCE, "python", ("python", "cpp"), "cpp"),
        (_CPP_SOURCE, "python", ("python", "c", "cpp"), "cpp"),
        (_CPP_SOURCE, "python", ("python", "c"), None),  # C++, which the run does not take
        (_C_SOURCE, "python", ("python", "c"), "c"),
        (_C_SOURCE, "python", ("python", "cpp", "c"), "c"),
        (_C_SOURCE, "python", ("python", "cpp"), "cpp"),  # C with no c listed: the first compiled one
        ("long s = 0;\nfor (int i = 0; i < 3; ++i) s += i;\nreturn s;", "python", ("python", "c", "cpp"), "c"),
        (_PY_ADD, "c", ("python", "c"), "python"),
        (_PY_ADD, "cpp", ("cpp", "c"), None),  # no python to send it as
        (_C_SOURCE, "c", ("python", "c"), None),
        ("a, b = 1, 2;\nprint(a + b);", "python", ("python", "cpp"), None),  # valid Python, however C-like
        ("n = int(input());\nm = n * 2;\nprint(m\n", "python", ("python", "cpp"), None),  # broken Python
        ("print(1", "python", ("python", "cpp"), None),  # broken, but not C
    ],
)
def test_evident_language(code, language, offered, evident):
    assert evident_language(code, language, offered) == evident


def test_a_problem_stating_no_time_limit_is_told_the_one_each_language_runs_under():
    env = _env(sandbox=_RecordingSandbox(), timeout_per_test=5, compiled_time_limit_scale=2.0)
    ids, _ = env.reset(["a+b", "a+b"], [{"answer": {"tests": [{"input": "2 3\n", "output": "5\n"}]}}, _ADD])
    unstated, stated = (t.messages[-1].content for t in env.get_trajectories(ids))
    assert unstated == "a+b\n\nEach test of this problem runs under 5 s in python, 10 s in cpp."
    assert stated == "a+b"
    single = CodeContestsEnvironment(language="python", sandbox=_RecordingSandbox(), timeout_per_test=4)
    ids, _ = single.reset(["p"], [{"answer": [{"input": "", "output": "5"}]}])
    assert single.get_trajectories(ids)[0].messages[-1].content == "p\n\nEach test of this problem runs under 4 s."


def test_the_tool_descriptions_state_the_toolchain_the_sandbox_runs(monkeypatch):
    """Read off the language registry and the interpreter the local sandbox runs, so a changed flag
    reaches the model; a backend that does not say (a stub, a remote service) states nothing."""
    monkeypatch.setitem(
        LANGUAGES,
        "cpp",
        replace(LANGUAGES["cpp"], compile_argv=("g++", "-O2", "-std=c++20", "-o", "main", "main.cpp")),
    )
    runtime = f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}"
    env = _env(sandbox=LocalSubprocessSandbox())
    clause = f" Here python runs on {runtime} and cpp is compiled with g++ -O2 -std=c++20."
    for tool in env.registry.list_tools():
        assert clause in tool.description, tool.description
    stub = _env(sandbox=_RecordingSandbox())
    assert all(" Here " not in tool.description for tool in stub.registry.list_tools())


def test_the_scratchpad_is_named_and_described_by_what_it_returns():
    fixed = CodeContestsEnvironment(language="python", sandbox=_RecordingSandbox())
    chosen = _env(sandbox=_RecordingSandbox())
    assert "python_repl to try it, submit_solution to be graded" in fixed.system_prompt
    assert "run_code to try it, submit_solution to be graded" in chosen.system_prompt
    description = chosen.registry.get("run_code").description
    assert "returns what it prints to stdout (stderr too when it fails)" in description


@pytest.mark.parametrize("scale", [0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_compiled_time_limit_scale_is_refused(scale):
    with pytest.raises(ValueError, match="compiled_time_limit_scale"):
        _env(sandbox=_RecordingSandbox(), compiled_time_limit_scale=scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
