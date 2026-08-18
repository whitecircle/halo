#!/usr/bin/env python
"""
CPU tests for Codeforces-style grading in the unified ``CodeContestsEnvironment``.

Covers the pieces that plain exact-match grading gets wrong and that the environment must get
right: token comparison, special-judge checkers, TLE handling, the per-problem
time limit, and the env's reset → grade → reward path with ``output_comparison="tokens"``. Grading
runs real solutions through the local subprocess sandbox, so these assertions fail if the grader,
the verdict selection, or the sandbox integration breaks.

Run:
    python tests/cpu/environments/test_codeforces.py
"""

import json
import sys

import pytest

from scripts.environments.preparation.prepare_code_dataset import rating_in_bounds
from src.environments.envs.tasks.coding.code_contests import DEFAULT_REASONING_EFFORT, CodeContestsEnvironment
from src.environments.envs.tasks.coding.datasets import (
    CODE_DATASET_ADAPTERS,
    format_codeforces_prompt,
    format_icpc_prompt,
    format_titled_statement,
    keep_hlce,
    keep_icpc,
    keep_livecodebench,
    pack_codeforces_verification,
    pack_hlce_verification,
    pack_icpc_verification,
    pack_livecodebench_verification,
)
from src.environments.envs.tasks.coding.grading import GradingSpec, compare_tokens, grade_solution
from src.environments.sandbox.resolve import resolve_sandbox

# A tolerance special-judge: accept any float within 1e-4 of the reference (argv = 3 file paths).
_CHECKER_TOLERANCE = (
    "import sys\n"
    "exp = float(open(sys.argv[2]).read().split()[0])\n"
    "got = open(sys.argv[3]).read().split()\n"
    "print(1 if got and abs(float(got[0]) - exp) <= 1e-4 else 0)\n"
)

_ADD_TESTS = [{"input": "2\n3\n", "output": "5\n"}, {"input": "10\n20\n", "output": "30\n"}]
_CORRECT_ADD = "a=int(input()); b=int(input()); print(a+b)"


def test_compare_tokens_whitespace_insensitive():
    assert compare_tokens("1\n9\n", "1\r\n9\r\n")
    assert compare_tokens("1 2 3", "1 2 3   \n")
    assert compare_tokens("YES\nNO\n", "YES \nNO\n\n")


def test_compare_tokens_rejects_real_differences():
    assert not compare_tokens("5", "6")
    assert not compare_tokens("1 2", "2 1")
    assert not compare_tokens("1 2 3", "1 2")


def test_compare_tokens_float_tolerance():
    # 1e-6 abs/rel window (AtCoder special-judge tolerance): print precision must not fail a correct float.
    assert compare_tokens("0.333333333333333", "0.333333333333")
    assert compare_tokens("3 0.5000000", "3 0.4999999")
    assert compare_tokens("1e-9", "0.0000000")
    assert not compare_tokens("0.333", "0.444")
    assert not compare_tokens("1.0 2.0", "1.0 2.5")
    assert not compare_tokens("5", "5.0000001")  # integer-expected tokens stay byte-exact


# Grading (token + checker + TLE) through the sandbox


def _token_spec() -> GradingSpec:
    """The Codeforces grading contract these cases share: a real sandbox, token comparison."""
    return GradingSpec(sandbox=resolve_sandbox(), comparison="tokens")


def test_grade_correct_solution_passes_all():
    passed, total, *_ = grade_solution(_CORRECT_ADD, _ADD_TESTS, _token_spec(), time_limit=2.0)
    assert (passed, total) == (2, 2)


def test_grade_wrong_solution_fails():
    wrong = "a=int(input()); b=int(input()); print(a*b)"
    passed, total, *_ = grade_solution(wrong, _ADD_TESTS, _token_spec(), time_limit=2.0)
    assert passed == 0 and total == 2


def test_grade_infinite_loop_is_tle():
    passed, _total, details, ran_ok, *_ = grade_solution("while True: pass", _ADD_TESTS, _token_spec(), time_limit=1.0)
    assert passed == 0
    assert "TIME LIMIT EXCEEDED" in details
    assert ran_ok == 0  # never ran to completion → no execution-progress credit


def test_ran_ok_separates_runnable_wrong_from_crash():
    """``ran_ok`` (the dense execution-progress signal) must count a solution that RUNS but is WRONG, and
    must NOT count one that CRASHES — this is the only within-group signal when every completion fails the
    hidden tests, so a regression here silently removes the reward variance GRPO relies on."""
    passed, total, _, ran_ok, *_ = grade_solution(
        "a=int(input()); b=int(input()); print(a*b)", _ADD_TESTS, _token_spec(), time_limit=2.0
    )
    assert (passed, ran_ok) == (0, total), (
        f"runnable-but-wrong should run all {total} tests: passed={passed} ran={ran_ok}"
    )
    _, _, _, ran_crash, *_ = grade_solution("raise SystemExit(1)", _ADD_TESTS, _token_spec(), time_limit=2.0)
    assert ran_crash == 0, f"a crashing solution must earn 0 execution progress, got {ran_crash}"


def test_empty_output_stub_earns_no_execution_progress():
    """A `pass`-style stub (clean exit, no output) must NOT count as ran_ok: on the execution rung it
    would tie every honest runnable-but-wrong attempt, making the stub a reward-floor attractor in
    all-fail groups (the observed terminal state of an unanchored run)."""
    result = grade_solution("pass", _ADD_TESTS, _token_spec(), time_limit=2.0)
    assert result.passed == 0
    assert result.ran_ok == 0, f"an empty-output stub must earn 0 execution progress, got {result.ran_ok}"


def test_backend_outage_counts_infra_errors():
    """A sandbox backend/transport failure is infrastructure's fault, not the program's: GradeResult
    must report it as infra_errors so the reward layer can withhold every rung (an all-infra-error
    grade carries no signal about the code)."""
    from src.environments.envs.tasks.coding.grading import run_solution_against_tests
    from src.environments.sandbox.base import SandboxResult

    class _DownSandbox:
        def run(self, code, **kwargs):
            return SandboxResult(stdout="", stderr="", returncode=None, timed_out=False, error="backend down")

    result = run_solution_against_tests("print(1)", _ADD_TESTS, sandbox=_DownSandbox())
    assert result.infra_errors == result.total == 2
    assert result.ran_ok == 0 and result.passed == 0


def test_checker_accepts_within_tolerance_rejects_outside():
    tests = [{"input": "", "output": "1.0\n"}]
    near, *_ = grade_solution("print('1.00003')", tests, _token_spec(), checker=_CHECKER_TOLERANCE, time_limit=2.0)
    far, *_ = grade_solution("print('2.0')", tests, _token_spec(), checker=_CHECKER_TOLERANCE, time_limit=2.0)
    assert near == 1  # token compare would reject "1.00003" vs "1.0"
    assert far == 0


def test_checker_overrides_token_mismatch():
    tests = [{"input": "", "output": "1.0\n"}]
    passed, *_ = grade_solution("print('0.99999')", tests, _token_spec(), checker=_CHECKER_TOLERANCE, time_limit=2.0)
    assert passed == 1
    no_checker, *_ = grade_solution("print('0.99999')", tests, _token_spec(), time_limit=2.0)
    assert no_checker == 0


# Dataset adapters


def test_prompt_and_verification_from_row():
    row = {
        "title": "Sum",
        "index": "A",
        "time_limit": 1.0,
        "memory_limit": 256.0,
        "description": "Print a+b.",
        "input_format": "Two integers.",
        "output_format": "Their sum.",
        "examples": [{"input": "2 3", "output": "5"}],
        "official_tests": [{"input": "2 3\n", "output": "5\n"}],
        "generated_checker": None,
        "input_mode": "stdio",
    }
    prompt = format_codeforces_prompt(row)
    assert "Print a+b." in prompt and "## Input" in prompt and "## Output" in prompt
    assert "time limit per test: 1 s" in prompt

    payload = pack_codeforces_verification(row)
    assert payload["tests"] == [{"input": "2 3\n", "output": "5\n"}]
    assert payload["checker"] is None
    assert payload["time_limit"] == 1.0


def test_verification_falls_back_to_examples():
    row = {"examples": [{"input": "1", "output": "1"}], "official_tests": [], "generated_checker": None}
    payload = pack_codeforces_verification(row)
    assert payload["tests"] == [{"input": "1", "output": "1"}]


def _encode_lcb_private(tests):
    """Reproduce LiveCodeBench's private_test_cases encoding: base64(zlib(pickle(json_str)))."""
    import base64
    import pickle
    import zlib

    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode("utf-8")


def test_livecodebench_combines_stdin_and_drops_functional():
    # Public tests are plain JSON, private ones compressed; functional (LeetCode) tests are dropped
    # because this env grades stdin/stdout only.
    row = {
        "question_title": "Sum",
        "question_content": "Read a and b, print a+b.",
        "public_test_cases": json.dumps([{"input": "2 3", "output": "5", "testtype": "stdin"}]),
        "private_test_cases": _encode_lcb_private(
            [
                {"input": "10 20", "output": "30", "testtype": "stdin"},
                {"input": "[1,2]", "output": "3", "testtype": "functional"},
            ]
        ),
    }
    assert keep_livecodebench(row) is True
    payload = pack_livecodebench_verification(row)
    assert payload["tests"] == [{"input": "2 3", "output": "5"}, {"input": "10 20", "output": "30"}]
    assert payload["checker"] is None
    assert "Sum" in format_titled_statement(row)


def test_livecodebench_drops_functional_only_problem():
    row = {
        "question_content": "Implement f(nums).",
        "public_test_cases": json.dumps([{"input": "[1]", "output": "1", "testtype": "functional"}]),
        "private_test_cases": json.dumps([]),
    }
    assert keep_livecodebench(row) is False
    assert pack_livecodebench_verification(row)["tests"] == []


def test_icpc_packs_pairs_and_scales_time_limit():
    row = {
        "type": "traditional",
        "problem_label": "A",
        "title": "Echo",
        "description": "Echo the input.",
        "input": "A line.",
        "output": "The same line.",
        "time_limit_ms": 2000,
        "test_cases": [["hi\n", "hi\n"], ["yo\n", "yo\n"]],
        "examples": [["hi\n", "hi\n"]],
    }
    assert keep_icpc(row) is True
    payload = pack_icpc_verification(row)
    assert payload["tests"] == [{"input": "hi\n", "output": "hi\n"}, {"input": "yo\n", "output": "yo\n"}]
    assert payload["time_limit"] == 2.0  # ms -> s
    assert "## Input" in format_icpc_prompt(row)


def test_icpc_drops_special_judge_problems():
    # ICPC-Eval ``spj`` problems carry a C++ special judge this env's python-checker contract can't run.
    row = {"type": "spj", "test_cases": [["1\n", "1\n"]], "spj_code": "#include <bits/stdc++.h>"}
    assert keep_icpc(row) is False


def test_hlce_packs_world_finals_tests():
    # Only the World Finals subset is gradable; IOI rows carry statement samples but no hidden tests.
    row = {
        "question_title": "Ship Traffic",
        "question_content": "Compute the longest safe interval.",
        "platform": "ICPC_world_final_2015",
        "test_cases": [{"input": "1 100\n", "output": "5\n"}, {"input": "2 3\n", "output": "0\n"}],
    }
    assert keep_hlce(row) is True
    payload = pack_hlce_verification(row)
    assert payload["tests"] == [{"input": "1 100\n", "output": "5\n"}, {"input": "2 3\n", "output": "0\n"}]
    assert payload["time_limit"] is None  # no per-problem limit -> env falls back to timeout_per_test
    assert "Ship Traffic" in format_titled_statement(row)
    assert keep_hlce({"test_cases": []}) is False


def test_code_dataset_adapters_registered():
    # Benchmarks carry a custom loader and their own report bucket; training pools load via load_dataset.
    assert CODE_DATASET_ADAPTERS["livecodebench"].load is not None
    assert CODE_DATASET_ADAPTERS["livecodebench"].group_field == "difficulty"
    assert CODE_DATASET_ADAPTERS["icpc"].load is not None
    assert CODE_DATASET_ADAPTERS["icpc"].group_field == "source"
    assert CODE_DATASET_ADAPTERS["hlce"].load is not None
    assert CODE_DATASET_ADAPTERS["hlce"].group_field == "platform"
    assert CODE_DATASET_ADAPTERS["codeforces"].load is None


def test_rating_bounds_drop_unrated_problems():
    # An unrated row can't be shown to sit in the band, so a rating-filtered pool must drop it.
    assert rating_in_bounds({"rating": 1000}, 800, 1200) is True
    assert rating_in_bounds({"rating": 700}, 800, 1200) is False
    assert rating_in_bounds({"rating": 1500}, 800, 1200) is False
    assert rating_in_bounds({"rating": None}, 800, None) is False
    assert rating_in_bounds({"rating": None}, None, 1200) is False
    assert rating_in_bounds({"rating": None}, None, None) is True
    # A rating-less dataset (deepcoder) ignores the bounds instead of emptying itself.
    assert rating_in_bounds({"problem": "..."}, 800, 1200) is True


# Environment: reset → grade → reward


def test_env_reset_parses_payload_and_reward_is_fraction():
    env = CodeContestsEnvironment(language="python", output_comparison="tokens")
    answer = {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}
    traj = env._reset_single("Print a+b.", {"answer": answer})
    assert traj.info["tests_total"] == 2
    assert traj.info["_time_limit"] == 2.0
    assert traj.info["has_checker"] is False

    half_right = "a=int(input()); b=int(input()); print(a+b if a==2 else a*b)"
    passed, total, *_ = env._grade_submission(half_right, traj)
    assert (passed, total) == (1, 2)

    traj.info["tests_passed"] = passed
    traj.info["tests_total"] = total
    traj.info["submission_result"] = "Passed 1/2 test cases."
    traj.info["completed"] = True
    reward = env._compute_reward(traj)
    assert reward == pytest.approx(0.5)


def test_python_test_tool_runs_complete_program_with_imports():
    """The python_repl test tool runs a complete program through the grading sandbox.

    It must allow the standard library and print, matching what ``submit_solution`` grades. Routed to
    the in-process restricted REPL instead, ``import sys`` is rejected ("imports are not allowed in
    the sandbox") and so is a bare ``print(...)`` ("name 'print' is not defined"), leaving a model
    unable to test a real stdin/stdout solution before submitting.
    """
    env = CodeContestsEnvironment(language="python")
    tool = env.registry.get("python_repl")
    assert tool is not None
    out = tool.execute(code="import sys\nfrom collections import Counter\nprint(sum(Counter([1, 1, 2]).values()))")
    assert out.strip() == "3"


def test_final_code_block_without_submit_is_not_graded():
    """A final ```python block with NO submit_solution call is not graded — submit_solution is the only
    graded channel, so an unsubmitted solution scores 0 even when it is correct. (Prevents a model from
    bypassing the submission cap by dumping a fenced block at the end.)
    """
    env = CodeContestsEnvironment(language="python", output_comparison="tokens")
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Read a and b on separate lines; print a+b."], [ctx])

    final = "After reasoning, the solution is:\n```python\na=int(input()); b=int(input()); print(a+b)\n```"
    env.step(eids, [final], [None])

    traj = env.get_trajectories(eids)[0]
    assert traj.done
    assert "submission_result" not in traj.info
    assert traj.info["tests_passed"] == 0
    assert traj.total_reward == pytest.approx(0.0)


def test_completed_without_submitting_scores_zero():
    """A finished episode that never submitted scores 0, not flat partial credit.

    ``submit_solution`` is the only graded channel, so a model that ends with prose (or runs out of
    turns) without submitting earns ``failure_reward`` — ``partial_reward`` would inflate the benchmark
    and, in training, reward a model for not submitting code.
    """
    env = CodeContestsEnvironment(language="python", output_comparison="tokens")
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Read a and b; print a+b."], [ctx])
    env.step(eids, ["I would add them, but I never call the submit tool."], [None])

    traj = env.get_trajectories(eids)[0]
    assert traj.done and traj.info.get("completed")
    assert "submission_result" not in traj.info
    assert traj.total_reward == pytest.approx(0.0)


def test_passing_submit_truncated_at_max_turns_still_rewarded():
    """A passing submit_solution on the last allowed turn is rewarded even though the episode truncates.

    submit_solution is a tool call, so it does not end the episode; if the model submits on its final
    turn it is truncated at max_turns with ``completed`` unset. Gating the reward on ``completed``
    scores a fully passing submission at the tool bonus alone (0.2). A correct solution must score the
    full pass fraction regardless — otherwise training punishes models that solve a problem while
    still using tools. (``max_submissions=2`` so the
    submit does not end the episode via the submission cap; max_turns truncation is what ends it here.)
    """
    env = CodeContestsEnvironment(max_turns=1, language="python", output_comparison="tokens", max_submissions=2)
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Read a and b; print a+b."], [ctx])

    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "submit_solution", "arguments": json.dumps({"code": _CORRECT_ADD})},
    }
    env.step(eids, [""], [{"tool_calls": [tool_call]}])

    traj = env.get_trajectories(eids)[0]
    assert traj.done and traj.truncated
    assert not traj.info.get("completed")
    assert traj.info["tests_passed"] == 2 and traj.info["tests_total"] == 2
    assert traj.total_reward == pytest.approx(1.0)  # pure pass fraction, not a partial/0 score


def _submit_call(cid, code):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})},
    }


def _repl_call(cid, code):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": "python_repl", "arguments": json.dumps({"code": code})},
    }


def test_test_tool_capped_per_episode():
    """The scratchpad test tool is capped at ``max_test_calls`` per episode; beyond it the call is
    rejected and the model is nudged to submit, so it cannot iterate in the sandbox without bound."""
    assert CodeContestsEnvironment(language="python").max_test_calls == 5  # default
    env = CodeContestsEnvironment(language="python", output_comparison="tokens", max_test_calls=2)
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Print a+b."], [ctx])
    for i in range(3):
        env.step(eids, [""], [{"tool_calls": [_repl_call(f"c{i}", "print('scratch')")]}])

    traj = env.get_trajectories(eids)[0]
    assert traj.info["test_call_count"] == 2
    tool_msgs = [m.content for m in traj.messages if m.role == "tool"]
    assert "scratch" in tool_msgs[0] and "scratch" in tool_msgs[1]
    assert "Test limit reached" in tool_msgs[2]
    assert not traj.done  # an exhausted test budget must not end the episode; the model can still submit


def test_single_submission_cap_ends_episode():
    """With ``max_submissions=1`` one submit_solution grades and ends the episode (the default cap is 2)."""
    assert CodeContestsEnvironment(language="python").max_submissions == 2  # default
    env = CodeContestsEnvironment(language="python", output_comparison="tokens", max_submissions=1)
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Read a and b; print a+b."], [ctx])
    env.step(eids, [""], [{"tool_calls": [_submit_call("c1", _CORRECT_ADD)]}])

    traj = env.get_trajectories(eids)[0]
    assert traj.done and traj.info["submission_count"] == 1
    assert traj.info["tests_passed"] == 2
    assert traj.total_reward == pytest.approx(1.0)


def test_submission_cap_rejects_extra_and_lifts_with_knob():
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    calls = [_submit_call("c1", "print(0)"), _submit_call("c2", _CORRECT_ADD)]  # wrong then correct

    env1 = CodeContestsEnvironment(language="python", output_comparison="tokens", max_submissions=1)
    e1, _ = env1.reset(["x"], [ctx])
    env1.step(e1, [""], [{"tool_calls": calls}])
    t1 = env1.get_trajectories(e1)[0]
    assert t1.info["submission_count"] == 1 and t1.info["tests_passed"] == 0

    env2 = CodeContestsEnvironment(language="python", output_comparison="tokens", max_submissions=2)
    e2, _ = env2.reset(["x"], [ctx])
    env2.step(e2, [""], [{"tool_calls": calls}])
    t2 = env2.get_trajectories(e2)[0]
    assert t2.info["submission_count"] == 2 and t2.info["tests_passed"] == 2


def test_sandbox_fault_during_grading_marks_the_episode_invalid():
    """A grade lost to the HOST must never train as a wrong program.

    The local backend reports such a fault by RAISING (missing interpreter, ``bwrap`` absent, fork
    table exhausted, ENOSPC), where the remote one returns ``SandboxResult(error=...)``. Unconverted,
    the raise escapes ``submit_solution`` as an ordinary tool error: the submission budget is already
    spent, so the episode completes ungraded and scores ``failure_reward`` — inside the GRPO group
    baseline, where it biases every sibling's advantage."""

    class _RaisingSandbox:
        def run(self, code, **kwargs):
            raise OSError(28, "No space left on device")

    env = CodeContestsEnvironment(
        language="python",
        output_comparison="tokens",
        max_submissions=1,
        success_reward=1.0,
        failure_reward=0.0,
        sandbox=_RaisingSandbox(),
    )
    ctx = {"answer": {"tests": _ADD_TESTS, "checker": None, "time_limit": 2.0}}
    eids, _ = env.reset(["Read a and b; print a+b."], [ctx])
    env.step(eids, [""], [{"tool_calls": [_submit_call("c1", _CORRECT_ADD)]}])

    traj = env.get_trajectories(eids)[0]
    assert traj.done and traj.info["submission_count"] == 1
    assert traj.info["tests_infra_errors"] == len(_ADD_TESTS)
    assert traj.info["tests_ran_ok"] == 0
    assert traj.episode_invalid is True
    assert traj.total_reward == pytest.approx(0.0)


def test_registry_codeforces_is_token_comparison_preset():
    from src.configs.environment_config import EnvironmentConfig
    from src.environments.registry import get_registered_environments, resolve_environment

    assert "codeforces" in get_registered_environments()
    env = resolve_environment(
        "codeforces", EnvironmentConfig(environment_type="codeforces", max_turns=4).to_env_config()
    )
    assert isinstance(env, CodeContestsEnvironment)
    assert env.grading_spec.comparison == "tokens"
    assert env.max_turns == 4

    cc = resolve_environment("code_contests", EnvironmentConfig(environment_type="code_contests").to_env_config())
    assert cc.grading_spec.comparison == "exact"


def test_reasoning_effort_default_is_medium():
    """The solver defaults to medium effort, and the prompt never mentions it (chat-template only)."""
    env = CodeContestsEnvironment(language="python")
    assert env.reasoning_effort == DEFAULT_REASONING_EFFORT == "medium"
    assert "reasoning" not in env.system_prompt.lower() and "effort" not in env.system_prompt.lower()


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(ValueError, match="reasoning_effort"):
        CodeContestsEnvironment(language="python", reasoning_effort="maximum")


def test_codeforces_preset_forwards_reasoning_effort():
    """The codeforces/code_contests registry presets thread reasoning_effort onto the env."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("codeforces", {"max_turns": 4, "reasoning_effort": "high"})
    assert env.reasoning_effort == "high"
    assert resolve_environment("code_contests", {}).reasoning_effort == "medium"


def test_non_coding_env_defines_no_reasoning_effort():
    """Only the coding env opts in; other envs leave the attribute unset so getattr(...) is None."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("native_math", {"max_turns": 2})
    assert getattr(env, "reasoning_effort", None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
