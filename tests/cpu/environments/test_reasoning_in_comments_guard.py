"""The code-contests guard on reasoning carried into a program's comments: the comment accounting by the
language's registered syntax, a program whose comments reach the bound and outweigh the rest refused unrun
on both tools with the call returned to its budget and the turn flagged untrainable, an honestly documented
program run, the offline re-grader skipping what the environment refused, and the metrics naming it."""

import json

import pytest

from scripts.environments.inference.regrade_trajectories import submitted_solutions
from src.environments.base import REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import REASONING_IN_COMMENTS_REPLY, CodeContestsEnvironment
from src.environments.envs.tasks.coding.comments import (
    REASONING_IN_COMMENTS_MIN_CHARS,
    carries_reasoning_in_comments,
    comment_chars,
)
from src.environments.episode import recovering_turn
from src.environments.sandbox.base import SandboxResult
from tests.common.code_contests import SINGLE_TEST_ANSWER, StubSandbox

MIN = REASONING_IN_COMMENTS_MIN_CHARS
# Comment lines past the bound beside eight characters of code: reasoning, not documentation.
LAUNDERED = "\n".join("# " + "r" * 100 for _ in range(MIN // 100)) + "\nprint(1)\n"
# 40 characters of comments beside 28 of code, far under the bound: documentation.
DOCUMENTED = "# reads n, prints it doubled\nn = int(input())\n# the answer\nprint(2 * n)\n"


def _call(name: str, **arguments) -> dict:
    return {"id": f"call_{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _env(**kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(
        sandbox=StubSandbox(SandboxResult(stdout="X\n", returncode=0)), tool_error_penalty=0.05, **kwargs
    )


def test_comment_chars_reads_each_languages_registered_syntax():
    """Characters inside comments, and every other character of the program, newlines included."""
    assert comment_chars("# a\nx = 1\n", "python") == (3, 7)
    assert comment_chars("x = 1  # trailing reasoning\n", "python") == (20, 8)
    assert comment_chars("// a\nint x;\n/* b\n c */\ny;", "cpp") == (14, 11)
    assert comment_chars("int x; /* mid-line */ y;", "c") == (14, 10)
    assert comment_chars("#if 0\nold code\n#endif\nz;", "cpp") == (21, 3)
    assert comment_chars("# a\necho hi\n", "bash") == (3, 9)
    assert comment_chars("", "python") == (0, 0)


def test_comment_chars_skips_string_literals_and_counts_python_bare_strings():
    grid = 'grid = """' + "\n".join("#....#" for _ in range(3000)) + '"""\nprint(grid)\n'
    assert comment_chars(grid, "python") == (0, len(grid))
    assert not carries_reasoning_in_comments(grid, "python")
    c_strings = "s = \"// not a comment\"; t = '# nor this';"
    assert comment_chars(c_strings, "cpp") == (0, len(c_strings))
    py_string = 's = "# not a comment"\nprint(s)\n'
    assert comment_chars(py_string, "python") == (0, len(py_string))
    docstring = '"""' + "d" * 50 + '"""'
    assert comment_chars(f"{docstring}\nx = 1\n", "python") == (56, 7)


def test_an_unregistered_language_has_no_comment_syntax():
    with pytest.raises(ValueError, match="language registry"):
        comment_chars("x", "fortran")


def test_the_bound_is_pinned_at_the_constant():
    assert carries_reasoning_in_comments("#" + "r" * (MIN - 1), "python")
    assert not carries_reasoning_in_comments("#" + "r" * (MIN - 2), "python")
    assert not carries_reasoning_in_comments("#" + "r" * MIN + "\n" + "x" * (MIN + 1), "python")


def test_a_laundered_program_is_refused_on_both_tools_and_the_budget_returned():
    env = _env(max_test_calls=1, max_submissions=1)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=LAUNDERED)]}])[0]
    traj = step.trajectory
    reply = traj.messages[-1]
    assert reply.role == "tool" and reply.content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    assert step.reward == pytest.approx(-0.05) and not step.done
    assert traj.info["reasoning_in_comments_calls"] == 1 and env._test_calls(traj) == 0
    # A turn of nothing but refusals trains only on a negative advantage, and the next turn is a retry.
    assert traj.messages[-2].role == "assistant" and traj.messages[-2].untrainable and recovering_turn(traj)

    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=LAUNDERED)]}])[0]
    assert traj.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="graded")
    assert not step.done and env._submissions(traj) == 0 and traj.info["reasoning_in_comments_calls"] == 2

    # The budget it returned is still there, and the accounting books the later calls as their own.
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=DOCUMENTED, stdin="3\n")]}])[0]
    assert traj.messages[-1].content.startswith("X") and traj.info["successful_tool_calls"] == 1
    assert not traj.messages[-2].untrainable and not recovering_turn(traj)
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=DOCUMENTED)]}])[0]
    assert step.done and traj.info["submission_result"].startswith("Passed")
    metrics = env.rollout_metrics(traj)
    assert metrics["episode/reasoning_in_comments_calls"] == 2.0
    laundered = comment_chars(LAUNDERED, "python")
    documented = comment_chars(DOCUMENTED, "python")
    comments, rest = 2 * laundered[0] + 2 * documented[0], 2 * laundered[1] + 2 * documented[1]
    assert metrics["episode/code_comment_share"] == pytest.approx(comments / (comments + rest))
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/turn_shaping"] == pytest.approx(-0.1)


def test_a_c_family_program_is_read_by_its_own_syntax():
    env = _env(language=["python", "cpp"])
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    laundered = "\n".join("// " + "r" * 100 for _ in range(MIN // 100)) + "\nint main(){}\n"
    step = env.step(ids, [""], [{"tool_calls": [_call("run_code", code=laundered, language="cpp")]}])[0]
    assert step.trajectory.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    block = "/*" + "\n".join("r" * 100 for _ in range(MIN // 100)) + "*/\nint main(){}\n"
    step = env.step(ids, [""], [{"tool_calls": [_call("run_code", code=block, language="cpp")]}])[0]
    assert step.trajectory.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    assert step.trajectory.info["reasoning_in_comments_calls"] == 2


def test_comments_under_the_bound_or_under_the_code_run():
    env = _env()
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=DOCUMENTED, stdin="3\n")]}])[0]
    assert step.trajectory.messages[-1].content.startswith("X") and step.reward == 0.0
    # Many comments beside more code than that: documentation at scale, not laundering.
    heavy = (
        "\n".join(f"# note {i}" for i in range(2000))
        + "\n"
        + "\n".join(f"x{i} = {i}" for i in range(4000))
        + "\nprint(1)\n"
    )
    assert comment_chars(heavy, "python")[0] >= MIN
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=heavy, stdin="3\n")]}])[0]
    assert step.trajectory.messages[-1].content.startswith("X")
    assert "reasoning_in_comments_calls" not in step.trajectory.info


def test_a_laundered_call_after_the_accept_takes_the_free_post_accept_reply():
    env = _env(max_submissions=3)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    calls = [_call("submit_solution", code=DOCUMENTED), {**_call("submit_solution", code=LAUNDERED), "id": "call_2"}]
    step = env.step(ids, [""], [{"tool_calls": calls}])[0]
    assert step.done and step.trajectory.info["submission_result"].startswith("Passed")
    assert "already passed" in step.trajectory.messages[-1].content and step.reward >= 0.0
    assert "reasoning_in_comments_calls" not in step.trajectory.info


def test_the_regrader_takes_no_slot_for_a_refused_program():
    env = _env(max_submissions=1)
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=LAUNDERED)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=DOCUMENTED)]},
        ]
    }
    assert submitted_solutions(episode, env) == [(DOCUMENTED, None)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
