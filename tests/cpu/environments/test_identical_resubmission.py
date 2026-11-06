"""A resubmission identical to a program this episode already graded, comments aside, takes that grade again
unregraded behind a note naming the earlier submission: the call is spent, priced as a resubmission and trains like
any submission; a changed program grades; a grade the backend lost records no program; a repeat beside the accept in
one turn takes the free post-accept reply; the offline re-grader gives the repeat its slot."""

import json
from unittest import mock

import pytest

from halo.entrypoints.code_contests_regrade import submitted_solutions
from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.results import SandboxResult
from halo.environments.tasks.coding import grading as grading_module
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tasks.coding.code_contests_prompts import (
    IDENTICAL_RESUBMISSION_NOTE,
    SUBMISSION_AFTER_ACCEPT_REPLY,
)
from halo.environments.tasks.coding.program_source import normalized_program, strip_comments
from halo.rewards.samples import REFUSED_CALL_IDS_KEY
from tests.common.code_contests import SINGLE_TEST_ANSWER

WRONG = "n = int(input())\nprint(n)\n"
WRONG_RESPACED = "n = int(input())   \nprint(n)  \n\n"
WRONG_COMMENTED = "# attempt two\nn = int(input())  # read n\nprint(n)\n"
RIGHT = "n = int(input())\nprint(2 * n)\n"


class _ScriptedSandbox(SandboxExecutor):
    """Answers the given results in order (the last one again past the end) and records every run."""

    isolated = True

    def __init__(self, *results: SandboxResult):
        self._results = list(results)
        self.runs: list[str] = []

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.runs.append(code)
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


def _call(name: str, **arguments) -> dict:
    return {"id": f"call_{name}_{len(arguments)}", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _env(sandbox: SandboxExecutor, **kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(sandbox=sandbox, tool_error_penalty=0.05, submission_reward=0.1, **kwargs)


def test_normalized_program_reads_the_program_without_comments_or_trailing_whitespace():
    assert normalized_program(WRONG, "python") == normalized_program(WRONG_RESPACED, "python")
    assert normalized_program(WRONG, "python") == normalized_program(WRONG_COMMENTED, "python")
    assert normalized_program(WRONG, "python") != normalized_program(RIGHT, "python")
    assert normalized_program(WRONG, "python") != normalized_program(WRONG, "cpp"), (
        "the same text is another program in another language"
    )
    assert normalized_program("  x = 1  # one\n", "python") == "python\n  x = 1"
    assert strip_comments("int x; /* c */ y; // tail\nz;", "cpp") == "int x;  y; \nz;"


def test_normalized_program_keeps_the_whitespace_inside_string_literals():
    """A blank line or trailing space inside a literal is program data, so two programs differing only there are
    two programs; outside a literal the same differences are still none."""
    triple = 'print("""a\n\nb""")\n'
    padded = 'print("""a\n\n\nb""")\n'
    trailing = 'print("""a \n\nb""")\n'
    assert len({normalized_program(code, "python") for code in (triple, padded, trailing)}) == 3
    assert normalized_program('x = "a  "\n\n\ny = 1   \n', "python") == 'python\nx = "a  "\ny = 1'
    assert normalized_program('print("""a\n\nb""")  \n\n', "python") == normalized_program(triple, "python")
    assert normalized_program("int main() {\n\n  return 0;  \n}\n", "cpp") == "cpp\nint main() {\n  return 0;\n}"


def test_strip_comments_reads_each_languages_registered_syntax_and_keeps_string_literals():
    assert strip_comments("# a\nx = 1  # note\n", "python") == "\nx = 1  \n"
    assert strip_comments("// a\nint x;\n/* b\n c */\ny;", "cpp") == "\nint x;\n\ny;"
    assert strip_comments("#if 0\nold code\n#endif\nz;", "cpp") == "\nz;"
    assert strip_comments("# a\necho hi\n", "bash") == "\necho hi\n"
    strings = "s = \"// not a comment\"; t = '# nor this';"
    assert strip_comments(strings, "cpp") == strings
    assert strip_comments('s = \'# kept\'\nt = "a \\" # kept"  # note\n', "python") == (
        's = \'# kept\'\nt = "a \\" # kept"  \n'
    )
    docstring = '"""# kept"""\nx = 1\n'
    assert strip_comments(docstring, "python") == docstring
    grid = 'grid = """\n#..#\n#..#\n"""\nprint(grid)\n'
    assert strip_comments(grid, "python") == grid
    assert normalized_program('print("# a")\n', "python") != normalized_program('print("")\n', "python"), (
        "a '#' inside a string is the program's, not a comment"
    )
    with pytest.raises(ValueError, match="unsupported language"):
        strip_comments("x", "fortran")


def test_an_identical_resubmission_takes_the_earlier_grade_spent_and_priced():
    sandbox = _ScriptedSandbox(SandboxResult(stdout="X\n", returncode=0))
    env = _env(sandbox, max_submissions=4, resubmission_penalty=0.2)
    ids, _ = env.reset(["solve it"], [{"answer": {"tests": [{"input": "", "output": "Y"}]}}])
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=WRONG)]}])[0]
    traj = step.trajectory
    first_result = traj.info["submission_result"]
    assert first_result.startswith("Passed 0/1") and len(sandbox.runs) == 1
    for count, repeat in enumerate((WRONG_RESPACED, WRONG_COMMENTED), start=2):
        step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=repeat)]}])[0]
        assert traj.messages[-1].content == f"{IDENTICAL_RESUBMISSION_NOTE.format(ordinal=1)}\n{first_result}"
        assert step.reward == 0.0 and not step.done, "a submission, not a tool error"
        assert len(sandbox.runs) == 1, "a repeat is never regraded"
        assert env._submissions(traj) == count and traj.info["submission_result"] == first_result
        assert not traj.messages[-2].calls_rejected and not traj.messages[-2].untrainable
    assert traj.info["submission_pass_fracs"] == [0.0, 0.0, 0.0]
    assert traj.info["identical_resubmissions"] == 2
    assert env._grade_episode(traj).shaping["resubmission"] == pytest.approx(-0.4)
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0]
    assert step.done and env._submissions(traj) == 4 and len(sandbox.runs) == 2
    assert env.rollout_metrics(traj)["episode/identical_resubmissions"] == 2.0


def test_a_repeat_of_an_earlier_program_restores_that_grade_not_the_last_one():
    """Two tests: the first program fails both, a second passes one, the first sent again draws the first grade."""
    sandbox = _ScriptedSandbox(
        SandboxResult(stdout="X\n", returncode=0),
        SandboxResult(stdout="X\n", returncode=0),
        SandboxResult(stdout="2\n", returncode=0),
        SandboxResult(stdout="X\n", returncode=0),
    )
    env = _env(sandbox, max_submissions=4)
    tests = [{"input": "1", "output": "2"}, {"input": "2", "output": "4"}]
    ids, _ = env.reset(["solve it"], [{"answer": {"tests": tests}}])
    traj = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=WRONG)]}])[0].trajectory
    first_result = traj.info["submission_result"]
    env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])
    assert traj.info["tests_passed"] == 1 and len(sandbox.runs) == 4
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=WRONG_COMMENTED)]}])[0]
    assert not step.done and env._submissions(traj) == 3
    assert traj.messages[-1].content == f"{IDENTICAL_RESUBMISSION_NOTE.format(ordinal=1)}\n{first_result}"
    info = traj.info
    assert (info["tests_passed"], info["tests_total"], info["submission_result"]) == (0, 2, first_result)
    assert info["submission_pass_fracs"] == [0.0, 0.5, 0.0] and len(sandbox.runs) == 4
    assert info["_submitted_code"] == WRONG_COMMENTED
    assert len(info["_failed_submission_runs_left"]) == 3, "the resend is booked as a failed submission"
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0]
    assert step.done and env._submissions(traj) == 4 and len(sandbox.runs) == 4, "the budget is spent unregraded"
    assert (
        traj.messages[-1].content
        == f"{IDENTICAL_RESUBMISSION_NOTE.format(ordinal=2)}\n" + (traj.info["submission_result"])
    )


def test_a_repeat_reaches_the_judge_as_an_ordinary_call():
    """The repeat was spent, not refused: the scorer's sample carries no refused call, so a judge reads it as an
    action like any submission."""
    sandbox = _ScriptedSandbox(SandboxResult(stdout="X\n", returncode=0))
    env = _env(sandbox, max_submissions=3)
    ids, _ = env.reset(["solve it"], [{"answer": {"tests": [{"input": "", "output": "Y"}]}}])
    env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=WRONG)]}])
    repeat = {**_call("submit_solution", code=WRONG_COMMENTED), "id": "repeat"}
    traj = env.step(ids, [""], [{"tool_calls": [repeat]}])[0].trajectory
    assert traj.messages[-2].refused_call_ids == ()
    sample = env._scoring_sample(traj)
    assert [m.get(REFUSED_CALL_IDS_KEY) for m in sample.completion if m["role"] == "assistant"] == [None, None]


def test_a_repeat_beside_the_accept_in_one_turn_takes_the_free_post_accept_reply():
    sandbox = _ScriptedSandbox(SandboxResult(stdout="X\n", returncode=0))
    env = _env(sandbox, max_submissions=3)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    calls = [_call("submit_solution", code=RIGHT), {**_call("submit_solution", code=RIGHT), "id": "call_2"}]
    step = env.step(ids, [""], [{"tool_calls": calls}])[0]
    assert step.done and step.trajectory.info["submission_result"].startswith("Passed 1/1")
    assert step.trajectory.messages[-1].content == SUBMISSION_AFTER_ACCEPT_REPLY and len(sandbox.runs) == 1
    assert step.reward == 0.0, "the refund, not the error price (the submission bonus settles with the episode)"
    assert "identical_resubmissions" not in step.trajectory.info


def test_a_program_whose_grade_the_backend_lost_may_come_back():
    sandbox = _ScriptedSandbox(SandboxResult(error="backend down"), SandboxResult(stdout="X\n", returncode=0))
    env = _env(sandbox, max_submissions=2)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0]
    traj = step.trajectory
    assert traj.info["tests_infra_errors"] == 1 and "_graded_programs" not in traj.info
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0]
    assert step.done and traj.info["submission_result"].startswith("Passed 1/1") and len(sandbox.runs) == 2
    assert "identical_resubmissions" not in traj.info


class _TickingSandbox(SandboxExecutor):
    """Passes every test and spends ``seconds`` of a shared clock on each run."""

    isolated = True

    def __init__(self, clock: dict[str, float], seconds: float):
        self._clock = clock
        self._seconds = seconds
        self.runs = 0

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.runs += 1
        self._clock["t"] += self._seconds
        return SandboxResult(stdout="2\n", returncode=0)


def test_a_grade_the_budget_cut_short_may_come_back():
    """Two of four tests passed and the grading budget ran out before any failed: no verdict on the program, so the
    same program is graded again, not answered with the cut-short grade."""
    clock = {"t": 0.0}
    sandbox = _TickingSandbox(clock, seconds=10.0)
    env = _env(sandbox, max_submissions=3, max_grading_seconds=15.0)
    tests = [{"input": str(i), "output": "2"} for i in range(4)]
    ids, _ = env.reset(["solve it"], [{"answer": {"tests": tests}}])
    with mock.patch.object(grading_module.time, "monotonic", lambda: clock["t"]):
        traj = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0].trajectory
        assert traj.info["grading_budget_hit"] and traj.info["tests_passed"] == 2 and sandbox.runs == 2
        assert "_graded_programs" not in traj.info
        env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])
    assert sandbox.runs == 4, "the retry graded again"
    assert "identical_resubmissions" not in traj.info


def test_the_regrader_gives_an_identical_program_its_slot():
    env = _env(_ScriptedSandbox(SandboxResult()), max_submissions=2)
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=WRONG)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=WRONG_COMMENTED)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=RIGHT)]},
        ]
    }
    assert submitted_solutions(episode, env) == [(WRONG, None), (WRONG_COMMENTED, None), (RIGHT, None)]


def test_the_regrader_pairs_each_turn_with_its_own_replies_when_an_engine_reuses_call_ids():
    """Every turn's call is ``call_0``: the refused one must not take the graded one's reply, nor the reverse."""

    def submit(code):
        return {
            "id": "call_0",
            "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})},
        }

    env = _env(_ScriptedSandbox(SandboxResult()), max_submissions=3)
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [submit(WRONG)]},
            {"role": "tool", "tool_call_id": "call_0", "content": "Error: Not graded: submission budget spent."},
            {"role": "assistant", "tool_calls": [submit(RIGHT)]},
            {"role": "tool", "tool_call_id": "call_0", "content": "Passed 1/1 tests."},
        ]
    }
    assert submitted_solutions(episode, env) == [(RIGHT, None)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
