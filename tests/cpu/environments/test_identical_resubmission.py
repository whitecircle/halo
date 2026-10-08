"""A resubmission identical to a program this episode already graded, comments aside, is refused unrun as a
tool error with the submission returned to its budget and the turn flagged like any refusal; a changed program
grades; a grade the backend lost records no program; a repeat beside the accept in one turn takes the free
post-accept reply; the offline re-grader takes no slot for the repeat."""

import json

import pytest

from scripts.environments.inference.regrade_trajectories import submitted_solutions
from src.environments.envs.tasks.coding.code_contests import (
    IDENTICAL_RESUBMISSION_REPLY,
    SUBMISSION_AFTER_ACCEPT_REPLY,
    CodeContestsEnvironment,
    normalized_program,
)
from src.environments.envs.tasks.coding.comments import strip_comments
from src.environments.episode import recovering_turn
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
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


def test_an_identical_resubmission_is_refused_unrun_and_a_changed_one_grades():
    sandbox = _ScriptedSandbox(SandboxResult(stdout="X\n", returncode=0))
    env = _env(sandbox, max_submissions=2)
    ids, _ = env.reset(["solve it"], [{"answer": {"tests": [{"input": "", "output": "Y"}]}}])
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=WRONG)]}])[0]
    traj = step.trajectory
    first_result = traj.info["submission_result"]
    assert (
        not step.done
        and env._submissions(traj) == 1
        and first_result.startswith("Passed 0/1")
        and len(sandbox.runs) == 1
    )
    for repeat in (WRONG_RESPACED, WRONG_COMMENTED):
        step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=repeat)]}])[0]
        assert traj.messages[-1].content == "Error: " + IDENTICAL_RESUBMISSION_REPLY
        assert step.reward == pytest.approx(-0.05) and not step.done
        assert len(sandbox.runs) == 1, "a refused program never reaches the sandbox"
        assert env._submissions(traj) == 1 and traj.info["submission_result"] == first_result
        assert traj.info["submission_pass_fracs"] == [0.0]
        assert traj.messages[-2].untrainable and recovering_turn(traj)
    assert traj.info["identical_resubmissions"] == 2
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=RIGHT)]}])[0]
    assert step.done and env._submissions(traj) == 2 and len(sandbox.runs) == 2
    assert env.rollout_metrics(traj)["episode/identical_resubmissions"] == 2.0


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


def test_the_regrader_takes_no_slot_for_an_identical_program():
    env = _env(_ScriptedSandbox(SandboxResult()), max_submissions=2)
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=WRONG)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=WRONG_COMMENTED)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=RIGHT)]},
        ]
    }
    assert submitted_solutions(episode, env) == [(WRONG, None), (RIGHT, None)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
