#!/usr/bin/env python
"""CPU tests: what the code-contests scratchpad tells the model about a program that failed.

The scratchpad is where the policy debugs, so its reply has to carry the diagnosis a judge would:
a compile error as the compiler's first diagnostics (the last stderr line of g++ is a caret gutter),
a crash as its signal and the traceback's tail, ahead of any stdout (the protocol cuts a long
observation from the end), and a program run under the same limits its graded tests get — the
problem's time limit, and a stack as large as the memory limit. A run that showed the model nothing
(no input, a clean exit, nothing on stdout) stays spent, and a turn of nothing else is flagged
untrainable, so the next turn retries on the recovery reserve. Every program here really compiles
and runs on the local backend.

Run: python tests/cpu/environments/test_scratchpad_feedback.py  (or pytest)
"""

import json
import resource
import shutil

import pytest

from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.code_contests import (
    NO_STDIN_NOTE,
    SCRATCHPAD_BUDGET_SPENT_REPLY,
    SCRATCHPAD_TIME_LIMIT_NOTE,
    SUBMIT_TOOL,
    CodeContestsEnvironment,
)
from src.environments.envs.tasks.coding.grading import run_solution_against_tests
from src.environments.episode import recovering_turn
from src.environments.registry import resolve_environment
from src.environments.sandbox.base import REPL_NO_OUTPUT_MESSAGE, SandboxExecutor, SandboxResult
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.repl import run_code_via_sandbox
from src.environments.tools.definitions import NativeToolCall
from tests.common.code_contests import counts_beside_budget_words, retired_budget_phrases

needs_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")

_UNDECLARED = """
#include <iostream>
int main() {
    int n;
    std::cin >> n;
    std::cout << undeclared_total + n << std::endl;
}
"""
_SEGFAULT = """
#include <cstdio>
int main() { std::puts("before"); std::fflush(stdout); volatile int* p = nullptr; *p = 1; }
"""
# Non-tail recursion 2e5 deep with a padded frame: ~80 MB of stack, past the 8 MB a container
# launches with and inside the 1 GiB memory limit a contest judge gives the stack.
_DEEP_RECURSION = """
#include <iostream>
int dfs(int d) {
    volatile char pad[400];
    pad[0] = static_cast<char>(d);
    if (d == 0) return 0;
    return dfs(d - 1) + (pad[0] & 1);
}
int main() { std::cout << dfs(200000) << std::endl; }
"""
_DEEP_RECURSION_STACK_BYTES = 256 * 1024 * 1024
# Reads its input and prints an answer only when it got one: given no stdin it exits cleanly and silently.
_READS_INPUT = "import sys\ndata = sys.stdin.read().split()\nif data:\n    print(int(data[0]) + 1)"
_SPIN_SECONDS = """
#include <chrono>
#include <iostream>
int main() {
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(%d);
    while (std::chrono::steady_clock::now() < end) {}
    std::cout << "done" << std::endl;
}
"""


def _env(language="cpp", **kwargs):
    kwargs.setdefault("sandbox", LocalSubprocessSandbox())
    return CodeContestsEnvironment(language=language, max_test_calls=6, **kwargs)


def _reset(env, time_limit=None):
    answer = {"tests": [{"input": "1\n", "output": "2\n"}], "time_limit": time_limit}
    ids, _ = env.reset(["solve it"], [{"answer": json.dumps(answer)}])
    return ids


def _episode(env, time_limit=None):
    return env.get_trajectories(_reset(env, time_limit))[0]


def _turn(env, ids, *calls: tuple[str, dict]):
    """One assistant turn through the protocol's step, its ``(tool, arguments)`` calls as the engine parses them."""
    tool_calls = [
        {"id": f"c{i}", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(calls)
    ]
    return env.step(ids, [""], [{"finish_reason": "stop", "tool_calls": tool_calls}])[0]


def _last_turn_flagged(traj) -> bool:
    return next(m for m in reversed(traj.messages) if m.role == "assistant").calls_rejected


def _scratchpad(env, traj, **arguments):
    """One scratchpad call through the protocol: admitted, counted and truncated as in a rollout."""
    call = NativeToolCall(id="c", name=env.test_tool_name, arguments=arguments)
    results, _ = env._execute_tool_calls([call], traj)
    return results[0].content


@needs_gpp
def test_a_compile_error_reply_carries_the_compilers_first_diagnostic():
    """g++'s last stderr line is the caret gutter under the offending token; the reply must lead with
    the error line that names the undeclared identifier."""
    env = _env()
    reply = _scratchpad(env, _episode(env), code=_UNDECLARED, stdin="1\n")
    assert reply.startswith("Error: compilation failed\n"), reply
    first = reply.splitlines()[1:4]
    assert any("error:" in line and "undeclared_total" in line for line in first), reply


@needs_gpp
def test_a_compile_error_is_not_mistaken_for_missing_stdin():
    env = _env()
    reply = _scratchpad(env, _episode(env), code=_UNDECLARED)
    assert reply.startswith("Error: compilation failed")
    assert NO_STDIN_NOTE not in reply


def test_a_crash_after_long_output_keeps_the_traceback_inside_the_observation_cap():
    """A program printing 100k debug lines and then crashing: the protocol keeps the first
    ``max_observation_chars`` of the reply, so the error must come before the stdout."""
    env = _env(language="python")
    code = "for i in range(100000):\n    print('debug', i)\n\ndef f(x):\n    return x[5]\n\nf([])\n"
    reply = _scratchpad(env, _episode(env), code=code, stdin="1\n")
    head = reply[: env.max_observation_chars]
    assert head.startswith("Error: Traceback (most recent call last):")
    assert "line 5, in f" in head and "IndexError: list index out of range" in head
    assert head.index("IndexError") < head.index("Output:\ndebug 0")


@needs_gpp
def test_a_segfault_names_the_signal_in_the_scratchpad_and_the_verdict():
    env = _env()
    reply = _scratchpad(env, _episode(env), code=_SEGFAULT, stdin="1\n")
    assert reply.startswith("Error: killed by SIGSEGV (invalid memory access or stack overflow)"), reply
    assert "Output:\nbefore" in reply
    grade = run_solution_against_tests(
        _SEGFAULT,
        [{"input": "1", "output": "2"}],
        sandbox=LocalSubprocessSandbox(),
        language="cpp",
        verdict_detail="full",
    )
    assert "RUNTIME ERROR (killed by SIGSEGV (invalid memory access or stack overflow))" in grade.details


@needs_gpp
def test_deep_recursion_runs_on_a_stack_as_large_as_the_memory_limit():
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    if hard != resource.RLIM_INFINITY and hard < _DEEP_RECURSION_STACK_BYTES:
        pytest.skip(f"this process's hard stack limit ({hard} bytes) caps what the sandbox may raise the stack to")
    sandbox = LocalSubprocessSandbox()
    res = sandbox.run(_DEEP_RECURSION, language="cpp", timeout=10.0)
    assert res.ok, res
    grade = run_solution_against_tests(
        _DEEP_RECURSION, [{"input": "", "output": "100000"}], sandbox=sandbox, language="cpp"
    )
    assert grade.passed == 1, grade.details


def test_an_interpreted_run_still_starts_threads():
    """glibc sizes a thread's stack by the stack limit, so raising it to the memory limit for an
    interpreter would fail every thread it starts under the address-space cap."""
    code = "import threading\nt = threading.Thread(target=print, args=('from a thread',))\nt.start(); t.join()"
    assert run_code_via_sandbox(code, LocalSubprocessSandbox()) == "from a thread"


@needs_gpp
def test_a_scratchpad_run_is_held_to_the_limit_it_is_graded_at():
    """A 1 s stated limit at a 2x compiled scale grades C++ at 2 s: a 5 s program must time out in the
    scratchpad at that limit, not run to completion under a generous REPL timeout."""
    env = _env(compiled_time_limit_scale=2.0)
    reply = _scratchpad(env, _episode(env, time_limit=1.0), code=_SPIN_SECONDS % 5, stdin="1\n")
    assert reply.startswith(f"Error: execution exceeded 2s timeout {SCRATCHPAD_TIME_LIMIT_NOTE}"), reply
    assert NO_STDIN_NOTE not in reply


@needs_gpp
def test_a_compile_that_outruns_its_timeout_is_a_compile_error_not_an_infra_fault():
    """A compile timeout is the source's doing: raised as an infra error it would charge the scratchpad
    call as a failed tool call and void the episode's grade."""
    sandbox = LocalSubprocessSandbox(compile_timeout=0.01)
    res = sandbox.run(_UNDECLARED, language="cpp")
    assert res.compile_failed and res.error is None, res
    assert "compilation timed out after 0.01 s" in res.stderr
    reply = run_code_via_sandbox(_UNDECLARED, sandbox, language="cpp")
    assert reply == "Error: compilation failed\ncompilation timed out after 0.01 s"
    grade = run_solution_against_tests(_UNDECLARED, [{"input": "1", "output": "2"}], sandbox=sandbox, language="cpp")
    assert grade.infra_errors == 0 and "COMPILATION ERROR" in grade.details


def test_no_scratchpad_reply_states_the_runs_left():
    """The cap is enforced, never stated: a reply is the run's output and its notes, the run that spends
    the last one reads like any other, and the next call is refused with the spent-budget reply, which
    names no count either and spends nothing."""
    env = _env(language="python")
    traj = _episode(env)
    replies = [_scratchpad(env, traj, code=code, stdin="1\n") for code in ("print(1)", "raise SystemExit(3)")]
    assert replies[0] == "1"
    assert replies[1].startswith("Error: ")
    for _ in range(4):
        replies.append(_scratchpad(env, traj, code="print(2)", stdin="1\n"))
    assert replies[-1] == "2" and env._test_calls(traj) == 6
    refused = _scratchpad(env, traj, code="print(3)", stdin="1\n")
    assert refused == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}" and env._test_calls(traj) == 6
    for reply in (*replies, refused):
        assert not retired_budget_phrases(reply) and not counts_beside_budget_words(reply), reply
    assert env._run_test("print(3)") == f"3\n{NO_STDIN_NOTE}", "a direct call outside an episode carries no budget"


def test_a_long_output_is_cut_so_the_notes_after_it_survive():
    """The protocol cuts an observation past ``max_observation_chars`` from its end, where the notes
    are; the program's output gives way instead, so the reply keeps them and fits the cap."""
    env = _env(language="python")
    reply = _scratchpad(env, _episode(env), code="print('x' * 2_000_000)")
    assert len(reply) <= env.max_observation_chars, len(reply)
    assert reply.startswith("xxx") and "\n…[truncated " in reply
    assert reply.endswith(f"\n{NO_STDIN_NOTE}"), reply[-300:]


class _PrintsSandbox(SandboxExecutor):
    """Answers every run with ``n`` characters of stdout and a clean exit."""

    def __init__(self, n: int):
        self.n = n

    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return SandboxResult(stdout="y" * self.n, returncode=0)


@pytest.mark.parametrize("past_the_fit", [-24, 0, 1])
def test_output_that_fits_beside_its_notes_is_not_cut(past_the_fit):
    """Only a reply that would pass the cap is cut: an output up to the cap less its no-stdin note line
    comes back whole, one character more is cut."""
    tail = f"\n{NO_STDIN_NOTE}"
    cap = _env(language="python").max_observation_chars
    n = cap - len(tail) + past_the_fit
    env = _env(language="python", sandbox=_PrintsSandbox(n))
    reply = _scratchpad(env, _episode(env), code="print(1)")
    assert len(reply) <= cap and reply.endswith(tail)
    assert (reply == "y" * n + tail) is (past_the_fit <= 0), reply[-120:]


def test_a_list_program_is_refused_unspent_on_both_tools():
    """A list ``code`` has no string reading, and staged into the sandbox it fails there as the
    backend's fault, voiding a graded episode: refused at binding, it spends neither budget and grades
    nothing."""
    env = _env(language="python")
    traj = _episode(env)
    for tool in (env.test_tool_name, SUBMIT_TOOL):
        call = NativeToolCall(id="c", name=tool, arguments={"code": ["print(2)"]})
        (result,), _ = env._execute_tool_calls([call], traj)
        assert result.content == f"Error: {tool}: code must be a string, got list" and not result.success
    assert env._test_calls(traj) == 0 and env._submissions(traj) == 0
    assert "submission_result" not in traj.info and not traj.episode_invalid


def test_a_scalar_stdin_runs_as_its_string_and_a_null_one_as_none():
    """``"stdin": 5`` (as SGLang's Gemma 4 parser emits it) feeds the program ``5``; ``"stdin": null``
    is a run on no input."""
    env = _env(language="python")
    traj = _episode(env)
    assert _scratchpad(env, traj, code="print(int(input()) * 2)", stdin=5) == "10"
    reply = _scratchpad(env, traj, code="print(int(input()) * 2)", stdin=None)
    assert reply.startswith("Error: ") and "EOFError" in reply and NO_STDIN_NOTE in reply, reply


def test_a_silent_input_less_run_is_booked_like_any_run():
    """A run given no input that prints nothing still ran: it spends its slot, counts as a successful call
    and earns ``tool_success_reward`` like the run beside it."""
    env = _env(language="python", tool_success_reward=0.05)
    traj = _episode(env)
    calls = [
        NativeToolCall(id="a", name=env.test_tool_name, arguments={"code": _READS_INPUT}),
        NativeToolCall(id="b", name=env.test_tool_name, arguments={"code": "print(1)", "stdin": "1\n"}),
    ]
    results, reward = env._execute_tool_calls(calls, traj)
    assert results[0].content == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}", results[0].content
    assert reward == pytest.approx(0.10), "both runs are paid"
    assert (traj.info["total_tool_calls"], traj.info["successful_tool_calls"]) == (2, 2)
    assert env._test_calls(traj) == 2


def test_output_computed_from_no_input_says_it_got_none():
    """A program that prints without reading the input it needed reads like a result; the note says
    no stdin was passed. Empty stdin is not refused: a self-test embedding its own input is a real use."""
    env = _env(language="python")
    reply = _scratchpad(env, _episode(env), code="import sys\nprint(len(sys.stdin.read()))")
    assert reply == (
        "0\n(No stdin was passed to this run; if the program reads input, pass it in the `stdin` argument.)"
    ), reply


@pytest.mark.parametrize(
    ("code", "error"), [("def f(:\n    pass", "SyntaxError"), ("if 1:\nprint(1)", "IndentationError")]
)
def test_a_python_source_that_does_not_compile_gets_no_missing_input_note(code, error):
    """The interpreter stops on a SyntaxError before reading anything, so a missing input cannot be its cause."""
    env = _env(language="python")
    reply = _scratchpad(env, _episode(env), code=code)
    assert reply.startswith("Error: ") and error in reply, reply
    assert NO_STDIN_NOTE not in reply and not retired_budget_phrases(reply), reply


def _starved_runs(env, traj) -> float:
    return env.rollout_metrics(traj)["episode/starved_test_runs"]


def test_every_silent_input_less_run_spends_its_run_and_counts_as_starved():
    """A solution run with no stdin reads nothing and prints nothing: it spends its run, the reply carries the
    plain no-stdin note, and ``episode/starved_test_runs`` counts it, again on every such run. A crash, output
    and a quiet run on real input each told the model something: they spend their runs and are not starved."""
    env = _env(language="python")
    traj = _episode(env)
    silent = _scratchpad(env, traj, code=_READS_INPUT)
    assert silent == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}", silent
    assert env._test_calls(traj) == 1 and _starved_runs(env, traj) == 1.0
    crash = _scratchpad(env, traj, code="print(int(input()) + 1)")
    assert crash.startswith("Error: ") and "EOFError" in crash and crash.endswith(f"\n{NO_STDIN_NOTE}"), crash
    printed = _scratchpad(env, traj, code="print(7)")
    assert printed == f"7\n{NO_STDIN_NOTE}", printed
    quiet_on_input = _scratchpad(env, traj, code="import sys\nsys.stdin.read()", stdin="1\n")
    assert quiet_on_input == REPL_NO_OUTPUT_MESSAGE, quiet_on_input
    assert env._test_calls(traj) == 4 and _starved_runs(env, traj) == 1.0
    again = _scratchpad(env, traj, code=_READS_INPUT)
    assert again == silent and env._test_calls(traj) == 5 and _starved_runs(env, traj) == 2.0


def test_a_clean_input_less_run_that_writes_only_to_stderr_is_starved():
    """A clean exit's reply leaves stderr out, so a program that only logged to stderr showed the model no
    output: it counts as starved, and spends its run."""
    env = _env(language="python")
    traj = _episode(env)
    reply = _scratchpad(
        env, traj, code="import sys\ndata = sys.stdin.read()\nsys.stderr.write('debug: read nothing\\n')"
    )
    assert reply == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}", reply
    assert env._test_calls(traj) == 1 and _starved_runs(env, traj) == 1.0


def test_a_turn_of_only_starved_runs_is_flagged_stays_spent_and_the_next_turn_recovers():
    """A turn whose every call was a silent input-less run showed the model nothing: it is flagged untrainable
    like a turn of refused calls, so the next turn runs on the recovery reserve instead of a fresh cap. Unlike a
    refusal the runs stay spent and booked as runs, their replies unchanged, and no recovery price is charged;
    ``episode/starved_turns`` counts the turn, ``episode/starved_test_runs`` every run."""
    env = _env(language="python", tool_success_reward=0.05, length_cutoff_penalty=0.3)
    ids = _reset(env)
    step = _turn(
        env,
        ids,
        (env.test_tool_name, {"code": _READS_INPUT}),
        (env.test_tool_name, {"code": "import sys\nsys.stdin.read()"}),
    )
    traj = step.trajectory
    assert _last_turn_flagged(traj) and recovering_turn(traj)
    replies = traj.messages[-2:]
    assert [m.content for m in replies] == [f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}"] * 2
    assert all(type(m.content) is str for m in replies), "the reply's marker never reaches the conversation"
    assert env._test_calls(traj) == 2 and traj.info["successful_tool_calls"] == 2
    assert step.reward == pytest.approx(0.10), "paid as runs: no refund, no charge"
    assert env._tool_use_shaping(traj) == 0.0, "not a cut or empty turn: no length_cutoff_penalty"
    metrics = env.rollout_metrics(traj)
    assert (metrics["episode/starved_test_runs"], metrics["episode/starved_turns"]) == (2.0, 1.0)

    step = _turn(env, ids, (env.test_tool_name, {"code": _READS_INPUT, "stdin": "1\n"}))
    assert step.trajectory.messages[-1].content == "2"
    assert not _last_turn_flagged(step.trajectory) and not recovering_turn(step.trajectory)
    assert env._test_calls(step.trajectory) == 3
    assert env.rollout_metrics(step.trajectory)["episode/starved_turns"] == 1.0


def test_a_starved_run_beside_a_productive_call_leaves_the_turn_trainable():
    """A run that printed on its input, or a graded submission, did something: the turn trains, and the starved
    run beside it is counted as a run, not as a starved turn."""
    env = _env(language="python")
    for productive in (
        (env.test_tool_name, {"code": "print(int(input()) + 1)", "stdin": "1\n"}),
        (SUBMIT_TOOL, {"code": "print(0)"}),
    ):
        step = _turn(env, _reset(env), (env.test_tool_name, {"code": _READS_INPUT}), productive)
        traj = step.trajectory
        assert not _last_turn_flagged(traj) and not recovering_turn(traj), productive
        metrics = env.rollout_metrics(traj)
        assert (metrics["episode/starved_test_runs"], metrics["episode/starved_turns"]) == (1.0, 0.0), productive


@pytest.mark.parametrize(
    "arguments",
    [
        {"code": "import sys\nsys.stdin.read()", "stdin": "1\n"},
        {"code": "print(7)"},
        {"code": "print(int(input()) + 1)"},
    ],
    ids=["silent-on-input", "output-on-no-input", "crash-on-no-input"],
)
def test_a_run_given_input_or_that_showed_something_is_not_flagged(arguments):
    env = _env(language="python")
    step = _turn(env, _reset(env), (env.test_tool_name, arguments))
    assert not _last_turn_flagged(step.trajectory) and not recovering_turn(step.trajectory)
    assert env.rollout_metrics(step.trajectory)["episode/starved_turns"] == 0.0


@pytest.mark.parametrize(
    ("language", "code", "starved"),
    [
        ("python", "def solve(n):\n    return n + 1\nassert solve(1) == 2\nassert solve(41) == 42", False),
        ("python", "# reads input() later\nx = 1", False),
        ("python", "import sys\nfor line in sys.stdin:\n    pass", True),
        ("python", "data = open(0).read()", True),
        (
            "python",
            'import io, sys\nsys.stdin = io.StringIO("1 2\\n")\nassert sum(map(int, input().split())) == 3',
            False,
        ),
    ],
    ids=["passing-assert-self-test", "input-only-in-a-comment", "sys-stdin-loop", "open-fd-0", "stringio-self-test"],
)
def test_only_a_silent_run_of_a_program_that_reads_input_is_starved(language, code, starved):
    """A silent input-less run counts as starved only when its program reads input: a self-test that embeds its
    own and passed told the model so, and a mention of an input call inside a comment reads nothing."""
    env = _env(language=language)
    step = _turn(env, _reset(env), (env.test_tool_name, {"code": code}))
    assert step.trajectory.messages[-1].content == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}"
    assert _last_turn_flagged(step.trajectory) is starved and recovering_turn(step.trajectory) is starved
    assert env.rollout_metrics(step.trajectory)["episode/starved_test_runs"] == (1.0 if starved else 0.0)


def test_starved_turns_separate_a_starved_run_flagging_its_turn_from_an_unknown_call_alone():
    """A turn of unknown calls is flagged without a starved run; one whose only other call is unknown is flagged
    for its starved run, and only that one counts in ``episode/starved_turns``."""
    env = _env(language="python")
    ids = _reset(env)
    step = _turn(env, ids, ("test_tool", {}))
    assert _last_turn_flagged(step.trajectory)
    assert env.rollout_metrics(step.trajectory)["episode/starved_turns"] == 0.0
    step = _turn(env, ids, ("test_tool", {}), (env.test_tool_name, {"code": _READS_INPUT}))
    assert _last_turn_flagged(step.trajectory) and recovering_turn(step.trajectory)
    assert env.rollout_metrics(step.trajectory)["episode/starved_turns"] == 1.0


def test_a_direct_call_outside_an_episode_gets_the_plain_note():
    env = _env(language="python")
    assert env._run_test(_READS_INPUT) == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}"


def test_a_config_spelling_the_starved_run_refund_cap_is_refused():
    """No knob returns a silent input-less run to the budget: ``max_starved_run_refunds`` is refused at
    construction, as any option the environment does not take is, never dropped."""
    config = {"sandbox": LocalSubprocessSandbox(), "max_starved_run_refunds": 0}
    with pytest.raises(TypeError, match=r"unexpected environment option\(s\) \['max_starved_run_refunds'\]"):
        resolve_environment("codeforces", config)


def test_a_garbled_argument_name_is_refused_unspent_and_named():
    """A doubled ``parameter=`` prefix arrives as an argument the tool does not declare: the call is refused
    before it runs or spends a run, and the reply names the bad key and the real ones, where a dropped key
    would have run the program without its input."""
    env = _env(language="python")
    traj = _episode(env)
    reply = _scratchpad(env, traj, code="print(int(input()) + 1)", **{"parameter=stdin": "1\n"})
    assert reply == f"Error: {env.test_tool_name}: unknown argument 'parameter=stdin'; its arguments are code, stdin"
    assert env._test_calls(traj) == 0


@pytest.mark.parametrize("name", ["EMPTY_TURN_NUDGE", "LENGTH_CUTOFF_NUDGE", "LENGTH_CUTOFF_IN_CALL_NUDGE"])
def test_the_recovery_nudges_name_the_graded_channel_and_offer_no_final_answer(name):
    """A final text answer ends a code-contests episode ungraded, so the nudge the protocol sends after
    an unproductive turn must ask for the tool call, naming submit_solution."""
    nudge = getattr(CodeContestsEnvironment, name)
    assert nudge != getattr(NativeToolUseEnvironment, name)
    assert SUBMIT_TOOL in nudge
    assert "final answer" not in nudge.lower() and "answer" not in nudge.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
