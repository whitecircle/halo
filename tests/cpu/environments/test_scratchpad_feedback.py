#!/usr/bin/env python
"""CPU tests: what the code-contests scratchpad tells the model about a program that failed.

The scratchpad is where the policy debugs, so its reply has to carry the diagnosis a judge would:
a compile error as the compiler's first diagnostics (the last stderr line of g++ is a caret gutter),
a crash as its signal and the traceback's tail, ahead of any stdout (the protocol cuts a long
observation from the end), and a program run under the same limits its graded tests get — the
problem's time limit, and a stack as large as the memory limit. Every program here really compiles
and runs on the local backend.

Run: python tests/cpu/environments/test_scratchpad_feedback.py  (or pytest)
"""

import json
import resource
import shutil
import time

import pytest

from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.code_contests import (
    NO_STDIN_NOTE,
    SCRATCHPAD_TIME_LIMIT_NOTE,
    STARVED_RUN_NOTE,
    SUBMIT_TOOL,
    CodeContestsEnvironment,
)
from src.environments.envs.tasks.coding.grading import run_solution_against_tests
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.repl import run_code_via_sandbox
from src.environments.tools.definitions import NativeToolCall

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


def _episode(env, time_limit=None):
    answer = {"tests": [{"input": "1\n", "output": "2\n"}], "time_limit": time_limit}
    ids, _ = env.reset(["solve it"], [{"answer": json.dumps(answer)}])
    return env.get_trajectories(ids)[0]


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
    started = time.monotonic()
    reply = _scratchpad(env, _episode(env, time_limit=1.0), code=_SPIN_SECONDS % 5, stdin="1\n")
    assert reply.startswith(f"Error: execution exceeded 2s timeout {SCRATCHPAD_TIME_LIMIT_NOTE}"), reply
    assert time.monotonic() - started < 4.5
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


def test_every_scratchpad_reply_states_the_runs_left():
    env = _env(language="python")
    traj = _episode(env)
    replies = [_scratchpad(env, traj, code=code, stdin="1\n") for code in ("print(1)", "raise SystemExit(3)")]
    assert replies[0] == "1\n(Scratchpad runs left: 5 of 6.)"
    assert replies[1].endswith("\n(Scratchpad runs left: 4 of 6.)")
    for _ in range(4):
        last = _scratchpad(env, traj, code="print(2)", stdin="1\n")
    assert last.endswith("(Scratchpad runs left: 0 of 6.)")
    assert "runs left" not in env._run_test("print(3)"), "a direct call outside an episode carries no budget"


def test_a_run_given_no_input_that_prints_nothing_spends_no_run():
    """A solution run with no stdin reads nothing and prints nothing: the run is returned to the budget and the
    reply says so. A starved run that crashes returns a traceback, and a run that prints (or is quiet on real
    input) told the model something, so each of those spends its run."""
    env = _env(language="python")
    traj = _episode(env)
    reads_input = "import sys\ndata = sys.stdin.read().split()\nif data:\n    print(int(data[0]) + 1)"
    silent = _scratchpad(env, traj, code=reads_input)
    assert STARVED_RUN_NOTE in silent and silent.endswith("(Scratchpad runs left: 6 of 6.)"), silent
    assert env._test_calls(traj) == 0
    crash = _scratchpad(env, traj, code="print(int(input()) + 1)")
    assert NO_STDIN_NOTE in crash and crash.endswith("(Scratchpad runs left: 5 of 6.)"), crash
    printed = _scratchpad(env, traj, code="print(7)")
    assert STARVED_RUN_NOTE not in printed and printed.endswith("(Scratchpad runs left: 4 of 6.)"), printed
    quiet_on_input = _scratchpad(env, traj, code=reads_input.replace("print", "len"), stdin="1\n")
    assert quiet_on_input.endswith("(Scratchpad runs left: 3 of 6.)"), quiet_on_input
    assert env.rollout_metrics(traj)["episode/starved_test_runs"] == 1.0


def test_only_the_first_silent_input_less_run_of_an_episode_is_returned():
    """Returning every such run would make one that reads nothing free to repeat: past
    ``max_starved_run_refunds`` the run spends its turn of the budget and gets the plain no-stdin note."""
    env = _env(language="python")
    traj = _episode(env)
    reads_input = "import sys\ndata = sys.stdin.read().split()\nif data:\n    print(int(data[0]) + 1)"
    first, second = _scratchpad(env, traj, code=reads_input), _scratchpad(env, traj, code=reads_input)
    assert STARVED_RUN_NOTE in first and first.endswith("(Scratchpad runs left: 6 of 6.)"), first
    assert NO_STDIN_NOTE in second and second.endswith("(Scratchpad runs left: 5 of 6.)"), second
    assert env.rollout_metrics(traj)["episode/starved_test_runs"] == 1.0
    never = _env(language="python", max_starved_run_refunds=0)
    reply = _scratchpad(never, _episode(never), code=reads_input)
    assert NO_STDIN_NOTE in reply and reply.endswith("(Scratchpad runs left: 5 of 6.)"), reply


@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_a_refund_cap_that_is_not_a_count_is_refused(cap):
    with pytest.raises(ValueError, match="max_starved_run_refunds"):
        _env(language="python", max_starved_run_refunds=cap)


def test_a_garbled_argument_name_is_refused_unspent_and_named():
    """A doubled ``parameter=`` prefix arrives as an argument the tool does not declare: the call is refused
    before it runs or spends a run, and the reply names the bad key and the real ones, where a dropped key
    would have run the program without its input."""
    env = _env(language="python")
    traj = _episode(env)
    reply = _scratchpad(env, traj, code="print(int(input()) + 1)", **{"parameter=stdin": "1\n"})
    assert reply.startswith(
        f"Error: {env.test_tool_name}: unknown argument 'parameter=stdin'; its arguments are code"
    ), reply
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
