#!/usr/bin/env python
"""
Tests for the pluggable code-execution sandbox backends.

Covers:
- LocalSubprocessSandbox: stdout capture, stdin plumbing, real imports, non-zero exit,
  wall-clock timeout, address-space limit, unsupported-language rejection, aux-file traversal guard.
  Multi-language (C/C++) compile-and-run and persistent sessions are covered in test_sandbox_multilang.py.
- RemoteSandbox: SandboxFusion response parsing (incl. a rejected compile as ``compile_failed`` and a
  compile time limit as a backend error), endpoint normalization, timeout/transport errors
  (via an injected fake session — no network).
- resolve_sandbox: env-var backend selection and the remote-url requirement.
- format_sandbox_repl_output / run_code_via_sandbox: REPL-style rendering.
- run_solution_against_tests: pass/fail/timeout grading on the local backend.

Run: python tests/cpu/environments/test_sandbox.py
"""

import os
import time

import pytest
import requests

from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import run_solution_against_tests
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.sandbox.base import LOCAL_NPROC_LIMIT, ExecutionGate, SandboxInfraError, SandboxResult
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox
from src.environments.sandbox.repl import format_sandbox_repl_output, run_code_via_sandbox
from src.environments.sandbox.resolve import resolve_sandbox
from tests.common.code_contests import RecordingSandboxSession

# LocalSubprocessSandbox


def test_local_output_that_is_not_utf8_is_captured_not_raised():
    """A program emitting bytes outside UTF-8 must come back as a result the grader can judge (a wrong
    answer), not as an exception that scores the test as an infra error."""
    sb = LocalSubprocessSandbox()
    res = sb.run("import sys\nsys.stdout.buffer.write(b'\\x80\\xffok\\n'); sys.stderr.buffer.write(b'\\x80')")
    assert res.returncode == 0 and res.error is None
    assert "ok" in res.stdout and "\ufffd" in res.stdout
    assert "\ufffd" in res.stderr


def test_local_runs_and_captures_stdout():
    sb = LocalSubprocessSandbox()
    res = sb.run("print(2 + 3)")
    assert res.ok, f"expected clean exit, got {res}"
    assert res.stdout.strip() == "5"
    assert res.returncode == 0


def test_local_feeds_stdin():
    sb = LocalSubprocessSandbox()
    res = sb.run("import sys\nfor line in sys.stdin:\n    print(line.strip()[::-1])", stdin="abc\n")
    assert res.stdout.strip() == "cba"


def test_local_allows_real_imports():
    """Unlike the in-process restricted REPL, the subprocess backend can import the stdlib."""
    sb = LocalSubprocessSandbox()
    res = sb.run("import math, json\nprint(json.dumps({'r': math.sqrt(16)}))")
    assert res.ok
    assert '"r": 4.0' in res.stdout


def test_local_nonzero_exit_captured():
    sb = LocalSubprocessSandbox()
    res = sb.run("raise ValueError('boom')")
    assert not res.ok
    assert res.returncode != 0
    assert "ValueError" in res.stderr and "boom" in res.stderr
    assert not res.timed_out


def test_local_times_out_on_infinite_loop():
    sb = LocalSubprocessSandbox()
    res = sb.run("while True:\n    pass", timeout=0.5)
    assert res.timed_out, f"expected timeout, got {res}"
    assert not res.ok


def _proc_state(pid: int) -> str | None:
    """Process state letter from /proc (None = no such process)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(") ", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def test_local_timeout_kills_forked_grandchildren():
    """On timeout the WHOLE process group must die: killing only the direct child leaves forked
    grandchildren (a fork bomb's population) running unbounded past every per-process limit."""
    sb = LocalSubprocessSandbox()
    code = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "print(pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    res = sb.run(code, timeout=1.0)
    assert res.timed_out
    grandchild = int(res.stdout.split()[0])
    # Dead or an unreaped zombie both mean the SIGKILL landed; 'S'leeping means it escaped.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _proc_state(grandchild) not in (None, "Z"):
        time.sleep(0.05)
    assert _proc_state(grandchild) in (None, "Z"), f"grandchild {grandchild} survived the group kill"


def test_a_child_that_escapes_the_group_does_not_hold_the_run_open():
    """A grandchild that setsid()s OUT of the process group survives the group kill and keeps its copy
    of the run's stdin and output files; the timed-out run returns without waiting for it."""
    code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"  # escape the process group: the killpg misses this one
        "    time.sleep(15)\n"
        "    os._exit(0)\n"
        "print('parent alive', flush=True)\n"
        "time.sleep(15)\n"
    )
    start = time.monotonic()
    res = LocalSubprocessSandbox().run(code, timeout=1.0)
    elapsed = time.monotonic() - start
    assert res.timed_out
    assert elapsed < 10.0, f"the run waited {elapsed:.1f}s for a child outside its group"


def test_limit_wrap_caps_process_count_for_run_step():
    """The run step's ulimit wrapper must carry -u (RLIMIT_NPROC) so a fork bomb's growth is capped
    until the group kill lands; the trusted compile step must NOT inherit that cap."""
    run_wrapped = LocalSubprocessSandbox._limit_wrap(["prog"], 2, 128, nproc=LOCAL_NPROC_LIMIT)
    assert f"ulimit -u {LOCAL_NPROC_LIMIT}" in run_wrapped[2]
    compile_wrapped = LocalSubprocessSandbox._limit_wrap(["cc"], 30, 2048)
    assert "ulimit -u" not in compile_wrapped[2]


def test_execution_gate_caps_concurrency():
    """The gate admits at most ``slots`` executions at once (a judge-style queue), so a wall-clock
    time limit is not distorted by oversubscription. One slot strictly serializes; three let runs
    overlap but never beyond the cap. Fails if the gate stops bounding concurrency."""
    import threading
    import time

    def peak_concurrency(gate: ExecutionGate, workers: int) -> int:
        live = 0
        peak = 0
        lock = threading.Lock()

        def work():
            nonlocal live, peak
            with gate.slot():
                with lock:
                    live += 1
                    peak = max(peak, live)
                time.sleep(0.03)
                with lock:
                    live -= 1

        threads = [threading.Thread(target=work) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return peak

    assert peak_concurrency(ExecutionGate(1), workers=5) == 1, "a 1-slot gate must serialize executions"
    peak3 = peak_concurrency(ExecutionGate(3), workers=6)
    assert 1 < peak3 <= 3, f"a 3-slot gate must allow >1 but never >3 concurrent, saw {peak3}"


def test_execution_slots_env_override_via_env_int():
    """HALO_SANDBOX_MAX_CONCURRENCY parses through src.env.env_int (single home for env parsing):
    a valid override wins (clamped to >=1), a malformed value warns inside env_int and falls back
    to the CPU-count default instead of raising mid-run."""
    from src.environments.sandbox.base import _resolve_execution_slots

    default_slots = max(1, os.cpu_count() or 1)
    saved = os.environ.get("HALO_SANDBOX_MAX_CONCURRENCY")
    try:
        os.environ["HALO_SANDBOX_MAX_CONCURRENCY"] = "3"
        assert _resolve_execution_slots() == 3
        os.environ["HALO_SANDBOX_MAX_CONCURRENCY"] = "0"
        assert _resolve_execution_slots() == 1  # clamped to at least one slot
        os.environ["HALO_SANDBOX_MAX_CONCURRENCY"] = "garbage"
        assert _resolve_execution_slots() == default_slots
        os.environ.pop("HALO_SANDBOX_MAX_CONCURRENCY")
        assert _resolve_execution_slots() == default_slots
    finally:
        if saved is None:
            os.environ.pop("HALO_SANDBOX_MAX_CONCURRENCY", None)
        else:
            os.environ["HALO_SANDBOX_MAX_CONCURRENCY"] = saved


def test_local_enforces_memory_limit():
    """An allocation past the address-space cap must fail the process, not the host."""
    sb = LocalSubprocessSandbox(memory_limit_mb=256)
    # ~1.5 GiB: RLIMIT_AS turns this into a MemoryError in the child.
    res = sb.run("x = bytearray(1500 * 1024 * 1024)\nprint(len(x))")
    assert not res.ok, "allocation beyond the memory cap should not succeed"
    assert not res.timed_out


def test_local_rejects_unsupported_language():
    sb = LocalSubprocessSandbox()
    res = sb.run("puts 1", language="ruby")
    assert res.error is not None
    assert "unsupported language" in res.error.lower()
    assert "ruby" in res.error.lower()


def test_local_writes_auxiliary_files():
    sb = LocalSubprocessSandbox()
    res = sb.run("print(open('data.txt').read().strip())", files={"data.txt": "hello-aux"})
    assert res.ok
    assert res.stdout.strip() == "hello-aux"


def test_local_rejects_unsafe_aux_path():
    sb = LocalSubprocessSandbox()
    res = sb.run("print(1)", files={"../escape.txt": "x"})
    assert res.error is not None
    assert "unsafe" in res.error.lower()


def test_local_isolated_mode_ignores_pythonpath():
    """`-I` means an injected PYTHONPATH module is NOT importable inside the sandbox."""
    sb = LocalSubprocessSandbox()
    res = sb.run("import definitely_not_a_real_module_xyz")
    assert not res.ok
    assert "ModuleNotFoundError" in res.stderr or "ImportError" in res.stderr


# RemoteSandbox (no network — injected fake session)


def test_remote_endpoint_normalization():
    sb_base = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession())
    sb_full = RemoteSandbox("http://sandbox:8080/run_code/", session=RecordingSandboxSession())
    assert sb_base.endpoint == "http://sandbox:8080/run_code"
    assert sb_full.endpoint == "http://sandbox:8080/run_code"


def test_remote_parses_success():
    payload = {
        "status": "Success",
        "run_result": {"status": "Finished", "stdout": "42\n", "stderr": "", "return_code": 0},
    }
    sess = RecordingSandboxSession(payload)
    sb = RemoteSandbox("http://sandbox:8080", session=sess)
    res = sb.run("print(42)", stdin="ignored", timeout=7)
    assert res.ok
    assert not res.compile_failed
    assert res.stdout.strip() == "42"
    # Request shape is SandboxFusion-compatible.
    (sent,) = sess.posts
    assert sent.payload["code"] == "print(42)"
    assert sent.payload["language"] == "python"
    assert sent.payload["run_timeout"] == 7
    assert sent.payload["stdin"] == "ignored"
    assert sent.url.endswith("/run_code")


def test_remote_coerces_string_return_code():
    """A numeric-string return_code is normalized to int so .ok / formatting read it correctly."""
    payload = {
        "status": "Success",
        "run_result": {"status": "Finished", "stdout": "ok\n", "stderr": "", "return_code": "0"},
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("print('ok')")
    assert res.returncode == 0
    assert res.ok


def test_remote_parses_program_error():
    payload = {
        "status": "Success",
        "run_result": {"status": "Finished", "stdout": "", "stderr": "Traceback ...", "return_code": 1},
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("raise SystemExit(1)")
    assert not res.ok
    assert res.returncode == 1
    assert "Traceback" in res.stderr


def test_remote_parses_timeout():
    payload = {
        "status": "Failed",
        "message": "time limit",
        "run_result": {"status": "TimeLimitExceeded", "stdout": "", "stderr": "", "return_code": None},
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("while True: pass")
    assert res.timed_out
    assert not res.ok


def test_remote_compile_failure_is_program_verdict_not_infra_error():
    """A compile the service's compiler rejected (a finished step, non-zero exit) is the submission's
    fault: ``compile_failed`` with the diagnostics, ``error`` unset — an ``error`` here would invalidate
    the whole graded episode."""
    payload = {
        "status": "Failed",
        "message": "",
        "compile_result": {
            "status": "Finished",
            "return_code": 1,
            "stdout": "",
            "stderr": "main.cpp:1:12: error: boom",
        },
        "run_result": None,
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("int main(){ boom }", language="cpp")
    assert res.compile_failed
    assert res.error is None
    assert not res.ok
    assert not res.timed_out
    assert "error: boom" in res.stderr
    assert res.returncode == 1


@pytest.mark.parametrize(
    "compile_result",
    (
        {"status": "Error", "stdout": "", "stderr": "g++: not found"},
        {"status": "Finished", "return_code": 127, "stdout": "", "stderr": "sh: g++: not found"},
        {"status": "", "return_code": None, "stdout": "", "stderr": ""},
    ),
    ids=("error-status", "exit-127", "no-status-no-exit"),
)
def test_remote_incomplete_compile_step_is_backend_error(compile_result):
    """A compiler step that did not run to completion — or whose compiler is absent (exit 127) — is
    the service's fault, never a verdict on the source: ``error`` set, ``compile_failed`` unset."""
    payload = {"status": "Failed", "compile_result": compile_result}
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("int main(){}", language="cpp")
    assert res.error is not None
    assert not res.compile_failed
    assert not res.ok


def test_remote_success_without_a_run_result_is_a_backend_error():
    """A ``Success`` body with no run block carries no program output; reading it as an empty clean run
    would pass a test whose expected output is empty."""
    for payload in (
        {"status": "Success"},
        {"status": "Success", "run_result": None},
        {"status": "Success", "run_result": "x"},
    ):
        sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
        res = sb.run("print(1)")
        assert res.error is not None and not res.ok, payload


def test_remote_compile_time_limit_is_a_compile_verdict():
    """A compile that hit the service's compile time limit is the source's verdict (an ``#include``
    bomb), like the local backend's compile timeout — never ``error``, which would let a program void
    its own episode, and not a run timeout."""
    payload = {
        "status": "Failed",
        "message": "",
        "compile_result": {"status": "TimeLimitExceeded", "return_code": None, "stdout": "", "stderr": ""},
        "run_result": None,
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("int main(){}", language="cpp")
    assert res.error is None
    assert res.compile_failed and not res.timed_out
    assert not res.ok
    assert format_sandbox_repl_output(res, timeout=5).startswith("Error:"), "the REPL shows it as a failed build"


def test_remote_clean_compile_step_reads_run_result():
    """A compiled-language success carries a finished, zero-exit ``compile_result``: the verdict must
    come from ``run_result`` exactly as for an interpreted run."""
    payload = {
        "status": "Success",
        "compile_result": {"status": "Finished", "return_code": 0, "stdout": "", "stderr": ""},
        "run_result": {"status": "Finished", "stdout": "42\n", "stderr": "", "return_code": 3},
    }
    sb = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload))
    res = sb.run("int main(){ return 3; }", language="cpp")
    assert not res.compile_failed
    assert res.error is None
    assert res.returncode == 3
    assert res.stdout.strip() == "42"


def test_remote_handles_transport_error():
    sess = RecordingSandboxSession(exc=requests.ConnectionError("refused"))
    sb = RemoteSandbox("http://sandbox:8080", session=sess)
    res = sb.run("print(1)")
    assert res.error is not None
    assert "remote sandbox error" in res.error


def test_remote_client_timeout_is_an_infra_error_not_the_programs_tle():
    """The service enforces ``run_timeout`` itself and reports it in ``run_result.status``, so the
    client deadline fires only when the service does not answer. Booked as ``timed_out`` it would
    grade as TIME LIMIT EXCEEDED — a wrong program, outside the infra-outage invalidation — and the
    REPL would render a timeout string instead of raising ``SandboxInfraError``."""
    sess = RecordingSandboxSession(exc=requests.Timeout("slow"))
    sb = RemoteSandbox("http://sandbox:8080", session=sess)
    res = sb.run("print(1)")
    assert res.timed_out is False
    assert res.error is not None and "timed out" in res.error
    with pytest.raises(SandboxInfraError):
        format_sandbox_repl_output(res, timeout=5.0)


# resolve_sandbox


def test_resolve_defaults_to_local():
    saved = os.environ.pop("HALO_SANDBOX_BACKEND", None)
    try:
        sb = resolve_sandbox()
        assert isinstance(sb, LocalSubprocessSandbox)
    finally:
        if saved is not None:
            os.environ["HALO_SANDBOX_BACKEND"] = saved


def test_resolve_remote_requires_url():
    saved_backend = os.environ.pop("HALO_SANDBOX_BACKEND", None)
    saved_url = os.environ.pop("HALO_SANDBOX_URL", None)
    try:
        raised = False
        try:
            resolve_sandbox(backend="remote")
        except ValueError:
            raised = True
        assert raised, "remote backend without a url must raise"
    finally:
        if saved_backend is not None:
            os.environ["HALO_SANDBOX_BACKEND"] = saved_backend
        if saved_url is not None:
            os.environ["HALO_SANDBOX_URL"] = saved_url


def test_resolve_remote_from_env():
    saved_backend = os.environ.get("HALO_SANDBOX_BACKEND")
    saved_url = os.environ.get("HALO_SANDBOX_URL")
    os.environ["HALO_SANDBOX_BACKEND"] = "remote"
    os.environ["HALO_SANDBOX_URL"] = "http://sandbox.internal:8080"
    try:
        sb = resolve_sandbox()
        assert isinstance(sb, RemoteSandbox)
        assert sb.endpoint == "http://sandbox.internal:8080/run_code"
    finally:
        for key, val in (("HALO_SANDBOX_BACKEND", saved_backend), ("HALO_SANDBOX_URL", saved_url)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_resolve_rejects_unknown_backend():
    raised = False
    try:
        resolve_sandbox(backend="banana")
    except ValueError:
        raised = True
    assert raised


def test_resolve_explicit_backend_overrides_env():
    saved = os.environ.get("HALO_SANDBOX_BACKEND")
    os.environ["HALO_SANDBOX_BACKEND"] = "remote"  # would require a url if it won
    try:
        sb = resolve_sandbox(backend="local")
        assert isinstance(sb, LocalSubprocessSandbox)
    finally:
        if saved is None:
            os.environ.pop("HALO_SANDBOX_BACKEND", None)
        else:
            os.environ["HALO_SANDBOX_BACKEND"] = saved


# Environment-level sandbox config (sandbox_backend / sandbox_url via environment_kwargs)


def test_env_sandbox_backend_config():
    for env in (
        CodeContestsEnvironment(sandbox_backend="local"),
        SweEnvironment(sandbox_backend="local"),
    ):
        assert isinstance(env.sandbox, LocalSubprocessSandbox)


def test_env_sandbox_remote_config_requires_url():
    raised = False
    try:
        CodeContestsEnvironment(sandbox_backend="remote")  # no url -> must raise
    except ValueError:
        raised = True
    assert raised, "remote sandbox_backend without a url must raise at env construction"


def test_env_sandbox_config_overrides_env():
    # Config beats the env var: remote-without-url would raise if the env var won.
    saved = os.environ.get("HALO_SANDBOX_BACKEND")
    os.environ["HALO_SANDBOX_BACKEND"] = "remote"
    try:
        env = CodeContestsEnvironment(sandbox_backend="local")
        assert isinstance(env.sandbox, LocalSubprocessSandbox)
    finally:
        if saved is None:
            os.environ.pop("HALO_SANDBOX_BACKEND", None)
        else:
            os.environ["HALO_SANDBOX_BACKEND"] = saved


# REPL formatting


def test_format_repl_stdout():
    assert format_sandbox_repl_output(SandboxResult(stdout="hi\n", returncode=0), 5) == "hi"


def test_format_repl_no_output():
    out = format_sandbox_repl_output(SandboxResult(stdout="", returncode=0), 5)
    assert "executed successfully" in out.lower()


def test_format_repl_timeout():
    out = format_sandbox_repl_output(SandboxResult(timed_out=True), 0.5)
    assert "Error" in out and "timeout" in out.lower()


def test_format_repl_error_tail():
    res = SandboxResult(stdout="", stderr="Traceback...\nValueError: boom", returncode=1)
    out = format_sandbox_repl_output(res, 5)
    assert out == "Error: ValueError: boom"


def test_format_repl_raises_on_backend_error_with_partial_output():
    """A backend/transport error must RAISE, even if a partial response carried stdout.

    Returning it as a string made the tool protocol score an infrastructure outage as a successful
    call (paying ``tool_success_reward`` for a run that never happened).
    """
    res = SandboxResult(stdout="partial", stderr="", returncode=None, error="remote sandbox error: 503")
    try:
        format_sandbox_repl_output(res, 5)
    except SandboxInfraError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("a backend failure must raise SandboxInfraError, not return a string")


def test_format_repl_program_failures_stay_strings():
    """A program-level failure is a verdict on the code, not an outage: it must NOT raise."""
    assert "timeout" in format_sandbox_repl_output(SandboxResult(timed_out=True), 0.5).lower()
    assert format_sandbox_repl_output(SandboxResult(stderr="ValueError: boom", returncode=1), 5).startswith("Error")


def test_run_code_via_sandbox_local():
    sb = LocalSubprocessSandbox()
    assert run_code_via_sandbox("print(6 * 7)", sb) == "42"
    assert run_code_via_sandbox("import math\nprint(math.gcd(12, 18))", sb) == "6"


# run_solution_against_tests on the local backend


_ECHO_DOUBLE = "n = int(input())\nprint(n * 2)"


def test_grading_all_pass():
    cases = [{"input": "3", "output": "6"}, {"input": "10", "output": "20"}]
    passed, total, details, *_ = run_solution_against_tests(_ECHO_DOUBLE, cases, sandbox=LocalSubprocessSandbox())
    assert (passed, total) == (2, 2)
    assert "Passed 2/2" in details


def test_grading_partial_fail():
    cases = [{"input": "3", "output": "6"}, {"input": "10", "output": "999"}]
    passed, total, *_ = run_solution_against_tests(_ECHO_DOUBLE, cases, sandbox=LocalSubprocessSandbox())
    assert (passed, total) == (1, 2)


def test_grading_timeout_counts_as_fail():
    cases = [{"input": "", "output": "done"}]
    code = "while True:\n    pass"
    passed, total, details, *_ = run_solution_against_tests(
        code, cases, timeout_per_test=1, sandbox=LocalSubprocessSandbox()
    )
    assert passed == 0
    assert "TIME LIMIT EXCEEDED" in details


def test_grading_no_test_cases():
    passed, total, details, *_ = run_solution_against_tests("print(1)", [], sandbox=LocalSubprocessSandbox())
    assert (passed, total) == (0, 0)
    assert "No test cases" in details


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
