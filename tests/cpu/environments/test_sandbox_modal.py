"""The Modal sandbox backend against a fake Modal client whose exec runs the real in-sandbox runner as a local
subprocess, so limits, verdicts and session state go through the production runner. The live test (one tiny
program in a real Modal sandbox) runs only with ``HALO_MODAL_LIVE=1``, the Modal SDK and Modal credentials."""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import types

import pytest

from halo.cli import ROLLOUT_BACKEND_ENV
from halo.environments.registry import build_environment
from halo.environments.sandbox import modal_sandbox as ms
from halo.environments.sandbox.factory import build_sandbox
from halo.environments.sandbox.languages import require_language
from halo.environments.sandbox.local import (
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    RLIMIT_CPU_SLACK_SECONDS,
    SANDBOX_DEFAULT_MEMORY_MB,
    LocalSubprocessSandbox,
    build_program_env,
)
from halo.environments.sandbox.modal_sandbox import ModalSandbox
from halo.environments.sandbox.results import SandboxInfraError
from halo.environments.tasks.coding.grading import run_solution_against_tests

_MIB = 1024 * 1024


class _Stdin:
    def __init__(self):
        self.data = ""

    def write(self, text):
        self.data += text

    def write_eof(self):
        pass

    def drain(self):
        pass


class _Stream:
    def __init__(self, proc, name):
        self._proc, self._name = proc, name

    def read(self):
        self._proc.finish()
        return getattr(self._proc, self._name)


class _FakeProcess:
    """Runs the runner locally; an expired exec ``timeout`` returns -1 as Modal's ``wait()`` does."""

    def __init__(self, modal, argv, timeout):
        self._modal, self._argv, self._timeout = modal, argv, timeout
        self._done = False
        self.stdin = _Stdin()
        self.stdout = _Stream(self, "out")
        self.stderr = _Stream(self, "err")

    def finish(self):
        if self._done:
            return
        self._done = True
        if self._modal.runner_returncode is not None:
            self.out, self.err, self.code = "", "Traceback: boom", self._modal.runner_returncode
            return
        request = json.loads(self.stdin.data)
        self._modal.requests.append(request)
        argv = [sys.executable, *self._argv[1:]]
        with self._modal.lock:
            self._modal.in_flight += 1
            self._modal.peak_in_flight = max(self._modal.peak_in_flight, self._modal.in_flight)
        try:
            proc = subprocess.run(argv, input=self.stdin.data, capture_output=True, text=True, timeout=self._timeout)
            self.out, self.err, self.code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            self.out, self.err, self.code = "", "", -1
        finally:
            with self._modal.lock:
                self._modal.in_flight -= 1

    def wait(self):
        self.finish()
        return self.code


class _FakeSandbox:
    def __init__(self, modal, kwargs):
        self.modal, self.kwargs = modal, kwargs
        self.exec_calls = 0
        self.terminated = False

    def exec(self, *argv, timeout=None):
        assert argv[:3] == ("python3", "-I", "-c") and argv[3] == ms._RUNNER_SOURCE
        self.exec_calls += 1
        if self.terminated:
            raise RuntimeError("sandbox terminated")
        if self.modal.fail_next_execs:
            self.modal.fail_next_execs -= 1
            raise ConnectionError("transport down")
        return _FakeProcess(self.modal, argv, timeout)

    def terminate(self):
        self.terminated = True


def _fake_modal():
    modal = types.SimpleNamespace(
        sandboxes=[],
        lookups=[],
        registry_images=[],
        requests=[],
        fail_next_execs=0,
        runner_returncode=None,
        lock=threading.Lock(),
        in_flight=0,
        peak_in_flight=0,
    )

    class Image:
        def __init__(self, label):
            self.label = label

        @staticmethod
        def debian_slim(python_version=None):
            return Image(f"debian_slim:{python_version}")

        @staticmethod
        def from_registry(tag):
            modal.registry_images.append(tag)
            return Image(f"registry:{tag}")

        def apt_install(self, *pkgs):
            return Image(f"{self.label}+{','.join(pkgs)}")

    def lookup(name, create_if_missing=False):
        modal.lookups.append((name, create_if_missing))
        return types.SimpleNamespace(name=name)

    def create(**kwargs):
        modal.sandboxes.append(_FakeSandbox(modal, kwargs))
        return modal.sandboxes[-1]

    modal.Image = Image
    modal.App = types.SimpleNamespace(lookup=lookup)
    modal.Sandbox = types.SimpleNamespace(create=create)
    return modal


@pytest.fixture
def modal(monkeypatch, tmp_path):
    fake = _fake_modal()
    monkeypatch.setattr(ms, "_require_modal", lambda: fake)
    monkeypatch.setattr(ms, "_WORKSPACE", str(tmp_path))
    return fake


@pytest.fixture
def executor(modal):
    sandbox = ModalSandbox()
    yield sandbox
    sandbox.close()


def _exec_count(modal):
    return sum(sb.exec_calls for sb in modal.sandboxes)


# ---------------------------------------------------------------------------
# Results and limits
# ---------------------------------------------------------------------------


def test_run_maps_stdout_stdin_and_exit(executor):
    res = executor.run("import sys\nprint(sys.stdin.read().strip()[::-1])", stdin="abc\n")
    assert res.ok and res.stdout == "cba\n" and res.returncode == 0

    res = executor.run("import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)")
    assert (res.returncode, res.stderr, res.error) == (3, "boom\n", None) and not res.ok


def test_program_timeout_is_its_own_verdict(executor):
    start = time.monotonic()
    res = executor.run("import time\ntime.sleep(30)", timeout=0.5)
    assert res.timed_out and res.returncode is None and res.error is None
    assert time.monotonic() - start < 10


def test_exec_deadline_expiry_is_infra_not_a_timeout(modal, executor, monkeypatch):
    # Slack shrunk so the exec deadline (1 s) passes before the runner's own 10 s limit.
    monkeypatch.setattr(ms, "_EXEC_OVERHEAD_SECONDS", -9.0)
    res = executor.run("import time\ntime.sleep(30)", timeout=10)
    assert res.error and "deadline" in res.error and not res.timed_out


def test_runner_crash_is_infra(modal, executor):
    modal.runner_returncode = 1
    res = executor.run("print(1)")
    assert res.error and "runner exited 1" in res.error and res.returncode is None


def test_request_carries_the_local_backends_limits_and_env(modal, executor):
    executor.run("print(1)", timeout=4)
    req = modal.requests[0]
    assert req["limits"] == {
        "RLIMIT_CPU": math.ceil(4 + RLIMIT_CPU_SLACK_SECONDS),
        "RLIMIT_FSIZE": LOCAL_FSIZE_LIMIT,
        "RLIMIT_CORE": 0,
        "RLIMIT_AS": SANDBOX_DEFAULT_MEMORY_MB * _MIB,
        "RLIMIT_NPROC": LOCAL_NPROC_LIMIT,
    }
    assert req["env"] == build_program_env(req["workdir"]) and "PATH" not in req["env"]
    cpp = executor.run_request(require_language("cpp"), timeout=4, build=True)
    assert cpp["limits"]["RLIMIT_STACK"] == SANDBOX_DEFAULT_MEMORY_MB * _MIB


def test_runner_enforces_the_requested_memory_limit(modal):
    executor = ModalSandbox(memory_limit_mb=256, compile_memory_limit_mb=256)
    res = executor.run("x = bytearray(1024 * 1024 * 1024)\nprint('allocated')")
    assert res.returncode not in (0, None) and "MemoryError" in res.stderr
    executor.close()


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_compile_verdicts_and_build_reuse(modal, executor):
    bad = executor.run("int main( {", language="c")
    assert bad.compile_failed and bad.error is None and bad.stderr

    good = '#include <stdio.h>\nint main(){int x; scanf("%d",&x); printf("%d\\n", 2*x); return 0;}'
    with executor.open_session() as session:
        assert session.run(good, stdin="21", language="c").stdout == "42\n"
        assert session.run(good, stdin="5", language="c").stdout == "10\n"
    builds = [req for req in modal.requests if req["op"] == "run" and req["compile_argv"]]
    assert [req["build"] for req in builds[-2:]] == [True, False]  # the second run reuses the binary


def test_missing_compiler_is_infra_not_a_verdict():
    step = {"timed_out": False, "returncode": 127, "stdout": "", "stderr": "gcc: not found"}
    res = ms._compile_verdict(step, require_language("c"), 30.0)
    assert res.error == "compiler not found: 'gcc'" and not res.compile_failed


def test_reset_to_prepared_drops_what_the_run_wrote(executor):
    with executor.open_session() as session:
        session.write_file("lib.py", "X = 1\n")
        session.run("import lib\nopen('out.txt', 'w').write(str(lib.X))")
        assert "out.txt" in session.list_files()
        session.reset_to_prepared()
        res = session.run("import os\nprint(sorted(os.listdir('.')))")
        assert res.stdout.strip() == str(sorted(["lib.py", "main.py"]))


def test_unknown_language_and_unsafe_paths_never_reach_modal(modal, executor):
    assert "unsupported language" in executor.run("x", language="cobol").error
    assert "unsafe auxiliary file path" in executor.run("x", files={"../evil": "1"}).error
    with pytest.raises(ValueError):
        executor.open_session().write_file("/etc/passwd", "x")
    assert modal.sandboxes == []


# ---------------------------------------------------------------------------
# One sandbox, sessions as directories, loss and cleanup
# ---------------------------------------------------------------------------


def test_runs_and_sessions_share_one_sandbox_and_closed_dirs_are_removed(modal, executor, tmp_path):
    with executor.open_session() as first:
        first.write_file("a.txt", "1")
        workdir = first.workdir
        second = executor.open_session()
        assert second.read_file("a.txt") is None  # sessions do not see each other's directory
    assert os.path.isdir(workdir)  # closing makes no remote call
    executor.run("print(1)")
    assert not os.path.isdir(workdir)  # the next exec removed it
    assert len(modal.sandboxes) == 1


def test_lost_sandbox_is_dropped_and_the_run_not_retried(modal, executor):
    executor.run("print(0)")
    runs_before = len(modal.requests)
    modal.fail_next_execs = 1
    res = executor.run("print(5)")
    assert res.error and "ConnectionError" in res.error and not res.ok
    assert len(modal.requests) == runs_before  # the program was not run a second time
    assert modal.sandboxes[0].terminated
    assert executor.run("print(6)").stdout == "6\n" and len(modal.sandboxes) == 2


def test_session_file_op_loss_raises_infra(modal, executor):
    session = executor.open_session()
    modal.fail_next_execs = 1
    with pytest.raises(SandboxInfraError):
        session.write_file("a.txt", "1")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_compiled_session_rebuilds_on_a_replaced_sandbox(modal, executor):
    code = '#include <stdio.h>\nint main(){puts("hi");return 0;}'
    session = executor.open_session()
    assert session.run(code, language="c").stdout == "hi\n"
    modal.fail_next_execs = 1
    assert session.run(code, language="c").error
    res = session.run(code, language="c")
    assert res.ok and res.stdout == "hi\n" and len(modal.sandboxes) == 2


def test_concurrent_runs_are_capped_at_max_concurrency(modal):
    executor = ModalSandbox(cpu=2.0, memory_mb=4096)
    threads = [threading.Thread(target=executor.run, args=("import time\ntime.sleep(0.4)",)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert executor.max_concurrency == 2 and modal.peak_in_flight == 2
    executor.close()


def test_sandbox_reserves_cpu_memory_and_reclaims_itself(modal):
    executor = ModalSandbox(cpu=8.0, memory_mb=16384, sandbox_timeout=90)
    executor.run("print(1)")
    kwargs = modal.sandboxes[0].kwargs
    assert (kwargs["cpu"], kwargs["memory"], kwargs["timeout"]) == (8.0, 16384, 90)
    assert kwargs["idle_timeout"] == ms._IDLE_TIMEOUT_SECONDS and kwargs["block_network"] is True
    executor.close()


def test_memory_must_hold_every_concurrent_run(modal):
    with pytest.raises(ValueError, match="max_concurrency=4"):
        ModalSandbox(cpu=4.0, memory_mb=4096)


def test_close_and_atexit_terminate_the_sandbox(modal):
    executor = ModalSandbox()
    executor.run("print(1)")
    executor.close()
    assert modal.sandboxes[0].terminated and executor not in ms._live_executors
    other = ModalSandbox()
    other.run("print(1)")
    ms._close_live_executors()
    assert modal.sandboxes[1].terminated


def test_missing_modal_sdk_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(ImportError, match=r"pip install 'halo\[modal\]'"):
        ModalSandbox()


# ---------------------------------------------------------------------------
# Selection: factory, YAML, launch flag
# ---------------------------------------------------------------------------


def test_factory_builds_modal_with_its_options(modal):
    executor = build_sandbox(backend="modal", app_name="advprop-test", image="python:3.12-slim")
    assert isinstance(executor, ModalSandbox) and executor.isolated
    assert modal.lookups == [("advprop-test", True)] and modal.registry_images == ["python:3.12-slim"]
    executor.close()


def test_env_var_selects_modal(modal, monkeypatch):
    monkeypatch.setenv("HALO_SANDBOX_BACKEND", "modal")
    executor = build_sandbox()
    assert isinstance(executor, ModalSandbox)
    executor.close()


@pytest.mark.parametrize("env_type", ["code_contests", "codeforces", "swe"])
def test_yaml_sandbox_options_reach_modal(modal, env_type):
    options = {"app_name": "advprop-yaml", "cpu": 2.0, "memory_mb": 4096}
    env = build_environment(env_type, {"sandbox_backend": "modal", "sandbox_options": options})
    assert isinstance(env.sandbox, ModalSandbox) and modal.lookups == [("advprop-yaml", True)]
    assert (env.sandbox.cpu, env.sandbox.memory_mb, env.sandbox.max_concurrency) == (2.0, 4096, 2)
    env.sandbox.close()


def test_yaml_backend_beats_the_env_var(modal, monkeypatch):
    monkeypatch.setenv("HALO_SANDBOX_BACKEND", "modal")
    env = build_environment("code_contests", {"sandbox_backend": "local"})
    assert isinstance(env.sandbox, LocalSubprocessSandbox) and modal.lookups == []


@pytest.fixture
def under_flag(modal, monkeypatch):
    """Exactly the environment ``halo launch --halo-rollout-backend modal`` sets."""
    for key, value in ROLLOUT_BACKEND_ENV["modal"].items():
        monkeypatch.setenv(key, value)
    return modal


def test_flag_runs_contest_test_tool_and_grading_on_modal(under_flag):
    env = build_environment("code_contests", {})
    assert isinstance(env.sandbox, ModalSandbox)
    assert "42" in env._run_test("print(6 * 7)")
    tests = [{"input": "3\n", "output": "6\n"}, {"input": "5\n", "output": "10\n"}]
    grade = run_solution_against_tests("print(int(input()) * 2)", tests, sandbox=env.sandbox)
    assert (grade.passed, grade.total) == (2, 2)
    assert _exec_count(under_flag) == 3
    env.sandbox.close()


def test_flag_runs_the_python_repl_tool_on_modal(under_flag):
    env = build_environment("native_coding", {})
    # The in-process REPL refuses imports; only a real interpreter prints the separator.
    assert env.registry.get("python_repl").execute(code="import os\nprint(os.sep)") == "/"
    assert _exec_count(under_flag) == 1


def test_without_the_flag_the_python_repl_stays_in_process(modal, monkeypatch):
    for key in ROLLOUT_BACKEND_ENV["modal"]:
        monkeypatch.delenv(key, raising=False)
    env = build_environment("native_coding", {})
    env.registry.get("python_repl").execute(code="print(1)")
    assert modal.sandboxes == []


# ---------------------------------------------------------------------------
# Live (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("HALO_MODAL_LIVE") != "1", reason="set HALO_MODAL_LIVE=1 to run against Modal")
def test_live_modal_sandbox_runs_programs():
    executor = ModalSandbox(
        app_name=os.environ.get("HALO_MODAL_APP", "advprop-halo-sandbox-test"),
        cpu=1.0,
        memory_mb=2048,
        sandbox_timeout=600,
    )
    try:
        start = time.monotonic()
        first = executor.run("import sys\nprint(int(sys.stdin.read()) * 2)", stdin="21")
        cold = time.monotonic() - start
        assert first.ok and first.stdout == "42\n", first
        warm = []
        for i in range(5):
            start = time.monotonic()
            res = executor.run(f"print({i})")
            warm.append(time.monotonic() - start)
            assert res.ok and res.stdout == f"{i}\n", res
        res = executor.run('#include <stdio.h>\nint main(){puts("c ok");return 0;}', language="c")
        assert res.ok and res.stdout == "c ok\n", res
        assert executor.run("import time\ntime.sleep(10)", timeout=1).timed_out
        print(f"\nmodal live: first run (sandbox start included) {cold:.2f}s; warm exec median {sorted(warm)[2]:.3f}s")
    finally:
        executor.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
