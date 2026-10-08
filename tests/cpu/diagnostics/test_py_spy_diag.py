"""``py_spy_diag.py`` against a stand-in ``py-spy`` on ``PATH``: a pid it cannot attach to fails the
run by name, and ``record`` samples every rank over the same window.

An attach that fails (no ptrace permission, another pid namespace, an exited rank) must not leave a
partial artifact directory behind a success line and exit 0. The ranks are sampled together: one
after another, the last of eight would be sampled seven windows after the first. And a sweep that
stops early, on its timeout or an interrupt, must leave no py-spy attached to a training rank.

Run: python tests/cpu/diagnostics/test_py_spy_diag.py
"""

import json
import os
import subprocess
import sys

import pytest

from src.diagnostics import debugging
from tests.common.utils import REPO_ROOT

_PIDS = (101, 102, 103)

# Records its own wall-clock window per pid, so the test can tell concurrent attaches from sequential ones.
_FAKE_PY_SPY = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
pid = args[args.index("--pid") + 1]
start = time.time()
if pid in os.environ["FAKE_PY_SPY_FAIL"].split(","):
    print("Error: Permission Denied: Try running again with elevated permissions", file=sys.stderr)
    sys.exit(1)
if args[0] == "record":
    time.sleep(float(args[args.index("--duration") + 1]))
    with open(args[args.index("--output") + 1], "w") as svg:
        svg.write("<svg/>")
else:
    print(f"Thread {{pid}} (idle): MainThread")
with open(os.path.join(os.environ["FAKE_PY_SPY_LOG"], pid + ".json"), "w") as window:
    json.dump([start, time.time()], window)
"""


def _run(tmp_path, command: str, *, failing: tuple[int, ...] = ()) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir, log_dir, out_dir = (tmp_path / name for name in ("bin", "log", "out"))
    bin_dir.mkdir()
    log_dir.mkdir()
    fake = bin_dir / "py-spy"
    fake.write_text(_FAKE_PY_SPY.format(python=sys.executable))
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PY_SPY_LOG": str(log_dir),
        "FAKE_PY_SPY_FAIL": ",".join(map(str, failing)),
    }
    argv = [sys.executable, "scripts/profiling/py_spy_diag.py", command, "--output-dir", str(out_dir)]
    argv += [arg for pid in _PIDS for arg in ("--pid", str(pid))]
    if command == "record":
        argv += ["--duration", "1"]
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=120)
    (artifacts,) = out_dir.iterdir()
    return proc, str(artifacts)


@pytest.mark.parametrize("command", ["dump", "record"])
def test_a_failed_attach_exits_non_zero_naming_the_pid(tmp_path, command):
    proc, artifacts = _run(tmp_path, command, failing=(102,))
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, f"{command} exited 0 with an attach failure:\n{output}"
    assert "pid 102" in output and "Permission Denied" in output, output
    assert "pid 101" not in output and "pid 103" not in output, output
    assert os.path.isfile(os.path.join(artifacts, "pid102.error"))
    if command == "record":
        assert all(os.path.isfile(os.path.join(artifacts, f"pid{pid}.svg")) for pid in (101, 103)), (
            "one failed attach stopped the others"
        )


def test_record_samples_every_rank_over_one_window(tmp_path):
    proc, artifacts = _run(tmp_path, "record")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    windows = [json.loads((tmp_path / "log" / f"{pid}.json").read_text()) for pid in _PIDS]
    assert max(start for start, _ in windows) < min(end for _, end in windows), (
        f"the ranks were sampled one after another, not together: {windows}"
    )
    assert sorted(os.listdir(artifacts)) == sorted(f"pid{pid}.{ext}" for pid in _PIDS for ext in ("svg", "log"))


_SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


@pytest.fixture
def started(monkeypatch) -> list[subprocess.Popen]:
    """Every process the runner launches, recorded so the test can see whether it is still running."""
    processes = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        processes.append(real_popen(*args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(debugging.subprocess, "Popen", recording_popen)
    return processes


def test_an_attach_past_its_timeout_is_killed_and_reported(tmp_path, started):
    capture = debugging._run_py_spy_per_pid(
        tmp_path, [7], lambda _pid, _target: _SLEEPER, timeout=1, output_suffix=".log"
    )

    assert capture.failures == {7: "py-spy did not finish within 1s"}
    assert started and all(process.poll() is not None for process in started), "a timed-out py-spy is still attached"


def test_a_sweep_interrupted_mid_launch_leaves_no_process_running(tmp_path, started):
    def command(pid: int, _target) -> list[str]:
        if pid == 2:
            raise KeyboardInterrupt
        return _SLEEPER

    with pytest.raises(KeyboardInterrupt):
        debugging._run_py_spy_per_pid(tmp_path, [1, 2], command, timeout=60, output_suffix=".log")

    assert started and all(process.poll() is not None for process in started), (
        "an interrupted sweep left py-spy running"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
