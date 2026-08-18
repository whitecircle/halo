"""Contract tests for the GPU-suite launcher in ``tests/gpu/conftest.py``.

A launcher that misclassifies a verdict voids the whole tier silently, so its classification is
tested directly rather than trusted. These are pytest-native (collected like ``test_suite.py``, not
launched through the manifest) and need no GPU — but they carry the ``gpu``/``core`` markers because
they gate the GPU tier and must run wherever it runs.

Each case pins one verdict the suite depends on:

* a failed check is a FAIL, not a pass;
* a crash before any harness code is an ERROR;
* a non-zero exit whose result line says ``pass`` is a FAIL (only rank 0 emits the line, so that
  combination means a non-zero rank failed) — never an infra ERROR, which would read as something
  to retry rather than as a real regression;
* the timeout path reaches workers torchrun placed in their own sessions, and reports any it could
  not kill (they keep holding GPUs and poison later nodes);
* zero visible GPUs is a usage error, never a green tier of skips.
"""

import contextlib
import json
import subprocess
import time
from unittest import mock

import pytest

from tests.common.reporting import RESULT_SENTINEL
from tests.gpu import conftest as launcher
from tests.gpu.manifest import TestSpec

pytestmark = [pytest.mark.gpu, pytest.mark.core]


def _case(monkeypatch, code, stdout, leaked=(), *, nproc=1):
    """A GPUCase whose launch is replaced by a canned ``(exit code, stdout, leaked pids)``."""
    case = launcher.GPUCase("fake/test_x.py", TestSpec(nproc=nproc, markers=("gpu",)), "")
    monkeypatch.setattr(case, "_launch_once", lambda tmp_path: (code, stdout, list(leaked)))
    return case


def _result_line(status: str, checks: dict) -> str:
    return f"{RESULT_SENTINEL} " + json.dumps({"status": status, "checks": checks, "metrics": {}})


def _verdict(case, tmp_path) -> tuple[str, str]:
    """Run the case; return ``("pass", "")`` or ``("fail", <report text>)``."""
    try:
        case.run(tmp_path)
    except BaseException as e:
        return "fail", str(getattr(e, "msg", e))
    return "pass", ""


def test_failed_check_is_a_failure(monkeypatch, tmp_path):
    case = _case(monkeypatch, 1, _result_line("fail", {"loss_matches": False}))
    outcome, msg = _verdict(case, tmp_path)
    assert outcome == "fail", "a result line with status=fail was reported as a pass"
    assert "loss_matches" in msg, f"the failing check must be named in the report, got: {msg}"


def test_crash_with_no_result_line_is_an_error(monkeypatch, tmp_path):
    case = _case(monkeypatch, 1, "Traceback ...\nImportError: no module named x\n")
    outcome, msg = _verdict(case, tmp_path)
    assert outcome == "fail", "a non-zero exit with no result line was reported as a pass"
    assert "no result line" in msg, msg


def test_non_zero_rank_failure_is_a_fail_not_an_infra_error(monkeypatch, tmp_path):
    stdout = "[Rank 1] FAILED CHECKS: ['grad_matches']\n" + _result_line("pass", {"grad_matches": True})
    case = _case(monkeypatch, 1, stdout, nproc=2)
    outcome, msg = _verdict(case, tmp_path)
    assert outcome == "fail", "a rank-1-only failure was swallowed"
    assert "non-zero rank failed" in msg, f"must be attributed to a non-zero rank, got: {msg}"
    assert "no result line" not in msg, f"misreported as a missing result line: {msg}"


def test_passing_run_passes(monkeypatch, tmp_path):
    case = _case(monkeypatch, 0, _result_line("pass", {"loss_matches": True}))
    assert _verdict(case, tmp_path)[0] == "pass"


def test_declined_run_is_skipped_through_the_rank_prefix(monkeypatch, tmp_path):
    """A script that declines to run must be SKIPPED, not reported as a green PASS.

    The scripts announce it through ``log()``, which stamps a ``[Rank N] `` prefix, and torchrun
    interleaves ranks mid-line — so the launcher has to scan for the marker rather than anchor at
    the start of the line. Anchoring is what makes an unrun test look like a passing one.
    """
    case = _case(monkeypatch, 0, "[Rank 0] SKIP: local model path missing: org/model\n")
    with pytest.raises(pytest.skip.Exception, match="local model path missing"):
        case.run(tmp_path)


def test_a_silent_exit_zero_is_still_a_pass(monkeypatch, tmp_path):
    """Anti-vacuity for the skip above: exit-code-only scripts have no other way to report success,
    so a clean exit with neither a result line nor a SKIP must stay a PASS."""
    assert _verdict(_case(monkeypatch, 0, "done\n"), tmp_path)[0] == "pass"


def test_timeout_reports_workers_it_could_not_kill(monkeypatch, tmp_path):
    case = _case(monkeypatch, None, "", leaked=[4242, 4243])
    outcome, msg = _verdict(case, tmp_path)
    assert outcome == "fail"
    assert "TIMEOUT" in msg and "4242" in msg, f"leaked workers must be named so the run is distrusted: {msg}"


def test_oom_on_a_small_tier_is_not_skipped(monkeypatch, tmp_path):
    """An OOM at nproc=2 is a real error: that config is required to fit."""
    case = _case(monkeypatch, 1, "torch.OutOfMemoryError: CUDA out of memory\n", nproc=2)
    outcome, msg = _verdict(case, tmp_path)
    assert outcome == "fail" and "OOM" in msg, msg


def test_kill_launch_sweeps_workers_outside_the_agent_process_group():
    """The timeout path must reach workers torchrun put in their own sessions.

    A ``killpg`` on the agent alone leaves them running — they keep holding GPUs and the launch's
    stdout pipe, which stalls the drain and poisons every later node. Stand up a real grandchild in
    its own session and check the sweep finds and kills it.
    """
    launch_id = "halo-launcher-contract-probe"
    env = {launcher._LAUNCH_ID_VAR: launch_id}
    agent = subprocess.Popen(["/bin/sh", "-c", "sleep 300"], env=env, start_new_session=True)
    # Its own session, exactly like a torchrun worker: outside the agent's process group.
    worker = subprocess.Popen(["/bin/sh", "-c", "sleep 300"], env=env, start_new_session=True)
    try:
        # Popen returns once the child is FORKED, not once it has exec'd, and the sweep reads
        # /proc/<pid>/environ — which carries the marker only after the exec. Poll for it rather
        # than sampling once, or the assertion below races the child's startup. Assert on what the
        # poll saw: re-surveying here would race the process-table churn all over again.
        deadline = time.monotonic() + 30
        survivors: list[int] = []
        while time.monotonic() < deadline:
            survivors = launcher._launch_survivors(launch_id)
            if agent.pid in survivors and worker.pid in survivors:
                break
            time.sleep(0.05)
        assert worker.pid in survivors, "the marker sweep did not see the worker"
        leaked = launcher._kill_launch(agent, launch_id)
        assert leaked == [], f"processes survived the timeout kill: {leaked}"
        assert worker.wait(timeout=30) is not None, (
            "the worker outside the agent's process group survived the timeout kill"
        )
    finally:
        for p in (agent, worker):
            with contextlib.suppress(ProcessLookupError):
                p.kill()
            p.wait(timeout=10)


def test_zero_visible_gpus_is_an_error_not_a_green_tier(monkeypatch):
    """0 GPUs must abort collection; skipping every node exits 0 and looks like a passing tier."""

    class FakeItem:
        def __init__(self, case):
            self.callspec = type("CallSpec", (), {"params": {"gpu_case": case}})()
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    monkeypatch.setattr(launcher, "_gpu_count", lambda: 0)
    monkeypatch.setattr(launcher, "unregistered_scripts", lambda: [])
    monkeypatch.setattr(launcher, "stale_entries", lambda: [])
    items = [FakeItem(launcher.GPUCase("fake/test_x.py", TestSpec(nproc=2, markers=("gpu",)), ""))]
    with pytest.raises(pytest.UsageError, match="no GPU is visible"):
        launcher.pytest_collection_modifyitems(None, items)
    assert items[0].markers == [], "nodes must not be silently skipped when no GPU is visible"


def test_server_tier_skips_when_its_engine_is_absent_and_names_how_to_start_it():
    """A node driving vLLM/SGLang cannot run on a host where that engine is down. Failing it there
    buries the tier's real signal under nine red herrings, so it skips — and the reason has to carry
    the command that fixes it, or the skip is just a quieter dead end."""
    launcher._server_is_up.cache_clear()
    with mock.patch.object(launcher, "_server_is_up", return_value=False):
        vllm = launcher._absent_server(("gpu", "full", "vllm_server"))
        sglang = launcher._absent_server(("gpu", "full", "sglang_server"))
    assert vllm and "docker-compose.vllm.yml" in vllm
    assert sglang and "docker-compose.sglang.yml" in sglang


def test_only_server_tier_nodes_are_skipped_and_only_while_the_engine_is_down():
    """The skip must not leak: an ordinary GPU node has no engine to wait for, and a server node
    whose engine IS up must run — otherwise a live tier reports green without testing anything."""
    with mock.patch.object(launcher, "_server_is_up", return_value=False):
        assert launcher._absent_server(("gpu", "core", "2gpu", "ep")) is None
    with mock.patch.object(launcher, "_server_is_up", return_value=True):
        assert launcher._absent_server(("gpu", "full", "vllm_server")) is None


def test_scheme_less_server_url_is_probed_not_condemned():
    """``VLLM_SERVER_URL=localhost:8000`` passes the Makefile's ``curl`` gate, and urllib raises
    ``ValueError`` on it — read as "down", the whole tier would silently skip seconds after curl
    said healthy. The probe must normalize the scheme so both gates ask the same question."""
    seen = []
    with mock.patch.object(
        launcher.urllib.request,
        "urlopen",
        side_effect=lambda url, timeout: seen.append(url) or (_ for _ in ()).throw(OSError()),
    ):
        launcher._server_is_up.cache_clear()
        launcher._server_is_up("localhost:8000")
    assert seen == ["http://localhost:8000/health"]
    launcher._server_is_up.cache_clear()
