#!/usr/bin/env python
"""A GPU row that fails on an open bug still runs, as a strict ``xfail`` naming that bug.

``TestSpec.known_failures`` maps rows of a suite's ``args_matrix`` to the bug each fails on, and
``tests/gpu/conftest.py`` marks exactly those rows strict ``xfail``: the failure stays visible, and a row
that starts passing fails the tier until its entry is removed. An entry naming no row would mark
nothing and read as coverage, so the spec refuses it.

The xfail accepts only the FAIL verdict, a script that ran and reported the failure. An ERROR on the row
(a crash with no result line, a timeout, a usage error before the launch) is not the bug, so it still
fails the row. pytest decides that through the mark's ``raises``, which each verdict below is checked
against. The harness keeps the verdict reportable: a body that faulted the device also faults the
cleanup after it, and that must not drop the result line.

    python tests/cpu/conventions/test_gpu_known_failures.py
"""

import json

import pytest

from tests.common import harness
from tests.common.reporting import RESULT_SENTINEL
from tests.gpu import conftest as gpu_conftest
from tests.gpu.manifest import TestSpec

_KNOWN_ROW = "--mode b"
_DEVICE_FAULT = "CUDA error: unspecified launch failure"


class _Metafunc:
    fixturenames = ("gpu_case",)

    def parametrize(self, name, params):
        self.params = params


def _generated_rows(monkeypatch) -> dict:
    spec = TestSpec(
        nproc=2, markers=("gpu",), args_matrix=("--mode a", _KNOWN_ROW), known_failures={_KNOWN_ROW: "bug"}
    )
    monkeypatch.setattr(gpu_conftest, "MANIFEST", {"suite.py": spec})
    metafunc = _Metafunc()
    gpu_conftest.pytest_generate_tests(metafunc)
    return {param.id: param for param in metafunc.params}


def _result_line(status: str, checks: dict, **extra) -> str:
    return f"{RESULT_SENTINEL} " + json.dumps({"status": status, "checks": checks, "metrics": {}, **extra})


def _launch_raises(monkeypatch, tmp_path, launch) -> tuple[BaseException, bool]:
    """What the known row raises when its launch returns ``launch`` (exit code, stdout, leaked pids), and
    whether its xfail accepts that, by pytest's rule for ``raises``: an instance of the named type."""
    row = _generated_rows(monkeypatch)[f"suite.py[{_KNOWN_ROW}]"]
    case = row.values[0]
    if launch is not None:
        monkeypatch.setattr(case, "_launch_once", lambda tmp_path: launch)
    with pytest.raises(BaseException) as raised:
        case.run(tmp_path)
    (xfail,) = [mark for mark in row.marks if mark.name == "xfail"]
    return raised.value, isinstance(raised.value, xfail.kwargs["raises"])


def test_a_known_failure_must_name_a_row_the_suite_runs():
    with pytest.raises(ValueError, match="does not run"):
        TestSpec(nproc=2, args_matrix=("--mode a",), known_failures={"--mode b": "open bug"})


def test_only_known_failure_rows_are_collected_as_strict_xfail(monkeypatch):
    rows = _generated_rows(monkeypatch)
    xfails = {row_id: [mark.kwargs for mark in param.marks if mark.name == "xfail"] for row_id, param in rows.items()}
    assert xfails == {
        "suite.py[--mode a]": [],
        "suite.py[--mode b]": [{"reason": "bug", "strict": True, "raises": gpu_conftest.ReportedFailure}],
    }


@pytest.mark.parametrize(
    "launch",
    [
        (1, _result_line("fail", {"loss_matches": False}), []),
        (1, _result_line("error", {}, error=f"AcceleratorError: {_DEVICE_FAULT}"), []),
        (1, "[Rank 1] FAILED CHECKS: ['grad_matches']\n" + _result_line("pass", {"grad_matches": True}), []),
    ],
    ids=["failed-check", "body-raised", "non-zero-rank-failed"],
)
def test_a_failure_the_script_reports_is_the_known_bug(monkeypatch, tmp_path, launch):
    exc, accepted = _launch_raises(monkeypatch, tmp_path, launch)
    assert str(getattr(exc, "msg", exc)).startswith("FAIL:"), exc
    assert accepted, f"a reported failure was not the known bug: {exc!r}"


@pytest.mark.parametrize(
    "launch",
    [
        (1, "Traceback (most recent call last):\nImportError: no module named x\n", []),
        (-6, f"[Rank 0] FATAL: AcceleratorError: {_DEVICE_FAULT}\n", []),
        (None, "", []),
        (1, "torch.OutOfMemoryError: CUDA out of memory\n", []),
    ],
    ids=["crash-before-the-harness", "crash-after-the-body-raised", "timeout", "oom"],
)
def test_an_error_on_a_known_failure_row_still_fails_it(monkeypatch, tmp_path, launch):
    exc, accepted = _launch_raises(monkeypatch, tmp_path, launch)
    assert str(getattr(exc, "msg", exc)).startswith("ERROR:"), exc
    assert not accepted, f"an ERROR passed as the known bug: {exc!r}"


def test_a_usage_error_before_the_launch_still_fails_a_known_failure_row(monkeypatch, tmp_path):
    """The real ``_launch_once`` refusing a basetemp too deep for a socket, before any process starts."""
    monkeypatch.setattr(gpu_conftest, "_AF_UNIX_MAX", 0)
    exc, accepted = _launch_raises(monkeypatch, tmp_path, launch=None)
    assert isinstance(exc, pytest.UsageError), exc
    assert not accepted, "a usage error passed as the known bug"


def _harnessed_run(monkeypatch, tmp_path, body) -> tuple[int, list[str]]:
    """Run ``body`` under the real ``gpu_test_main`` on one CPU rank whose cleanup faults the way a faulted
    device does; return the exit code and the statuses the result line carried."""
    monkeypatch.setattr(harness, "init_distributed", lambda: (0, 1, 0))
    monkeypatch.setattr(
        harness, "setup_cache_dirs", lambda prefix, rank: (str(tmp_path / "out"), str(tmp_path / "cache"))
    )

    def faulted_cleanup():
        raise RuntimeError(_DEVICE_FAULT)

    monkeypatch.setattr(harness, "cleanup_memory", faulted_cleanup)
    emitted = []
    monkeypatch.setattr(harness, "emit_result", lambda status, **kwargs: emitted.append(status))
    with pytest.raises(SystemExit) as exited:
        harness.gpu_test_main(partial_state=False)(body)()
    return exited.value.code, emitted


def _raising_body(ctx):
    raise RuntimeError(_DEVICE_FAULT)


def _failing_body(ctx):
    return {"checks": {"loss_matches": False}}


@pytest.mark.parametrize(
    ("body", "status"), [(_raising_body, "error"), (_failing_body, "fail")], ids=["raised", "failed"]
)
def test_a_failed_body_is_reported_even_when_its_cleanup_faults(monkeypatch, tmp_path, body, status):
    """Without the result line the launcher reads the row as an infra ERROR, which no known failure accepts."""
    assert _harnessed_run(monkeypatch, tmp_path, body) == (1, [status])


def test_a_passing_body_does_not_survive_a_faulted_cleanup(monkeypatch, tmp_path):
    """A device fault surfacing only at cleanup may be the body's own, so a pass is never reported past it."""
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        _harnessed_run(monkeypatch, tmp_path, lambda ctx: {"checks": {"loss_matches": True}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
