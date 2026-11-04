#!/usr/bin/env python
"""``gpu_test_main`` reports a failed body's verdict even when the cleanup after it faults. Raising there drops the
result line, which the launcher reads as an infra ERROR; a pass must not survive the fault, which may be the
body's own."""

import pytest

from tests.common import harness

_DEVICE_FAULT = "CUDA error: unspecified launch failure"


def _harnessed_run(monkeypatch, tmp_path, body) -> tuple[int, list[str]]:
    """Run ``body`` under the real ``gpu_test_main`` on one CPU rank whose cleanup faults; return the exit code and
    the reported statuses."""
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
    assert _harnessed_run(monkeypatch, tmp_path, body) == (1, [status])


def test_a_passing_body_does_not_survive_a_faulted_cleanup(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        _harnessed_run(monkeypatch, tmp_path, lambda ctx: {"checks": {"loss_matches": True}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
