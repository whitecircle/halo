"""CPU tests for ``export_profiler_artifacts``: only real artifacts survive, and the log says which.

``torch.profiler.profile.export_stacks`` writes an empty file and returns successfully when no
event carries a Python stack (the case on the image's torch), so an exporter that trusts its
return value leaves 0-byte files behind and still logs "wrote trace + stacks".

Run: python tests/cpu/diagnostics/test_profiler_artifacts.py
"""

import logging
import os
import sys

import pytest

from src.diagnostics.profiling import export_profiler_artifacts


class _FakeProfile:
    """Stand-in for ``torch.profiler.profile`` whose exporters write fixed payloads."""

    def __init__(self, *, stacks_payload: str):
        self._stacks_payload = stacks_payload

    def export_chrome_trace(self, path):
        with open(path, "w") as fh:
            fh.write('{"traceEvents": []}')

    def export_stacks(self, path, metric):
        with open(path, "w") as fh:
            fh.write(self._stacks_payload)

    def export_memory_timeline(self, path, device):
        with open(path, "w") as fh:
            fh.write("<html></html>")

    def key_averages(self):
        return self

    def table(self, sort_by, row_limit):
        return "aten::mm  1.0ms\n"


def _written_list(log: str) -> str:
    """The artifact names the summary line claims to have written."""
    return log.split("wrote ", 1)[1].split(" → ", 1)[0]


def _export(tmp_path, caplog, *, stacks_payload):
    with caplog.at_level(logging.INFO, logger="src.diagnostics.profiling"):
        export_profiler_artifacts(
            _FakeProfile(stacks_payload=stacks_payload),
            str(tmp_path),
            label="trace",
            with_stack=True,
            profile_memory=True,
            export_memory_timeline=True,
        )
    return sorted(os.listdir(tmp_path)), caplog.text


def test_empty_stacks_are_dropped_not_advertised(tmp_path, caplog):
    """A no-op export_stacks must leave no file and must not be reported as written."""
    files, log = _export(tmp_path, caplog, stacks_payload="")

    assert not [name for name in files if ".stacks_" in name], files
    assert files == ["trace-rank00.memory_timeline.html", "trace-rank00.top_ops.txt", "trace-rank00.trace.json.gz"]
    assert "stacks_cuda.txt came back empty" in log
    assert "stacks_cpu.txt came back empty" in log
    assert "stacks" not in _written_list(log)


def test_non_empty_stacks_are_kept_and_reported(tmp_path, caplog):
    """The drop must be driven by emptiness alone — real stacks still land and are logged."""
    files, log = _export(tmp_path, caplog, stacks_payload="frame;frame 1\n")

    assert "trace-rank00.stacks_cuda.txt" in files
    assert "trace-rank00.stacks_cpu.txt" in files
    assert "came back empty" not in log
    assert "trace-rank00.stacks_cuda.txt" in _written_list(log)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
