"""CPU tests for scripts/profiling/trace_report.py: trace-set discovery and report guards.

Discovery must group the exporter's ``<label>-rankNN.trace.json[.gz]`` artifacts by label and
rank and ignore everything else in the directory; the collective-report guards (non-dense rank
sets, mixed suffixes, bad filenames) must reject *before* touching TraceLens, so they are
testable without the profiling dependency group installed. The report happy paths need real
GPU traces + TraceLens and are exercised by the GPU-side smoke, not here.

Run: python tests/cpu/parallelism/test_trace_report.py
"""

import gzip
import sys
from pathlib import Path

import pytest

from scripts.profiling import trace_report
from scripts.profiling.trace_report import (
    discover_trace_sets,
    has_cpu_ops,
    write_collective_report,
    write_rank_report,
)

_KERNEL_EVENT = b'{"ph": "X", "cat": "kernel", "name": "gemm", "dur": 1.0}'
_CPU_OP_EVENT = b'{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 2.0}'


def _write_trace(path: Path, *events: bytes, pad: int = 0) -> Path:
    """Write a minimal Chrome trace, gzipped when the name says so, padded to force chunked reads."""
    body = b'{"traceEvents": [' + b",".join((*events, b'{"ph": "M", "name": "%b"}' % (b"x" * pad))) + b"]}"
    path.write_bytes(gzip.compress(body) if path.name.endswith(".gz") else body)
    return path


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_bytes(b"{}")


def test_discovery_groups_by_label_and_rank(tmp_path):
    _touch(
        tmp_path,
        "trace-cycle1-rank00.trace.json.gz",
        "trace-cycle1-rank01.trace.json.gz",
        "gen-rank00.trace.json",
        # Sibling profiler artifacts and noise must be ignored.
        "trace-cycle1-rank00.top_ops.txt",
        "trace-cycle1-rank00.stacks_cuda.txt",
        "mem-trace-rank00.pickle",
        "notes.md",
    )
    sets = discover_trace_sets(tmp_path)
    assert set(sets) == {"trace-cycle1", "gen"}
    assert list(sets["trace-cycle1"]) == [0, 1]
    assert sets["gen"][0].name == "gen-rank00.trace.json"


def test_discovery_keeps_zero_padded_rank_order(tmp_path):
    _touch(tmp_path, *[f"t-rank{r:02d}.trace.json.gz" for r in (10, 2, 0)])
    assert list(discover_trace_sets(tmp_path)["t"]) == [0, 2, 10]


def test_discovery_empty_dir(tmp_path):
    assert discover_trace_sets(tmp_path) == {}


def test_rank_report_rejects_non_trace_filename(tmp_path):
    _touch(tmp_path, "random.json")
    with pytest.raises(ValueError, match="not a profiler trace artifact"):
        write_rank_report(tmp_path / "random.json")


def test_rank_report_rejects_cuda_only_capture(tmp_path):
    """cpu_activity=False traces cannot be reported on — say so, not a pandas KeyError from TraceLens."""
    trace = _write_trace(tmp_path / "t-rank00.trace.json.gz", _KERNEL_EVENT)
    with pytest.raises(ValueError, match="no cpu_op events"):
        write_rank_report(trace)


@pytest.mark.parametrize("name", ["t-rank00.trace.json", "t-rank00.trace.json.gz"])
def test_has_cpu_ops_detects_marker_across_read_boundaries(tmp_path, monkeypatch, name):
    """The scan is chunked, so the marker must still be found when it straddles two reads."""
    monkeypatch.setattr(trace_report, "_SCAN_CHUNK", 8)
    assert has_cpu_ops(_write_trace(tmp_path / name, _CPU_OP_EVENT, _KERNEL_EVENT, pad=64))
    assert not has_cpu_ops(_write_trace(tmp_path / f"other-{name}", _KERNEL_EVENT, pad=64))


def test_collective_report_skips_sparse_rank_set(tmp_path):
    """A ranks="0,8" capture has no dense 0..N-1 rank set — must skip, not mis-map ranks."""
    _touch(tmp_path, "t-rank00.trace.json.gz", "t-rank08.trace.json.gz")
    ranks = discover_trace_sets(tmp_path)["t"]
    assert write_collective_report("t", ranks) is None


def test_collective_report_skips_mixed_suffixes(tmp_path):
    _touch(tmp_path, "t-rank00.trace.json.gz", "t-rank01.trace.json")
    ranks = discover_trace_sets(tmp_path)["t"]
    assert write_collective_report("t", ranks) is None


def _run_main(tmp_path, monkeypatch, failing: set[str], *, keep_going: bool = False) -> list[str]:
    """Run the CLI over ``tmp_path`` with the TraceLens call stubbed; returns the reported labels."""
    reported: list[str] = []

    def fake_rank_report(path: Path) -> Path:
        label = trace_report._TRACE_NAME_RE.match(path.name)["label"]
        if label in failing:
            raise ValueError(f"{label}: no cpu_op events")
        reported.append(label)
        return path.with_suffix(".xlsx")

    argv = ["trace_report.py", str(tmp_path)] + (["--keep-going"] if keep_going else [])
    monkeypatch.setattr(trace_report, "write_rank_report", fake_rank_report)
    monkeypatch.setattr(sys, "argv", argv)
    trace_report.main()
    return reported


def test_one_unreportable_label_does_not_cancel_the_others(tmp_path, monkeypatch):
    """A CUDA-only (or truncated) capture is per-label: it must not take every other label's report
    with it — the directory holds a whole run's cycles, and re-capturing is a GPU job.

    ``--keep-going`` is what accepts the partial result; the sibling labels are reported either way
    (see the exit-code test below), so the flag changes the verdict, never the work done.
    """
    _touch(tmp_path, "cuda-only-rank00.trace.json", "good-rank00.trace.json", "later-rank00.trace.json")

    assert _run_main(tmp_path, monkeypatch, failing={"cuda-only"}, keep_going=True) == ["good", "later"]


def test_a_partial_run_exits_nonzero_by_default(tmp_path, monkeypatch):
    """Exiting 0 on a partial run reports missing workbooks as generated ones: the caller analyses
    whichever labels happened to survive without knowing any are gone. The surviving labels are
    still reported — the failure is the verdict, not a cancellation."""
    _touch(tmp_path, "cuda-only-rank00.trace.json", "good-rank00.trace.json", "later-rank00.trace.json")

    with pytest.raises(SystemExit, match="1 of 3 trace set"):
        _run_main(tmp_path, monkeypatch, failing={"cuda-only"})


def test_every_label_failing_exits_nonzero(tmp_path, monkeypatch):
    """Nothing was produced — the CLI must not report success."""
    _touch(tmp_path, "a-rank00.trace.json", "b-rank00.trace.json")

    with pytest.raises(SystemExit, match="2 of 2 trace set"):
        _run_main(tmp_path, monkeypatch, failing={"a", "b"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
