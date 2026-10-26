"""CPU tests for ``halo/diagnostics/profiling.py``: artifact export, profiler construction, the CUDA
memory read and the allocation-history region.

``torch.profiler.profile.export_stacks`` writes an empty file and returns successfully when no
event carries a Python stack (the case on the image's torch), so an exporter that trusts its
return value leaves 0-byte files behind and still logs "wrote trace + stacks". Both profilers are
built by ``build_profiler``, so a setting it drops is lost to the session and the callback alike;
each caller is pinned on what it passes in (the callback in ``tests/cpu/callbacks/test_torch_profiler.py``).
Torch traces every supported activity when given none, so the activity choice is read off that call.

Run: python tests/cpu/diagnostics/test_profiler_artifacts.py
"""

import logging
import os

import pytest
import torch
from torch.profiler import ProfilerActivity

from halo.diagnostics.profiling import (
    CudaMemoryGiB,
    build_profiler,
    cuda_memory_history,
    export_profiler_artifacts,
    log_cuda_memory,
    read_cuda_memory,
    torch_profiler_session,
)

# Allocator counters in bytes, each a distinct GiB value so a swapped field shows.
_CUDA_COUNTERS = {
    "memory_allocated": 2**30,
    "memory_reserved": 5 * 2**29,
    "max_memory_allocated": 3 * 2**30,
    "max_memory_reserved": 9 * 2**29,
}


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
    with caplog.at_level(logging.INFO, logger="halo.diagnostics.profiling"):
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


def test_the_session_profiles_only_the_selected_ranks_region(tmp_path):
    """The one-shot session profiles its region on a selected rank and writes that rank's artifacts;
    a rank outside ``ranks`` gets ``None`` and writes nothing."""
    with torch_profiler_session(str(tmp_path / "skipped"), ranks="1", label="region") as prof:
        assert prof is None
    assert not (tmp_path / "skipped").exists()

    with torch_profiler_session(str(tmp_path), ranks="0", label="region", with_stack=False) as prof:
        torch.mm(torch.ones(8, 8), torch.ones(8, 8))
    files = sorted(os.listdir(tmp_path))
    assert "region-rank00.trace.json.gz" in files, files
    assert "aten::mm" in (tmp_path / "region-rank00.top_ops.txt").read_text()


def test_the_session_builds_its_profiler_from_its_settings(tmp_path, monkeypatch):
    built: list[dict[str, object]] = []

    def recording_build_profiler(output_dir, **settings):
        built.append(settings)
        return build_profiler(output_dir, **settings)

    monkeypatch.setattr("halo.diagnostics.profiling.build_profiler", recording_build_profiler)
    with torch_profiler_session(
        str(tmp_path),
        ranks="0",
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
        export_memory_timeline=False,
    ):
        pass
    assert built == [
        {
            "cpu_activity": True,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
            "with_flops": True,
        }
    ]


def test_build_profiler_creates_the_directory_and_forwards_every_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=1)

    def on_trace_ready(prof) -> None:
        pass

    prof = build_profiler(
        str(tmp_path / "traces"),
        cpu_activity=True,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
    )

    assert (tmp_path / "traces").is_dir()
    assert prof.activities == {ProfilerActivity.CPU}
    assert (prof.record_shapes, prof.profile_memory, prof.with_stack, prof.with_flops) == (False, True, False, True)
    assert prof.schedule is schedule
    assert prof.on_trace_ready is on_trace_ready


@pytest.mark.parametrize(
    ("cpu_activity", "expected"),
    [(True, {ProfilerActivity.CPU, ProfilerActivity.CUDA}), (False, {ProfilerActivity.CUDA})],
)
def test_build_profiler_traces_cuda_when_available(tmp_path, monkeypatch, cpu_activity, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    prof = build_profiler(
        str(tmp_path), cpu_activity=cpu_activity, record_shapes=False, profile_memory=False, with_stack=False
    )
    assert prof.activities == expected


def _fake_cuda_counters(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, count in _CUDA_COUNTERS.items():
        monkeypatch.setattr(torch.cuda, name, lambda *args, count=count, **kwargs: count)


def test_read_cuda_memory_reports_each_counter_in_gib(monkeypatch):
    _fake_cuda_counters(monkeypatch)
    assert read_cuda_memory() == CudaMemoryGiB(allocated=1.0, reserved=2.5, peak_allocated=3.0, peak_reserved=4.5)


def test_log_cuda_memory_logs_this_ranks_counters(monkeypatch, caplog):
    _fake_cuda_counters(monkeypatch)
    with caplog.at_level(logging.INFO, logger="halo.diagnostics.profiling"):
        log_cuda_memory("after forward")
    assert caplog.messages == [
        "[mem rank00] after forward: allocated=1.00 reserved=2.50 peak_alloc=3.00 peak_reserved=4.50 GiB"
    ]


def test_without_cuda_there_is_no_reading_and_no_log_line(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.INFO, logger="halo.diagnostics.profiling"):
        assert read_cuda_memory() is None
        log_cuda_memory("after forward")
    assert caplog.messages == []


def _record_memory_history(monkeypatch) -> list[tuple]:
    """Stub CUDA and the history recorder; returns the recorder calls in order."""
    events: list[tuple] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "halo.diagnostics.profiling.start_memory_history",
        lambda max_entries: events.append(("start", max_entries)) or True,
    )
    monkeypatch.setattr(
        "halo.diagnostics.profiling.dump_memory_snapshot", lambda path: events.append(("dump", path)) or path
    )
    monkeypatch.setattr("halo.diagnostics.profiling.stop_memory_history", lambda: events.append(("stop",)))
    return events


def test_memory_history_dumps_the_region_and_then_stops_recording(tmp_path, monkeypatch):
    events = _record_memory_history(monkeypatch)
    with cuda_memory_history(str(tmp_path / "mem"), ranks="0", max_entries=7, label="step"):
        events.append(("region",))
    assert events == [("start", 7), ("region",), ("dump", str(tmp_path / "mem" / "step-rank00.pickle")), ("stop",)]
    assert (tmp_path / "mem").is_dir()


def test_memory_history_fails_on_a_bad_directory_before_recording(tmp_path, monkeypatch):
    """The region never runs and recording never starts, so the error cannot surface at exit with
    allocation history still on."""
    events = _record_memory_history(monkeypatch)
    (tmp_path / "file").write_text("")
    with pytest.raises(NotADirectoryError):
        with cuda_memory_history(str(tmp_path / "file" / "mem"), ranks="0"):
            events.append(("region",))
    assert events == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
