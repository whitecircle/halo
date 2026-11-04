"""py-spy stack dumps and CPU flame graphs of this node's torchrun ranks, every rank over one window."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from halo.env import data_path
from halo.runtime.ranks import rank_tag

# Beside the other profiler artifacts, so a capture outlives a container whose TMPDIR is not mounted.
STACK_DUMP_DIR = data_path("profiling", "py_spy", "stacks")
FLAMEGRAPH_DIR = data_path("profiling", "py_spy", "flamegraphs")

# A record needs headroom on top of its own ``duration``; pgrep's budget elapses only on a wedged host.
_PY_SPY_DUMP_TIMEOUT_S = 10
_PY_SPY_RECORD_GRACE_S = 30
_PGREP_TIMEOUT_S = 5


@dataclass(frozen=True)
class _PySpyCapture:
    """One py-spy sweep: its artifact ``directory`` and, by pid, why each failed attach failed."""

    directory: Path
    failures: dict[int, str]


def torchrun_python_pids() -> set[int]:
    """Best-effort: training Python processes (torchrun children) on this node."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "torchrun"], text=True, timeout=_PGREP_TIMEOUT_S)
    except (subprocess.SubprocessError, OSError):
        return set()
    parents = {int(p) for p in out.split() if p}
    pids: set[int] = set()
    for parent in parents:
        try:
            children = subprocess.check_output(["pgrep", "-P", str(parent)], text=True, timeout=_PGREP_TIMEOUT_S)
        except (subprocess.SubprocessError, OSError):
            continue
        for p in children.split():
            if p:
                pids.add(int(p))
    return pids


def _py_spy_artifact_dir(output_dir: str | Path) -> Path:
    target = Path(output_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{rank_tag()}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _py_spy_target_pids(pids: Iterable[int] | None, *, include_children: bool) -> set[int]:
    """Explicit ``pids`` verbatim, else this process plus, with ``include_children``, torchrun's children."""
    if pids is not None:
        return set(pids)
    targets = {os.getpid()}
    if include_children:
        targets.update(torchrun_python_pids())
    return targets


def _stop_running(processes: dict[int, subprocess.Popen]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.kill()
            process.wait()


def _run_py_spy_per_pid(
    target: Path,
    pids: Iterable[int],
    build_command: Callable[[int, Path], list[str]],
    *,
    timeout: int,
    output_suffix: str,
) -> _PySpyCapture:
    """Run every pid at once so all ranks share one window; output lands in ``pid<pid><output_suffix>``, and a failed
    attach writes ``pid<pid>.error`` while the rest run on."""
    running: dict[int, subprocess.Popen] = {}
    failures: dict[int, str] = {}
    with ExitStack() as stack:
        # Unwinds last, so no py-spy stays attached past the timeout or an early exit.
        stack.callback(_stop_running, running)
        for pid in sorted(pids):
            output = stack.enter_context((target / f"pid{pid}{output_suffix}").open("w"))
            try:
                running[pid] = subprocess.Popen(build_command(pid, target), stdout=output, stderr=subprocess.STDOUT)
            except OSError as exc:
                failures[pid] = f"{type(exc).__name__}: {exc}"
        deadline = time.monotonic() + timeout
        for pid, process in running.items():
            try:
                returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                failures[pid] = f"py-spy did not finish within {timeout}s"
                continue
            if returncode != 0:
                lines = (target / f"pid{pid}{output_suffix}").read_text().strip().splitlines()
                failures[pid] = f"py-spy exited {returncode}: {lines[-1] if lines else '(no output)'}"
    for pid, reason in failures.items():
        (target / f"pid{pid}.error").write_text(f"{reason}\n")
    return _PySpyCapture(target, dict(sorted(failures.items())))


def dump_node_stacks(
    output_dir: str | Path = STACK_DUMP_DIR,
    *,
    pids: Iterable[int] | None = None,
) -> _PySpyCapture | None:
    """``py-spy dump`` of every Python rank on this node into a timestamped sub-directory; ``None`` without py-spy.
    A hang on one rank dumps the whole node unless explicit ``pids`` are given."""
    if not shutil.which("py-spy"):
        return None

    return _run_py_spy_per_pid(
        _py_spy_artifact_dir(output_dir),
        _py_spy_target_pids(pids, include_children=True),
        lambda pid, _target: ["py-spy", "dump", "--pid", str(pid)],
        timeout=_PY_SPY_DUMP_TIMEOUT_S,
        output_suffix=".txt",
    )


def record_flamegraphs(
    output_dir: str | Path = FLAMEGRAPH_DIR,
    *,
    duration: int = 30,
    rate: int = 100,
    native: bool = False,
    this_rank_only: bool = True,
    pids: Iterable[int] | None = None,
) -> _PySpyCapture | None:
    """Record a CPU flame graph (SVG) over ``duration`` seconds, for dataloader or tokenization stalls, where
    :func:`dump_node_stacks` snapshots a hang. ``this_rank_only=False`` profiles every torchrun child; explicit
    ``pids`` override discovery. ``None`` without py-spy."""
    if not shutil.which("py-spy"):
        return None

    def _record_command(pid: int, dest: Path) -> list[str]:
        command = [
            "py-spy",
            "record",
            "--format",
            "flamegraph",
            "--output",
            str(dest / f"pid{pid}.svg"),
            "--pid",
            str(pid),
            "--duration",
            str(duration),
            "--rate",
            str(rate),
        ]
        if native:
            command.append("--native")
        return command

    return _run_py_spy_per_pid(
        _py_spy_artifact_dir(output_dir),
        _py_spy_target_pids(pids, include_children=not this_rank_only),
        _record_command,
        timeout=duration + _PY_SPY_RECORD_GRACE_S,
        output_suffix=".log",
    )
