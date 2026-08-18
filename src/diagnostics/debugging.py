"""Opt-in diagnostics for distributed failures: cross-rank consistency checks and py-spy captures.

``assert_consistent`` / ``assert_tensor_shape_consistent`` are **manual instrumentation** — the
toolkit ships no call sites, so ``HALO_TP_CONSISTENCY_CHECK=1`` (raise instead of warn) only bites
once the call is added to the code under investigation. Nothing here costs anything when unused.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from src.distributed.runtime import rank_tag
from src.env import env_flag

logger = logging.getLogger(__name__)

__all__ = [
    "FLAMEGRAPH_DIR",
    "STACK_DUMP_DIR",
    "assert_consistent",
    "assert_tensor_shape_consistent",
    "dump_distributed_stacks",
    "record_distributed_flamegraph",
    "torchrun_python_pids",
]

# Default py-spy destinations, on the node's own temp filesystem rather than under HALO_DATA_ROOT: a
# hang diagnosis must not itself block on a shared filesystem. py_spy_diag.py reads them back.
STACK_DUMP_DIR = os.path.join(tempfile.gettempdir(), "halo_diag_stacks")
FLAMEGRAPH_DIR = os.path.join(tempfile.gettempdir(), "halo_diag_flamegraphs")

# py-spy attach budgets: a dump snapshots an already-stopped process, a record needs headroom on
# top of its own ``duration``. pgrep is a local /proc walk — its budget elapses only on a wedged host.
_PY_SPY_DUMP_TIMEOUT_S = 10
_PY_SPY_RECORD_GRACE_S = 30
_PGREP_TIMEOUT_S = 5


def assert_consistent(
    value: Any,
    *,
    group: dist.ProcessGroup | None,
    label: str,
    strict: bool | None = None,
) -> None:
    """Assert ``value`` is identical on every rank of ``group`` (hashes gathered).

    Call it from the code under investigation; the toolkit has no built-in call sites.
    ``group=None`` is a no-op. ``strict`` raises on mismatch (vs warn); defaults to
    ``HALO_TP_CONSISTENCY_CHECK``.
    """
    if group is None:
        return
    world = dist.get_world_size(group=group)
    if world <= 1:
        return
    if strict is None:
        strict = env_flag("HALO_TP_CONSISTENCY_CHECK")

    rank = dist.get_rank(group=group)
    digest = _hash_value(value)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest, group=group)

    distinct = {d for d in digests if d is not None}
    if len(distinct) <= 1:
        return

    summary = ", ".join(f"rank{i}={d[:12] if d else '<missing>'}" for i, d in enumerate(digests))
    message = f"[assert_consistent:{label}] divergent values across ranks: {summary}"
    if strict:
        raise RuntimeError(message + f" (rank {rank}: digest={digest[:12]})")
    logger.warning(message)


def assert_tensor_shape_consistent(
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None,
    label: str,
    strict: bool | None = None,
) -> None:
    """``assert_consistent`` comparing only ``tensor.shape`` — for TP/EP collective deadlocks,
    which almost always start as a shape mismatch."""
    assert_consistent(tuple(tensor.shape), group=group, label=label, strict=strict)


def dump_distributed_stacks(
    output_dir: str | Path = STACK_DUMP_DIR,
    *,
    pids: Iterable[int] | None = None,
) -> Path | None:
    """Capture a ``py-spy dump`` for every Python rank on this node.

    One file per pid in a timestamped sub-dir; returns that path, or ``None`` without py-spy. One
    rank's hang dumps the whole node (this process plus torchrun's other children), and explicit
    ``pids`` overrides that discovery (the out-of-process py_spy_diag.py path).
    """
    if not shutil.which("py-spy"):
        return None

    target = _py_spy_artifact_dir(output_dir)
    _run_py_spy_per_pid(
        target,
        _py_spy_target_pids(pids, include_self=True, include_children=True),
        lambda pid, _target: ["py-spy", "dump", "--pid", str(pid)],
        timeout=_PY_SPY_DUMP_TIMEOUT_S,
        stdout_suffix=".txt",
    )
    return target


def record_distributed_flamegraph(
    output_dir: str | Path = FLAMEGRAPH_DIR,
    *,
    duration: int = 30,
    rate: int = 100,
    native: bool = False,
    this_rank_only: bool = True,
    pids: Iterable[int] | None = None,
) -> Path | None:
    """Record a ``py-spy`` CPU flame graph (SVG) for this node's ranks.

    Samples over ``duration`` seconds — CPU-side *bottlenecks* (dataloader stalls, tokenization) —
    where :func:`dump_distributed_stacks` snapshots a *hang*. One ``pid<pid>.svg`` per process;
    ``this_rank_only=False`` profiles every torchrun child, explicit ``pids`` overrides discovery.
    """
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

    target = _py_spy_artifact_dir(output_dir)
    _run_py_spy_per_pid(
        target,
        _py_spy_target_pids(pids, include_self=True, include_children=not this_rank_only),
        _record_command,
        timeout=duration + _PY_SPY_RECORD_GRACE_S,
    )
    return target


def torchrun_python_pids() -> set[int]:
    """Best-effort enumeration of training Python processes (torchrun children) on this node."""
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


def _hash_value(value: Any) -> str:
    """Stable hash: tensors as ``(shape, dtype)`` + SHA256 of raw bytes; containers element-wise;
    else the pickle.

    Tensor bytes go through a ``torch.uint8`` view, never ``.numpy()``: bf16/fp8 have no numpy
    equivalent, and a one-rank ``TypeError`` inside an ``all_gather_object`` flow hangs the peers.
    """
    hasher = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        hasher.update(repr(value.shape).encode())
        hasher.update(repr(value.dtype).encode())
        raw = value.detach().to("cpu").contiguous().view(-1)
        if raw.numel():
            hasher.update(raw.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (list, tuple)):
        # Recurse over EVERY element (no first-element sniffing); type/length prefix keeps [x] vs [[x]].
        hasher.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            hasher.update(_hash_value(item).encode())
    else:
        hasher.update(pickle.dumps(value, protocol=4))
    return hasher.hexdigest()


def _py_spy_artifact_dir(output_dir: str | Path) -> Path:
    """Create and return this rank's timestamped artifact sub-dir under ``output_dir``."""
    target = Path(output_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{rank_tag()}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _py_spy_target_pids(
    pids: Iterable[int] | None,
    *,
    include_self: bool,
    include_children: bool,
) -> set[int]:
    """Processes to attach to: explicit ``pids`` verbatim, else this process and/or torchrun's children."""
    if pids is not None:
        return set(pids)
    targets = {os.getpid()} if include_self else set()
    if include_children:
        targets.update(torchrun_python_pids())
    return targets


def _run_py_spy_per_pid(
    target: Path,
    pids: Iterable[int],
    build_command: Callable[[int, Path], list[str]],
    *,
    timeout: int,
    stdout_suffix: str | None = None,
) -> None:
    """Run one ``build_command(pid, target)`` invocation per pid, recording failures beside the output.

    A failed attach must not abort the sweep — the point is capturing every rank of a hanging job —
    so the error goes to ``pid<pid>.error`` and the loop continues. ``stdout_suffix`` captures the
    tool's stdout+stderr into ``pid<pid><suffix>`` for the subcommands that write to stdout.
    """
    for pid in sorted(pids):
        command = build_command(pid, target)
        try:
            if stdout_suffix is None:
                subprocess.run(command, timeout=timeout, check=False)
            else:
                with (target / f"pid{pid}{stdout_suffix}").open("w") as out_file:
                    subprocess.run(command, stdout=out_file, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            (target / f"pid{pid}.error").write_text(f"{type(exc).__name__}: {exc}\n")
