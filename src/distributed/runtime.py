"""torch.distributed core: rank/world-size getters, launcher-derived node math, barriers and sync
context managers, the cross-rank rejection/consensus seams, the shared-filesystem flags, the
process-group timeouts and DTensor resolution.

Leaf of the package — :mod:`~src.distributed.nvlink` and :mod:`~src.distributed.filesystem` import
it, never the reverse — which is why the shared-filesystem flags live here rather than beside the
coordination that reads them: :func:`init_distributed` agrees them as the group comes up.
"""

import datetime
import gc
import logging
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
from torch.distributed.tensor import DTensor

from src.env import env_flag, resolve_nccl_timeout_minutes, resolve_store_timeout_hours
from src.log import warn_once

logger = logging.getLogger(__name__)

# Process-wide: the fallback resolves the same way on every call, so one line says it.
_LWS_FALLBACK_ANNOUNCED: set[str] = set()

# Each side falls back to the umbrella while UNSET; a side var that IS set wins over it.
_SHARED_FILESYSTEM_VAR = "DIST_SHARED_FILESYSTEM"
_INPUT_SHARED_FILESYSTEM_VAR = "DIST_INPUT_SHARED_FILESYSTEM"
_OUTPUT_SHARED_FILESYSTEM_VAR = "DIST_OUTPUT_SHARED_FILESYSTEM"

# Rank-agreed values for the three vars above, set once by resolve_shared_filesystem_consensus().
_SHARED_FILESYSTEM_CONSENSUS: dict[str, bool] | None = None

# NCCL has no 16-bit-integer type; such tensors ride the wire as a lossless uint8 bit view.
_NCCL_UNSUPPORTED_DTYPES = (torch.int16, torch.uint16)


def current_device() -> torch.device:
    """This rank's CUDA device, or CPU when CUDA is unavailable."""
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def collective_device() -> torch.device:
    """Where a hand-built collective tensor must live for the default group's BACKEND — not this
    rank's compute device.

    The two differ on every gloo group running on a GPU box: gloo moves host memory, so a CUDA
    tensor is at best a staging copy, while :func:`current_device` still answers ``cuda:N``.
    """
    backend = dist.get_backend() if dist.is_available() and dist.is_initialized() else None
    if backend is not None and backend.lower() == dist.Backend.NCCL and torch.cuda.is_available():
        return current_device()
    return torch.device("cpu")


def get_global_rank() -> int:
    """Global rank of this process, or 0 if distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_global_world_size() -> int:
    """Total process count across all nodes, or 1 if not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def launcher_global_rank() -> int:
    """Global rank from the live process group, else the launcher's ``RANK`` / ``SLURM_PROCID``.

    Valid BEFORE ``init_process_group``, unlike :func:`get_global_rank`, which the node-local rank
    math needs: reporting 0 pre-init would bind every rank of the node to ``cuda:0``. ``RANK`` wins
    over ``SLURM_PROCID`` because ``srun torchrun`` sets both and the SLURM value counts srun tasks
    (one per node) there, not ranks.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)


def launcher_global_world_size() -> int:
    """World size from the live process group, else ``WORLD_SIZE`` / ``SLURM_NTASKS`` (1 when none).

    Pre-init counterpart of :func:`get_global_world_size`, same precedence. Under a bare
    ``srun --ntasks-per-node=8`` neither ``RANK`` nor ``WORLD_SIZE`` exists, and without the SLURM
    fallback every rank would read world 1, build no group, believe itself main and write one
    ``output_dir``. ``SLURM_NTASKS`` counts only alongside ``SLURM_PROCID``: ``sbatch``/``salloc``
    export the former without the latter, where a plain ``python`` launch really is world 1.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    slurm_world = os.environ.get("SLURM_NTASKS") if "SLURM_PROCID" in os.environ else None
    return int(os.environ.get("WORLD_SIZE") or slurm_world or 1)


def _warn_bad_local_world_size(local_world_size: int) -> None:
    """Warn that the node's process count is not a positive number, so node-local rank math degrades.

    A nonsensical value must not silently collapse every rank onto node 0 / ``cuda:0``.
    """
    logger.warning(
        "Local world size resolved to %d, which is not a positive process count — node-local rank "
        "math degrades to a single node with this rank at index 0. Set LOCAL_WORLD_SIZE explicitly "
        "(or pass gpus_per_node to ParallelismConfig).",
        local_world_size,
    )


def get_local_rank() -> int:
    """Local rank within this node, from LOCAL_RANK / SLURM_LOCALID, else derived from the global
    rank modulo this node's process count.

    Valid before ``init_process_group``, which reads it to pick this rank's CUDA device.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    local_world_size = get_local_world_size()
    if local_world_size <= 0:
        _warn_bad_local_world_size(local_world_size)
        return 0
    return launcher_global_rank() % local_world_size


def get_local_world_size() -> int:
    """Processes on this node, from LOCAL_WORLD_SIZE / SLURM_NTASKS_PER_NODE.

    Falls back to the visible CUDA device count (one process per GPU), then to the global world
    size — a last resort that collapses num_nodes to 1 and builds wrong node-local EP/CP groups on a
    real multi-node job, so it warns once.
    """
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    # Bare srun: SLURM may report a compact/list form ("8(x2)", "4,8"), so parse leading int.
    slurm_ntpn = os.environ.get("SLURM_NTASKS_PER_NODE")
    if slurm_ntpn:
        m = re.match(r"\d+", slurm_ntpn.strip())
        if m:
            return int(m.group())

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return count

    # Same last resort before and after init_process_group, so the answer cannot change mid-run.
    world = launcher_global_world_size()
    if world > 1:
        warn_once(
            logger,
            _LWS_FALLBACK_ANNOUNCED,
            "local_world_size",
            "Could not determine local world size from LOCAL_WORLD_SIZE / "
            "SLURM_NTASKS_PER_NODE / CUDA device count; falling back to global "
            f"world_size={world} (assuming single node). On a real multi-node job "
            "this builds WRONG node-local EP/CP groups — set LOCAL_WORLD_SIZE "
            "explicitly or pass gpus_per_node to ParallelismConfig.",
        )
    return world


def get_num_nodes() -> int:
    """Number of nodes in the distributed setup, or 1 if not distributed.

    Launcher-derived like :func:`get_local_rank`: mixing a launcher-derived node size with a
    post-init-only world size would report 0 nodes before ``init_process_group``.
    """
    local_world_size = get_local_world_size()
    if local_world_size <= 0:
        _warn_bad_local_world_size(local_world_size)
        return 1
    return max(1, launcher_global_world_size() // local_world_size)


def get_node_rank() -> int:
    """Rank of the current node, or 0 if not distributed (launcher-derived, see :func:`get_num_nodes`)."""
    local_world_size = get_local_world_size()
    if local_world_size <= 0:
        _warn_bad_local_world_size(local_world_size)
        return 0
    return launcher_global_rank() // local_world_size


def is_multi_rank_run() -> bool:
    """Whether a collective would actually reach a peer: an initialized group of more than one rank.

    Guards code that computes something locally and then gathers it: a single-rank run has nothing
    to gather, and an uninitialized one would raise inside the collective.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_global_main_process() -> bool:
    """Whether this is the global main process (rank 0)."""
    return get_global_rank() == 0


def is_local_main_process() -> bool:
    """Whether this is the main process on its node (local rank 0)."""
    return get_local_rank() == 0


def fs_aware_save_rank() -> bool:
    """Whether this rank should write shared checkpoint files (index, config, weights).

    Shared output FS → only global rank 0 writes (avoids NFS races); non-shared → each node's
    local rank 0 writes its own copy.
    """
    return is_global_main_process() if is_output_shared_filesystem() else is_local_main_process()


def fs_aware_load_rank() -> bool:
    """Whether this rank should perform shared read-side work (model/dataset download, cache fill).

    Shared input FS → only global rank 0 fetches; non-shared → each node's local rank 0 fetches
    its own copy.
    """
    return is_global_main_process() if is_input_shared_filesystem() else is_local_main_process()


def barrier(group: dist.ProcessGroup | None = None):
    """Synchronize all processes in ``group`` (default group if None)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)


@contextmanager
def barrier_on_exit(group: dist.ProcessGroup | None = None):
    """Run a save-rank-only write, then barrier — **even if the write raised**.

    One rank writes while every rank must reach the barrier, so an unfenced writer turns a local I/O
    failure (ENOSPC, EIO, a stale NFS handle) into a job-wide hang whose traceback names the barrier
    rather than the disk. The writer's exception still propagates; this only releases the peers first.
    """
    try:
        yield
    finally:
        barrier(group)


def nccl_safe_broadcast(tensor: torch.Tensor, src: int, group: dist.ProcessGroup | None = None) -> None:
    """In-place ``dist.broadcast`` that routes NCCL-unsupported dtypes through a uint8 bit view.

    The view shares storage, so the receiver's tensor is written through it, and the bit
    reinterpret round-trips exactly (``-1`` sentinels survive). A non-contiguous tensor of an
    unsupported dtype fails loud in ``Tensor.view`` rather than silently broadcasting a copy.
    """
    wire = tensor.view(torch.uint8) if tensor.dtype in _NCCL_UNSUPPORTED_DTYPES else tensor
    dist.broadcast(wire, src=src, group=group)


def broadcast_from_rank0(value):
    """Return rank 0's ``value`` on every rank; unchanged when not distributed / world size 1.

    Use where a per-rank-derived value (strftime output_dir, run id, a checkpoint decision)
    must agree across ranks to avoid split-brain checkpoints/logs/collective hangs.
    """
    if is_multi_rank_run():
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    return value


def raise_gathered_reasons(reasons: list[str | None], what: str, exc_type: type[Exception]) -> None:
    """Uniform raise from a world-gathered reason list — shared tail of the two reject transports."""
    failed = [(rank, reason) for rank, reason in enumerate(reasons) if reason]
    if failed:
        rank, reason = failed[0]
        raise exc_type(f"{what} failed on {len(failed)} of {len(reasons)} rank(s). First (rank {rank}): {reason}")


def reject_across_ranks(local_reason: str | None, what: str, exc_type: type[Exception] = RuntimeError) -> None:
    """Raise on EVERY rank when any rank reports a reason. COLLECTIVE — every rank must call it.

    The seam for a rank-local verdict (one rank's filesystem, one rank's dataset map, one stage's key
    set) sitting between collectives: a rank raising alone leaves its peers in the next collective
    with no diagnostic, and its own traceback never prints because teardown blocks there too. The
    reason travels, so every rank reports the real cause rather than a timeout, and ``exc_type``
    keeps the caller's error contract (a gate documenting ``ValueError`` still raises ``ValueError``).

    The gather is a collective, bounded by the process-group timeout: where the work preceding the
    join is unbounded single-rank time, use
    :func:`~src.distributed.filesystem.store_reject_across_ranks` instead.
    """
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    reasons: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reasons, local_reason)
    raise_gathered_reasons(reasons, what, exc_type)


def divergent_settings(gathered: list[dict[str, object]]) -> dict[str, list[str]]:
    """Setting names whose value is not identical across the per-rank mappings, as sorted spellings.

    The whole verdict of :func:`reject_divergent_settings`, pure so it is testable without a process
    group. A name absent from one rank's mapping is a divergence like any other.
    """
    if not gathered:
        return {}
    reference = gathered[0]
    names = {name for rank_values in gathered for name in rank_values}
    return {
        name: sorted({str(rank_values.get(name)) for rank_values in gathered})
        for name in sorted(names)
        if any(rank_values.get(name) != reference.get(name) for rank_values in gathered)
    }


def reject_divergent_settings(values: dict[str, object], what: str, guidance: str) -> None:
    """Raise on EVERY rank when the ranks disagree on a RESOLVED setting. COLLECTIVE.

    The seam for env-derived knobs that decide how many collectives a rank runs, or the wire
    parameters both ends of an all-to-all share: a partial per-node rollout is a realistic mistake at
    512 GPUs whose symptom is a hang naming nothing.

    ``values`` are RESOLVED, never raw strings: an absent variable and one set to its own default
    mean the same thing, so comparing strings would reject a launcher exporting a default on the
    head node only. ``guidance`` is appended verbatim so the caller says what to align.
    """
    if not is_multi_rank_run():
        return
    gathered: list[dict[str, object]] = [{} for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    differing = divergent_settings(gathered)
    if differing:
        raise ValueError(
            f"{what} differs across ranks: {differing}. {guidance} Set them identically on every "
            f"node (rank 0 has {dict(gathered[0])})."
        )


def rank_consensus(local_ok: bool) -> tuple[bool, bool]:
    """``(all_ok, any_ok)`` across ranks in one SUM all-reduce. COLLECTIVE — every rank must call it.

    The seam for an all-or-nothing decision taken from a per-rank observation (a checkpoint file
    present on some nodes only): such a decision gates collectives, so a split verdict leaves half
    the world in a DTensor gather the other half never enters. Unlike :func:`reject_across_ranks`
    it does not raise — the caller chooses what a partial result means.
    """
    if is_multi_rank_run():
        count = torch.tensor([1 if local_ok else 0], device=collective_device())
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        n = int(count.item())
        return n == dist.get_world_size(), n > 0
    return local_ok, local_ok


class DeferredRankFailure:
    """Defer a rank-local failure to the next collective, so the ranks after it are not left hanging.

    For a write INTERLEAVED with collectives (a checkpoint streaming to disk between per-layer
    gathers), where :func:`barrier_on_exit` fences one that precedes a barrier: raising at layer *k*
    would leave the peers in layer *k+1*'s all-gather until the watchdog fires. Wrap each local step
    in :meth:`run` and close the region with :meth:`reject` — the collectives still run everywhere.

    Once a step has failed the rest are skipped: they would write to the filesystem that just failed,
    and the first reason is the diagnostic one.
    """

    def __init__(self, what: str, exc_type: type[Exception] = RuntimeError) -> None:
        self.what = what
        self.exc_type = exc_type
        self.reason: str | None = None

    def run(self, step: Callable[[], Any]) -> Any:
        """Run a rank-local ``step``, recording rather than raising. Returns None if it failed or was skipped."""
        if self.reason is not None:
            return None
        try:
            return step()
        except Exception as exc:  # any local failure must reach the peers as a reason
            # Only the message survives to :meth:`reject`; a genuine bug needs its traceback here.
            logger.exception("%s failed on this rank; deferring to the next collective", self.what)
            self.reason = f"{type(exc).__name__}: {exc}"
            return None

    def reject(self) -> None:
        """COLLECTIVE — every rank must call it. Raise on all ranks if any recorded a failure."""
        reject_across_ranks(self.reason, self.what, exc_type=self.exc_type)


def _shared_filesystem_flag(var: str, default: bool) -> bool:
    """One shared-filesystem flag: the rank-agreed value once consensus has run, else this rank's env."""
    if _SHARED_FILESYSTEM_CONSENSUS is not None:
        return _SHARED_FILESYSTEM_CONSENSUS[var]
    return env_flag(var, default=default)


def is_shared_filesystem() -> bool:
    """Umbrella shared-filesystem flag, from DIST_SHARED_FILESYSTEM (default "1"=shared).

    The default both sides fall back to; the coordination helpers read
    :func:`is_input_shared_filesystem` / :func:`is_output_shared_filesystem`. Answers the rank-agreed
    value once :func:`resolve_shared_filesystem_consensus` has run, else this rank's raw env. Never
    a collective itself — callers like :func:`fs_aware_save_rank` run inside rank-gated branches.
    """
    return _shared_filesystem_flag(_SHARED_FILESYSTEM_VAR, True)


def is_input_shared_filesystem() -> bool:
    """Whether the read side (model/dataset downloads, dataset map/pack, HF caches) is shared.

    ``DIST_INPUT_SHARED_FILESYSTEM``, falling back to the ``DIST_SHARED_FILESYSTEM`` umbrella. Picks
    the coordination scope of :func:`fs_aware_main_first` and the rank of :func:`fs_aware_load_rank`.
    """
    return _shared_filesystem_flag(_INPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def is_output_shared_filesystem() -> bool:
    """Whether the write side (checkpoints, run.log, dumped artifacts) is on a shared filesystem.

    ``DIST_OUTPUT_SHARED_FILESYSTEM``, falling back to the ``DIST_SHARED_FILESYSTEM`` umbrella. Picks
    the rank of :func:`fs_aware_save_rank`.
    """
    return _shared_filesystem_flag(_OUTPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def _env_shared_filesystem_flags() -> dict[str, bool]:
    """This rank's own three shared-filesystem flags, read straight from the environment.

    Bypasses the memo deliberately: it is the INPUT to the consensus, and reading it through the
    getters would return whatever was agreed last time.
    """
    umbrella = env_flag(_SHARED_FILESYSTEM_VAR, default=True)
    return {
        _SHARED_FILESYSTEM_VAR: umbrella,
        _INPUT_SHARED_FILESYSTEM_VAR: env_flag(_INPUT_SHARED_FILESYSTEM_VAR, default=umbrella),
        _OUTPUT_SHARED_FILESYSTEM_VAR: env_flag(_OUTPUT_SHARED_FILESYSTEM_VAR, default=umbrella),
    }


def resolve_shared_filesystem_consensus() -> dict[str, bool]:
    """Agree the shared-filesystem flags across ranks once and memoize them. COLLECTIVE.

    The flags pick the coordination SCOPE — world-wide vs per-node — so a per-rank divergence (a
    heterogeneous ``--env-file``, per-node env injection) would split one tag's participants across
    two scopes where each waits out the full store timeout. All three are agreed, not just the
    umbrella: a side var that IS set wins over it. Rank 0's values win, loudly.

    The memo is written ONCE, at the end: a concurrent reader (a dataloader worker, the profiler
    thread) must never observe a half-resolved state.
    """
    global _SHARED_FILESYSTEM_CONSENSUS
    local = _env_shared_filesystem_flags()
    agreed = broadcast_from_rank0(local)
    disagreed = [var for var, value in agreed.items() if local[var] != value]
    if disagreed:
        logger.warning(
            "Shared-filesystem flags disagree across ranks: this rank resolved %s but global rank 0 "
            "resolved %s, and rank 0's values win. Every rank must see the same values — fix the "
            "launch env (a per-node override is the usual cause) or main-first coordination scopes "
            "will not match what this node's filesystem actually does.",
            {var: local[var] for var in disagreed},
            {var: agreed[var] for var in disagreed},
        )
    _SHARED_FILESYSTEM_CONSENSUS = agreed
    # Getting it wrong is silent: an umbrella left shared on per-node local disks makes every node
    # recompute the dataset — correct output, N× the CPU and disk, no error.
    if is_global_main_process():
        logger.info(
            "Shared-filesystem scope: %s (set DIST_SHARED_FILESYSTEM=0 for per-node local storage).",
            {var: ("shared" if value else "per-node") for var, value in agreed.items()},
        )
    return agreed


def ensure_shared_filesystem_consensus() -> None:
    """Agree the flags if no consensus has been taken yet. COLLECTIVE on that first call only.

    The seam for a coordinated phase that can precede ``init_distributed`` (a hub-metadata read
    naming the run); it must not re-broadcast once the scope is settled.
    """
    if _SHARED_FILESYSTEM_CONSENSUS is None:
        resolve_shared_filesystem_consensus()


def reset_shared_filesystem_consensus() -> None:
    """Test-only seam: drop the memoized consensus so the flags re-read this rank's environment.

    The flags are agreed ONCE per process, so a test suite that varies ``DIST_*_SHARED_FILESYSTEM``
    would otherwise keep asserting against the scope an earlier case agreed — silently, since every
    getter still answers. Never for training code: a job's coordination scope is fixed for the run.
    """
    global _SHARED_FILESYSTEM_CONSENSUS
    _SHARED_FILESYSTEM_CONSENSUS = None


def move_model_to_local_device(model: torch.nn.Module) -> torch.nn.Module:
    """Move ``model`` to this rank's CUDA device (or CPU) and free the stale CPU copy."""
    model = model.to(current_device())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def log_global_load_duration_seconds(
    *,
    tag: str,
    method: str,
    t_start_wall: float,
    t_end_wall: float,
) -> float:
    """Global load span ``max(t_end) - min(t_start)`` across ranks, each passing its interval.

    Collective — all ranks must call this so the ``all_reduce`` completes. Wall-clock times
    are only comparable with NTP-synced clocks. Returns the global span (same on every rank).
    """
    local_sec = t_end_wall - t_start_wall
    if not is_multi_rank_run():
        logger.info(
            f"[{tag}] model load timing: method={method} global_load_duration_sec={local_sec:.3f} (single process)"
        )
        return local_sec

    ws = dist.get_world_size()
    device = collective_device()

    st = torch.tensor([t_start_wall], dtype=torch.float64, device=device)
    en = torch.tensor([t_end_wall], dtype=torch.float64, device=device)
    dist.all_reduce(st, op=dist.ReduceOp.MIN)
    dist.all_reduce(en, op=dist.ReduceOp.MAX)
    # No barrier: reading ``en`` already blocked until every rank contributed to the all_reduce.
    global_sec = max(0.0, en.item() - st.item())

    if is_global_main_process():
        logger.info(
            f"[{tag}] model load timing: method={method} global_load_duration_sec={global_sec:.3f} "
            f"(world_size={ws} wall span min_start→max_end across ranks)"
        )
    return global_sec


def get_nccl_timeout() -> datetime.timedelta:
    """NCCL collective-watchdog timeout as a ``timedelta``.

    ``DIST_NCCL_TIMEOUT_MINUTES`` env → ``DEFAULT_NCCL_TIMEOUT_MINUTES`` (30). Only honoured via the
    ``timeout=`` kwarg of ``init_process_group`` (the watchdog reads nothing from the environment).
    """
    return datetime.timedelta(minutes=resolve_nccl_timeout_minutes())


def get_store_timeout() -> datetime.timedelta:
    """Wall-clock bound for c10d-store coordination waits, as a ``timedelta``.

    ``DIST_STORE_TIMEOUT_HOURS`` env → ``DEFAULT_STORE_TIMEOUT_HOURS`` (4). Deliberately hours-scale
    and independent of ``DIST_NCCL_TIMEOUT_MINUTES``: these waits bound one rank's download/packing
    work, not a collective, so the NCCL watchdog scale does not apply.
    """
    return datetime.timedelta(hours=resolve_store_timeout_hours())


def _world_group_timeout() -> datetime.timedelta | None:
    """Current NCCL watchdog timeout of the (already-initialized) world group, or None if
    unreadable (private torch internals — same access tier as ``c10d._set_pg_timeout``)."""
    try:
        backend = dist.group.WORLD._get_backend(torch.device("cuda"))
        return backend.options._timeout
    except Exception:  # introspection only; the caller degrades to setting anyway
        return None


def apply_default_pg_timeout(timeout: datetime.timedelta) -> None:
    """Make ``timeout`` the default NCCL watchdog for every *subsequent* ``new_group``.

    ``init_process_group(timeout=...)`` sets the world group's timeout only; ``new_group`` falls
    back to PyTorch's 10 min, too short for EP/CP/TP subgroups at 100B+/cross-node scale. Reassigning
    ``distributed_c10d.default_pg_nccl_timeout`` before any subgroup is built makes them inherit it —
    chiefly ``init_device_mesh`` (DP/HSDP/TP axes), which takes no timeout kwarg.

    The name is READ before it is written, and its absence RAISES: assigning an attribute torch no
    longer defines would leave every mesh subgroup on the 10-minute default with nothing to show.
    """
    if not hasattr(c10d, "default_pg_nccl_timeout"):
        raise RuntimeError(
            "torch.distributed.distributed_c10d.default_pg_nccl_timeout is absent on this PyTorch "
            f"build ({torch.__version__}), so DIST_NCCL_TIMEOUT_MINUTES cannot reach the subgroups: "
            "every device-mesh (DP/HSDP/TP) and EP/CP group would fall back to the 10 min NCCL "
            "watchdog and abort long cross-node collectives (a gathered save at 100B+). Point the "
            "pin in src/distributed/runtime.py at the renamed symbol."
        )
    c10d.default_pg_nccl_timeout = timeout

    # The pin reaches only *future* new_group calls, so extend an existing world group too.
    # Extend-only: shortening one would resurface the straggler aborts.
    if not dist.is_initialized():
        return
    current = _world_group_timeout()
    if current is None:
        # Setting blindly could SHORTEN an external watchdog (accelerate ddp_timeout).
        if is_global_main_process():
            logger.warning(
                "Could not read the world group's current NCCL timeout; skipping the retroactive "
                f"timeout update to {timeout} (future subgroups still inherit it)."
            )
    elif current >= timeout:
        if is_global_main_process():
            logger.info(f"World process group already has a >= timeout ({current}); not shortening to {timeout}.")
    else:
        # Best-effort and private: the world group already carries init_process_group's timeout, so
        # a rename here costs the extension, not the pin above.
        try:
            c10d._set_pg_timeout(timeout)
        except Exception:
            logger.warning("Could not extend the world group's NCCL timeout (non-fatal)", exc_info=True)


def require_rendezvous_env() -> None:
    """Fail loud when the launcher declares a multi-rank world but no ``env://`` rendezvous.

    A bare ``srun --ntasks-per-node=8`` sets ``SLURM_PROCID``/``SLURM_NTASKS`` and nothing else,
    where c10d's own "environment variable MASTER_ADDR expected" names neither launcher nor fix.
    """
    missing = [var for var in ("MASTER_ADDR", "MASTER_PORT") if not os.environ.get(var)]
    if not missing:
        return
    raise RuntimeError(
        f"The launcher declares world_size={launcher_global_world_size()} but {missing} is unset, so "
        f"no process group can be built. Bare 'srun --ntasks-per-node=N' provides no env:// "
        f"rendezvous: launch one torchrun per node instead — 'srun --ntasks-per-node=1 torchrun "
        f"--nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=N "
        f"--master_addr=<head> --master_port=<port> <script> <config>' — or export MASTER_ADDR / "
        f"MASTER_PORT identically on every task (see agent-docs/parallelism/launch-recipes.md)."
    )


def init_distributed(backend: str = "nccl") -> bool:
    """Initialize the default process group for a torchrun/SLURM launch.

    Single entry point for every training script: it passes ``timeout=get_nccl_timeout()`` (the only
    mechanism extending PyTorch's 10-min watchdog), pins the same timeout for later ``new_group``
    so EP/CP/TP subgroups inherit it, and eagerly binds this rank's CUDA device via ``device_id=``.

    Runs whenever the launcher declares a rank or a multi-rank world, which includes an
    ``accelerate launch``: building the group here rather than in ``PartialState`` is what applies
    those two. A single process with no launcher vars is the only no-op.

    Returns whether it initialized; an already-initialized group still gets the subgroup timeout
    pin and the shared-filesystem consensus.
    """
    timeout = get_nccl_timeout()
    # A multi-rank world counts even without RANK: that is exactly the bare-srun shape, where
    # skipping init leaves every rank at world 1, "main", and writing the same output_dir.
    world = launcher_global_world_size()
    if ("RANK" not in os.environ and world <= 1) or dist.is_initialized():
        if dist.is_initialized():
            apply_default_pg_timeout(timeout)
            resolve_shared_filesystem_consensus()
        return False
    if world > 1:
        require_rendezvous_env()

    kwargs = {"backend": backend, "timeout": timeout}
    if "RANK" not in os.environ:
        # Bare srun declares the world through SLURM_* alone, while c10d's env:// handler reads
        # RANK / WORLD_SIZE itself and raises on their absence.
        kwargs.update(rank=launcher_global_rank(), world_size=world)
    if backend == "nccl" and torch.cuda.is_available():
        # Launcher's rank, not is_global_main_process(): pre-init the latter is 0 on every rank.
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1" and launcher_global_rank() == 0:
            logger.warning(
                "CUDA_DEVICE_MAX_CONNECTIONS != 1 (currently %r). Expert Parallelism that forms "
                "more than one DeepEP dispatch group per NVLink domain (ep_group_size < "
                "nvlink_domain_size) may deadlock the combine barrier against FSDP2's DP-wide "
                "collectives. Set it in the process environment (the prebuilt image already "
                "does); a Python-level override here is too late (driver init at import).",
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            )
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        kwargs["device_id"] = torch.device("cuda", local_rank)

    dist.init_process_group(**kwargs)
    apply_default_pg_timeout(timeout)
    # The flag picks the coordination scope, so it must be one value for the whole job.
    resolve_shared_filesystem_consensus()
    if is_global_main_process():
        logger.info("Initialized distributed: backend=%s timeout=%s world_size=%d", backend, kwargs["timeout"], world)
    return True


def materialize_dtensor(data: torch.Tensor | None) -> torch.Tensor | None:
    """Materialize a DTensor shard as its full tensor; pass ``None``/plain tensors through.

    ``full_tensor()`` is a COLLECTIVE across the device mesh — every rank holding a shard must call
    it — and autograd-aware, so unlike :func:`resolve_param_tensor` it preserves gradient.
    """
    if isinstance(data, DTensor):
        return data.full_tensor()
    return data


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's shard of a DTensor; plain tensors pass through.

    Purely local, unlike its neighbours here — no collective. The custom optimizers step through it
    because ``view(-1)`` and their Triton kernels reject a DTensor.
    """
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def local_numel(param: torch.Tensor) -> int:
    """Element count held BY THIS RANK (``DTensor.numel()`` reports the global size).

    Mixing this with a plain ``numel()`` across a model sums two scopes: FSDP2/TP params are
    DTensors (global), EP expert params are FSDP-ignored plain tensors (already per-rank).
    """
    return to_local(param).numel()


def resolve_param_tensor(param_data: torch.Tensor) -> torch.Tensor:
    """Resolve a parameter tensor to a plain CPU tensor.

    DTensor params reconstruct via ``full_tensor()`` — a COLLECTIVE, so every rank of the mesh must
    call it for the same param; plain tensors just detach to CPU.
    """
    if isinstance(param_data, DTensor):
        return param_data.full_tensor().cpu()
    return param_data.detach().cpu()


def fs_aware_makedirs(path: str, writer_rank: Callable[[], bool] = fs_aware_save_rank) -> None:
    """Create a directory on the writer rank, then barrier — shared FS: global rank 0; per-node:
    each node's local rank 0.

    ``writer_rank`` picks the side: :func:`fs_aware_save_rank` (the default — checkpoints, adapters,
    dumps) or :func:`fs_aware_load_rank` for a read-side cache dir, which follows
    ``DIST_INPUT_SHARED_FILESYSTEM`` instead. The barrier is in a ``finally`` for
    :func:`barrier_on_exit`'s reason: a writer dying in ``makedirs`` must not hang its peers.
    """
    try:
        if writer_rank():
            os.makedirs(path, exist_ok=True)
    finally:
        barrier()


def rank_tag() -> str:
    """``rankNN`` tag for the current global rank, zero-padded to world-size width."""
    rank = get_global_rank()
    width = max(2, len(str(max(0, get_global_world_size() - 1))))
    return f"rank{rank:0{width}d}"
