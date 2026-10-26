"""Shared-vs-per-node filesystem scope: the ``DIST_*_SHARED_FILESYSTEM`` flags and their rank
consensus, the save/load ranks they pick, main-rank-first ordering for read-side work, and the probe of
the declared output sharing."""

import datetime
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch.distributed as dist

from halo.env import env_flag, store_timeout
from halo.runtime.collectives import barrier, broadcast_from_rank0
from halo.runtime.consensus import (
    StorePhase,
    agree_probe_across_ranks,
    rank_consensus,
    reject_across_ranks,
    store_join_recorded_failure,
)
from halo.runtime.ranks import (
    get_global_rank,
    get_global_world_size,
    get_local_rank,
    get_local_world_size,
    get_node_rank,
    get_num_nodes,
    has_process_group,
    is_global_main_process,
    is_local_main_process,
)

logger = logging.getLogger(__name__)

# Each side falls back to the umbrella while unset; a side var that is set wins over it.
_SHARED_FILESYSTEM_VAR = "DIST_SHARED_FILESYSTEM"
_INPUT_SHARED_FILESYSTEM_VAR = "DIST_INPUT_SHARED_FILESYSTEM"
_OUTPUT_SHARED_FILESYSTEM_VAR = "DIST_OUTPUT_SHARED_FILESYSTEM"

# Far below the store default: a hub-metadata wait this long means the ranks disagree on the phase.
_HUB_METADATA_TIMEOUT = datetime.timedelta(minutes=30)

# Sentinel the output-filesystem probe writes under output_dir; dot-prefixed and inert if one leaks.
OUTPUT_FS_PROBE_PREFIX = ".halo_fs_probe_"

# NFS's cached negative lookup can hide the sentinel up to ``acdirmax`` (60 s); only a wrongly declared
# run waits the whole budget out.
_OUTPUT_FS_PROBE_TIMEOUT_S = 60.0
_OUTPUT_FS_PROBE_INTERVAL_S = 0.25

_shared_filesystem_consensus: dict[str, bool] | None = None

# An entry script probes before resume detection and every trainer again; the repeat must cost nothing.
_probed_output_dirs: set[str] = set()


def _shared_filesystem_flag(var: str, default: bool) -> bool:
    if _shared_filesystem_consensus is not None:
        return _shared_filesystem_consensus[var]
    return env_flag(var, default=default)


def is_shared_filesystem() -> bool:
    """The ``DIST_SHARED_FILESYSTEM`` umbrella (default shared): the rank-agreed value once
    :func:`resolve_shared_filesystem_consensus` has run, else this rank's env. Never a collective."""
    return _shared_filesystem_flag(_SHARED_FILESYSTEM_VAR, True)


def is_input_shared_filesystem() -> bool:
    """Whether the read side (downloads, dataset map/pack, HF caches) is shared:
    ``DIST_INPUT_SHARED_FILESYSTEM``, else the umbrella."""
    return _shared_filesystem_flag(_INPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def is_output_shared_filesystem() -> bool:
    """Whether the write side (checkpoints, run.log, dumps) is shared: ``DIST_OUTPUT_SHARED_FILESYSTEM``,
    else the umbrella."""
    return _shared_filesystem_flag(_OUTPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def resolve_shared_filesystem_consensus() -> None:
    """COLLECTIVE: agree the three flags across ranks, rank 0's values winning with a warning, and memoize
    them once, at the end, so no reader sees a half-resolved state. The flags pick the coordination
    scope, so ranks that disagree would wait in different scopes."""
    global _shared_filesystem_consensus
    # Read from the environment, bypassing the memo: this rank's flags are the consensus input.
    umbrella = env_flag(_SHARED_FILESYSTEM_VAR, default=True)
    local = {
        _SHARED_FILESYSTEM_VAR: umbrella,
        _INPUT_SHARED_FILESYSTEM_VAR: env_flag(_INPUT_SHARED_FILESYSTEM_VAR, default=umbrella),
        _OUTPUT_SHARED_FILESYSTEM_VAR: env_flag(_OUTPUT_SHARED_FILESYSTEM_VAR, default=umbrella),
    }
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
    _shared_filesystem_consensus = agreed
    # Getting it wrong is silent: an umbrella left shared on per-node local disks makes every node
    # recompute the dataset — correct output, N× the CPU and disk, no error.
    if is_global_main_process():
        logger.info(
            "Shared-filesystem scope: %s (set DIST_SHARED_FILESYSTEM=0 for per-node local storage).",
            {var: ("shared" if value else "per-node") for var, value in agreed.items()},
        )


def is_fs_save_rank() -> bool:
    """Whether this rank writes shared files: global rank 0 on a shared output filesystem, else each
    node's local rank 0."""
    return is_global_main_process() if is_output_shared_filesystem() else is_local_main_process()


def is_fs_load_rank() -> bool:
    """Whether this rank does shared read-side work (downloads, cache fills): global rank 0 on a shared
    input filesystem, else each node's local rank 0."""
    return is_global_main_process() if is_input_shared_filesystem() else is_local_main_process()


def fs_aware_makedirs(path: str, writer_rank: Callable[[], bool] = is_fs_save_rank) -> None:
    """COLLECTIVE: create ``path`` on ``writer_rank`` (:func:`is_fs_load_rank` for a read-side cache dir),
    then barrier, even when ``makedirs`` raised."""
    try:
        if writer_rank():
            os.makedirs(path, exist_ok=True)
    finally:
        barrier()


@contextmanager
def fs_aware_main_first(tag: str, timeout: datetime.timedelta | None = None):
    """Run the body on the main rank first, then on every other rank, which reuses what it cached.

    The scope follows the input filesystem: global rank 0 leads the world, or each node's local rank 0
    its node. Waiters block on a store key, so the body must issue no collective (``fs_aware_makedirs``
    barriers). ``tag`` names the call site under the :class:`StorePhase` invariant.
    """
    if not has_process_group():
        yield
        return

    if is_input_shared_filesystem():
        scope, participant, num = "shared", get_global_rank(), get_global_world_size()
        is_main = is_global_main_process()
    else:
        scope, participant, num = f"node{get_node_rank()}", get_local_rank(), get_local_world_size()
        is_main = is_local_main_process()
    phase = StorePhase(f"main_first/{tag}", scope, participant, num, timeout or store_timeout())

    try:
        if is_main:
            try:
                yield
            finally:
                phase.set("done")
        else:
            phase.wait(["done"])
            yield
    finally:
        phase.finish(["done"])


def hub_metadata_main_first(tag: str, fetch: Callable[[], Any]) -> Any:
    """Run a small hub metadata read (``AutoConfig``, ``AutoTokenizer``, ...) main-rank-first and return
    its result, under a short bound so a tag entered from a rank-dependent branch fails by name. It can
    be the run's first coordinated phase, before ``init_distributed``, so it takes the shared-filesystem
    consensus when none has been taken yet."""
    if _shared_filesystem_consensus is None:
        resolve_shared_filesystem_consensus()
    with fs_aware_main_first(f"hub_meta/{tag}", timeout=_HUB_METADATA_TIMEOUT):
        return fetch()


def output_filesystem_contradiction(declared_shared: bool, seen: list[bool]) -> str | None:
    """Why the declared output sharing contradicts what the ranks saw, or None when it holds.

    ``seen[rank]`` is whether that rank saw the sentinel global rank 0 wrote under ``output_dir``.
    """
    if not seen:
        return None
    if not seen[0]:
        return (
            "Output-filesystem probe: global rank 0 could not read back the sentinel it just wrote "
            "under output_dir. The output directory is not durably readable by the process that "
            "writes checkpoints — check the mount, its permissions and its free space."
        )
    observed_shared = all(seen)
    if declared_shared == observed_shared:
        return None
    if declared_shared:
        blind = [rank for rank, ok in enumerate(seen) if not ok]
        return (
            f"Output filesystem is declared SHARED but {len(blind)} of {len(seen)} ranks (first: "
            f"rank {blind[0]}) cannot see a file global rank 0 wrote under output_dir. Only rank 0 "
            f"would write trainer_state.json / scheduler.pt / rng_state, so no checkpoint would resume "
            f"on the other nodes. Set "
            f"DIST_OUTPUT_SHARED_FILESYSTEM=0 (or DIST_SHARED_FILESYSTEM=0 for both sides) so each "
            f"node writes its own copy, or point output_dir at the shared mount."
        )
    return (
        f"Output filesystem is declared PER-NODE but all {len(seen)} ranks can see a file global "
        f"rank 0 wrote under output_dir, so it is shared. Every node's local rank 0 would write the "
        f"same checkpoint paths concurrently. Set DIST_OUTPUT_SHARED_FILESYSTEM=1 (or "
        f"DIST_SHARED_FILESYSTEM=1 for both sides), or give each node its own output_dir."
    )


def _visible_within(path: str, seconds: float) -> bool:
    """Whether ``path`` shows up within ``seconds``, polling past NFS/Lustre attribute caching."""
    deadline = time.monotonic() + seconds
    while True:
        if os.path.exists(path):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_OUTPUT_FS_PROBE_INTERVAL_S)


def verify_output_filesystem_sharing(output_dir: str) -> None:
    """COLLECTIVE: probe whether ``output_dir`` really is shared across nodes and raise on a declaration
    that contradicts it, once per directory per process. Both wrong declarations are otherwise
    silent: shared on per-node disks leaves the other nodes no resumable checkpoint, per-node on a
    shared mount races every node's writer on the same paths."""
    world = get_global_world_size()
    if world <= 1 or get_local_world_size() >= world or not output_dir:
        return
    probed = os.path.abspath(output_dir)
    if probed in _probed_output_dirs:
        return
    declared_shared = is_output_shared_filesystem()

    # Broadcast so every rank polls the same name — a per-rank one would probe nothing.
    sentinel = os.path.join(output_dir, OUTPUT_FS_PROBE_PREFIX + broadcast_from_rank0(f"{time.time_ns():x}"))
    write_failure = None
    if is_global_main_process():
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(sentinel, "w") as handle:
                handle.write("halo")
        except OSError as exc:
            write_failure = f"could not write {sentinel}: {exc}"
    # Joins rank 0's write failure AND orders every peer's poll after the write.
    reject_across_ranks(write_failure, "output-filesystem probe", exc_type=OSError)

    # Only a rank that EXPECTS to see the sentinel polls for it, ruling out a cached negative lookup;
    # under a per-node declaration seeing the file at all is already proof, and not seeing it agrees.
    seen: list[bool] = [False] * world
    dist.all_gather_object(seen, _visible_within(sentinel, _OUTPUT_FS_PROBE_TIMEOUT_S if declared_shared else 0.0))
    if is_global_main_process():
        try:
            os.remove(sentinel)
        except OSError:
            logger.warning("Could not remove the output-filesystem probe sentinel %s (inert).", sentinel)

    reason = output_filesystem_contradiction(declared_shared, seen)
    if reason:
        raise RuntimeError(reason)
    _probed_output_dirs.add(probed)
    if is_global_main_process():
        logger.info(
            "Output filesystem probed across %d nodes: %s, matching the declared flags.",
            get_num_nodes(),
            "shared" if all(seen) else "per-node",
        )


def agree_input_probe_across_ranks(probe: Callable[[], bool], subject, probe_name: str) -> bool:
    """COLLECTIVE: :func:`agree_probe_across_ranks` for a probe of the run's input, run only on each input
    scope's load rank (:func:`is_fs_load_rank`) while the rest abstain with ``False``. A probe that raises
    is joined over the store first, so its cause raises on every rank."""
    probing = is_fs_load_rank()
    local, failure = False, None
    if probing:
        try:
            local = probe()
        except BaseException as exc:  # joined below, so every rank raises with it
            failure = exc
    store_join_recorded_failure(f"input_probe/{probe_name}", failure, f"Probing {subject} ({probe_name})")
    if probing:
        return agree_probe_across_ranks(local, subject, probe_name)
    return rank_consensus(False).any_ok
