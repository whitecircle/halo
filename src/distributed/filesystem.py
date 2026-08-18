"""Filesystem-aware coordination: the c10d-store phase primitive, the store-carried rejection joins
built on it, main-rank-first ordering for read-side work, the shared-output-filesystem probe, and
the per-node load throttle.

Every wait here is bounded by wall clock (``DIST_STORE_TIMEOUT_HOURS``) rather than the NCCL
watchdog, because the work it covers is unbounded single-rank filesystem time — a snapshot download,
a whole-corpus pack — that a collective would turn into a watchdog abort on every peer.
"""

import datetime
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from src.distributed.runtime import (
    broadcast_from_rank0,
    ensure_shared_filesystem_consensus,
    get_global_rank,
    get_global_world_size,
    get_local_rank,
    get_local_world_size,
    get_node_rank,
    get_num_nodes,
    get_store_timeout,
    is_global_main_process,
    is_input_shared_filesystem,
    is_local_main_process,
    is_multi_rank_run,
    is_output_shared_filesystem,
    raise_gathered_reasons,
    reject_across_ranks,
)

logger = logging.getLogger(__name__)

# Ceiling on the ranks per node admitted to weight materialization at once when
# ``max_concurrent_loading`` is unset; an explicit setting is not capped by it.
MAX_CONCURRENT_LOADING_CAP = 4

# Far below the hours-scale store default: a hub-metadata phase moves a handful of small files, so a
# wait this long means the participants disagree on the phase rather than that the writer is busy.
_HUB_METADATA_TIMEOUT = datetime.timedelta(minutes=30)

# Sentinel the output-filesystem probe writes under output_dir; dot-prefixed and inert if one leaks.
OUTPUT_FS_PROBE_PREFIX = ".halo_fs_probe_"
# The run's own directory under ``output_dir``: run.log plus the decoded dataset samples.
RUN_LOG_DIR_NAME = "log"
# Poll budget for a rank waiting to SEE the sentinel: NFS's cached negative lookup can hide it for up
# to the directory attribute-cache ceiling (``acdirmax``, 60 s), and judging on the first stat would
# fail a healthy shared mount. Only a wrongly declared run ever pays the whole budget.
_OUTPUT_FS_PROBE_TIMEOUT_S = 60.0
_OUTPUT_FS_PROBE_INTERVAL_S = 0.25


class _StorePhase:
    """One collectively-ordered use ("phase") of a c10d-store coordination tag — self-cleaning.

    Each participant's own store-held entry counter supplies the phase number and namespaces the
    phase's keys, so repeated uses of one tag never see a previous use's keys even while its cleanup
    lags, with no module-level state. The LAST participant to leave (:meth:`finish`) deletes the
    transient keys, bounding a tag's store footprint at the per-participant counters.

    **Invariant: every participant of a scope must enter a given tag the same number of times, in
    the same order** — the phase number is each participant's private count, so one extra entry is
    permanently off-by-one thereafter, and only half of that reports itself (an off-by-one READER
    waits on a key nobody writes; a WRITER leaves a stale key that releases a later peer). Guarantee
    it at the call site: one tag per call site, never entered from a rank-dependent branch.
    """

    def __init__(self, tag: str, scope: str, participant: int, num_participants: int, timeout: datetime.timedelta):
        if not 0 <= participant < num_participants:
            # An UNDER-counted num_participants collects the phase's keys once that many ranks leave,
            # stranding the rest for the whole store timeout; this rank's index disproves it.
            raise RuntimeError(
                f"Store coordination '{tag}/{scope}' got participant index {participant} of only "
                f"{num_participants} participants, which cannot both be true. The participant count "
                f"is wrong — typically a launcher that does not set LOCAL_WORLD_SIZE on a host whose "
                f"visible CUDA device count is smaller than its rank count. Set LOCAL_WORLD_SIZE "
                f"explicitly (or pass gpus_per_node to ParallelismConfig)."
            )
        self._store = c10d._get_default_store()
        self._num = num_participants
        self._timeout = timeout
        self._label = f"{tag}/{scope}"
        root = f"_halo_phase/{tag}/{scope}"
        self._phase = self._store.add(f"{root}/entered{participant}", 1)
        self._prefix = f"{root}/p{self._phase}"

    def set(self, name: str, value: str = "1") -> None:
        self._store.set(f"{self._prefix}/{name}", value)

    def get_all(self, names) -> list[str]:
        """Values of phase keys this participant has already waited on, in ``names`` order (bytes
        decoded to str) — ONE store round-trip.

        The only reader: a key-per-participant loop is O(world²) sequential requests through the
        single store server (~262k for one join at world=512), and a pipeline runs several joins.
        """
        values = self._store.multi_get([f"{self._prefix}/{name}" for name in names])
        return [value.decode() if isinstance(value, bytes) else str(value) for value in values]

    def wait(self, names) -> None:
        keys = [f"{self._prefix}/{name}" for name in names]
        if not keys:
            return
        try:
            self._store.wait(keys, self._timeout)
        except Exception as e:
            raise RuntimeError(
                f"Store coordination '{self._label}' timed out after {self._timeout} waiting for "
                f"{keys} (this participant is in phase {self._phase} of {self._num}). Either the "
                f"peer that writes these keys is still working — raise DIST_STORE_TIMEOUT_HOURS — or "
                f"the participants disagree on how many times they entered this tag / on their "
                f"coordination scope, which leaves this one waiting on a key nobody will write."
            ) from e

    def finish(self, transient_names) -> None:
        """Mark this participant done with the phase; the last one deletes the phase's keys."""
        if self._store.add(f"{self._prefix}/exited", 1) == self._num:
            for name in (*transient_names, "exited"):
                self._store.delete_key(f"{self._prefix}/{name}")


def store_reject_across_ranks(
    tag: str,
    local_reason: str | None,
    what: str,
    exc_type: type[Exception] = RuntimeError,
    timeout: datetime.timedelta | None = None,
) -> None:
    """:func:`~src.distributed.runtime.reject_across_ranks` over the c10d store instead of a
    collective, for joins whose preceding work is unbounded single-rank wall-clock time.

    A fresh-cache dataset map or a first-run shard download that only some ranks pay would hold the
    peers inside NCCL/gloo for its whole duration and die at ``DIST_NCCL_TIMEOUT_MINUTES`` blaming
    the collective; here they wait on store keys bounded by ``DIST_STORE_TIMEOUT_HOURS``.

    World-scoped and collective-EQUIVALENT: every rank must call it with the same ``tag``\\ s in the
    same order (the :class:`_StorePhase` invariant). Same verdict and message contract.
    """
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    rank, world = get_global_rank(), get_global_world_size()
    reason_keys = [f"reason{i}" for i in range(world)]
    phase = _StorePhase(f"reject/{tag}", "world", rank, world, timeout or get_store_timeout())
    try:
        # "" is the no-failure sentinel; reject_across_ranks treats an empty reason as none too.
        phase.set(f"reason{rank}", local_reason or "")
        phase.wait(reason_keys)
        reasons: list[str | None] = [reason or None for reason in phase.get_all(reason_keys)]
    finally:
        # Keys survive until the LAST rank leaves (finish deletes only then), so every rank reads
        # them before any deletion; a raise below still releases this rank's exit slot.
        phase.finish(reason_keys)
    raise_gathered_reasons(reasons, what, exc_type)


def store_join_recorded_failure(tag: str, failure: BaseException | None, what: str) -> None:
    """Join a RECORDED rank-local failure across the world over the store.

    The seam for "one rank ran work the others depend on": the failing rank re-raises its OWN
    exception, keeping the type and traceback callers rely on, while every other rank takes the
    uniform ``RuntimeError`` naming it. The uniform rejection is chained under the original so a
    transport failure of the join itself is not swallowed. Same tag rules as the join above.
    """
    if failure is None:
        store_reject_across_ranks(tag, None, what)
        return
    try:
        store_reject_across_ranks(tag, f"{type(failure).__name__}: {failure}", what)
    except RuntimeError as uniform:
        raise failure from uniform
    raise failure


@contextmanager
def fs_aware_main_first(tag: str, timeout: datetime.timedelta | None = None):
    """Order the body main-rank-first: the main rank runs it alone, then everyone else runs it.

    **Every rank runs the body** — the main rank goes first and populates a cache (a hub snapshot, a
    packed arrow file), the peers then run the same code and hit what it left behind. Scope follows
    the INPUT filesystem: shared → global rank 0 leads the world, per-node → each node's local rank 0
    leads its own node and the nodes proceed independently.

    Waiters block on a store key rather than in a collective, because the body is unbounded
    single-rank work. The body must therefore issue NO collective itself, directly or through a
    helper (``fs_aware_makedirs`` barriers), or the main rank blocks alone until the NCCL watchdog
    fires. ``tag`` namespaces the call site, under the :class:`_StorePhase` equal-entry invariant.
    """
    if not (dist.is_available() and dist.is_initialized()):
        yield
        return

    if is_input_shared_filesystem():
        scope, participant, num = "shared", get_global_rank(), get_global_world_size()
        is_main = is_global_main_process()
    else:
        scope, participant, num = f"node{get_node_rank()}", get_local_rank(), get_local_world_size()
        is_main = is_local_main_process()
    phase = _StorePhase(f"main_first/{tag}", scope, participant, num, timeout or get_store_timeout())

    try:
        if is_main:
            try:
                yield
            finally:
                # Release even on failure — the exception aborts the job; waiters must not hang.
                phase.set("done")
        else:
            phase.wait(["done"])
            yield
    finally:
        phase.finish(["done"])


def hub_metadata_main_first(tag: str, fetch: Callable[[], Any]) -> Any:
    """Run a checkpoint-metadata read main-rank-first and return its result.

    The one seam for the small hub reads preceding the weight download — ``AutoConfig``,
    ``AutoProcessor``, ``AutoTokenizer`` — each cheap per rank and ruinous world-wide: at 512 ranks
    an uncoordinated fetch is 512 simultaneous hub requests (HTTP 429, then a fallback path that
    disagrees with the config answer), and under ``trust_remote_code`` the ranks also race to
    populate transformers' unlocked dynamic-module cache, where a peer can import a truncated module.

    Same call-site rules as :func:`fs_aware_main_first`. The wait is bounded to
    :data:`_HUB_METADATA_TIMEOUT`, so a tag entered from a branch that is not in fact rank-uniform
    fails by name instead of stalling for the hours-scale store default.
    """
    # These reads can precede ``init_distributed`` (the entry scripts probe the checkpoint's modality
    # to name the run), so this may be the run's FIRST coordinated phase — and its scope comes from
    # the shared-filesystem flags, which a per-node override would otherwise split in two.
    ensure_shared_filesystem_consensus()
    with fs_aware_main_first(f"hub_meta/{tag}", timeout=_HUB_METADATA_TIMEOUT):
        return fetch()


def output_filesystem_contradiction(declared_shared: bool, seen: list[bool]) -> str | None:
    """Why the declared output-FS sharing contradicts what the ranks observed, or None if it holds.

    ``seen[rank]`` is whether that rank could see the sentinel global rank 0 wrote under
    ``output_dir``. The whole decision of :func:`verify_output_filesystem_sharing`, pure so it is
    testable without a job, and identical on every rank because the gathered list is.

    Both contradictions corrupt a multi-node run silently: declared shared on per-node storage leaves
    nodes 1..N without ``trainer_state.json`` / ``scheduler.pt`` / ``rng_state`` (they resume at step
    0), declared per-node on shared storage has every node's rank 0 write the SAME paths at once.
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
            f"would write trainer_state.json / scheduler.pt / rng_state, so every other node "
            f"resumes at global_step=0 and the run desyncs. Set "
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
    """Whether ``path`` shows up within ``seconds``. Polls: NFS/Lustre attribute caching can hide a
    just-created file for a beat, and judging on the first ``stat`` would fail a healthy shared mount."""
    deadline = time.monotonic() + seconds
    while True:
        if os.path.exists(path):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_OUTPUT_FS_PROBE_INTERVAL_S)


def verify_output_filesystem_sharing(output_dir: str) -> None:
    """Probe whether ``output_dir`` really is shared and reject a declaration that contradicts it.

    COLLECTIVE — every rank must call it, unconditionally. ``DIST_SHARED_FILESYSTEM`` and its
    output-side override are pure declarations that nothing else checks against the filesystem, and
    both ways of getting them wrong are silent (:func:`output_filesystem_contradiction`). Only
    meaningful across nodes: one node's ranks share their mounts by construction.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    if world <= 1 or get_local_world_size() >= world or not output_dir:
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
    if is_global_main_process():
        logger.info(
            "Output filesystem probed across %d nodes: %s, matching the declared flags.",
            get_num_nodes(),
            "shared" if all(seen) else "per-node",
        )


def resolve_load_concurrency(max_concurrent: int | None, local_world_size: int) -> int:
    """Ranks per node admitted to weight materialization at once, resolved against the node's width.

    ``None`` derives from the node: half its width, capped at
    :data:`MAX_CONCURRENT_LOADING_CAP` — 4 on an 8-GPU node, 2 on a 4-GPU tray, where a flat
    4 would equal the node width and admit every rank at once (the CPU-RAM OOM this prevents).

    Every explicit value passes through untouched, ``0`` ("no throttle") and ``4`` included, which is
    why the unset default is ``None``: an in-band sentinel would make ``max_concurrent_loading: 4``
    on a 4-GPU tray mean the opposite of what it says.
    """
    if max_concurrent is not None:
        return max_concurrent
    return min(MAX_CONCURRENT_LOADING_CAP, max(1, local_world_size // 2))


@contextmanager
def sequential_load_within_node(tag: str = "model", max_concurrent: int | None = 1):
    """Throttle a node's local ranks in batches of ``max_concurrent`` (0=all, 1=sequential).

    For loading large models, where simultaneous CPU allocation by every rank would OOM. Coordinates
    over the store rather than NCCL barriers, which a long load would time out, and a repeated
    ``tag`` stays throttled because :class:`_StorePhase` isolates it from the previous done-keys.
    ``None`` resolves node-width-aware through :func:`resolve_load_concurrency`.
    """
    if not (dist.is_available() and dist.is_initialized()):
        yield
        return

    local_rank = get_local_rank()
    local_world_size = get_local_world_size()
    max_concurrent = resolve_load_concurrency(max_concurrent, local_world_size)

    if local_world_size <= 1 or max_concurrent == 0 or max_concurrent >= local_world_size:
        yield
        return

    phase = _StorePhase(f"seq_load/{tag}", f"node{get_node_rank()}", local_rank, local_world_size, get_store_timeout())

    batch = local_rank // max_concurrent
    try:
        # Inside the try: a timed-out wait must still mark this rank done, or every later batch on
        # the node blocks for the full store timeout.
        phase.wait([f"rank{r}_done" for r in range(batch * max_concurrent)])
        yield
    finally:
        phase.set(f"rank{local_rank}_done")
        phase.finish([f"rank{r}_done" for r in range(local_world_size)])
