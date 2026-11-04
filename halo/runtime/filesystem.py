"""Shared-vs-per-node filesystem scope: the ``DIST_*_SHARED_FILESYSTEM`` flags, consensus, probes, main-rank-first."""

import datetime
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from halo.env import env_flag, fs_probe_timeout, store_timeout
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import (
    StorePhase,
    agree_probe_across_ranks,
    barrier_on_exit,
    rank_consensus,
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

# Dot-prefixed and inert if one leaks.
FS_PROBE_PREFIX = ".halo_fs_probe_"

_FS_PROBE_INTERVAL_S = 0.25

_shared_filesystem_consensus: dict[str, bool] | None = None

# An entry script probes before resume detection and every trainer again; the repeat must cost nothing.
_probed_output_dirs: set[str] = set()


@dataclass(frozen=True)
class FilesystemSide:
    """One side of the shared-filesystem declaration, in the words its probe's refusals use."""

    name: str
    flag: str
    # Who must read back what global rank 0 wrote there.
    reader: str
    # What a shared declaration over per-node storage breaks.
    unshared_loss: str
    # What a per-node declaration over shared storage races.
    shared_race: str
    # Each node's own copy, and every node sharing one; ``{name}`` is the flagged location.
    per_node_remedy: str
    shared_remedy: str
    # What else the side's declaration binds, closing each of its refusals.
    coverage: str = ""


OUTPUT_SIDE = FilesystemSide(
    name="Output",
    flag=_OUTPUT_SHARED_FILESYSTEM_VAR,
    reader="the process that writes checkpoints",
    unshared_loss="Only rank 0 would write trainer_state.json / scheduler.pt / rng_state, so no checkpoint would "
    "resume on the other nodes.",
    shared_race="every node's local rank 0 would write the same checkpoint paths concurrently",
    per_node_remedy="give each node its own {name}",
    shared_remedy="point {name} at the shared mount",
)
INPUT_SIDE = FilesystemSide(
    name="Input",
    flag=_INPUT_SHARED_FILESYSTEM_VAR,
    reader="the rank that fills the caches",
    unshared_loss="Only global rank 0 would download and map into it, so the other nodes would read a cache "
    "nobody filled: their Hub reads fail and each of their ranks recomputes every dataset map.",
    shared_race="every node's local rank 0 would fill the same cache files concurrently",
    per_node_remedy="put {name} on each node's local storage",
    shared_remedy="point {name} at the shared mount",
    coverage="The one declaration covers HF_HUB_CACHE and HF_DATASETS_CACHE alike, so both must be on the same kind "
    "of storage.",
)


def _shared_filesystem_flag(var: str, default: bool) -> bool:
    if _shared_filesystem_consensus is not None:
        return _shared_filesystem_consensus[var]
    return env_flag(var, default=default)


def is_shared_filesystem() -> bool:
    """The ``DIST_SHARED_FILESYSTEM`` umbrella: rank-agreed after :func:`resolve_shared_filesystem_consensus`, else
    this rank's env. Never a collective."""
    return _shared_filesystem_flag(_SHARED_FILESYSTEM_VAR, True)


def is_input_shared_filesystem() -> bool:
    """Read side (downloads, dataset map/pack, HF caches): ``DIST_INPUT_SHARED_FILESYSTEM``, else the umbrella."""
    return _shared_filesystem_flag(_INPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def is_output_shared_filesystem() -> bool:
    """Write side (checkpoints, run.log, dumps): ``DIST_OUTPUT_SHARED_FILESYSTEM``, else the umbrella."""
    return _shared_filesystem_flag(_OUTPUT_SHARED_FILESYSTEM_VAR, is_shared_filesystem())


def resolve_shared_filesystem_consensus() -> None:
    """COLLECTIVE: agree the three flags, rank 0's values winning with a warning. They pick the coordination scope, so
    ranks that disagree would wait in different scopes. Memoized at the end, so no reader sees half a state."""
    global _shared_filesystem_consensus
    # Bypass the memo: this rank's env is the consensus input.
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
    # A wrong umbrella is silent: left shared on per-node disks, every node recomputes the dataset at N× CPU and disk.
    if is_global_main_process():
        logger.info(
            "Shared-filesystem scope: %s (set DIST_SHARED_FILESYSTEM=0 for per-node local storage).",
            {var: ("shared" if value else "per-node") for var, value in agreed.items()},
        )


def is_fs_save_rank() -> bool:
    return is_global_main_process() if is_output_shared_filesystem() else is_local_main_process()


def is_fs_load_rank() -> bool:
    """Whether this rank does shared read-side work (downloads, cache fills)."""
    return is_global_main_process() if is_input_shared_filesystem() else is_local_main_process()


def fs_aware_makedirs(path: str, writer_rank: Callable[[], bool] = is_fs_save_rank) -> None:
    """COLLECTIVE: create ``path`` on ``writer_rank``; a ``makedirs`` failure raises on every rank."""
    with barrier_on_exit(f"Creating {path}"):
        if writer_rank():
            os.makedirs(path, exist_ok=True)


@contextmanager
def fs_aware_main_first(tag: str, timeout: datetime.timedelta | None = None):
    """Run the body on global rank 0 (shared input FS) or each node's local rank 0 first, then on the others. Waiters
    block on a store key, so the body must issue no collective (``fs_aware_makedirs`` barriers). ``tag`` names the call
    site under the :class:`StorePhase` invariant."""
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


def hub_metadata_main_first(tag: str, fetch: Callable[..., Any]) -> Any:
    """Main-rank-first hub metadata read (``AutoConfig``, ...) under a short bound, so a tag entered from a
    rank-dependent branch fails by name. ``fetch(local_files_only=...)`` reads only the cache off the load rank, one
    Hub query per scope; before ``init_distributed`` every rank fetches. Takes the consensus if none ran yet."""
    if _shared_filesystem_consensus is None:
        resolve_shared_filesystem_consensus()
    with fs_aware_main_first(f"hub_meta/{tag}", timeout=_HUB_METADATA_TIMEOUT):
        return fetch(local_files_only=has_process_group() and not is_fs_load_rank())


def filesystem_contradiction(
    side: FilesystemSide, name: str, declared_shared: bool, seen: list[bool], *, path: str | None = None
) -> str | None:
    """Why ``side``'s declared sharing contradicts ``seen`` (per rank, whether it saw rank 0's sentinel under
    ``name``), or None when it holds."""
    if not seen:
        return None
    location = f"{name} ({path})" if path else name
    closing = f" {side.coverage}" if side.coverage else ""
    if not seen[0]:
        return (
            f"{side.name}-filesystem probe: global rank 0 could not read back the sentinel it just wrote "
            f"under {location}, which is then not durably readable by {side.reader} — check the mount, "
            f"its permissions and its free space."
        )
    observed_shared = all(seen)
    if declared_shared == observed_shared:
        return None
    if declared_shared:
        blind = [rank for rank, ok in enumerate(seen) if not ok]
        return (
            f"{side.name} filesystem is declared SHARED but {len(blind)} of {len(seen)} ranks (first: "
            f"rank {blind[0]}) cannot see a file global rank 0 wrote under {location}. {side.unshared_loss} "
            f"If {location} is on a shared mount whose attribute cache outlasts the probe (NFS actimeo or "
            f"acdirmax above DIST_FS_PROBE_TIMEOUT_SECONDS, or lookupcache keeping the negative lookup), "
            f"raise DIST_FS_PROBE_TIMEOUT_SECONDS: a per-node declaration there means {side.shared_race}. "
            f"Otherwise set {side.flag}=0 (or DIST_SHARED_FILESYSTEM=0 for both sides) so each node keeps "
            f"its own copy, or {side.shared_remedy.format(name=name)}.{closing}"
        )
    return (
        f"{side.name} filesystem is declared PER-NODE but all {len(seen)} ranks can see a file global "
        f"rank 0 wrote under {location}, so it is shared: {side.shared_race}. Set {side.flag}=1 (or "
        f"DIST_SHARED_FILESYSTEM=1 for both sides), or {side.per_node_remedy.format(name=name)}.{closing}"
    )


def _visible_within(paths: list[str], seconds: float) -> list[bool]:
    """Which of ``paths`` show up within ``seconds``, polling past NFS/Lustre attribute caching."""
    deadline = time.monotonic() + seconds
    while True:
        seen = [os.path.exists(path) for path in paths]
        if all(seen) or time.monotonic() >= deadline:
            return seen
        time.sleep(_FS_PROBE_INTERVAL_S)


def _write_sentinels(directories: list[str]) -> tuple[str, list[str | None]]:
    """Rank 0's half of the probe: the sentinel token and each directory's write failure, None where it succeeded."""
    token = f"{time.time_ns():x}"
    failures: list[str | None] = []
    for directory in directories:
        sentinel = os.path.join(directory, FS_PROBE_PREFIX + token)
        try:
            os.makedirs(directory, exist_ok=True)
            with open(sentinel, "w") as handle:
                handle.write("halo")
            failures.append(None)
        except OSError as exc:
            failures.append(f"could not write {sentinel}: {exc}")
    return token, failures


def _sentinel_visibility(directories: list[str], declared_shared: bool) -> tuple[list[str | None], list[list[bool]]]:
    """COLLECTIVE: rank 0 writes a sentinel under each directory and every rank looks for it; returns rank 0's write
    failures and ``seen[directory][rank]`` (all False where it wrote none)."""
    # Broadcast after the writes, so peers poll only names that exist.
    token, failures = broadcast_from_rank0(_write_sentinels(directories) if is_global_main_process() else None)
    sentinels = [os.path.join(directory, FS_PROBE_PREFIX + token) for directory in directories]
    written = [sentinel for sentinel, failure in zip(sentinels, failures, strict=True) if failure is None]
    # Only a shared declaration polls, ruling out a cached negative lookup; per-node, seeing the file at all is proof
    # and not seeing it agrees.
    budget = fs_probe_timeout().total_seconds() if declared_shared else 0.0
    gathered: list[list[bool]] = [[] for _ in range(get_global_world_size())]
    dist.all_gather_object(gathered, _visible_within(written, budget))
    if is_global_main_process():
        for sentinel in written:
            try:
                os.remove(sentinel)
            except OSError:
                logger.warning("Could not remove the filesystem probe sentinel %s (inert).", sentinel)
    # Transposed to one row per written sentinel, each indexed by rank.
    rows = iter([list(row) for row in zip(*gathered, strict=True)])
    seen = [next(rows) if failure is None else [False] * len(gathered) for failure in failures]
    return failures, seen


def _is_multi_node_run() -> bool:
    world = get_global_world_size()
    return world > 1 and get_local_world_size() < world


def verify_output_filesystem_sharing(output_dir: str) -> None:
    """COLLECTIVE: raise when ``output_dir``'s real sharing contradicts the declaration, once per directory per
    process. Otherwise silent: shared on per-node disks leaves other nodes no resumable checkpoint, per-node on a
    shared mount races every node's writer."""
    if not _is_multi_node_run() or not output_dir:
        return
    probed = os.path.abspath(output_dir)
    if probed in _probed_output_dirs:
        return
    declared_shared = is_output_shared_filesystem()
    [failure], [seen] = _sentinel_visibility([output_dir], declared_shared)
    if failure is not None:
        raise OSError(f"Output-filesystem probe: {failure}. output_dir must be writable by global rank 0.")
    reason = filesystem_contradiction(OUTPUT_SIDE, "output_dir", declared_shared, seen)
    if reason:
        raise RuntimeError(reason)
    _probed_output_dirs.add(probed)
    if is_global_main_process():
        logger.info(
            "Output filesystem probed across %d nodes: %s, matching the declared flags.",
            get_num_nodes(),
            "shared" if all(seen) else "per-node",
        )


def verify_input_filesystem_sharing(caches: Mapping[str, str]) -> None:
    """COLLECTIVE: raise when a read-side cache directory (``caches`` maps variable to path) contradicts
    ``DIST_INPUT_SHARED_FILESYSTEM``. A cache rank 0 cannot write is filled by nobody and is skipped."""
    if not _is_multi_node_run():
        return
    declared_shared = is_input_shared_filesystem()
    names, directories = list(caches), list(caches.values())
    failures, seen = _sentinel_visibility(directories, declared_shared)
    for name, directory, failure, rank_seen in zip(names, directories, failures, seen, strict=True):
        if failure is not None:
            if is_global_main_process():
                logger.info("Not probing %s: %s, so no rank fills it.", name, failure)
            continue
        reason = filesystem_contradiction(INPUT_SIDE, name, declared_shared, rank_seen, path=directory)
        if reason:
            raise RuntimeError(reason)
    if is_global_main_process():
        logger.info(
            "Input caches probed across %d nodes: %s declared, matching %s.",
            get_num_nodes(),
            "shared" if declared_shared else "per-node",
            ", ".join(name for name, failure in zip(names, failures, strict=True) if failure is None) or "none",
        )


def agree_input_probe_across_ranks(probe: Callable[[], bool], subject, probe_name: str) -> bool:
    """COLLECTIVE: :func:`agree_probe_across_ranks` for a probe run only on each input scope's load rank, the rest
    abstaining with ``False``. A raising probe is joined over the store first, so its cause raises on every rank."""
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
