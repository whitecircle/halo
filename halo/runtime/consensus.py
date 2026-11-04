"""Cross-rank verdicts, all COLLECTIVE, and c10d-store joins for waits that outlast the NCCL watchdog."""

import datetime
import hashlib
import logging
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from halo.env import store_timeout
from halo.log import KEY_PREVIEW_COUNT
from halo.runtime.collectives import broadcast_from_rank0, collective_device
from halo.runtime.ranks import get_global_rank, get_global_world_size, is_global_main_process, is_multi_rank_run

logger = logging.getLogger(__name__)

# A digest below 2**62 negates without overflowing int64, so [d, -d] rides one MAX all-reduce.
_AGREEMENT_DIGEST_BITS = 62
# The [d, -d] of a rank whose object cannot be digested: 2**62 is above every real digest and 0 at or above every
# real -d, so max(d) != -max(-d) whoever else takes part.
_UNDIGESTIBLE = (2**_AGREEMENT_DIGEST_BITS, 0)
# Splits a store-join failure record "<rank><sep><reason>": a rank number never holds it, a reason may.
_FAILURE_RECORD_SEP = "\n"

_T = TypeVar("_T")


def run_on_rank0_raising_everywhere(
    step: Callable[[], Any], describe: Callable[[Exception], str], exc_type: type[Exception] = RuntimeError
) -> None:
    """COLLECTIVE: run ``step`` on global rank 0 and raise its failure on every rank as ``exc_type``."""
    failure: str | None = None
    cause: Exception | None = None
    if is_global_main_process():
        try:
            step()
        except Exception as e:  # every failure must reach the peers
            cause, failure = e, describe(e)
    failure = broadcast_from_rank0(failure)
    if failure is not None:
        raise exc_type(failure) from cause


def refusal_text(error: Exception) -> str:
    """A check's own refusal (a ``ValueError``) verbatim; any other failure with its type, which ``str`` drops."""
    return str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"


def gathered_failure_summary(reasons: list[str | None], what: str) -> str | None:
    """``what`` failed on which ranks, with the first one's reason; None when no rank reported one."""
    failed = [rank for rank, reason in enumerate(reasons) if reason]
    if not failed:
        return None
    shown = ", ".join(str(rank) for rank in failed[:KEY_PREVIEW_COUNT])
    more = ", …" if len(failed) > KEY_PREVIEW_COUNT else ""
    return (
        f"{what} failed on {len(failed)} of {len(reasons)} rank(s) [{shown}{more}]. "
        f"First (rank {failed[0]}): {reasons[failed[0]]}"
    )


def _raise_gathered_reasons(reasons: list[str | None], what: str, exc_type: type[Exception]) -> None:
    summary = gathered_failure_summary(reasons, what)
    if summary is not None:
        raise exc_type(summary)


def gather_rank_reasons(local_reason: str | None) -> list[str | None]:
    """COLLECTIVE: every rank's ``local_reason``, indexed by global rank."""
    if not is_multi_rank_run():
        return [local_reason]
    reasons: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reasons, local_reason)
    return reasons


def rank_max(values: list[int], group: dist.ProcessGroup | None = None) -> list[int]:
    """COLLECTIVE: each int64 value's MAX across ``group`` (default: the world) in one fixed-size all-reduce."""
    if not is_multi_rank_run():
        return list(values)
    reduced = torch.tensor(values, dtype=torch.int64, device=collective_device(group))
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX, group=group)
    return reduced.tolist()


def rank_min(values: list[int]) -> list[int]:
    """COLLECTIVE: each int64 value's MIN across the world."""
    return [-value for value in rank_max([-value for value in values])]


def reject_across_ranks(local_reason: str | None, what: str, exc_type: type[Exception] = RuntimeError) -> None:
    """COLLECTIVE: raise on every rank when any rank reports a reason, since a rank raising alone strands its peers in
    the next collective. After unbounded single-rank work use :func:`store_reject_across_ranks`."""
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    if not rank_max([int(bool(local_reason))])[0]:
        return
    _raise_gathered_reasons(gather_rank_reasons(local_reason), what, exc_type)


def _raise_joined_failure(failure: BaseException | None, join: Callable[[str | None], None]) -> None:
    """``join`` a recorded ``failure``; the failing rank re-raises it, chained under the join's error."""
    if failure is None:
        join(None)
        return
    try:
        join(f"{type(failure).__name__}: {failure}")
    except RuntimeError as uniform:
        raise failure from uniform
    raise failure


@contextmanager
def barrier_on_exit(what: str) -> Iterator[None]:
    """COLLECTIVE: run a rank-gated write, then join its failure with one all-reduce where a barrier would stand, so it
    raises on every rank. After unbounded single-rank work use :func:`store_join_recorded_failure`."""
    failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:  # joined below, so every rank raises with it
        failure = exc
    _raise_joined_failure(failure, lambda reason: reject_across_ranks(reason, what))


def divergent_settings(gathered: list[dict[str, object]]) -> dict[str, list[str]]:
    """Setting names whose value differs across the per-rank mappings (a missing name included), with spellings."""
    if not gathered:
        return {}
    reference = gathered[0]
    names = {name for rank_values in gathered for name in rank_values}
    return {
        name: sorted({str(rank_values.get(name)) for rank_values in gathered})
        for name in sorted(names, key=str)
        if any(rank_values.get(name) != reference.get(name) for rank_values in gathered)
    }


def _agreement_digest(obj: object) -> int:
    """62-bit blake2b of ``obj``'s canonical repr. Never Python's ``hash``, which is salted per process."""
    canonical = sorted(obj.items()) if isinstance(obj, dict) else obj
    digest = hashlib.blake2b(repr(canonical).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> (64 - _AGREEMENT_DIGEST_BITS)


def agree_across_ranks(obj: object, *, group: dist.ProcessGroup | None = None) -> bool:
    """COLLECTIVE: whether every rank of ``group`` holds an equal ``obj``, by one :func:`rank_max` of the digest pair
    ``[d, -d]``: they agree iff ``max(d) == -max(-d)``. An object that cannot be digested forces a mismatch."""
    if not is_multi_rank_run():
        return True
    try:
        digest = _agreement_digest(obj)
        local = [digest, -digest]
    except Exception:  # a rank-local raise here would strand the peers in the reduce
        local = list(_UNDIGESTIBLE)
    max_digest, max_negated = rank_max(local, group)
    return max_digest == -max_negated


def reject_divergent_settings(values: dict[str, object], what: str, guidance: str) -> None:
    """COLLECTIVE: raise ``ValueError`` on every rank when the ranks disagree on a setting. Pass resolved values, so an
    absent variable and one set to its default agree. The digest is a fast path (``30`` and ``30.0`` differ); a
    mismatch gathers the values and :func:`divergent_settings` decides."""
    if not is_multi_rank_run() or agree_across_ranks(values):
        return
    gathered: list[dict[str, object]] = [{} for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    differing = divergent_settings(gathered)
    if differing:
        raise ValueError(
            f"{what} differs across ranks: {differing}. {guidance} Set them identically on every "
            f"node (rank 0 has {dict(gathered[0])})."
        )


@dataclass(frozen=True)
class RankConsensus:
    all_ok: bool
    any_ok: bool


def rank_consensus(local_ok: bool, group: dist.ProcessGroup | None = None) -> RankConsensus:
    """COLLECTIVE: whether every and whether any rank of ``group`` is ok, in one SUM all-reduce."""
    if is_multi_rank_run():
        count = torch.tensor([1 if local_ok else 0], device=collective_device())
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
        n = int(count.item())
        return RankConsensus(n == dist.get_world_size(group), n > 0)
    return RankConsensus(local_ok, local_ok)


class DeferredRankFailure:
    """Defer rank-local failures to a closing collective, for local work interleaved with collectives. Once a step
    fails the rest are skipped, since the first reason is the diagnostic one."""

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
        """COLLECTIVE: raise on every rank if any recorded a failure."""
        reject_across_ranks(self.reason, self.what, exc_type=self.exc_type)

    def store_reject(self, tag: str) -> None:
        """COLLECTIVE over the c10d store: :meth:`reject` after unbounded file I/O, where the NCCL watchdog would abort
        the slowest writer's peers."""
        store_reject_across_ranks(tag, self.reason, self.what, exc_type=self.exc_type)


def rank0_verdict(probe: Callable[[], _T], what: str, exc_type: type[Exception] = RuntimeError) -> _T:
    """COLLECTIVE: ``probe``'s result from global rank 0 on every rank; a raise on rank 0 raises on every rank."""
    guard = DeferredRankFailure(what, exc_type)
    local = guard.run(probe) if is_global_main_process() else None
    verdict = broadcast_from_rank0(local)
    guard.reject()
    return verdict


class StorePhase:
    """One self-cleaning use of a c10d-store tag; each participant's entry counter numbers the phase and namespaces its
    keys. Every participant of a scope must enter a tag the same number of times in the same order (one tag per call
    site, never from a rank-dependent branch), or one waits on a key nobody writes."""

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
        self._arrived = False

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}"

    def set(self, name: str, value: str = "1") -> None:
        self._store.set(self._key(name), value)

    def add(self, name: str, amount: int = 1) -> int:
        """Add ``amount`` to the phase's counter ``name`` and return the sum; ``0`` reads it."""
        return self._store.add(self._key(name), amount)

    def get_all(self, names) -> list[str]:
        """Values of keys set before a key this participant waited on, in one round trip (per key: O(world²))."""
        keys = [self._key(name) for name in names]
        if not keys:
            return []
        values = self._store.multi_get(keys)
        return [value.decode() if isinstance(value, bytes) else str(value) for value in values]

    def wait(self, names) -> None:
        keys = [self._key(name) for name in names]
        if not keys:
            return
        try:
            self._store.wait(keys, self._timeout)
        except Exception as e:
            raise self._timeout_error(str(keys)) from e

    def arrive_and_wait(self, failure: str | None = None) -> list[str]:
        """Every participant's ``failure`` records, in slot order, once all have arrived. A failure takes its slot
        before arriving, so the count is final by then. Arrivals count on one key and the last sets the key the rest
        wait on: O(1) store requests per participant, where a key per participant costs each O(participants)."""
        self._arrived = True
        if failure:
            slot = self.add("failures")
            self.set(f"failed{slot}", failure)
        if self.add("arrived") == self._num:
            self.set("all_arrived")
        try:
            self._store.wait([self._key("all_arrived")], self._timeout)
        except Exception as e:
            arrived = self.add("arrived", 0)
            raise self._timeout_error(f"all {self._num} participants to arrive ({arrived} did)") from e
        return self.get_all(self._failure_slots())

    def _failure_slots(self) -> list[str]:
        return [f"failed{slot}" for slot in range(1, self.add("failures", 0) + 1)]

    def finish(self, transient_names) -> None:
        """Mark this participant done; the last one deletes the phase's keys. It reads the failure count itself, final
        by then even when its own wait timed out."""
        if self.add("exited") == self._num:
            arrival_names = ("arrived", "all_arrived", "failures", *self._failure_slots()) if self._arrived else ()
            for name in (*transient_names, *arrival_names, "exited"):
                self._store.delete_key(self._key(name))

    def _timeout_error(self, waiting_for: str) -> RuntimeError:
        return RuntimeError(
            f"Store coordination '{self._label}' timed out after {self._timeout} waiting for "
            f"{waiting_for} (this participant is in phase {self._phase} of {self._num}). Either a peer "
            f"is still working — raise DIST_STORE_TIMEOUT_HOURS — or the participants disagree on how "
            f"many times they entered this tag / on their coordination scope, which leaves this one "
            f"waiting on a key nobody will write."
        )


def store_reject_across_ranks(
    tag: str,
    local_reason: str | None,
    what: str,
    exc_type: type[Exception] = RuntimeError,
    timeout: datetime.timedelta | None = None,
) -> None:
    """COLLECTIVE over the c10d store (same tags, same order on every rank): :func:`reject_across_ranks` bounded by
    ``DIST_STORE_TIMEOUT_HOURS`` instead of the NCCL watchdog, for a join after unbounded single-rank work (a dataset
    map, a shard download)."""
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    rank, world = get_global_rank(), get_global_world_size()
    phase = StorePhase(f"reject/{tag}", "world", rank, world, timeout or store_timeout())
    try:
        records = phase.arrive_and_wait(f"{rank}{_FAILURE_RECORD_SEP}{local_reason}" if local_reason else None)
    finally:
        phase.finish([])
    reasons: list[str | None] = [None] * world
    for record in records:
        failed_rank, reason = record.split(_FAILURE_RECORD_SEP, 1)
        reasons[int(failed_rank)] = reason
    _raise_gathered_reasons(reasons, what, exc_type)


def store_join_recorded_failure(tag: str, failure: BaseException | None, what: str) -> None:
    """COLLECTIVE over the c10d store: the failing rank re-raises its own exception, the others a ``RuntimeError``."""
    _raise_joined_failure(failure, lambda reason: store_reject_across_ranks(tag, reason, what))


def agree_probe_across_ranks(local: bool, subject, probe: str) -> bool:
    """COLLECTIVE: ``True`` if any rank's probe read ``True`` (an S3 probe errs only toward ``False``)."""
    agreed = rank_consensus(local).any_ok
    if agreed and not local:
        logger.warning(
            f"{probe} disagreed across ranks for {subject}: this rank read False, another read True. "
            f"Using the agreed verdict so every rank takes the same data path and the coordinated "
            f"dataset operations stay in lock-step."
        )
    return agreed


def _value_digest(value: Any) -> str:
    """SHA-256 hex digest of a tensor's shape, dtype and bytes (uint8 view, since bf16/fp8 have no numpy dtype), a
    sequence element-wise, anything else its pickle."""
    hasher = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        hasher.update(repr(value.shape).encode())
        hasher.update(repr(value.dtype).encode())
        raw = value.detach().to("cpu").contiguous().view(-1)
        if raw.numel():
            hasher.update(raw.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (list, tuple)):
        # The type/length prefix separates [x] from [[x]].
        hasher.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            hasher.update(_value_digest(item).encode())
    else:
        hasher.update(pickle.dumps(value, protocol=4))
    return hasher.hexdigest()


def assert_rank_uniform(value: Any, *, group: dist.ProcessGroup | None, label: str, strict: bool = False) -> None:
    """COLLECTIVE: warn, or raise under ``strict``, when ``value`` differs across the ranks of ``group``; a no-op
    without a group. Pass ``tuple(t.shape)`` to compare a shape, the usual first sign of a TP/EP deadlock."""
    if group is None:
        return
    world = dist.get_world_size(group=group)
    if world <= 1:
        return

    digest = _value_digest(value)
    if agree_across_ranks(digest, group=group):
        return
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest, group=group)
    summary = ", ".join(f"rank{i}={d[:12] if d else '<missing>'}" for i, d in enumerate(digests))
    message = f"[assert_rank_uniform:{label}] divergent values across ranks: {summary}"
    if strict:
        raise RuntimeError(message + f" (rank {dist.get_rank(group=group)}: digest={digest[:12]})")
    logger.warning(message)
