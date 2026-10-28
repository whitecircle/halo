"""Cross-rank verdicts, every one COLLECTIVE: raise a rank-local failure on every rank, agree on a value,
count a per-rank observation or take its world MIN/MAX, and the c10d-store joins for waits that outlast the
NCCL watchdog."""

import datetime
import hashlib
import logging
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

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
# MAX's neutral element: an abstaining rank's slot never wins.
_INT64_MIN = -(2**63)
# The [d, -d] pair of a rank whose object cannot be digested: 2**62 is above every real digest while
# 0 is at or above every real -d, so max(d) != -max(-d) whoever else takes part.
_UNDIGESTIBLE = (2**_AGREEMENT_DIGEST_BITS, 0)
# Splits a store-join failure record "<rank><sep><reason>": a rank number never holds it, a reason may.
_FAILURE_RECORD_SEP = "\n"

_T = TypeVar("_T")


def raise_rank0_failure(
    step: Callable[[], Any], describe: Callable[[Exception], str], exc_type: type[Exception] = RuntimeError
) -> None:
    """COLLECTIVE: run ``step`` on global rank 0 and raise its failure on every rank as ``exc_type``, with
    the message ``describe`` makes of it (chained on rank 0)."""
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
    """Raise from a world-gathered reason list, naming the failing ranks and the first one's reason."""
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
    """COLLECTIVE: each int64 value's MAX across ``group`` (default: the world) in one fixed-size all-reduce
    on the backend's device; ``values`` unchanged without a peer."""
    if not is_multi_rank_run():
        return list(values)
    reduced = torch.tensor(values, dtype=torch.int64, device=collective_device(group))
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX, group=group)
    return reduced.tolist()


def rank_min(values: list[int], group: dist.ProcessGroup | None = None) -> list[int]:
    """COLLECTIVE: each int64 value's MIN across ``group``: :func:`rank_max` of the negated values."""
    return [-value for value in rank_max([-value for value in values], group)]


def reject_across_ranks(local_reason: str | None, what: str, exc_type: type[Exception] = RuntimeError) -> None:
    """COLLECTIVE: raise ``exc_type`` on every rank when any rank reports a reason, naming the failing ranks.

    A rank raising alone would leave its peers in the next collective with no diagnostic. After
    unbounded single-rank work use :func:`store_reject_across_ranks`.
    """
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    if not rank_max([int(bool(local_reason))])[0]:
        return
    _raise_gathered_reasons(gather_rank_reasons(local_reason), what, exc_type)


def _raise_joined_failure(failure: BaseException | None, join: Callable[[str | None], None]) -> None:
    """``join`` a recorded rank-local ``failure`` (its type and message as the reason); the failing rank re-raises
    its own exception, chained under the uniform ``RuntimeError`` the join raised."""
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
    """COLLECTIVE: run a rank-gated write, then join its failure across ranks where a barrier would stand (one
    all-reduce on every rank): a failure on any rank raises on every rank, the failing rank its own exception
    and the others a ``RuntimeError`` naming the failing ranks. After unbounded single-rank work use
    :func:`store_join_recorded_failure`."""
    failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:  # joined below, so every rank raises with it
        failure = exc
    _raise_joined_failure(failure, lambda reason: reject_across_ranks(reason, what))


def divergent_settings(gathered: list[dict[str, object]]) -> dict[str, list[str]]:
    """Setting names whose value differs across the per-rank mappings (a missing name included), with
    each one's sorted spellings."""
    if not gathered:
        return {}
    reference = gathered[0]
    names = {name for rank_values in gathered for name in rank_values}
    return {
        name: sorted({str(rank_values.get(name)) for rank_values in gathered})
        for name in sorted(names, key=str)
        if any(rank_values.get(name) != reference.get(name) for rank_values in gathered)
    }


class RankAgreement(NamedTuple):
    """What :func:`agree_across_ranks` learns in its one all-reduce."""

    agreed: bool
    # Lowest rank that did not abstain; None when every rank did.
    first_present: int | None
    any_abstained: bool


def _agreement_digest(obj: object) -> int:
    """62-bit blake2b digest of ``obj``'s canonical repr (a mapping's items sorted). Never Python's
    ``hash``, which is salted per process."""
    canonical = sorted(obj.items()) if isinstance(obj, dict) else obj
    digest = hashlib.blake2b(repr(canonical).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> (64 - _AGREEMENT_DIGEST_BITS)


def agree_across_ranks(obj: object, *, abstain: bool = False, group: dist.ProcessGroup | None = None) -> RankAgreement:
    """COLLECTIVE: whether every rank of ``group`` that holds ``obj`` holds an equal one.

    One :func:`rank_max` of ``[d, -d, -rank, abstained]`` over the canonical digest ``d``: the ranks
    agree iff ``max(d) == -max(-d)``. An object that cannot be digested forces a mismatch rather than
    raising on one rank inside the collective. ``abstain`` takes a rank out of the comparison;
    ``first_present`` names a rank the abstainers can learn the object from.
    """
    if not is_multi_rank_run():
        return RankAgreement(True, None if abstain else 0, abstain)
    if abstain:
        local = [_INT64_MIN] * 3 + [1]
    else:
        try:
            digest = _agreement_digest(obj)
            pair = [digest, -digest]
        except Exception:  # a rank-local raise here would strand the peers in the reduce
            pair = list(_UNDIGESTIBLE)
        local = [*pair, -dist.get_rank(group), 0]
    max_digest, max_negated, max_negated_rank, any_abstained = rank_max(local, group)
    if max_negated_rank == _INT64_MIN:
        return RankAgreement(True, None, True)
    return RankAgreement(max_digest == -max_negated, -max_negated_rank, bool(any_abstained))


def reject_divergent_settings(values: dict[str, object], what: str, guidance: str) -> None:
    """COLLECTIVE: raise ``ValueError`` on every rank when the ranks disagree on a resolved setting.

    ``values`` are resolved, not raw strings: an absent variable and one set to its default agree. The
    :func:`agree_across_ranks` digest is only a fast path (``30`` and ``30.0`` digest differently); a
    mismatch gathers the values and :func:`divergent_settings` decides. ``guidance`` is appended.
    """
    if not is_multi_rank_run() or agree_across_ranks(values).agreed:
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
    """What :func:`rank_consensus` learns: whether every rank, and whether any rank, reported ok."""

    all_ok: bool
    any_ok: bool


def rank_consensus(local_ok: bool, group: dist.ProcessGroup | None = None) -> RankConsensus:
    """COLLECTIVE: whether every and whether any rank of ``group`` (default: the world) is ok, in one SUM
    all-reduce. The caller decides what a partial result means."""
    if is_multi_rank_run():
        count = torch.tensor([1 if local_ok else 0], device=collective_device())
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
        n = int(count.item())
        return RankConsensus(n == dist.get_world_size(group), n > 0)
    return RankConsensus(local_ok, local_ok)


class DeferredRankFailure:
    """Defer rank-local failures to a closing collective, for local work interleaved with collectives.

    Wrap each local step in :meth:`run` and close the region with :meth:`reject`. Once a step fails the
    rest are skipped: the first reason is the diagnostic one.
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
        """COLLECTIVE: raise on every rank if any recorded a failure."""
        reject_across_ranks(self.reason, self.what, exc_type=self.exc_type)

    def store_reject(self, tag: str) -> None:
        """COLLECTIVE over the c10d store under ``tag`` (:func:`store_reject_across_ranks`): :meth:`reject` for a
        region of unbounded file I/O, whose slowest writer the NCCL watchdog would abort its peers over."""
        store_reject_across_ranks(tag, self.reason, self.what, exc_type=self.exc_type)


def rank0_verdict(probe: Callable[[], _T], what: str, exc_type: type[Exception] = RuntimeError) -> _T:
    """COLLECTIVE: run ``probe`` on global rank 0 and return its result on every rank; a raise on rank 0
    is joined after the broadcast, so every rank raises ``exc_type`` naming it."""
    guard = DeferredRankFailure(what, exc_type)
    local = guard.run(probe) if is_global_main_process() else None
    verdict = broadcast_from_rank0(local)
    guard.reject()
    return verdict


class StorePhase:
    """One self-cleaning use ("phase") of a c10d-store coordination tag.

    Each participant's store-held entry counter numbers the phase and namespaces its keys; the last to
    :meth:`finish` deletes them. Invariant: every participant of a scope enters a tag the same number of
    times, in the same order (one tag per call site, never from a rank-dependent branch), or one waits
    on a key nobody writes.
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
        self._arrived = False

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}"

    def set(self, name: str, value: str = "1") -> None:
        self._store.set(self._key(name), value)

    def add(self, name: str, amount: int = 1) -> int:
        """Add ``amount`` to the phase's counter ``name`` and return the sum; ``0`` reads it."""
        return self._store.add(self._key(name), amount)

    def get_all(self, names) -> list[str]:
        """Values of keys set before a key this participant waited on, in ``names`` order, in one store
        round trip (a key-per-participant loop is O(world²) requests); no request for no names."""
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
        """Return the participants' ``failure`` records, in slot order, once every participant has arrived.

        A failure takes a numbered slot before its participant arrives, so the slot count is final once all
        have. The arrivals count on one key and the last one sets the single key the rest wait on: O(1) store
        requests per participant, plus one read of the slots when any participant failed (a wait on a key
        per participant costs each one O(participants)).
        """
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
        """Mark this participant done with the phase; the last one deletes the phase's keys, those of
        :meth:`arrive_and_wait` included. The last one reads the failure count itself: every participant has
        exited by then, so the count is final even when its own wait timed out."""
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
    """COLLECTIVE over the c10d store (every rank calls it with the same tags in the same order):
    :func:`reject_across_ranks` bounded by ``DIST_STORE_TIMEOUT_HOURS`` rather than the NCCL watchdog,
    for a join after unbounded single-rank work (a fresh dataset map, a shard download).

    A failing rank records its reason in a numbered slot before it arrives
    (:meth:`StorePhase.arrive_and_wait`): a clean join costs each rank O(1) store requests, a failed one
    also a read of the failing ranks' slots.
    """
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
    """COLLECTIVE over the c10d store: join a recorded rank-local failure; the failing rank re-raises its
    own exception (chained under the uniform one), every other rank a ``RuntimeError`` naming it."""
    _raise_joined_failure(failure, lambda reason: store_reject_across_ranks(tag, reason, what))


def agree_probe_across_ranks(local: bool, subject, probe: str) -> bool:
    """COLLECTIVE: ``True`` if any rank's probe read ``True`` (an S3 probe errs only toward ``False``, one
    image makes a corpus multimodal), warning on the ranks that read ``False``."""
    agreed = rank_consensus(local).any_ok
    if agreed and not local:
        logger.warning(
            f"{probe} disagreed across ranks for {subject}: this rank read False, another read True. "
            f"Using the agreed verdict so every rank takes the same data path and the coordinated "
            f"dataset operations stay in lock-step."
        )
    return agreed


def _value_digest(value: Any) -> str:
    """SHA-256 hex digest: a tensor's shape, dtype and bytes (through a uint8 view, since bf16/fp8 have
    no numpy dtype), a list or tuple element-wise, anything else its pickle."""
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
    """COLLECTIVE (one :func:`agree_across_ranks` all-reduce, the digests gathered only to name a
    divergence): warn, or raise under ``strict``, when ``value`` differs across the ranks of ``group``; a
    no-op for ``group=None`` or one rank. Pass ``tuple(t.shape)`` to compare a shape, the usual first sign
    of a TP/EP deadlock.
    """
    if group is None:
        return
    world = dist.get_world_size(group=group)
    if world <= 1:
        return

    digest = _value_digest(value)
    if agree_across_ranks(digest, group=group).agreed:
        return
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest, group=group)
    summary = ", ".join(f"rank{i}={d[:12] if d else '<missing>'}" for i, d in enumerate(digests))
    message = f"[assert_rank_uniform:{label}] divergent values across ranks: {summary}"
    if strict:
        raise RuntimeError(message + f" (rank {dist.get_rank(group=group)}: digest={digest[:12]})")
    logger.warning(message)
