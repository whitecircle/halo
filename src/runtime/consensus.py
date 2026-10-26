"""Cross-rank verdicts, every one COLLECTIVE: raise a rank-local failure on every rank, agree on a value,
count a per-rank observation, and the c10d-store joins for waits that outlast the NCCL watchdog."""

import datetime
import hashlib
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from src.env import env_flag, store_timeout
from src.log import KEY_PREVIEW_COUNT
from src.runtime.collectives import broadcast_from_rank0, collective_device
from src.runtime.ranks import get_global_rank, get_global_world_size, is_global_main_process, is_multi_rank_run

logger = logging.getLogger(__name__)

# A digest below 2**62 negates without overflowing int64, so [d, -d] rides one MAX all-reduce.
_AGREEMENT_DIGEST_BITS = 62
# MAX's neutral element: an abstaining rank's slot never wins.
_INT64_MIN = -(2**63)
# The [d, -d] pair of a rank whose object cannot be digested: 2**62 is above every real digest while
# 0 is at or above every real -d, so max(d) != -max(-d) whoever else takes part.
_UNDIGESTIBLE = (2**_AGREEMENT_DIGEST_BITS, 0)

_T = TypeVar("_T")


def raise_rank0_failure(
    step: Callable[[], Any], describe: Callable[[Exception], str], exc_type: type[Exception] = RuntimeError
) -> None:
    """Run ``step`` on global rank 0 and raise its failure on every rank as ``exc_type``, with the
    message ``describe`` makes of it (chained on rank 0). COLLECTIVE."""
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


def raise_gathered_reasons(reasons: list[str | None], what: str, exc_type: type[Exception]) -> None:
    """Raise from a world-gathered reason list, naming the failing ranks and the first one's reason."""
    summary = gathered_failure_summary(reasons, what)
    if summary is not None:
        raise exc_type(summary)


def gather_rank_reasons(local_reason: str | None) -> list[str | None]:
    """Every rank's ``local_reason``, indexed by global rank. COLLECTIVE."""
    if not is_multi_rank_run():
        return [local_reason]
    reasons: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reasons, local_reason)
    return reasons


def _rank_max(slots: list[int], group: dist.ProcessGroup | None = None) -> list[int]:
    """Each int64 slot's MAX across ``group`` in one fixed-size all-reduce. COLLECTIVE."""
    reduced = torch.tensor(slots, dtype=torch.int64, device=collective_device(group))
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX, group=group)
    return reduced.tolist()


def reject_across_ranks(local_reason: str | None, what: str, exc_type: type[Exception] = RuntimeError) -> None:
    """Raise ``exc_type`` on every rank when any rank reports a reason, naming the failing ranks. COLLECTIVE.

    A rank raising alone would leave its peers in the next collective with no diagnostic. After
    unbounded single-rank work use :func:`store_reject_across_ranks`.
    """
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    if not _rank_max([int(bool(local_reason))])[0]:
        return
    raise_gathered_reasons(gather_rank_reasons(local_reason), what, exc_type)


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
    """Whether every rank of ``group`` that holds ``obj`` holds an equal one. COLLECTIVE.

    One :func:`_rank_max` of ``[d, -d, -rank, abstained]`` over the canonical digest ``d``: the ranks
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
    max_digest, max_negated, max_negated_rank, any_abstained = _rank_max(local, group)
    if max_negated_rank == _INT64_MIN:
        return RankAgreement(True, None, True)
    return RankAgreement(max_digest == -max_negated, -max_negated_rank, bool(any_abstained))


def reject_divergent_settings(values: dict[str, object], what: str, guidance: str) -> None:
    """Raise ``ValueError`` on every rank when the ranks disagree on a resolved setting. COLLECTIVE.

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
    """Whether every and whether any rank of ``group`` (default: the world) is ok, in one SUM
    all-reduce. COLLECTIVE. The caller decides what a partial result means."""
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
        """Raise on every rank if any recorded a failure. COLLECTIVE."""
        reject_across_ranks(self.reason, self.what, exc_type=self.exc_type)


def rank0_verdict(probe: Callable[[], _T], what: str, exc_type: type[Exception] = RuntimeError) -> _T:
    """Run ``probe`` on global rank 0 and return its result on every rank. COLLECTIVE: a raise on rank 0
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

    def set(self, name: str, value: str = "1") -> None:
        self._store.set(f"{self._prefix}/{name}", value)

    def get_all(self, names) -> list[str]:
        """Values of keys this participant already waited on, in ``names`` order, in one store round
        trip (a key-per-participant loop is O(world²) requests)."""
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
    """:func:`reject_across_ranks` over the c10d store, bounded by ``DIST_STORE_TIMEOUT_HOURS`` rather
    than the NCCL watchdog, for a join after unbounded single-rank work (a fresh dataset map, a shard
    download). Collective-equivalent: every rank calls it with the same tags in the same order.
    """
    if not is_multi_rank_run():
        if local_reason:
            raise exc_type(local_reason)
        return
    rank, world = get_global_rank(), get_global_world_size()
    reason_keys = [f"reason{i}" for i in range(world)]
    phase = StorePhase(f"reject/{tag}", "world", rank, world, timeout or store_timeout())
    try:
        # "" is the no-failure sentinel.
        phase.set(f"reason{rank}", local_reason or "")
        phase.wait(reason_keys)
        reasons: list[str | None] = [reason or None for reason in phase.get_all(reason_keys)]
    finally:
        phase.finish(reason_keys)
    raise_gathered_reasons(reasons, what, exc_type)


def store_join_recorded_failure(tag: str, failure: BaseException | None, what: str) -> None:
    """Join a recorded rank-local failure over the store: the failing rank re-raises its own exception
    (chained under the uniform one), every other rank a ``RuntimeError`` naming it."""
    if failure is None:
        store_reject_across_ranks(tag, None, what)
        return
    try:
        store_reject_across_ranks(tag, f"{type(failure).__name__}: {failure}", what)
    except RuntimeError as uniform:
        raise failure from uniform
    raise failure


def agree_probe_across_ranks(local: bool, subject, probe: str) -> bool:
    """``True`` if any rank's probe read ``True`` (an S3 probe errs only toward ``False``, one image
    makes a corpus multimodal), warning on the ranks that read ``False``."""
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


def assert_rank_uniform(
    value: Any, *, group: dist.ProcessGroup | None, label: str, strict: bool | None = None
) -> None:
    """Warn, or raise under ``strict`` (default ``HALO_TP_CONSISTENCY_CHECK``), when ``value`` differs
    across the ranks of ``group``; a no-op for ``group=None`` or one rank. COLLECTIVE: one
    :func:`agree_across_ranks` all-reduce, the digests gathered only to name a divergence. Pass
    ``tuple(t.shape)`` to compare a shape, the usual first sign of a TP/EP deadlock.
    """
    if group is None:
        return
    world = dist.get_world_size(group=group)
    if world <= 1:
        return
    if strict is None:
        strict = env_flag("HALO_TP_CONSISTENCY_CHECK")

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
