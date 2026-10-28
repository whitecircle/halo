#!/usr/bin/env python
"""``store_reject_across_ranks`` — the failure-carrying join over the c10d store, not a collective.

The coordinated dataset ops join their two phases with this: the writer rank's map is unbounded
single-rank work (a fresh-cache tokenization of a large corpus), and a join spelled
``all_gather_object`` parks the peers INSIDE the process group for that whole duration — a map
longer than ``DIST_NCCL_TIMEOUT_MINUTES`` then kills the run with a watchdog abort blaming the
collective, not the map. The store transport keeps the ``reject_across_ranks`` contract (uniform raise, real cause,
caller's ``exc_type``) while bounding the wait by ``DIST_STORE_TIMEOUT_HOURS``.

Proven on real gloo groups: (1) a clean join passes on both ranks; (2) one rank's reason raises on
EVERY rank, naming the failing rank and cause; (3) **a rank may arrive later than the process-group
timeout** and the early rank still completes — the property the transport exists for, which
``all_gather_object`` cannot provide. Thread participants, each on its own client of one real store
server, prove the verdict across repeated phases of one tag with no, one and many failing ranks, that the
phases leave no key behind (also when the last participant out had timed out), and that a clean join
costs each participant a constant number of key operations at any world size: every checkpoint save joins
twice, and the one store server serves each key a participant waits on or reads. FakeStore paths pin the
verdict logic, the order a failing rank records and arrives in, the empty-reason contract and the timeout
diagnostic single-process.

    python tests/cpu/runtime/test_store_reject.py
"""

import datetime
import os
import threading
import time
from collections.abc import Callable

import pytest
import torch.distributed as dist

from halo.runtime import consensus
from halo.runtime.consensus import gathered_failure_summary, store_reject_across_ranks
from tests.common.distributed import FakeStore
from tests.common.gloo import run_gloo_ranks
from tests.common.ports import free_port

WORLD_SIZE = 2

# The late rank must be allowed to outlive the process-group timeout: that is the whole point of
# joining on the store. Kept far apart so a slow CI box cannot blur the two, and the PG timeout
# itself wide enough that spawn + torch-import skew between the ranks on a loaded host cannot trip
# the rendezvous before the join is even reached.
PG_TIMEOUT_SEC = 5
LATE_JOIN_SEC = 8

# Every store wait is explicitly bounded: under the 4 h production default a regression would surface
# only as a rank killed at the join deadline, not as the wait's own error.
WAIT_TIMEOUT = datetime.timedelta(seconds=30)

STORE_HOST = "127.0.0.1"
# The prefix init_process_group puts the default store under.
DEFAULT_PG_PREFIX = "default_pg"
# Bounds a thread participant end to end; only a participant stuck past its own bounded waits reaches it.
THREAD_JOIN_TIMEOUT_S = 120.0
THREAD_WORLD_SIZE = 32
WHAT = "unit-test join"
# The phases one tag runs in turn: no failure, one, many (more than the message previews), all, none again.
FAILING_RANKS_PER_PHASE = (
    frozenset(),
    frozenset({5}),
    frozenset(range(1, THREAD_WORLD_SIZE, 3)),
    frozenset(range(THREAD_WORLD_SIZE)),
    frozenset(),
)
# The most key operations one participant makes in a clean join, whatever the world: its entry, arrival,
# wait, failure-count read and exit, the last arrival's set, and the last exit's count read and delete of
# the four keys.
CLEAN_JOIN_KEY_OPS = 11
# The bound of the participant the cleanup test makes give up waiting; the test orders the rest by events.
GIVE_UP_WAIT = datetime.timedelta(seconds=1)


def _worker(rank: int, tmp_dir: str, failing_rank: int) -> None:
    reason = "ValueError: boom" if rank == failing_rank else None
    try:
        store_reject_across_ranks("t", reason, WHAT, timeout=WAIT_TIMEOUT)
        result = "NO RAISE"
    except RuntimeError as e:
        result = f"RuntimeError: {e}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def _late_rank_worker(rank: int, tmp_dir: str) -> None:
    """Rank 0 arrives at the join LATE_JOIN_SEC late, past the PG timeout; both must still pass."""
    try:
        if rank == 0:
            time.sleep(LATE_JOIN_SEC)
        store_reject_across_ranks("late", None, "late-join", timeout=datetime.timedelta(seconds=60))
        result = "PASS"
    except Exception as e:  # the failure mode under test is exactly "a join that raises"
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    with open(os.path.join(tmp_dir, f"late_{rank}.txt"), "w") as fh:
        fh.write(result)


def test_clean_join_passes_on_both_ranks(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), -1)
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            assert fh.read() == "NO RAISE", f"rank {rank} raised on a clean join"


def test_one_ranks_reason_raises_on_every_rank_with_the_cause(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), 1)
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result != "NO RAISE", f"rank {rank} sailed past a failed join"
        assert "rank 1" in result and "boom" in result, f"rank {rank} was not told the real cause: {result}"


def test_late_rank_may_outlive_the_process_group_timeout(tmp_path):
    """A rank arriving later than the PG timeout must NOT fail the early rank.

    Fails when the join is a collective: gloo aborts the early rank's recv at PG_TIMEOUT_SEC — the
    30-min NCCL-watchdog kill of a real fresh-cache map, scaled down.
    """
    run_gloo_ranks(_late_rank_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"late_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


class _OpLog:
    """A store that logs each request in order with the keys it names: a wait or read naming W keys costs the
    one store server W key operations, which a request count would hide."""

    def __init__(self, store):
        self._store = store
        self.ops: list[tuple[str, list[str]]] = []

    @property
    def key_ops(self) -> int:
        return sum(len(keys) for _, keys in self.ops)

    def add(self, key, amount):
        self.ops.append(("add", [key]))
        return self._store.add(key, amount)

    def set(self, key, value):
        self.ops.append(("set", [key]))
        self._store.set(key, value)

    def delete_key(self, key):
        self.ops.append(("delete", [key]))
        return self._store.delete_key(key)

    def wait(self, keys, timeout):
        self.ops.append(("wait", list(keys)))
        self._store.wait(keys, timeout)

    def multi_get(self, keys):
        self.ops.append(("multi_get", list(keys)))
        return self._store.multi_get(keys)


class _OrderedStore:
    """A participant's store ordered against a peer's by events: an increment of a key ending in a ``hold``
    suffix first waits for that suffix's event, a wait that times out sets ``timed_out``, and an exit sets
    ``exited``."""

    def __init__(self, store, hold: dict[str, threading.Event], timed_out: threading.Event, exited: threading.Event):
        self._store = store
        self._hold = hold
        self._timed_out = timed_out
        self._exited = exited

    def __getattr__(self, name):
        return getattr(self._store, name)

    def add(self, key, amount):
        for suffix, event in self._hold.items():
            if amount and key.endswith(suffix):
                assert event.wait(THREAD_JOIN_TIMEOUT_S), f"the peer never released {key}"
        total = self._store.add(key, amount)
        if key.endswith("/exited"):
            self._exited.set()
        return total

    def wait(self, keys, timeout):
        try:
            self._store.wait(keys, timeout)
        except Exception:
            self._timed_out.set()
            raise


def _run_thread_world(
    monkeypatch,
    world: int,
    body: Callable[[int], object],
    wrap: Callable[[int, dist.Store], object] = lambda rank, store: store,
) -> tuple[list[object], list[object], dist.TCPStore]:
    """Run ``body(rank)`` for ``world`` ranks as threads, each on its own client connection to one real store
    server under the default group's prefix, as each process of a job holds one to the launcher's store.

    Returns each rank's result and store (``wrap(rank, prefix store)``), and the server.
    """
    server = dist.TCPStore(STORE_HOST, free_port(), is_master=True, timeout=WAIT_TIMEOUT, wait_for_workers=False)
    current = threading.local()
    monkeypatch.setattr(consensus, "is_multi_rank_run", lambda: True)
    monkeypatch.setattr(consensus, "get_global_rank", lambda: current.rank)
    monkeypatch.setattr(consensus, "get_global_world_size", lambda: world)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: current.store)
    results: list[object] = [None] * world
    stores: list[object] = [None] * world

    def participant(rank: int) -> None:
        try:
            client = dist.TCPStore(STORE_HOST, server.port, is_master=False, timeout=WAIT_TIMEOUT)
            current.rank = rank
            current.store = stores[rank] = wrap(rank, dist.PrefixStore(DEFAULT_PG_PREFIX, client))
            results[rank] = body(rank)
        except BaseException as exc:  # re-raised on the test's thread below
            results[rank] = exc

    threads = [threading.Thread(target=participant, args=(rank,), daemon=True) for rank in range(world)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + THREAD_JOIN_TIMEOUT_S
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    stuck = [rank for rank, thread in enumerate(threads) if thread.is_alive()]
    assert not stuck, f"participants {stuck} of {world} still in the join after {THREAD_JOIN_TIMEOUT_S}s"
    raised = [result for result in results if isinstance(result, BaseException)]
    if raised:
        raise raised[0]
    return results, stores, server


def _phase_reason(rank: int, phase: int) -> str:
    # Two lines: the reason survives whole, separator included.
    return f"OSError: phase {phase} write failed on rank {rank}\nNo space left on device"


def _join_every_phase(rank: int) -> list[str | None]:
    """This rank's verdict in each phase of one tag: the raised message, or None for a clean join."""
    verdicts = []
    for phase, failing in enumerate(FAILING_RANKS_PER_PHASE):
        reason = _phase_reason(rank, phase) if rank in failing else None
        try:
            store_reject_across_ranks("phases", reason, WHAT, timeout=WAIT_TIMEOUT)
            verdicts.append(None)
        except RuntimeError as e:
            verdicts.append(str(e))
    return verdicts


def test_every_rank_raises_the_gathered_message_in_every_phase_of_a_tag(monkeypatch):
    """No, one, many and all failing ranks, in turn under one tag: every rank raises exactly the message
    the world's reasons gather to, which names the lowest failing ranks and the lowest one's reason
    whichever rank recorded first, and the phases leave behind only the per-participant entry counters."""
    verdicts, _, server = _run_thread_world(monkeypatch, THREAD_WORLD_SIZE, _join_every_phase)

    for phase, failing in enumerate(FAILING_RANKS_PER_PHASE):
        reasons = [_phase_reason(rank, phase) if rank in failing else None for rank in range(THREAD_WORLD_SIZE)]
        expected = gathered_failure_summary(reasons, WHAT)
        for rank in range(THREAD_WORLD_SIZE):
            assert verdicts[rank][phase] == expected, f"rank {rank}, phase {phase}"
    assert verdicts[0][1] == (
        f"{WHAT} failed on 1 of {THREAD_WORLD_SIZE} rank(s) [5]. First (rank 5): {_phase_reason(5, 1)}"
    )
    assert server.num_keys() == THREAD_WORLD_SIZE, "a phase left keys beyond the entry counters behind"


def test_a_last_exit_whose_wait_timed_out_still_deletes_the_failure_slots(monkeypatch):
    """Rank 0 gives up waiting just before failing rank 1 arrives, and exits after rank 1. The last exit then
    never read a failure count of its own and must still delete rank 1's slot with the phase's other keys,
    or every such abort leaves it on the store for the rest of the job."""
    rank0_timed_out, rank1_exited = threading.Event(), threading.Event()
    orders = {
        0: ({"/exited": rank1_exited}, rank0_timed_out, threading.Event()),
        1: ({"/arrived": rank0_timed_out}, threading.Event(), rank1_exited),
    }

    def join(rank: int) -> str | None:
        try:
            store_reject_across_ranks(
                "gave_up",
                "OSError: disk full" if rank == 1 else None,
                WHAT,
                timeout=GIVE_UP_WAIT if rank == 0 else WAIT_TIMEOUT,
            )
        except RuntimeError as e:
            return str(e)
        return None

    verdicts, _, server = _run_thread_world(
        monkeypatch, 2, join, wrap=lambda rank, store: _OrderedStore(store, *orders[rank])
    )

    assert "'reject/gave_up/world' timed out" in verdicts[0] and "participants to arrive" in verdicts[0], verdicts[0]
    assert verdicts[1] == f"{WHAT} failed on 1 of 2 rank(s) [1]. First (rank 1): OSError: disk full"
    assert server.num_keys() == 2, "the timed-out last exit left the failure slot behind"


@pytest.mark.parametrize("world", [8, 64])
def test_a_clean_join_costs_each_participant_constant_key_operations(monkeypatch, world):
    """Fails on a join where every participant writes its own key and waits on and reads every peer's: each
    one then names O(world) keys, O(world²) for the one store server (seconds per join at 1024 ranks)."""

    def clean_join(rank: int) -> None:
        store_reject_across_ranks("clean", None, WHAT, timeout=WAIT_TIMEOUT)

    _, stores, _ = _run_thread_world(monkeypatch, world, clean_join, wrap=lambda rank, store: _OpLog(store))

    key_ops = [store.key_ops for store in stores]
    assert max(key_ops) <= CLEAN_JOIN_KEY_OPS, f"key operations per participant at world {world}: {key_ops}"


@pytest.fixture()
def fake_world(monkeypatch):
    """One rank of a 2-rank world against a shared in-memory store, peer keys pre-seedable."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(consensus, "get_global_rank", lambda: 0)
    monkeypatch.setattr(consensus, "get_global_world_size", lambda: 2)
    return store


def _phase_prefix(tag: str) -> str:
    # Rank 0's next entry of the tag is its first → phase 1.
    return f"_halo_phase/reject/{tag}/world/p1"


def _seed_peer_arrival(store: FakeStore, tag: str, reason: str | None = None) -> None:
    """Rank 1 has arrived at rank 0's next phase of ``tag``, having recorded ``reason`` in the first slot."""
    prefix = _phase_prefix(tag)
    if reason:
        store.kv[f"{prefix}/failures"] = 1
        store.kv[f"{prefix}/failed1"] = f"1{consensus._FAILURE_RECORD_SEP}{reason}"
    store.kv[f"{prefix}/arrived"] = 1


def test_verdict_names_the_failing_rank_and_honors_exc_type(fake_world):
    _seed_peer_arrival(fake_world, "vt", "OSError: disk full")
    with pytest.raises(ValueError, match=r"rank 1.*disk full"):
        store_reject_across_ranks("vt", None, "unit join", exc_type=ValueError, timeout=WAIT_TIMEOUT)


def test_an_empty_reason_is_no_failure(fake_world):
    """The same falsy contract reject_across_ranks has: an empty reason records nothing and passes."""
    _seed_peer_arrival(fake_world, "et")
    store_reject_across_ranks("et", "", "unit join", timeout=WAIT_TIMEOUT)
    assert fake_world.kv[f"{_phase_prefix('et')}/failures"] == 0


def test_a_failing_rank_records_its_slot_before_it_arrives(fake_world, monkeypatch):
    """The slot count is final once every rank has arrived only because each failing rank takes and writes
    its slot first: one that arrived first could complete the arrivals before its slot exists, and a peer
    they release would read a count without its failure and pass a failed join."""
    log = _OpLog(fake_world)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: log)
    _seed_peer_arrival(fake_world, "order")
    with pytest.raises(RuntimeError, match="disk full"):
        store_reject_across_ranks("order", "OSError: disk full", "unit join", timeout=WAIT_TIMEOUT)
    prefix = _phase_prefix("order")
    writes = [(op, keys[0]) for op, keys in log.ops if op in ("add", "set")]
    take, record, arrive = (
        writes.index(("add", f"{prefix}/failures")),
        writes.index(("set", f"{prefix}/failed1")),
        writes.index(("add", f"{prefix}/arrived")),
    )
    assert take < record < arrive, f"store writes out of order: {writes}"


def test_own_reason_raises_even_when_peers_are_clean(fake_world):
    _seed_peer_arrival(fake_world, "ot")
    with pytest.raises(RuntimeError, match=r"rank 0.*KeyError: 'prompt'"):
        store_reject_across_ranks("ot", "KeyError: 'prompt'", "unit join", timeout=WAIT_TIMEOUT)


def test_the_first_reason_is_the_lowest_failing_rank_s_not_the_first_recorded(fake_world):
    """Rank 1 took the first slot; the message still leads with rank 0, as a rank-indexed gather does."""
    _seed_peer_arrival(fake_world, "lo", "OSError: disk full")
    with pytest.raises(RuntimeError) as raised:
        store_reject_across_ranks("lo", "KeyError: 'prompt'", "unit join", timeout=WAIT_TIMEOUT)
    assert str(raised.value) == "unit join failed on 2 of 2 rank(s) [0, 1]. First (rank 0): KeyError: 'prompt'"


def test_a_clean_join_reads_no_reason_and_a_failed_one_reads_them_in_one_round_trip(fake_world):
    _seed_peer_arrival(fake_world, "clean")
    store_reject_across_ranks("clean", None, "unit join", timeout=WAIT_TIMEOUT)
    assert fake_world.multi_get_keys == [] and fake_world.get_keys == []

    _seed_peer_arrival(fake_world, "failed", "OSError: disk full")
    with pytest.raises(RuntimeError, match=r"2 of 2 rank\(s\)"):
        store_reject_across_ranks("failed", "KeyError: 'prompt'", "unit join", timeout=WAIT_TIMEOUT)
    prefix = _phase_prefix("failed")
    assert fake_world.multi_get_keys == [[f"{prefix}/failed1", f"{prefix}/failed2"]]
    assert fake_world.get_keys == []


def test_a_timed_out_join_reports_how_many_arrived(fake_world):
    """The diagnostic counts the arrivals rather than listing a key per rank, so it stays one line at any
    world size and still says whether a peer is late or the ranks diverged."""
    with pytest.raises(RuntimeError, match=r"'reject/tt/world'.*all 2 participants to arrive \(1 did\)"):
        store_reject_across_ranks("tt", None, "unit join", timeout=WAIT_TIMEOUT)


def test_not_distributed_raises_locally_and_passes_clean():
    with pytest.raises(ValueError, match="local cause"):
        store_reject_across_ranks("nd", "local cause", "unit join", exc_type=ValueError)
    store_reject_across_ranks("nd", None, "unit join")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
