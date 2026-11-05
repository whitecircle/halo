"""One NCCL group across several rollout servers: the trainer is rank 0, each server's workers a contiguous block of
ranks, and every chunk is broadcast once. The first server's client owns the transport and the chunk buffer; every
server's client keeps its own pause, session and declare RPCs."""

import atexit
import itertools
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import NoReturn

import torch

from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    SERVER_ERROR_GRACE_S,
    GroupMembershipLost,
    WeightSyncClientBase,
)
from halo.weight_sync.clients.server_calls import AsyncCall, wait_for_calls
from halo.weight_sync.group_layout import reject_split_nodes, resolve_group_layout
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S
from halo.weight_sync.transport.rendezvous import is_local_address, open_group_rendezvous

logger = logging.getLogger(__name__)

# c10d registers group names process-globally, and an engine refuses a second join under a live name.
_generations = itertools.count(1)


class WeightSyncGroup:
    """The weight-sync clients of several servers behind one client's interface. Until the first chunk is declared an
    update can be cancelled or replayed over a re-formed group; from then on a failure leaves every server paused and
    unusable, as parked receives cannot be freed from the trainer."""

    def __init__(
        self,
        members: Sequence[WeightSyncClientBase],
        *,
        nodes: Sequence[str],
        group_port: int,
        group_host: str | None,
    ):
        """``members`` and ``nodes`` are in config order; formation sorts both by node."""
        self._members = list(members)
        self._nodes = list(nodes)
        self._group_port = group_port
        self._group_host = group_host
        self._group_name = ""
        self._offsets: list[int] = []
        self._engine_worlds: list[int] = []
        self._world_size = 0
        self._chunks_sent = 0
        self._update_open = False
        self._unusable_reason: str | None = None

    @property
    def world_size(self) -> int:
        """Ranks in the group, the trainer's included; zero before formation."""
        return self._world_size

    @property
    def rank_blocks(self) -> list[tuple[int, int]]:
        """``(first rank, engine ranks)`` of each server in rank order."""
        return list(zip(self._offsets, self._engine_worlds, strict=True))

    @property
    def is_unusable(self) -> bool:
        """Whether a mid-update failure left every server paused and refusing syncs."""
        return self._unusable_reason is not None

    @property
    def leader(self) -> WeightSyncClientBase:
        """The client that owns the transport and the chunk buffer."""
        return self._members[0]

    def init_communicator(self, device: torch.device | str | int = 0) -> None:
        """Form one group over every server, the trainer's half on ``device``; any failure leaves nothing held."""
        sync_device = resolve_sync_device(device)
        worlds = self._require_all(self._run_on_members(self._read_engine_layout), "read its layout")
        reject_split_nodes(self._nodes, [member.host for member in self._members], self._urls)
        layout = resolve_group_layout(self._nodes, worlds)
        self._members = [self._members[i] for i in layout.order]
        self._offsets = [layout.rank_offsets[i] for i in layout.order]
        self._engine_worlds = [worlds[i] for i in layout.order]
        self._world_size = layout.world_size
        self.leader.sync_device = sync_device
        self._form()
        atexit.register(self.close_communicator, at_exit=True)
        logger.info(
            f"Weight-sync group of {len(self._members)} {self.leader.BACKEND_NAME} servers formed: "
            + ", ".join(
                f"{url} ranks {offset}-{offset + world - 1}"
                for url, offset, world in zip(self._urls, self._offsets, self._engine_worlds, strict=True)
            )
        )

    def flush_chunk(self) -> None:
        """Send what is buffered as one chunk into the open update; an incomplete co-load group stays buffered."""
        chunk, held = self.leader.pending_chunk()
        if not chunk:
            return
        self._open_update()
        self._transmit_chunk(chunk)
        self.leader.commit_sent_chunk(held)

    def finish_weight_update(self) -> None:
        """Send the tail chunk, close every server's update and resume it; no-op if nothing was buffered."""
        if not self.leader.has_buffered_params and not self._update_open:
            return
        self.leader.require_no_held_params()
        self._open_update()
        self._close_update(self.leader.drain_param_buffer())
        self._chunks_sent = 0

    def abort_weight_update(self) -> None:
        """Drop the buffer and close an update the sync cannot finish; never raises. Once a chunk went out the servers
        hold neither policy, so they stay paused."""
        self.leader.drain_param_buffer()
        if not self._update_open:
            return
        self._update_open = False
        if self._chunks_sent:
            self._mark_all_unusable(self._chunks_sent, "the sync was interrupted between chunks")
            return
        self._run_on_members(lambda member: member.cancel_engine_update("after an interrupted sync"))

    def close_communicator(self, *, at_exit: bool = False) -> None:
        """Close an interrupted update, lift every pause and release both halves of the group."""
        self.abort_weight_update()
        halves = self._destroy_remote_halves()
        self._run_on_members(lambda member: member.close_communicator(at_exit=at_exit))
        self._join_all(halves)
        atexit.unregister(self.close_communicator)

    @property
    def _urls(self) -> list[str]:
        return [member.base_url for member in self._members]

    def _read_engine_layout(self, member: WeightSyncClientBase) -> int:
        world = member.fetch_engine_world_size()
        member.reject_group_membership()
        return world

    def _form(self) -> None:
        """Rendezvous, every server's join and the trainer's half run together, as each blocks until the others
        arrive."""
        self._group_name = f"weight_update_group_g{next(_generations)}"
        leader = self.leader
        # A loopback address must be refused when any server is remote, and a route read toward a remote one.
        reference = next((member for member in self._members if not is_local_address(member.host)), leader)
        master_address, listener = open_group_rendezvous(
            engine=leader.BACKEND_NAME,
            server_url=reference.base_url,
            server_host=reference.host,
            advertised_host=self._group_host,
            host_env=leader.GROUP_HOST_ENV,
            port=self._group_port,
        )
        with listener:
            joins: list[AsyncCall] = []
            local: AsyncCall | None = None
            try:
                joins = self._require_all(
                    self._run_on_members(
                        lambda member: member.request_engine_join(
                            master_address,
                            listener.port,
                            self._world_size,
                            self._offsets[self._members.index(member)],
                            self._group_name,
                        )
                    ),
                    "start joining the group",
                )
                local = AsyncCall(
                    lambda: leader.prepare_local_formation(master_address, listener, self._world_size)(),
                    name="the trainer's half of the weight-sync group",
                )
                wait_for_calls([*joins, local], timeout=GROUP_FORMATION_TIMEOUT_S)
                leader.adopt_transport(local)
            except Exception as e:
                halves = self._destroy_remote_halves()
                leader.discard_transport(local)
                self._join_all(halves)
                raise self._formation_error(e, joins, master_address, listener.port) from e

    def _formation_error(self, error: Exception, joins: list[AsyncCall], master_address: str, port: int) -> Exception:
        """The failure naming the server whose join failed; else the address and its knob, as a wrong-but-routable
        master surfaces as a bare store or handshake timeout."""
        if not joins:
            return error
        for member, join in zip(self._members, joins, strict=True):
            if error is join.error:
                return RuntimeError(
                    f"{member.BACKEND_NAME} server {member.base_url} failed to join the weight-sync group of "
                    f"{len(self._members)} servers: {type(error).__name__}: {error}"
                )
        return RuntimeError(self.leader.formation_failure(master_address, port, ", ".join(self._urls), error))

    def _destroy_remote_halves(self) -> list[AsyncCall]:
        """Every engine's half of the group is destroyed while the trainer's is, as under cuMem each side's finalize
        waits for the other."""
        return [AsyncCall(member.destroy_remote_group) for member in self._members]

    @staticmethod
    def _join_all(calls: list[AsyncCall]) -> None:
        for call in calls:
            call.join(CLEANUP_TIMEOUT_S)

    def _open_update(self) -> None:
        """Probe every server, then quiesce them together. A server that restarted since the group formed is found
        before any chunk is declared, so the group re-forms once with every server paused and the update replays."""
        if self._update_open:
            return
        if self._unusable_reason is not None:
            raise RuntimeError(self._unusable_reason)
        self._chunks_sent = 0
        self.leader.require_transport()
        stale = self._stale_members(lambda member: member.verify_engine_identity())
        self._quiesce_members()
        try:
            stale = stale or self._stale_members(lambda member: member.verify_group_membership())
            if stale:
                self._reform(stale)
                stale = self._stale_members(lambda member: member.verify_engine_identity()) or self._stale_members(
                    lambda member: member.verify_group_membership()
                )
                if stale:
                    raise RuntimeError(
                        f"The weight-sync group still lacks {[url for url, _ in stale]} after one re-form: {stale[0][1]}"
                    )
        except Exception:
            self._run_on_members(lambda member: member.cancel_engine_update("after a failed group probe"))
            raise
        self._update_open = True

    def _stale_members(self, probe: Callable[[WeightSyncClientBase], None]) -> list[tuple[str, Exception]]:
        """``(url, error)`` of each server that lost the group; any other failure raises, naming its server."""
        outcomes = self._run_on_members(probe)
        self._require_all(
            [(None, None if isinstance(error, GroupMembershipLost) else error) for _, error in outcomes], "be probed"
        )
        return [
            (member.base_url, error)
            for member, (_, error) in zip(self._members, outcomes, strict=True)
            if error is not None
        ]

    def _quiesce_members(self) -> None:
        """Pause and open every server's session together; one failing cancels the rest, so none is left paused."""
        started = time.monotonic()

        def open_one(member: WeightSyncClientBase) -> float:
            member.open_engine_update()
            return time.monotonic() - started

        outcomes = self._run_on_members(open_one)
        if any(error is not None for _, error in outcomes):
            paused = [member for member, (_, error) in zip(self._members, outcomes, strict=True) if error is None]
            self._run_on_members(
                lambda member: member.cancel_engine_update("after another server failed to pause"), paused
            )
        pauses = self._require_all(outcomes, "pause")
        timings = ", ".join(f"{url} {seconds:.2f}s" for url, seconds in zip(self._urls, pauses, strict=True))
        logger.info(
            f"Weight-sync pause {timings}; the first chunk waited {max(pauses) - min(pauses):.2f}s on the slowest"
        )

    def _reform(self, stale: list[tuple[str, Exception]]) -> None:
        """Re-form the group over servers that are paused: the sessions are closed, both halves destroyed and the group
        joined again under a new generation, then the sessions reopen."""
        logger.warning(f"Re-forming the weight-sync group: {'; '.join(f'{url}: {error}' for url, error in stale)}")
        self._run_on_members(lambda member: member.abandon_update_session())
        halves = self._destroy_remote_halves()
        self.leader.discard_transport(None)
        self._join_all(halves)
        self._require_all(
            self._run_on_members(lambda member: member.adopt_engine_identity()), "read its launch after the restart"
        )
        self._form()
        self._require_all(self._run_on_members(lambda member: member.open_update_session()), "reopen its session")

    def _transmit_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool = False) -> None:
        """Declare ``chunk`` on every server and broadcast it once. From the first declare on a failure is global. The
        send runs on its own thread, as a server that refused its declare would otherwise leave the trainer blocked in
        a broadcast that cannot finish until its deadline."""
        if not chunk:
            return
        self.leader.prepare_chunk_send(chunk)
        declared: list[AsyncCall] = []
        try:
            declared = [member.declare_chunk(chunk, final) for member in self._members]
            sent = AsyncCall(partial(self._send_on_leader, chunk), name="the group's broadcast")
            wait_for_calls([*declared, sent], timeout=CHUNK_SYNC_TIMEOUT_S)
        except BaseException as e:
            self._fail_group(e, declared)
        self._chunks_sent += 1

    def _send_on_leader(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        device = self.leader.sync_device
        if device is not None and device.type == "cuda":
            torch.cuda.set_device(device)
        self.leader.send_chunk(chunk)

    def _fail_group(self, error: BaseException, declared: list[AsyncCall]) -> NoReturn:
        """Abort the trainer's half, which frees its kernels but not the servers parked in a receive, mark every server
        unusable and raise naming the failed one."""
        self.leader.abort_transport()
        deadline = time.monotonic() + SERVER_ERROR_GRACE_S
        for call in declared:
            call.join(max(0.0, deadline - time.monotonic()))
        failed = [
            member.base_url for member, call in zip(self._members, declared, strict=False) if call.error is not None
        ]
        cause = next((call.error for call in declared if call.error is not None), error)
        rest = [url for url in self._urls if url not in failed]
        self._mark_all_unusable(self._chunks_sent + 1, f"a chunk failed: {type(cause).__name__}: {cause}")
        raise RuntimeError(
            f"{self.leader.BACKEND_NAME} weight sync to {len(self._members)} servers failed mid-stream "
            f"({type(cause).__name__}: {cause}). One broken group leaves every server holding "
            f"{self.leader.INTERRUPTED_UPDATE_STATE}, so all are left PAUSED and refuse further syncs. RESTART the "
            f"{self.leader.BACKEND_NAME} servers: {', '.join([*failed, *rest])}."
        ) from error

    def _mark_all_unusable(self, chunks_sent: int, why: str) -> None:
        logger.error(f"Weight-sync group of {len(self._members)} servers is unusable: {why}")
        self._update_open = False
        self._unusable_reason = (
            f"The {self.leader.BACKEND_NAME} weight-sync group was left mid-update ({why}), so every server is left "
            f"PAUSED. RESTART the servers: {', '.join(self._urls)}."
        )
        for member in self._members:
            member.mark_unusable_after(chunks_sent)

    def _close_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send the final chunk, close every session and resume every server, whatever the tail does."""
        snapshot_error: Exception | None = None
        try:
            self.leader.complete_snapshots()
        except Exception as e:
            # Only the close can end the open sessions. The tail goes, as its copies may not have landed.
            logger.error(f"Weight-sync snapshots did not complete: {e}")
            snapshot_error, tail = e, []
            if self._chunks_sent:
                self._mark_all_unusable(self._chunks_sent, "the tail's snapshots did not complete")
        try:
            self._transmit_chunk(tail, final=True)
            self.leader.release_send_buffers()
            self._require_all(
                self._run_on_members(
                    lambda member: member.close_engine_update(flushed=bool(tail), chunks_sent=self._chunks_sent)
                ),
                "close its update",
            )
        finally:
            self._update_open = False
        if snapshot_error is not None:
            raise snapshot_error

    def _run_on_members(
        self, step: Callable[[WeightSyncClientBase], object], members: Sequence[WeightSyncClientBase] | None = None
    ) -> list[tuple[object, Exception | None]]:
        """``(result, error)`` of ``step`` on ``members`` (default all), run together."""

        def run(member: WeightSyncClientBase) -> tuple[object, Exception | None]:
            try:
                return step(member), None
            except Exception as e:
                return None, e

        targets = self._members if members is None else members
        if not targets:
            return []
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            return list(executor.map(run, targets))

    def _require_all(self, outcomes: Sequence[tuple[object, Exception | None]], what: str) -> list:
        """The results of ``outcomes``, or a raise naming each server that failed to ``what``."""
        failures = [
            f"{member.base_url}: {type(error).__name__}: {error}"
            for member, (_, error) in zip(self._members, outcomes, strict=True)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                f"{self.leader.BACKEND_NAME} weight-sync group: {len(failures)}/{len(self._members)} server(s) "
                f"failed to {what}. {'; '.join(failures)}"
            ) from next(error for _, error in outcomes if error is not None)
        return [result for result, _ in outcomes]
