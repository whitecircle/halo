"""One NCCL group across several rollout servers: the trainer is rank 0, each server's workers a contiguous block of
ranks, and every chunk is broadcast once. The first server's client owns the transport; every server's client keeps
its own pause, session and declare RPCs."""

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
from halo.weight_sync.clients.server_calls import AsyncCall, announced_wait, wait_announced, wait_for_calls
from halo.weight_sync.group_layout import (
    reject_mixed_localities,
    reject_pipeline_stages,
    reject_split_nodes,
    resolve_group_layout,
)
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S
from halo.weight_sync.transport.rendezvous import is_local_address, open_group_rendezvous
from halo.weight_sync.update_stream import UpdateStream

logger = logging.getLogger(__name__)

# c10d registers group names process-globally, and an engine refuses a second join under a live name.
_generations = itertools.count(1)

# CUDA's error text for a device index the process cannot see, which NCCL surfaces when a peer GPU is hidden from it.
_HIDDEN_PEER_GPU = "invalid device ordinal"


class WeightSyncGroup(UpdateStream):
    """The weight-sync clients of several servers behind one update stream. Until the first chunk is declared an update
    can be cancelled or replayed over a re-formed group; from then on a failure leaves every server paused and
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
        self.BACKEND_NAME = members[0].BACKEND_NAME
        self.CO_LOADED_PARAM_GROUPS = members[0].CO_LOADED_PARAM_GROUPS
        super().__init__()
        self._members = list(members)
        for member in self._members:
            member.group_releases_remote_half = True
        self._nodes = list(nodes)
        self._group_port = group_port
        self._group_host = group_host
        self._group_name = ""
        self._offsets: list[int] = []
        self._engine_worlds: list[int] = []
        self._pipeline_stages: list[int] = []
        self._world_size = 0

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
        """The client that owns the transport."""
        return self._members[0]

    def init_communicator(self, device: torch.device | str | int = 0) -> None:
        """Form one group over every server, the trainer's half on ``device``; any failure leaves nothing held."""
        sync_device = resolve_sync_device(device)
        self._resolve_layout(require=True)
        self.leader.sync_device = sync_device
        self._form()
        atexit.register(self.close_communicator, at_exit=True)
        logger.info(
            f"Weight-sync group of {len(self._members)} {self.BACKEND_NAME} servers formed: "
            + ", ".join(
                f"{url} ranks {offset}-{offset + world - 1}"
                for url, offset, world in zip(self._urls, self._offsets, self._engine_worlds, strict=True)
            )
        )

    def abort_weight_update(self) -> None:
        self.drain_param_buffer()
        if not self._update_open:
            return
        self._update_open = False
        if self._chunks_sent:
            self._mark_unusable("the sync was interrupted between chunks")
            return
        self._run_on_members(lambda member: member.cancel_engine_update("after an interrupted sync"))

    def close_communicator(self, *, at_exit: bool = False) -> None:
        self.abort_weight_update()
        halves = self._destroy_remote_halves()
        self._run_on_members(lambda member: member.close_communicator(at_exit=at_exit))
        self._join_all(halves)
        atexit.unregister(self.close_communicator)

    def resume_engine(self) -> None:
        self._run_on_members(lambda member: member.resume_engine())

    def require_resumed(self) -> None:
        self._require_all(self._run_on_members(lambda member: member.require_resumed()), "resume")

    @property
    def _urls(self) -> list[str]:
        return [member.base_url for member in self._members]

    def _resolve_layout(self, *, require: bool, read: Sequence[WeightSyncClientBase] | None = None) -> None:
        """Read the layout of ``read`` (default every server) and give each server a block of ranks, node by node. A
        re-form reads only the servers that lost the group, as a restarted one may come back with another layout while
        one that kept the group is still the engine it formed with."""
        known = dict(zip(self._urls, zip(self._engine_worlds, self._pipeline_stages, strict=True), strict=False))

        def read_layout(member: WeightSyncClientBase) -> tuple[int, int]:
            if read is not None and member not in read:
                return known[member.base_url]
            with announced_wait(member.base_url, "its parallel layout", member.layout_read_bound_s):
                return member.read_engine_layout(require=require)

        readings = self._require_all(self._run_on_members(read_layout), "read its layout")
        worlds = [world for world, _ in readings]
        reject_pipeline_stages(self.BACKEND_NAME, self._urls, [stages for _, stages in readings])
        reject_split_nodes(self._nodes, [member.host for member in self._members], self._urls)
        reject_mixed_localities(
            self._urls, [is_local_address(member.host) for member in self._members], self._group_host
        )
        layout = resolve_group_layout(self._nodes, worlds)
        self._members = [self._members[i] for i in layout.order]
        self._nodes = [self._nodes[i] for i in layout.order]
        self._offsets = [layout.rank_offsets[i] for i in layout.order]
        self._engine_worlds = [worlds[i] for i in layout.order]
        self._pipeline_stages = [readings[i][1] for i in layout.order]
        self._world_size = layout.world_size

    def _form(self, adopt: Sequence[WeightSyncClientBase] | None = None) -> None:
        """Rendezvous, every server's join and the trainer's half run together, as each blocks until the others
        arrive. ``adopt`` are the servers whose launch is read first (default every server); one that kept the group
        is the launch it formed with."""

        def adopt_launch(member: WeightSyncClientBase) -> None:
            if adopt is None or member in adopt:
                member.adopt_engine_identity()

        self._require_all(self._run_on_members(adopt_launch), "read its launch")
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
                    leader.prepare_local_formation(master_address, listener, self._world_size),
                    name=leader.FORMATION_STEP,
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
        """Each call is a server's destroy request, which announces its own expiry."""
        for call in calls:
            call.join(CLEANUP_TIMEOUT_S)

    def _start_engine_update(self) -> None:
        """Probe every server, then quiesce them together. A server that restarted since the group formed is found
        before any chunk is declared, so the group re-forms once with every server paused and the update replays."""
        self.leader.verify_transport()
        stale = self._stale_members(lambda member: member.verify_engine_identity())
        self._quiesce_members()
        try:
            stale = stale or self._stale_members(lambda member: member.verify_group_membership())
            if stale:
                reformed = self._reform(stale)
                stale = self._stale_members(
                    lambda member: member.verify_engine_identity(), reformed
                ) or self._stale_members(lambda member: member.verify_group_membership())
                if stale:
                    raise RuntimeError(
                        f"The weight-sync group still lacks {[url for url, _ in stale]} after one re-form: {stale[0][1]}"
                    )
        except Exception:
            self._run_on_members(lambda member: member.cancel_engine_update("after a failed group probe"))
            if not self.leader.has_transport:
                self._unusable_reason = (
                    f"The {self.BACKEND_NAME} weight-sync group was lost in a failed re-form. RESTART the "
                    f"servers: {', '.join(self._urls)}."
                )
            raise

    def _stale_members(
        self, probe: Callable[[WeightSyncClientBase], None], only: Sequence[WeightSyncClientBase] | None = None
    ) -> list[tuple[str, Exception]]:
        """``(url, error)`` of each server that lost the group, probing ``only`` (default every server); any other
        failure raises, naming its server."""
        outcomes = self._run_on_members(
            lambda member: None if only is not None and member not in only else probe(member)
        )
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

    def _reform(self, stale: list[tuple[str, Exception]]) -> list[WeightSyncClientBase]:
        """Re-form the group over servers that are paused: the sessions are closed, both halves destroyed and the group
        joined again under a new generation, then the sessions reopen. Only the servers that lost the group have their
        layout and launch read again; returns them."""
        lost = [member for member in self._members if member.base_url in {url for url, _ in stale}]
        logger.warning(f"Re-forming the weight-sync group: {'; '.join(f'{url}: {error}' for url, error in stale)}")
        self._run_on_members(lambda member: member.abandon_update_session())
        # A lost member never enters its half, so a destroy of ours would wait on it and strand the survivors' halves.
        self.leader.abort_transport()
        halves = self._destroy_remote_halves()
        self.leader.discard_transport(None)
        self._join_all(halves)
        self._resolve_layout(require=False, read=lost)
        self._form(lost)
        self._require_all(self._run_on_members(lambda member: member.open_update_session()), "reopen its session")
        return lost

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare ``chunk`` on every server and broadcast it once. From the first declare on a failure is global. The
        send runs on its own thread, as a server that refused its declare would otherwise leave the trainer blocked in
        a broadcast that cannot finish until its deadline."""
        self.leader.prepare_chunk_send(chunk)
        declared: list[AsyncCall] = []
        try:
            for member in self._members:
                declared.append(member.declare_chunk(chunk, final))
            sent = AsyncCall(partial(self._send_on_leader, chunk), name="the group's broadcast")
            wait_for_calls([*declared, sent], timeout=CHUNK_SYNC_TIMEOUT_S)
        except BaseException as e:
            self._fail_group(e, declared)

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
        for member, call in zip(self._members, declared, strict=False):
            if call.is_alive():
                wait_announced(
                    call, member.base_url, "its answer to the failed chunk", max(0.0, deadline - time.monotonic())
                )
        failed = [
            member.base_url for member, call in zip(self._members, declared, strict=False) if call.error is not None
        ]
        cause = next((call.error for call in declared if call.error is not None), error)
        rest = [url for url in self._urls if url not in failed]
        self._chunks_sent += 1
        self._mark_unusable(f"a chunk failed: {type(cause).__name__}: {cause}")
        remedy = ""
        if any(_HIDDEN_PEER_GPU in str(call.error) for call in declared if call.error is not None):
            remedy = (
                " A server reported an invalid device ordinal: servers that each see only their own GPU cannot share "
                "one weight-update group, because NCCL peer-to-peer needs every peer's GPU visible. Mount all GPUs "
                "in each server and select with CUDA_VISIBLE_DEVICES, as the compose files do, or set "
                "rollout_sync_topology: per_server."
            )
        raise RuntimeError(
            f"{self.BACKEND_NAME} weight sync to {len(self._members)} servers failed mid-stream "
            f"({type(cause).__name__}: {cause}). One broken group leaves every server holding "
            f"{self.leader.INTERRUPTED_UPDATE_STATE}, so all are left PAUSED and refuse further syncs. RESTART the "
            f"{self.BACKEND_NAME} servers: {', '.join([*failed, *rest])}."
            f"{remedy}{self.leader.chunk_failure_remedy([call.error for call in declared if call.error is not None])}"
        ) from error

    def _mark_unusable(self, why: str = "a sync failed after its first chunk") -> None:
        if self._unusable_reason is not None:
            return
        logger.error(f"Weight-sync group of {len(self._members)} servers is unusable: {why}")
        self._update_open = False
        self._unusable_reason = (
            f"The {self.BACKEND_NAME} weight-sync group was left mid-update ({why}), so every server is left "
            f"PAUSED. RESTART the servers: {', '.join(self._urls)}."
        )
        for member in self._members:
            member.mark_unusable_after(self._chunks_sent)

    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        self._transmit_chunk(tail, final=True)
        self.leader.release_send_buffers()
        self._require_all(
            self._run_on_members(lambda member: member.end_engine_update(flushed=bool(tail))), "close its update"
        )

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
