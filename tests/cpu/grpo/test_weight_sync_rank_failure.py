#!/usr/bin/env python
"""A weight-sync send that fails on the forwarding rank must raise on EVERY rank, never hang the rest.

The push is a loop of GROUP-WIDE gathers — one ``full_tensor()`` per dense param, one gather per EP
layer — with the forwarding rank's client calls interleaved between them. Those calls are the only
part of the sync that can fail on one rank alone: an HTTP error from the engine, a host OOM pinning
the snapshot, a tensor the engine's key space rejects. A rank that raises out of that loop stops
entering the gathers, so every peer blocks in the next one until the NCCL watchdog fires half an hour
later, with a traceback naming a collective rather than the engine. At 512 GPUs that is 511 ranks
parked on one rank's HTTP 500, and the real cause is in a log nobody reads.

So the sends run under a :class:`DeferredRankFailure`: the forwarding rank records the failure, keeps
entering every remaining collective, and the verdict is taken at a rank-uniform ``reject`` that
raises on all ranks with the originating rank and its real cause. Both halves of the push are covered
here — the streamed sends and the buffered flush (``reset_prefix_cache``, where the broadcast
actually lands) — plus the retaining expert gather, whose assembly is the largest rank-local
allocation in the sync and therefore the likeliest thing to fail on the forwarding rank alone.

Proven on a real 2-rank gloo group whose params are DTensors, so the gathers are genuine collectives
and the regression manifests as the hang it is: rank 1 can only be *told* the cause if rank 0 stayed
in every gather after the one that failed. The process-group timeout and the join are both bounded —
a regression fails this suite instead of wedging it.

    python tests/cpu/grpo/test_weight_sync_rank_failure.py
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from src.distributed.runtime import DeferredRankFailure
from src.trainers.grpo.rollout.weight_sync import _send_ep_expert_weights, sync_weights_to_client
from tests.common.ports import free_port

WORLD_SIZE = 2
NUM_PARAMS, ROWS, COLS = 5, 8, 4
# Mid-loop by construction: gathers must still run for the params after it.
FAIL_AT = 3

# A stuck rank must surface as a failed assertion, not a wedged worker: gloo aborts the blocked
# collective at the group timeout, the rank writes what it saw, and the bounded join covers the
# process itself. Both are far above the work they bound (spawn + torch import, then microseconds of
# gloo) because the suite runs 8-way under xdist, where a starved rendezvous is not a regression.
PG_TIMEOUT = datetime.timedelta(seconds=90)
JOIN_TIMEOUT_S = 420.0


class _EngineRejected(RuntimeError):
    """What the client raises mid-push: an engine 500, a refused dtype, a host OOM on the snapshot."""


class _FakeClient:
    """Records what it was handed, and fails where the scenario says to.

    Non-forwarding ranks get one too, so the test also pins that ``sync_weights_to_client``'s rank
    gate keeps them from sending anything.
    """

    def __init__(self, fail_send_at: int = 0, fail_flush: bool = False):
        self._fail_send_at = fail_send_at
        self._fail_flush = fail_flush
        self.sent: list[str] = []
        self.flushes = 0
        self.aborts = 0

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent.append(name)
        if self._fail_send_at and len(self.sent) == self._fail_send_at:
            raise _EngineRejected(f"engine refused parameter {len(self.sent)}")

    def reset_prefix_cache(self) -> None:
        self.flushes += 1
        if self._fail_flush:
            raise _EngineRejected("engine refused the buffered broadcast")

    def abort_weight_update(self) -> None:
        self.aborts += 1


class _ShardedPolicy(nn.Module):
    """A policy whose params are DTensors, as ``fully_shard`` leaves them — so the sync's
    ``materialize_dtensor`` is a real group-wide collective per parameter."""

    def __init__(self, mesh):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(COLS, ROWS, bias=False) for _ in range(NUM_PARAMS))
        for layer in self.layers:
            layer.weight = nn.Parameter(distribute_tensor(layer.weight.detach(), mesh, [Shard(0)]))

    def forward(self, x):  # pragma: no cover - never called
        return x


def _worker(rank: int, tmp_dir: str, port: int, mode: str) -> None:
    """One rank of the sync. Writes its verdict to a file — an exception here must not be the signal,
    since the failure under test is a rank that never returns at all."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE, timeout=PG_TIMEOUT)
    try:
        model = _ShardedPolicy(init_device_mesh("cpu", (WORLD_SIZE,)))
        client = _FakeClient(fail_send_at=FAIL_AT if mode == "send" else 0, fail_flush=mode == "flush")
        try:
            sync_weights_to_client(model, client, is_main=(rank == 0), is_tp_main=True)
            result = "NO RAISE"
        except Exception as e:
            result = f"{type(e).__name__}: {e}"
        with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
            fh.write(result)
        with open(os.path.join(tmp_dir, f"sent_{rank}.txt"), "w") as fh:
            fh.write(f"{len(client.sent)} {client.flushes} {client.aborts}")
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _run_ranks(tmp_path, mode: str) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    """Run both ranks to completion (bounded) and return their verdicts and client counters."""
    ctx = mp.start_processes(
        _worker, args=(str(tmp_path), free_port(), mode), nprocs=WORLD_SIZE, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    while not ctx.join(timeout=max(0.1, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in ctx.processes:
                process.terminate()
            pytest.fail(f"the ranks did not finish within {JOIN_TIMEOUT_S}s — a rank is stuck in a collective")

    results, counters = {}, {}
    for rank in range(WORLD_SIZE):
        result_file, sent_file = tmp_path / f"result_{rank}.txt", tmp_path / f"sent_{rank}.txt"
        results[rank] = result_file.read_text() if result_file.exists() else "NO RESULT (the rank never returned)"
        counts = sent_file.read_text().split() if sent_file.exists() else ("-1", "-1", "-1")
        counters[rank] = tuple(int(value) for value in counts)
    return results, counters


def test_a_mid_loop_send_failure_raises_on_every_rank(tmp_path):
    """The P0: rank 0's client fails on the 3rd param while rank 1 still has two gathers to go.

    Rank 1 reaching the uniform error is the whole proof — it can only get there if rank 0 entered
    every gather after the failure and joined the reject. Without the deferral rank 1 sits in
    ``full_tensor()`` for param 4 until gloo aborts it, and reports a transport error instead.
    """
    results, counters = _run_ranks(tmp_path, "send")

    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past a failed weight sync: {results}"
        assert "engine refused parameter 3" in result, f"rank {rank} was not told the real cause: {results}"
        assert "rank 0" in result, f"rank {rank} was not told which rank failed: {results}"

    assert counters[0][0] == FAIL_AT, "the failing rank kept sending after the failure (steps must be skipped)"
    assert counters[1][0] == 0, "a non-forwarding rank sent weights — the rank gate is gone"
    assert counters[0][2] == 1, (
        "the forwarding rank did not close the update it had opened — the engine stays quiesced "
        "behind an open reload, refusing every later sync and queueing every rollout"
    )
    assert counters[1][2] == 0, "a non-forwarding rank talked to the engine"


def test_a_failing_flush_raises_on_every_rank(tmp_path):
    """The buffered broadcast is where the weights actually go out, and it fails on rank 0 alone.

    It sits after every gather, so nothing hangs — but a peer that sails past it drives its next
    rollout round against an engine left paused mid-update, and the run dies later somewhere else.
    """
    results, counters = _run_ranks(tmp_path, "flush")

    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past a failed flush: {results}"
        assert "engine refused the buffered broadcast" in result, f"rank {rank} was not told the cause: {results}"
        assert "rank 0" in result, f"rank {rank} was not told which rank failed: {results}"

    assert counters[1][1] == 0, "a non-forwarding rank flushed the engine — the rank gate is gone"
    # The flush closes the update only on the paths that REACH the close: a raise before it (the
    # D2H completion of the tail's snapshots, a chunk the engine rejects) leaves the engine quiesced
    # behind an open reload, and nothing else on this path ends it.
    assert counters[0][2] == 1, "the failing flush left the engine's update open — no abort followed it"


class _FakeEPLayer:
    """An EP layer whose gather assembles ONLY when it retains, as the real ones do.

    Every rank runs the same collectives inside the gather; only the forwarding rank builds the
    per-expert copies from their outputs, which is what doubles a layer's peak there.
    """

    def __init__(self, oom_when_retaining: bool = False):
        self._oom = oom_when_retaining
        self.retains: list[bool] = []

    def gather_expert_state_dict(self, device: str, merge_lora: bool = False, retain: bool = True) -> dict:
        self.retains.append(retain)  # the collectives ran, retained or not
        if not retain:
            return {}
        if self._oom:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory: tried to allocate 28.00 GiB")
        return {"experts.gate_up_proj": torch.zeros(2, 2)}


class _CountingForwarder:
    """Stands in for ``_HubForwarder``: only its presence decides that this rank retains."""

    def __init__(self):
        self.sent: list[str] = []

    def send(self, name: str, tensor: torch.Tensor) -> None:
        self.sent.append(name)

    def flush(self) -> None:
        pass


def test_a_retaining_gather_that_ooms_defers_and_stops_retaining():
    """The forwarding rank's ASSEMBLY is the sync's largest rank-local allocation, and it sits
    between two group-wide gathers.

    Raising out of it takes this rank out of the loop while every peer blocks in the next layer's
    gather until the watchdog fires — the exact hang the deferral exists for, and the reason the
    gathers must run under the same guard the sends do. Once the failure IS recorded there is
    nothing left to send, so retaining must stop: paying the doubled peak for every remaining layer
    is what turns one OOM into an OOM on every layer.
    """
    layers = {f"layers.{i}": _FakeEPLayer(oom_when_retaining=i == 1) for i in range(4)}
    guard = DeferredRankFailure("weight-sync push to the rollout engine")

    _send_ep_expert_weights(layers, _CountingForwarder(), guard)

    assert [len(layer.retains) for layer in layers.values()] == [1, 1, 1, 1], (
        "a layer's gather was skipped after the failure — its peers block in that collective"
    )
    assert [layer.retains[0] for layer in layers.values()] == [True, True, False, False], (
        "the forwarding rank kept assembling layers it can no longer send"
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        guard.reject()


def test_a_healthy_gather_retains_and_forwards_every_layer():
    """Anti-vacuity: the guard must not turn retention off on a sync nobody failed."""
    layers = {f"layers.{i}": _FakeEPLayer() for i in range(4)}
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    forwarder = _CountingForwarder()

    _send_ep_expert_weights(layers, forwarder, guard)
    guard.reject()

    assert all(layer.retains == [True] for layer in layers.values())
    assert forwarder.sent == [f"layers.{i}.experts.gate_up_proj" for i in range(4)]


def test_a_clean_sync_still_pushes_every_param(tmp_path):
    """Anti-vacuity: the guard must not swallow a healthy sync, nor reject one nobody failed."""
    results, counters = _run_ranks(tmp_path, "clean")

    assert results == {0: "NO RAISE", 1: "NO RAISE"}, f"a clean sync raised: {results}"
    assert counters[0] == (NUM_PARAMS, 1, 0), f"the forwarding rank did not push and flush every param: {counters[0]}"
    assert counters[1] == (0, 0, 0), "a non-forwarding rank talked to the engine"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
