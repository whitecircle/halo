#!/usr/bin/env python
"""A weight-sync send that fails on the forwarding rank must raise on every rank. The push is a loop of group-wide
gathers, and a rank leaving it parks every peer in the next gather until the NCCL watchdog fires."""

from __future__ import annotations

import datetime
import os
import types
from typing import ClassVar

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from halo.distributed.expert_parallel.moe_layer import gather as gather_module
from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.grpo.rollout.weight_push import _send_ep_expert_weights, sync_weights_to_client
from tests.common.ep_stubs import ReadyGatherMixin
from tests.common.gloo import new_groups, run_gloo_ranks

WORLD_SIZE = 2
NUM_PARAMS, ROWS, COLS = 5, 8, 4
# Mid-loop by construction: gathers must still run for the params after it.
FAIL_AT = 3

# A stuck rank surfaces as a failed assertion, since gloo aborts the blocked collective at the group timeout. That
# timeout is far above the work it bounds because the suite runs 8-way under xdist, where a starved rendezvous is no
# regression.
PG_TIMEOUT = datetime.timedelta(seconds=90)


class _EngineRejected(RuntimeError):
    """What the client raises mid-push: an engine 500, a refused dtype, a host OOM on the snapshot."""


class _FakeClient:
    """Records what it was handed and fails where the scenario says. Non-forwarding ranks get one too, to pin the rank
    gate in ``sync_weights_to_client``."""

    def __init__(self, fail_send_at: int = 0, fail_flush: bool = False):
        self._fail_send_at = fail_send_at
        self._fail_flush = fail_flush
        self.sent: list[str] = []
        self.flushes = 0
        self.aborts = 0

    def scope_co_load_groups(self, module_names) -> None:
        """Nothing in this policy is co-loaded; the forwarding rank scopes its client before the push."""

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent.append(name)
        if self._fail_send_at and len(self.sent) == self._fail_send_at:
            raise _EngineRejected(f"engine refused parameter {len(self.sent)}")

    def finish_weight_update(self) -> None:
        self.flushes += 1
        if self._fail_flush:
            raise _EngineRejected("engine refused the buffered broadcast")

    def abort_weight_update(self) -> None:
        self.aborts += 1


class _ShardedPolicy(nn.Module):
    """Params are DTensors, as ``fully_shard`` leaves them, so ``materialize_dtensor`` is a group-wide collective."""

    def __init__(self, mesh):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(COLS, ROWS, bias=False) for _ in range(NUM_PARAMS))
        for layer in self.layers:
            layer.weight = nn.Parameter(distribute_tensor(layer.weight.detach(), mesh, [Shard(0)]))

    def forward(self, x):  # pragma: no cover - never called
        return x


def _worker(rank: int, tmp_dir: str, mode: str) -> None:
    """Writes its verdict to a file, since the failure under test is a rank that never returns."""
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


def _run_ranks(tmp_path, mode: str) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), mode, pg_timeout=PG_TIMEOUT)
    results, counters = {}, {}
    for rank in range(WORLD_SIZE):
        result_file, sent_file = tmp_path / f"result_{rank}.txt", tmp_path / f"sent_{rank}.txt"
        results[rank] = result_file.read_text() if result_file.exists() else "NO RESULT (the rank never returned)"
        counts = sent_file.read_text().split() if sent_file.exists() else ("-1", "-1", "-1")
        counters[rank] = tuple(int(value) for value in counts)
    return results, counters


def test_a_mid_loop_send_failure_raises_on_every_rank(tmp_path):
    """Rank 1 reaching the uniform error is the proof: it gets there only if rank 0 entered every gather after the
    failure and joined the reject. Without the deferral it sits in ``full_tensor()`` for param 4 until gloo aborts."""
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
    """The flush sits after every gather, so nothing hangs, but a peer that sails past drives its next rollout round
    against an engine left paused mid-update."""
    results, counters = _run_ranks(tmp_path, "flush")

    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past a failed flush: {results}"
        assert "engine refused the buffered broadcast" in result, f"rank {rank} was not told the cause: {results}"
        assert "rank 0" in result, f"rank {rank} was not told which rank failed: {results}"

    assert counters[1][1] == 0, "a non-forwarding rank flushed the engine — the rank gate is gone"
    # The flush closes the update only on paths that reach the close. A raise before it leaves the engine quiesced
    # behind an open reload, and nothing else on this path ends it.
    assert counters[0][2] == 1, "the failing flush left the engine's update open — no abort followed it"


class _FakeEPLayer(ReadyGatherMixin):
    """A gather that assembles only when it retains. Every rank runs the collectives; only the forwarding rank builds
    the per-expert copies, which doubles a layer's peak there."""

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

    def send(self, name: str, tensor: torch.Tensor, *, bounded: bool = True) -> None:
        self.sent.append(name)

    def flush(self) -> None:
        pass


def test_a_retaining_gather_that_ooms_defers_and_stops_retaining():
    """The assembly sits between two group-wide gathers, so raising out of it parks every peer in the next layer's
    gather until the watchdog fires. Once a failure is recorded retaining must stop, or one OOM hits every layer."""
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
    layers = {f"layers.{i}": _FakeEPLayer() for i in range(4)}
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    forwarder = _CountingForwarder()

    _send_ep_expert_weights(layers, forwarder, guard)
    guard.reject()

    assert all(layer.retains == [True] for layer in layers.values())
    assert forwarder.sent == [f"layers.{i}.experts.gate_up_proj" for i in range(4)]


def test_a_clean_sync_still_pushes_every_param(tmp_path):
    results, counters = _run_ranks(tmp_path, "clean")

    assert results == {0: "NO RAISE", 1: "NO RAISE"}, f"a clean sync raised: {results}"
    assert counters[0] == (NUM_PARAMS, 1, 0), f"the forwarding rank did not push and flush every param: {counters[0]}"
    assert counters[1] == (0, 0, 0), "a non-forwarding rank talked to the engine"


# The real gather: the injected failure is rank 0's Nth receive, the rank holding earlier attributes through it.
_SEPARATE = MoEFamilySpec(
    model_types=("rank_failure_separate",),
    expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_GLU, per_expert_keys=("gate_proj", "up_proj", "down_proj")),
)
_FUSED = MoEFamilySpec(model_types=("rank_failure_fused",), expert_layout=ExpertLayout(ExpertLayoutKind.FUSED))
_LAYOUTS = {"separate": _SEPARATE, "fused": _FUSED}


class _GatherLayer(EPGatherMixin, nn.Module):
    has_expert_lora = False

    def __init__(self, spec, world_group, ep_group, ep_size, etp_group, etp_size, fill):
        nn.Module.__init__(self)
        self.FAMILY = spec
        self.ep_config = types.SimpleNamespace(process_group=world_group, dispatch_ep_group=ep_group, ep_size=ep_size)
        self.expert_tp_group, self.expert_tp_size = etp_group, etp_size
        self.expert_lora_scaling, self._expert_lora_attrs = 1.0, frozenset()
        self.gate_proj = nn.Parameter(torch.full((2, 6, 8), float(fill)))
        self.up_proj = nn.Parameter(torch.full((2, 6, 8), float(fill)))
        self.down_proj = nn.Parameter(torch.full((2, 8, 6), float(fill)))

    def _base_expert_named_params(self):
        return [("gate_proj", self.gate_proj), ("up_proj", self.up_proj), ("down_proj", self.down_proj)]

    expert_named_params = _base_expert_named_params


def _gather_worker(rank: int, out: str, ep_size: int, etp_size: int, fail_at: int | None, layout: str) -> None:
    real_gather, real_to_rank = gather_module.all_gather_cat, gather_module.gather_cat_to_rank
    real_hub = gather_module.to_hub_expert_tensors
    receives = {"count": 0}

    def count_receive(world_size: int) -> None:
        if world_size > 1 and rank == 0:
            receives["count"] += 1
            if receives["count"] == fail_at:
                raise torch.cuda.OutOfMemoryError(f"CUDA out of memory: injected at receive {fail_at}")

    def failing_gather(tensor, dim, group, world_size):
        count_receive(world_size)
        return real_gather(tensor, dim, group, world_size)

    def failing_gather_to_rank(tensor, group, world_size, *, dst, receive=True):
        count_receive(world_size)
        return real_to_rank(tensor, group, world_size, dst=dst, receive=receive)

    gather_module.all_gather_cat = failing_gather
    gather_module.gather_cat_to_rank = failing_gather_to_rank
    # The sync's destination device; this host has none.
    gather_module.to_hub_expert_tensors = lambda spec, tensors, *, device: real_hub(spec, tensors, device="cpu")
    world = ep_size * etp_size
    etp_ranks = [[r for r in range(world) if r // etp_size == i] for i in range(ep_size)]
    ep_ranks = [[r for r in range(world) if r % etp_size == i] for i in range(etp_size)]
    etp_group = new_groups(rank, etp_ranks, PG_TIMEOUT) if etp_size > 1 else None
    ep_group = new_groups(rank, ep_ranks, PG_TIMEOUT)
    world_group = new_groups(rank, [list(range(world))], PG_TIMEOUT)
    layers = {
        f"layers.{i}": _GatherLayer(_LAYOUTS[layout], world_group, ep_group, ep_size, etp_group, etp_size, rank + 1)
        for i in range(3)
    }
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    try:
        _send_ep_expert_weights(layers, _CountingForwarder() if rank == 0 else None, guard)
        guard.reject()
        result = "NO RAISE"
    except Exception as exc:
        result = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(out, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


@pytest.mark.parametrize(
    ("ep_size", "etp_size", "layout", "fail_at"),
    [
        (2, 1, "fused", 2),
        (2, 1, "separate", 2),
        (2, 1, "separate", 3),
        (2, 2, "separate", 3),
        (2, 2, "separate", 4),
        (2, 2, "fused", 3),
    ],
    ids=["ep2-fused", "ep2-separate-up", "ep2-separate-down", "ep2-etp2-etp-stage", "ep2-etp2-ep-stage", "etp2-fused"],
)
def test_a_failed_retained_receive_raises_on_every_rank_naming_the_cause(tmp_path, ep_size, etp_size, layout, fail_at):
    """The forwarding rank alone holds earlier attributes through a receive, so it is the rank that runs out. Moving on
    one collective short pairs the peers with the wrong tensor, silently where sizes match, until a group timeout with
    no cause on any peer."""
    run_gloo_ranks(
        _gather_worker, ep_size * etp_size, str(tmp_path), ep_size, etp_size, fail_at, layout, pg_timeout=PG_TIMEOUT
    )
    for rank in range(ep_size * etp_size):
        result = (tmp_path / f"result_{rank}.txt").read_text()
        assert "CUDA out of memory" in result, f"rank {rank}: {result}"
        assert "rank 0" in result, f"rank {rank} was not told which rank failed: {result}"


def test_a_clean_gather_over_the_real_groups_raises_nowhere(tmp_path):
    run_gloo_ranks(_gather_worker, 4, str(tmp_path), 2, 2, None, "separate", pg_timeout=PG_TIMEOUT)
    assert {(tmp_path / f"result_{rank}.txt").read_text() for rank in range(4)} == {"NO RAISE"}


class _ScriptedGather(EPGatherMixin):
    EXPERT_TP_SHARD_DIMS: ClassVar[dict[str, int]] = {}
    expert_tp_group, expert_tp_size = None, 1
    ep_config = types.SimpleNamespace(dispatch_ep_group=None, ep_size=2)


def test_a_failed_receive_drops_what_it_held_and_still_joins_every_collective(monkeypatch):
    calls = []

    def gather(tensor, dim, group, world_size):
        calls.append(tensor.shape[0])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return tensor

    monkeypatch.setattr(gather_module, "all_gather_cat", gather)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _ScriptedGather().gather_expert_tensors({"a": torch.zeros(1), "b": torch.zeros(2), "c": torch.zeros(3)})
    # The failed attempt and its retry for "b", then "c": the group saw each attribute once.
    assert calls == [1, 2, 2, 3]


def test_a_receive_with_nothing_to_drop_raises_at_once(monkeypatch):
    """That is the footprint every peer has too, so no retry could differ."""
    calls = []

    def gather(tensor, dim, group, world_size):
        calls.append(tensor)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(gather_module, "all_gather_cat", gather)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _ScriptedGather().gather_expert_tensors({"a": torch.zeros(1), "b": torch.zeros(2)})
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
