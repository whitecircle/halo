#!/usr/bin/env python
"""The environmental trainer's live weight sync runs fenced, once per optimizer step, and holds every rank until the
push lands. ``barrier_on_exit`` in ``sync_trainer_weights`` is the path's only barrier; online GRPO relies on it."""

import contextlib
import datetime
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

import halo.trainers.grpo.environmental.async_rollouts as async_mod
import halo.trainers.grpo.rollout.weight_push as weight_push
from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime import consensus
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights, sync_weights_to_client
from tests.common.gloo import run_gloo_ranks
from tests.common.weight_sync import RecordingSender

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=90)
# The forwarding rank's flush time. It is far above the peer's own return path, so an early release reads as a
# return ahead of the landing and cannot be a scheduling race.
_FLUSH_S = 0.5


def _stepping_trainer(monkeypatch, synced_at_step: int):
    """``synced_at_step`` is train-begin's forced push: 0 on a fresh run, the restored step on a resume."""
    monkeypatch.setattr(DistributedTrainerMixin, "training_step", lambda self, model, inputs, n=None: "loss")
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.state = SimpleNamespace(global_step=synced_at_step)
    trainer._last_sync_attempt_step = synced_at_step
    trainer._routing_injector = None
    trainer.fenced, trainer.unfenced = [], []
    trainer._sync_weights_to_engine_fenced = lambda force=False: trainer.fenced.append(trainer.state.global_step)
    trainer._sync_weights_to_engine = lambda force=False: trainer.unfenced.append(trainer.state.global_step)
    return trainer


@pytest.mark.parametrize("synced_at_step", [0, 7], ids=["fresh", "resumed"])
def test_each_optimizer_step_enters_one_fenced_sync(monkeypatch, synced_at_step):
    """Three microbatches per optimizer step: the first after each step syncs, its siblings and the
    microbatches of the step train-begin already pushed do not."""
    trainer = _stepping_trainer(monkeypatch, synced_at_step)
    for step in (synced_at_step, synced_at_step + 1, synced_at_step + 2):
        trainer.state.global_step = step
        assert [trainer.training_step(None, {}) for _ in range(3)] == ["loss"] * 3
    assert trainer.fenced == [synced_at_step + 1, synced_at_step + 2]
    assert trainer.unfenced == [], "the step entered the push without the fence that joins its verdict"


class _SlowFlush(RecordingSender):
    def finish_weight_update(self) -> None:
        time.sleep(_FLUSH_S)
        self.landed = time.monotonic()


class _ShardedPolicy(nn.Module):
    """Two DTensor params, as ``fully_shard`` leaves them, so the push gathers through real collectives."""

    def __init__(self, mesh):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 8, bias=False) for _ in range(2))
        for layer in self.layers:
            layer.weight = nn.Parameter(distribute_tensor(layer.weight.detach(), mesh, [Shard(0)]))

    def forward(self, x):  # pragma: no cover - never called
        return x


def _push_worker(rank: int, tmp_dir: str) -> None:
    fences = []
    real_fence = weight_push.barrier_on_exit

    @contextlib.contextmanager
    def counting_fence(what):
        with real_fence(what):
            yield
        fences.append(rank)

    weight_push.barrier_on_exit = counting_fence
    client = _SlowFlush() if rank == 0 else None
    trainer = SimpleNamespace(
        model=_ShardedPolicy(init_device_mesh("cpu", (WORLD_SIZE,))),
        parallelism_config=ParallelismConfig(),
        accelerator=SimpleNamespace(is_main_process=rank == 0),
        state=SimpleNamespace(global_step=1),
    )
    sync_trainer_weights(trainer, client)
    record = {"returned": time.monotonic(), "fences": len(fences)}
    if client is not None:
        record.update(landed=client.landed, sent=client.names)
    with open(os.path.join(tmp_dir, f"rank_{rank}.json"), "w") as fh:
        json.dump(record, fh)


def test_the_push_releases_no_rank_before_it_lands(tmp_path):
    run_gloo_ranks(_push_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    forwarding, peer = (json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(WORLD_SIZE))
    assert forwarding["sent"] == ["layers.0.weight", "layers.1.weight"]
    assert peer["returned"] >= forwarding["landed"], "a peer left the sync while the engine was mid-update"
    assert forwarding["fences"] == peer["fences"] == 1, "every rank joins the one fence after the push"


class _EngineClient(RecordingSender):
    """What ``_init_weight_sync_client`` builds on the main process, with no server behind it."""

    BACKEND_NAME = "vLLM"
    BUILT: list["_EngineClient"] = []
    FAIL = False

    def __init__(self, base_url: str, group_port: int, connection_timeout: float, group_host: str | None):
        if self.FAIL:
            raise ConnectionError("the NCCL group to the engine did not form")
        super().__init__()
        self.flushes, self.closed = 0, False
        self.BUILT.append(self)

    def init_communicator(self, device) -> None:
        self.sync_device = device

    buffer_param = RecordingSender.update_named_param

    def finish_weight_update(self) -> None:
        self.flushes += 1

    def close_communicator(self) -> None:
        self.closed = True


def _counted_verdicts(what: str) -> list[str]:
    """The rank-uniform verdicts named ``what`` this rank joins from here on."""
    joined = []
    real_reject = consensus.reject_across_ranks

    def counting_reject(reason, name, exc_type=RuntimeError):
        if name == what:
            joined.append(name)
        return real_reject(reason, name, exc_type=exc_type)

    consensus.reject_across_ranks = counting_reject
    return joined


def _trainer_outside_train(rank: int):
    """As a fresh construction or a returned ``train()``'s cleanup leaves it: no group, no client on any rank."""
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.model = _ShardedPolicy(init_device_mesh("cpu", (WORLD_SIZE,)))
    trainer.parallelism_config = ParallelismConfig()
    trainer.accelerator = SimpleNamespace(is_main_process=rank == 0, device=torch.device("cpu"))
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.async_config = SimpleNamespace(
        rollout_group_port=51216,
        sync_weights_every_n_steps=1,
        rollout_backend="vllm",
        rollout_server_url="http://engine:8000",
        rollout_server_configs=None,
        rollout_connection_timeout=1.0,
    )
    trainer._weight_sync_client = trainer._rollout_manager = trainer._loop = None
    trainer._prefetch_thread = trainer._engine_rescore_clients = None
    return trainer


def _record_outcome(tmp_dir: str, rank: int, step) -> None:
    """Run ``step`` and write what it raised, or ``NO RAISE``: a rank stuck in a collective writes nothing."""
    try:
        step()
        outcome = "NO RAISE"
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"outcome_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def _outcomes(tmp_path) -> list[str]:
    paths = [tmp_path / f"outcome_{rank}.txt" for rank in range(WORLD_SIZE)]
    return [path.read_text() if path.exists() else "NO RESULT (the rank never returned)" for path in paths]


def _reform_worker(rank: int, tmp_dir: str) -> None:
    async_mod.resolve_weight_sync_client = lambda backend: _EngineClient
    formations = _counted_verdicts("Weight-sync client formation")
    trainer = _trainer_outside_train(rank)
    pushed = [trainer._sync_weights_to_engine(force=True)]
    trainer._cleanup_async_components()
    pushed += [trainer._sync_weights_to_engine(force=True), trainer._sync_weights_to_engine(force=True)]
    record = {
        "pushed": pushed,
        "formations": len(formations),
        "clients": [{"sent": c.names, "flushes": c.flushes, "closed": c.closed} for c in _EngineClient.BUILT],
    }
    with open(os.path.join(tmp_dir, f"rank_{rank}.json"), "w") as fh:
        json.dump(record, fh)


def test_a_push_with_no_group_formed_forms_it_on_every_rank_and_sends(tmp_path):
    """``train()``'s cleanup drops the group and the next push forms it again; a push into a formed group joins no
    formation verdict."""
    run_gloo_ranks(_reform_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    forwarding, peer = (json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(WORLD_SIZE))
    assert forwarding["pushed"] == peer["pushed"] == [True, True, True]
    assert forwarding["formations"] == peer["formations"] == 2, "the formation is a verdict every rank joins"
    names = ["layers.0.weight", "layers.1.weight"]
    assert forwarding["clients"] == [
        {"sent": names, "flushes": 1, "closed": True},
        {"sent": names * 2, "flushes": 2, "closed": False},
    ]
    assert peer["clients"] == [], "a non-forwarding rank built an engine client"


def _failed_formation_worker(rank: int, tmp_dir: str) -> None:
    async_mod.resolve_weight_sync_client = lambda backend: _EngineClient
    _EngineClient.FAIL = rank == 0
    trainer = _trainer_outside_train(rank)
    _record_outcome(tmp_dir, rank, lambda: trainer._sync_weights_to_engine(force=True))


def test_a_client_that_cannot_form_raises_on_every_rank(tmp_path):
    """The main process forms the client alone; raised there, it would leave its peer in the gather."""
    run_gloo_ranks(_failed_formation_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    outcomes = _outcomes(tmp_path)
    for outcome in outcomes:
        assert "the NCCL group to the engine did not form" in outcome and "rank 0" in outcome, outcomes


def _clientless_worker(rank: int, tmp_dir: str) -> None:
    model = _ShardedPolicy(init_device_mesh("cpu", (WORLD_SIZE,)))
    client = None if rank == 0 else RecordingSender()
    _record_outcome(tmp_dir, rank, lambda: sync_weights_to_client(model, client, is_main=rank == 0, is_tp_main=True))


def test_a_forwarding_rank_with_no_client_raises_on_every_rank(tmp_path):
    """Gathering the whole policy and sending none of it would leave the engine on its old weights behind a
    push that returned normally."""
    run_gloo_ranks(_clientless_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    outcomes = _outcomes(tmp_path)
    for outcome in outcomes:
        assert "no engine client" in outcome and "rank 0" in outcome, outcomes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
