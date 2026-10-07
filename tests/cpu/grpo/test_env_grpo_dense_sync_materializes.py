#!/usr/bin/env python
"""Environmental GRPO's weight sync must never hand the engine a sharded or unmerged param.

Forwarding ``named_parameters()`` untouched would hand over DTensors under FSDP2: a DTensor reports
the GLOBAL shape/``numel()`` while ``data_ptr()`` addresses only the LOCAL shard, so the packed
broadcast reads past the end of the shard; and for PEFT, unmerged ``lora_*`` tensors the engine cannot
map. ``_sync_weights_to_engine`` has one path for every shape, single process and multi-server
included: the gather, which materializes (``materialize_dtensor``) and folds the adapters before
forwarding.

A one-rank gloo mesh is enough: what is asserted is the TYPE and the value that reach the client, not
the sharding arithmetic.

Run: ``python tests/cpu/grpo/test_env_grpo_dense_sync_materializes.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributed.tensor import DTensor, Shard, distribute_tensor, init_device_mesh

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

ROWS, COLS = 8, 4


class _FSDPStyleModel(nn.Module):
    """A model whose params are DTensors, as ``fully_shard`` leaves them (``nn.Parameter`` of a
    DTensor stays a DTensor — that is what ``_ParameterMeta`` exists for)."""

    def __init__(self, mesh):
        super().__init__()
        self.proj = nn.Linear(COLS, ROWS, bias=False)
        self.full_weight = torch.randn(ROWS, COLS)
        self.proj.weight = nn.Parameter(distribute_tensor(self.full_weight, mesh, [Shard(0)]))

    def forward(self, x):  # pragma: no cover - never called
        return self.proj(x)


class _RecordingClient:
    """Records what the gather forwards, in place of the NCCL transport."""

    def __init__(self):
        self.sent: list[tuple[str, torch.Tensor]] = []
        self.flushed = False

    def scope_co_load_groups(self, module_names) -> None:
        """Nothing in these models is co-loaded; the push scopes every client before it sends."""

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent.append((name, weights))

    def reset_prefix_cache(self) -> None:
        self.flushed = True


def _stub_trainer(model: nn.Module, client: _RecordingClient):
    """The trainer's sync entrypoint, on the smallest object that satisfies it."""
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda m: m)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.async_config = SimpleNamespace(sync_weights_every_n_steps=1, rollout_backend="vllm")
    trainer.parallelism_config = ParallelismConfig()
    trainer._weight_sync_client = client
    trainer._prefetch_enabled = False
    return trainer


@pytest.fixture
def mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_dense_sync_forwards_materialized_tensors(mesh):
    model = _FSDPStyleModel(mesh)
    assert isinstance(model.proj.weight, DTensor), "the fixture no longer reproduces FSDP2's param type"
    client = _RecordingClient()

    _stub_trainer(model, client)._sync_weights_to_engine(force=True)

    assert client.sent, "nothing was forwarded to vLLM"
    sharded = [name for name, tensor in client.sent if isinstance(tensor, DTensor)]
    assert not sharded, f"raw DTensor(s) handed to the weight-sync client: {sharded}"


def test_dense_sync_forwards_the_full_tensor(mesh):
    """Materializing must produce the whole weight, not rank 0's local shard under a global name."""
    model = _FSDPStyleModel(mesh)
    client = _RecordingClient()

    _stub_trainer(model, client)._sync_weights_to_engine(force=True)

    sent = dict(client.sent)
    assert "proj.weight" in sent, f"the dense weight was not forwarded: {sorted(sent)}"
    weight = sent["proj.weight"]
    assert type(weight) is torch.Tensor, f"forwarded a {type(weight).__name__}, not a materialized tensor"
    assert weight.shape == (ROWS, COLS)
    assert torch.equal(weight, model.full_weight)
    assert client.flushed, "the buffered broadcast was never flushed (reset_prefix_cache)"


def test_a_peft_push_sends_merged_weights():
    """``base_model.model.*`` / ``lora_A``-named tensors are names vLLM cannot map; the gather folds the
    adapters into each base weight first."""

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(COLS, ROWS, bias=False)

        def forward(self, x):  # pragma: no cover - never called
            return self.proj(x)

    model = get_peft_model(_Tiny(), LoraConfig(target_modules=["proj"], r=2))
    client = _RecordingClient()

    _stub_trainer(model, client)._sync_weights_to_engine(force=True)

    assert client.sent, "nothing was forwarded to vLLM"
    assert not [name for name, _ in client.sent if "lora_" in name], "unmerged adapter keys were sent"


def test_plain_dense_model_still_syncs():
    """Anti-over-rejection: the same branch must keep working for un-sharded params (no mesh)."""
    model = nn.Linear(COLS, ROWS, bias=False)
    client = _RecordingClient()

    _stub_trainer(model, client)._sync_weights_to_engine(force=True)

    assert torch.equal(dict(client.sent)["weight"], model.weight.data)


def test_a_declined_sync_reports_that_it_did_not_push():
    """The cadence gate declines most steps, and the trainer reads this verdict to decide whether the
    step owes a post-sync barrier: reporting a push that never happened puts an every-microbatch
    host-blocking barrier back on the step path (and would claim the engines hold weights nobody
    sent them)."""
    model = nn.Linear(COLS, ROWS, bias=False)
    client = _RecordingClient()
    trainer = _stub_trainer(model, client)
    trainer.async_config.sync_weights_every_n_steps = 4
    trainer.state.global_step = 3

    assert trainer._sync_weights_to_engine(force=False) is False
    assert client.sent == [], "the gate declined, yet weights went out"

    trainer.state.global_step = 4
    assert trainer._sync_weights_to_engine(force=False) is True
    assert client.sent, "anti-vacuity: an on-cadence step must still push"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
