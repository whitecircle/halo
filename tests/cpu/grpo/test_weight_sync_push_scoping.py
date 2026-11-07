#!/usr/bin/env python
"""Every weight-sync push scopes the engine's co-load groups to the model it sends. SGLang's MLA loaders drop a half
of the pair that arrives alone; an unscoped client falsely refuses a model without ``q_a_proj`` or splits the pair."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import halo.weight_sync.clients.base as base_module
import halo.weight_sync.pool as pool_module
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights, sync_weights_to_client
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync import ServerlessSGLangClient, Wire, offline_client

# Each projection is a 2x2 bf16 weight, 8 bytes: a 16-byte budget holds two, so the cut falls between
# the halves of the pair and only the scoped groups keep it whole.
_BUDGET = 16
_MLA_LEAVES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")


def _param(leaf: str) -> str:
    return f"model.layers.0.self_attn.{leaf}.weight"


class _Attention(nn.Module):
    def __init__(self, leaves: tuple[str, ...]):
        super().__init__()
        for leaf in leaves:
            setattr(self, leaf, nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))


class _MLAModel(nn.Module):
    """``model.layers.0.self_attn.<leaf>``: the naming the engine's MLA loader keys on."""

    def __init__(self, leaves: tuple[str, ...]):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _Attention(leaves)

    def forward(self, x):  # pragma: no cover - never called
        return x


def _stub_trainer(model: nn.Module, client):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.async_config = SimpleNamespace(sync_weights_every_n_steps=1, rollout_backend="sglang")
    trainer.parallelism_config = ParallelismConfig()
    trainer._weight_sync_client = client
    trainer.first_push_verified = True
    return trainer


def _sglang_pool(num_servers: int) -> WeightSyncPool:
    """A pool whose clients are built by its own ``init_communicator``, as the trainer builds them."""
    pool = WeightSyncPool(
        [{"url": f"http://localhost:{30000 + index}"} for index in range(num_servers)],
        connection_timeout=0.0,
        client_cls=ServerlessSGLangClient,
        base_group_port=51216,
    )
    pool.init_communicator("cpu")
    return pool


@pytest.fixture(autouse=True)
def _two_params_per_chunk(monkeypatch):
    for module in (base_module, pool_module):  # a client's own budget, and the pool's
        monkeypatch.setattr(module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)


def test_the_streamed_push_does_not_refuse_a_model_without_q_a_proj():
    """An MLA block without ``q_lora_rank`` has no ``q_a_proj``, and holding its ``kv_a_proj_with_mqa`` for a partner
    refuses a sound sync at the close."""
    model = _MLAModel(("q_b_proj", "kv_a_proj_with_mqa", "o_proj"))
    wire = Wire()
    client = wire.attach(offline_client(SGLangWeightSyncClient))

    _stub_trainer(model, client)._sync_weights_to_engine(force=True)

    assert [name for name, _ in wire.sent] == [_param(leaf) for leaf in ("q_b_proj", "kv_a_proj_with_mqa", "o_proj")]
    assert client._co_load_groups == (("self_attn.kv_a_proj_with_mqa.weight",),)


def test_the_streamed_push_keeps_a_declared_pair_in_one_request():
    wire = Wire()
    client = wire.attach(offline_client(SGLangWeightSyncClient))

    _stub_trainer(_MLAModel(_MLA_LEAVES), client)._sync_weights_to_engine(force=True)

    assert wire.chunk_names == [
        [_param("q_b_proj")],
        [_param("q_a_proj"), _param("kv_a_proj_with_mqa")],
        [_param("o_proj")],
    ]


def test_a_pool_push_keeps_a_declared_pair_in_one_request_on_every_server():
    """The pool drains each chunk on every server; a request that splits the pair loses a half."""
    pool = _sglang_pool(2)

    _stub_trainer(_MLAModel(_MLA_LEAVES), pool)._sync_weights_to_engine(force=True)

    for client in pool._clients:
        assert client.wire.chunk_names == [
            [_param("q_b_proj")],
            [_param("q_a_proj"), _param("kv_a_proj_with_mqa")],
            [_param("o_proj")],
        ], f"{client.base_url} received the MLA pair split across requests"


def test_a_pool_push_does_not_refuse_a_model_without_q_a_proj():
    pool = _sglang_pool(2)
    model = _MLAModel(("q_b_proj", "kv_a_proj_with_mqa", "o_proj"))

    _stub_trainer(model, pool)._sync_weights_to_engine(force=True)

    for client in pool._clients:
        assert [name for name, _ in client.wire.sent] == [
            _param(leaf) for leaf in ("q_b_proj", "kv_a_proj_with_mqa", "o_proj")
        ]
        assert client._co_load_groups == (("self_attn.kv_a_proj_with_mqa.weight",),)


class _EventClient:
    """Records scope calls and sends in one ordered log."""

    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def scope_co_load_groups(self, module_names) -> None:
        self.events.append(("scope", tuple(module_names)))

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.events.append(("send", name))

    def finish_weight_update(self) -> None:
        self.events.append(("close", None))

    def abort_weight_update(self) -> None:  # pragma: no cover - the push does not fail here
        self.events.append(("abort", None))


def test_the_collective_push_scopes_its_forwarding_client_once_before_sending():
    model = _MLAModel(_MLA_LEAVES)
    client = _EventClient()
    trainer = SimpleNamespace(
        model=model,
        parallelism_config=ParallelismConfig(),
        accelerator=SimpleNamespace(is_main_process=True),
        state=SimpleNamespace(global_step=1),
    )

    sync_trainer_weights(trainer, client)

    scopes = [payload for kind, payload in client.events if kind == "scope"]
    assert len(scopes) == 1, f"the forwarding client was scoped {len(scopes)} times"
    assert "model.layers.0.self_attn.q_a_proj" in scopes[0]
    assert client.events[0][0] == "scope", "a tensor went out before the groups were scoped"
    assert [payload for kind, payload in client.events if kind == "send"] == [_param(leaf) for leaf in _MLA_LEAVES]


def test_a_rank_that_does_not_forward_leaves_the_client_untouched():
    client = _EventClient()

    sync_weights_to_client(_MLAModel(_MLA_LEAVES), client, is_main=True, is_tp_main=False)

    assert client.events == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
