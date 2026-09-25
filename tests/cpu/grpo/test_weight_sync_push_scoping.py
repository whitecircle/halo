#!/usr/bin/env python
"""Every weight-sync push scopes the engine's co-load groups to the model it sends.

SGLang's MLA loaders fuse ``q_a_proj`` and ``kv_a_proj_with_mqa`` from a cache local to one request
and drop a half that arrives alone, so the client holds a declared pair in one chunk. The pair exists
only where the model has both projections, which is why the groups are scoped to the pushed module
tree. Unscoped, a client either waits for a partner the model never declares (a false "missing their
partners" refusal of a model without ``q_a_proj``) or, scoped to no model at all, holds nothing and
sends the halves in separate requests — which the engine drops, leaving it serving stale attention
weights with no error.

What is pinned: the environmental trainer's single-process push scopes on both of its branches (one
SGLang client through the streamed gather; a pool of SGLang clients through the rolling one), a
manager's clients keep the engine's full groups until a push scopes them, and the collective push
scopes its forwarding client exactly once.

Run: ``python tests/cpu/grpo/test_weight_sync_push_scoping.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.distributed.nccl.clients.base as base_module
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.rollout.weight_sync import sync_trainer_weights, sync_weights_to_client
from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager
from tests.common.weight_sync import Wire, offline_sglang_client

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


class _ServerlessSGLangClient(SGLangWeightSyncClient):
    """The real SGLang client with no server behind it: the engine side is a recording :class:`Wire`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wire = Wire()
        self.wire.attach(self)

    def check_server(self, total_timeout: float = 0.0):
        """No server to probe."""

    def init_communicator(self, device=0):
        self._resolve_sync_device(device)


def _stub_trainer(model: nn.Module, client, *, multi_server: bool):
    """The environmental trainer's single-process sync entrypoint, on the smallest object it needs."""
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda m: m)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.async_config = SimpleNamespace(sync_weights_every_n_steps=1, rollout_backend="sglang")
    trainer.parallelism_config = ParallelismConfig()
    trainer._weight_sync_client = client
    trainer._multi_server_mode = multi_server
    trainer._init_weight_sync_client = lambda: None
    return trainer


def _sglang_pool(num_servers: int) -> InferenceClientManager:
    """A manager whose clients are built by its own ``init_communicators``, as the trainer builds them."""
    manager = InferenceClientManager(
        [{"url": f"http://localhost:{30000 + index}"} for index in range(num_servers)],
        connection_timeout=0.0,
        client_cls=_ServerlessSGLangClient,
        base_group_port=51216,
    )
    manager.init_communicators("cpu")
    return manager


@pytest.fixture(autouse=True)
def _two_params_per_chunk(monkeypatch):
    monkeypatch.setattr(base_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)


def test_the_streamed_push_does_not_refuse_a_model_without_q_a_proj():
    """An MLA block without ``q_lora_rank`` has no ``q_a_proj``: its ``kv_a_proj_with_mqa`` loads on
    its own, so holding it for a partner refuses a sound sync at the close."""
    model = _MLAModel(("q_b_proj", "kv_a_proj_with_mqa", "o_proj"))
    wire = Wire()
    client = wire.attach(offline_sglang_client())

    _stub_trainer(model, client, multi_server=False)._sync_weights_to_engine_single(force=True)

    assert [name for name, _ in wire.sent] == [_param(leaf) for leaf in ("q_b_proj", "kv_a_proj_with_mqa", "o_proj")]
    assert client._co_load_groups == (("self_attn.kv_a_proj_with_mqa.weight",),)


def test_the_streamed_push_keeps_a_declared_pair_in_one_request():
    """Anti-over-scoping: where the model declares both halves, the scoped groups still hold the pair."""
    wire = Wire()
    client = wire.attach(offline_sglang_client())

    _stub_trainer(_MLAModel(_MLA_LEAVES), client, multi_server=False)._sync_weights_to_engine_single(force=True)

    assert wire.chunk_names == [
        [_param("q_b_proj")],
        [_param("q_a_proj"), _param("kv_a_proj_with_mqa")],
        [_param("o_proj")],
    ]


def test_a_pool_keeps_the_engines_groups_until_a_push_scopes_them():
    """A client built before any push must hold the pair and refuse it incomplete, not send it apart."""
    manager = _sglang_pool(2)

    assert all(client._co_load_groups == SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS for client in manager._clients)


def test_the_rolling_push_keeps_a_declared_pair_in_one_request_on_every_server():
    """The raw-model branch sends one server at a time; each request that splits the pair loses a half."""
    manager = _sglang_pool(2)

    _stub_trainer(_MLAModel(_MLA_LEAVES), manager, multi_server=True)._sync_weights_to_engine_single(force=True)

    for client in manager._clients:
        assert client.wire.chunk_names == [
            [_param("q_b_proj")],
            [_param("q_a_proj"), _param("kv_a_proj_with_mqa")],
            [_param("o_proj")],
        ], f"{client.base_url} received the MLA pair split across requests"


def test_the_rolling_push_does_not_refuse_a_model_without_q_a_proj():
    manager = _sglang_pool(2)
    model = _MLAModel(("q_b_proj", "kv_a_proj_with_mqa", "o_proj"))

    _stub_trainer(model, manager, multi_server=True)._sync_weights_to_engine_single(force=True)

    for client in manager._clients:
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

    def reset_prefix_cache(self) -> None:
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
