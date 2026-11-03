#!/usr/bin/env python
"""The per-forward generation token behind the EP capacity dedup: bumped too often it is a per-layer collective, too
rarely it under-sizes. The cache holds its group, cleared at EP teardown."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.distributed.expert_parallel.deepep import backends as backends_mod
from halo.distributed.expert_parallel.deepep import capacity as capacity_mod
from halo.distributed.expert_parallel.deepep import dispatcher as dispatcher_mod
from halo.distributed.expert_parallel.deepep import settings as settings_mod
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from tests.common.pp_fakes import hand_built_stage


class _TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Linear(4, 4)
        self.block2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


def _generation() -> int:
    return capacity_mod.capacity_scope.generation


class _StubBackbone(nn.Module):
    """A stage forward reads ``last_hidden_state`` off the returned object."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, **kwargs) -> SimpleNamespace:
        hidden = kwargs.get("input_ids")
        if hidden is None:
            hidden = kwargs["inputs_embeds"]
        return SimpleNamespace(last_hidden_state=self.layer(hidden))


def _mid_pipeline_stage() -> PipelineStageModule:
    return hand_built_stage(_StubBackbone(), None, is_first=False, is_last=False, layer_offset=8)


class _StubDispatcher:
    def __init__(self, group: object, hidden_dim: int):
        self.ep_group = group
        self.hidden_dim = hidden_dim
        self.spans_domains = False


def _isolated_backend(module, monkeypatch, group: object, *, dedup: bool) -> object:
    """The wire width trips the 32-bit index guard right after the cache write, the cheapest exit that needs no
    DeepEP buffer."""
    capacity_mod.capacity_scope.clear()
    monkeypatch.setattr(module, "EP_CAPACITY_DEDUP", dedup)
    hidden = capacity_mod.DEEPEP_INDEX_LIMIT // capacity_mod.EP_CAPACITY_ALIGN
    return module._ElasticBackend(_StubDispatcher(group, hidden))


def _stub_capacity_all_reduce(module, monkeypatch) -> list:
    calls: list = []
    real_tensor = torch.tensor
    monkeypatch.setattr(module.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **kw: calls.append(a))
    return calls


def _size_capacity(backend, num_tokens: int = 8, num_topk: int = 1) -> None:
    with pytest.raises(ValueError, match="wire-index limit"):
        backend.ensure(num_tokens, num_topk, deterministic=False)


def test_forward_bumps_generation_exactly_once_per_forward():
    model = _TwoLayerModel()
    capacity_mod.register_forward_generation_hook(model)

    x = torch.zeros(2, 4)
    before = _generation()
    model(x)
    assert _generation() == before + 1, "a forward must bump the generation exactly once"

    model(x)
    model(x)
    assert _generation() == before + 3, "each subsequent forward bumps once more"


def test_registration_is_idempotent():
    """EP patching can be entered more than once."""
    model = _TwoLayerModel()
    capacity_mod.register_forward_generation_hook(model)
    capacity_mod.register_forward_generation_hook(model)
    capacity_mod.register_forward_generation_hook(model)

    x = torch.zeros(2, 4)
    before = _generation()
    model(x)
    assert _generation() == before + 1, "duplicate registration must not add a second hook"


def test_a_pipeline_stage_bumps_the_generation_it_is_never_registered_for():
    """EP patching registers on the un-sliced CausalLM ``build_pipeline_stage`` discards; unregistered, the first
    longer batch is refused."""
    stage = _mid_pipeline_stage()
    hidden = torch.zeros(2, 4)

    before = _generation()
    stage(hidden)
    assert _generation() == before + 1, "a pipeline stage forward must bump the capacity generation"

    stage(hidden)
    assert _generation() == before + 2, "every microbatch is its own forward with its own token count"


def test_capacity_dedup_can_be_disabled_by_env(monkeypatch):
    """Restores the per-layer all-reduce."""
    monkeypatch.setenv("HALO_EP_CAPACITY_DEDUP", "0")
    reloaded = importlib.reload(settings_mod)
    try:
        assert reloaded.EP_CAPACITY_DEDUP is False
        monkeypatch.setattr(capacity_mod, "EP_CAPACITY_DEDUP", reloaded.EP_CAPACITY_DEDUP)

        model = _TwoLayerModel()
        capacity_mod.register_forward_generation_hook(model)
        assert not model._forward_pre_hooks

        before = _generation()
        model(torch.zeros(2, 4))
        assert _generation() == before, "no hook must be registered when disabled"
    finally:
        monkeypatch.delenv("HALO_EP_CAPACITY_DEDUP", raising=False)
        importlib.reload(settings_mod)


def test_the_cache_holds_the_group_its_key_names(monkeypatch):
    """Without the reference, a rebuilt group at the freed address gets an under-sized wire buffer."""
    group = object()
    backend = _isolated_backend(backends_mod, monkeypatch, group, dedup=True)
    _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)

    (entry,) = capacity_mod.capacity_scope._entries.values()
    assert entry[0] is group


def test_teardown_clears_the_cache_so_no_ep_group_is_pinned(monkeypatch):
    """A cache surviving teardown pins the communicator for the life of the process."""
    group = object()
    backend = _isolated_backend(backends_mod, monkeypatch, group, dedup=True)
    _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)
    assert capacity_mod.capacity_scope._entries, "precondition: the first layer cached its capacity"

    dispatcher_mod.destroy_all_dispatchers()

    assert capacity_mod.capacity_scope._entries == {}


def test_a_later_layer_reuses_the_first_layers_capacity(monkeypatch):
    """Per-layer counts are data-dependent per rank, so only a count over the capacity is refused."""
    backend = _isolated_backend(backends_mod, monkeypatch, object(), dedup=True)
    calls = _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend, num_tokens=8)
    assert len(calls) == 1, "the first layer of a forward sizes the capacity with one all-reduce"
    (entry,) = capacity_mod.capacity_scope._entries.values()
    capacity = entry[3]
    assert capacity >= 8

    _size_capacity(backend, num_tokens=capacity)
    assert len(calls) == 1, "a later layer of the same forward must reuse it, not re-reduce"

    with pytest.raises(RuntimeError, match="EP capacity dedup"):
        backend.ensure(capacity + 1, 1, deterministic=False)  # More tokens than the shared arena holds.


def test_disabled_dedup_writes_nothing_to_the_cache(monkeypatch):
    """With the feature off the cache is never read, so writing to it is a leak."""
    backend = _isolated_backend(backends_mod, monkeypatch, object(), dedup=False)
    calls = _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)
    _size_capacity(backend)

    assert capacity_mod.capacity_scope._entries == {}
    assert len(calls) == 2, "every layer sizes itself when the dedup is off"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
