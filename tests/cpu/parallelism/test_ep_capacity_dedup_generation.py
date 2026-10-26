#!/usr/bin/env python
"""CPU tests for the per-forward generation token behind the EP capacity dedup.

``_ElasticBackend.ensure`` all-reduces the per-rank token count once per forward and lets the
remaining MoE layers reuse that capacity. The generation token is what scopes the cache to one
forward, so two properties are load-bearing:

* A top-level forward bumps it **exactly once** — bump more often and the dedup degrades back to a
  per-layer collective; bump not at all and every later step reuses the first step's capacity, which
  under-sizes the wire buffer as soon as a batch grows.
* Every rank bumps on the same forward, so the first-layer cache miss (and hence the collective)
  lands on all ranks together. A hook that fires per *submodule* would break that.

``HALO_EP_CAPACITY_DEDUP=0`` is the documented escape hatch and must leave the model unhooked — and
must write nothing to the cache, which is never read while it is off.

Under a pipeline the hook has to ride the module the schedule calls: EP patching runs on the whole
CausalLM, but ``build_pipeline_stage`` then keeps only the backbone inside a
:class:`PipelineStageModule`, and the patched root's forward is never called again.

The cache itself is keyed on ``id(ep_group)``, so it must hold the group: a released group's id can
be recycled onto a rebuilt one, which would then be served another group's capacity — under-sized,
i.e. a raise on NVLink and an illegal access on cross-node Gin. That strong ref makes clearing the
cache at EP teardown part of the contract, or the communicator outlives the job's teardown.

Run: ``python tests/cpu/parallelism/test_ep_capacity_dedup_generation.py`` (or ``pytest -m cpu``).
"""

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
    """Stand-in for a model whose forward runs several MoE layers (submodules)."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Linear(4, 4)
        self.block2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


def _generation() -> int:
    return capacity_mod.capacity_scope.generation


class _StubBackbone(nn.Module):
    """The backbone contract a stage forward reads: a ``last_hidden_state`` on the returned object."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, **kwargs) -> SimpleNamespace:
        hidden = kwargs.get("input_ids")
        if hidden is None:
            hidden = kwargs["inputs_embeds"]
        return SimpleNamespace(last_hidden_state=self.layer(hidden))


def _mid_pipeline_stage() -> PipelineStageModule:
    """A stage holding neither the embedding nor the head — the plainest module the schedule runs."""
    return hand_built_stage(_StubBackbone(), None, is_first=False, is_last=False, layer_offset=8)


class _StubDispatcher:
    """The fields ``_ElasticBackend.ensure`` reads before it reaches DeepEP."""

    def __init__(self, group: object, hidden_dim: int):
        self.ep_group = group
        self.hidden_dim = hidden_dim
        self.spans_domains = False


def _isolated_backend(module, monkeypatch, group: object, *, dedup: bool) -> object:
    """An ``_ElasticBackend`` whose ``ensure`` can run its capacity sizing on a CPU box.

    The wire width is set so that ``needed × num_topk × padded_hidden`` trips the 32-bit index guard
    — the statement right AFTER the cache write, and the cheapest exit that needs no DeepEP buffer.
    The capacity all-reduce is stubbed (there is no process group here) and the scope is cleared so
    the assertions see only this test's writes.
    """
    capacity_mod.capacity_scope.clear()
    monkeypatch.setattr(module, "EP_CAPACITY_DEDUP", dedup)
    hidden = capacity_mod.DEEPEP_INDEX_LIMIT // capacity_mod.EP_CAPACITY_ALIGN
    return module._ElasticBackend(_StubDispatcher(group, hidden))


def _stub_capacity_all_reduce(module, monkeypatch) -> list:
    """Replace the capacity all-reduce and keep its tensor on the host; returns the call log."""
    calls: list = []
    real_tensor = torch.tensor
    monkeypatch.setattr(module.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **kw: calls.append(a))
    return calls


def _size_capacity(backend, num_tokens: int = 8, num_topk: int = 1) -> None:
    with pytest.raises(ValueError, match="wire-index limit"):
        backend.ensure(num_tokens, num_topk, deterministic=False)


def test_forward_bumps_generation_exactly_once_per_forward():
    """One bump per top-level forward, regardless of how many submodules run inside it."""
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
    """Re-registering must not double-bump: EP patching can be entered more than once."""
    model = _TwoLayerModel()
    capacity_mod.register_forward_generation_hook(model)
    capacity_mod.register_forward_generation_hook(model)
    capacity_mod.register_forward_generation_hook(model)

    x = torch.zeros(2, 4)
    before = _generation()
    model(x)
    assert _generation() == before + 1, "duplicate registration must not add a second hook"


def test_a_pipeline_stage_bumps_the_generation_it_is_never_registered_for():
    """The schedule calls the STAGE, so the stage is what must carry the hook.

    EP patching registers it on the model it patched, which under PP is the un-sliced CausalLM
    ``build_pipeline_stage`` then discards. Without the registration on the stage the generation
    never advances past the run's first forward, every later microbatch is served the first
    forward's capacity from the cache, and the first longer batch is refused outright by the
    dedup guard instead of growing the arena.
    """
    stage = _mid_pipeline_stage()
    hidden = torch.zeros(2, 4)

    before = _generation()
    stage(hidden)
    assert _generation() == before + 1, "a pipeline stage forward must bump the capacity generation"

    stage(hidden)
    assert _generation() == before + 2, "every microbatch is its own forward with its own token count"


def test_capacity_dedup_can_be_disabled_by_env(monkeypatch):
    """``HALO_EP_CAPACITY_DEDUP=0`` leaves the model unhooked, restoring the per-layer all-reduce."""
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
        # Restore the default-on flag for the rest of the session.
        monkeypatch.delenv("HALO_EP_CAPACITY_DEDUP", raising=False)
        importlib.reload(settings_mod)


def test_the_cache_holds_the_group_its_key_names(monkeypatch):
    """The key is ``id(ep_group)``, so the entry must own a reference to that group.

    Without it the allocator can hand a rebuilt group the freed one's address (EP teardown then
    rebuild inside one process), and the stale entry serves it a capacity sized for another group's
    token count — an under-sized wire buffer, not a wrong number.
    """
    group = object()
    backend = _isolated_backend(backends_mod, monkeypatch, group, dedup=True)
    _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)

    (entry,) = capacity_mod.capacity_scope._entries.values()
    assert entry[0] is group


def test_teardown_clears_the_cache_so_no_ep_group_is_pinned(monkeypatch):
    """``destroy_all_dispatchers`` must drop the entries, or the strong group ref never dies.

    The entry holds its EP group, so a cache that survives teardown pins that communicator for the
    life of the process — every group a job rebuilds (a second model, an EP teardown/rebuild cycle)
    stays alive, against the repo's teardown contract.
    """
    group = object()
    backend = _isolated_backend(backends_mod, monkeypatch, group, dedup=True)
    _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)
    assert capacity_mod.capacity_scope._entries, "precondition: the first layer cached its capacity"

    dispatcher_mod.destroy_all_dispatchers()

    assert capacity_mod.capacity_scope._entries == {}


def test_a_later_layer_reuses_the_first_layers_capacity(monkeypatch):
    """The reuse guard reads the cached entry back — generation, top-k and the aligned capacity.

    A later layer that dispatches a different token count still fits as long as the arena holds it
    (per-layer counts are data-dependent per rank); only a count over the capacity is refused.
    """
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
        backend.ensure(capacity + 1, 1, deterministic=False)  # more tokens than the shared arena holds


def test_disabled_dedup_writes_nothing_to_the_cache(monkeypatch):
    """With the feature off the cache is never read, so writing to it is a leak that never pays."""
    backend = _isolated_backend(backends_mod, monkeypatch, object(), dedup=False)
    calls = _stub_capacity_all_reduce(backends_mod, monkeypatch)

    _size_capacity(backend)
    _size_capacity(backend)

    assert capacity_mod.capacity_scope._entries == {}
    assert len(calls) == 2, "every layer sizes itself when the dedup is off"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
