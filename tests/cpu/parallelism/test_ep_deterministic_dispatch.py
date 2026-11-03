#!/usr/bin/env python
"""The EP dispatcher follows torch's deterministic-algorithms mode into DeepEP. Default dispatch reorders gradient
sums per step, and DeepEP asserts against ``fill_uninitialized_memory``."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from halo.distributed.expert_parallel.deepep import backends as backends_mod
from halo.distributed.expert_parallel.deepep import dispatcher as dispatcher_mod
from halo.distributed.expert_parallel.deepep.capacity import capacity_scope
from tests.common.distributed import pin_deterministic_ep_dispatch

NUM_EXPERTS, HIDDEN, TOPK = 8, 256, 2


class _RecordingBuffer:
    """Records each build (with the fill flag it found) and free."""

    built: list[dict] = []
    destroyed: list[dict] = []

    def __init__(self, group, **kwargs):
        self.kwargs = {**kwargs, "fill_at_build": torch.utils.deterministic.fill_uninitialized_memory}
        _RecordingBuffer.built.append(self.kwargs)

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return 24

    @staticmethod
    def get_dispatch_config(num_ranks):
        return SimpleNamespace(get_nvl_buffer_size_hint=lambda hidden_bytes, num_ranks: 1 << 20)

    get_combine_config = get_dispatch_config

    def destroy(self):
        _RecordingBuffer.destroyed.append(self.kwargs)


class _DispatcherShell:
    dispatch_group = object()
    hidden_dim = HIDDEN
    num_experts = NUM_EXPERTS
    ep_size = 2
    spans_domains = False
    ep_config = SimpleNamespace(expert_tp_size=1)
    _live = False

    def __init__(self, backend_cls):
        self.backend = backend_cls(self)

    def ensure_buffer(self):
        dispatcher_mod.DeepEPDispatcher._ensure_buffer(self, 64, TOPK)


@pytest.fixture(autouse=True)
def recorded_deep_ep(monkeypatch):
    _RecordingBuffer.built, _RecordingBuffer.destroyed = [], []
    monkeypatch.setattr(
        backends_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=_RecordingBuffer, Buffer=_RecordingBuffer)
    )
    monkeypatch.setattr(backends_mod._Arena, "_registry", {})
    capacity_scope.clear()
    monkeypatch.setattr(dispatcher_mod, "_live_dispatchers", weakref.WeakValueDictionary())
    real_tensor = torch.tensor
    monkeypatch.setattr(backends_mod.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(backends_mod.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setenv("EP_DISABLE_GIN", "1")
    # pin_deterministic_ep_dispatch replaces the seam for the rest of a process.
    monkeypatch.setattr(dispatcher_mod, "_deterministic_dispatch", dispatcher_mod._deterministic_dispatch)


@pytest.fixture
def deterministic_mode():
    """As ``enable_full_determinism`` leaves it: the fill still on."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def test_outside_the_mode_the_default_buffer_is_built_and_the_fill_left_alone():
    fill = torch.utils.deterministic.fill_uninitialized_memory
    _DispatcherShell(backends_mod._ElasticBackend).ensure_buffer()
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False]
    assert torch.utils.deterministic.fill_uninitialized_memory is fill


@pytest.mark.parametrize(
    "backend_cls", [backends_mod._ElasticBackend, backends_mod._LegacyBackend], ids=["elastic", "legacy"]
)
def test_under_the_mode_the_fill_is_off_before_any_deepep_buffer_exists(backend_cls, deterministic_mode):
    _DispatcherShell(backend_cls).ensure_buffer()
    (kwargs,) = _RecordingBuffer.built
    assert kwargs["fill_at_build"] is False
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    if backend_cls is backends_mod._ElasticBackend:
        assert kwargs["deterministic"] is True


def test_a_mode_change_moves_the_layer_to_a_buffer_of_the_new_mode():
    """A buffer built ahead of ``Trainer.__init__`` must not keep serving the deterministic run."""
    backend = _DispatcherShell(backends_mod._ElasticBackend).backend
    backend.ensure(64, TOPK, deterministic=False)
    backend.ensure(64, TOPK, deterministic=True)
    backend.ensure(64, TOPK, deterministic=True)
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False, True]
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.destroyed] == [False]


def test_the_test_pin_builds_deterministic_buffers_through_the_same_seam():
    """A replay test pins the mode without turning torch's deterministic algorithms on."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    pin_deterministic_ep_dispatch()
    _DispatcherShell(backends_mod._ElasticBackend).ensure_buffer()
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [True]
    assert torch.utils.deterministic.fill_uninitialized_memory is fill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
