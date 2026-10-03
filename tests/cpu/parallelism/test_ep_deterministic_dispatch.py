#!/usr/bin/env python
"""The EP dispatcher follows torch's deterministic-algorithms mode into DeepEP.

DeepEP's default dispatch claims receive slots with atomics, so every expert weight gradient is summed
in a different order on every step. Under the mode HF's ``full_determinism`` turns on, the
``ElasticBuffer`` must therefore be built in DeepEP's deterministic mode, and a buffer built before the
mode turned on must not keep serving the run. DeepEP's dispatch and combine also assert, on both
buffers, that the mode does not run beside ``torch.utils.deterministic.fill_uninitialized_memory``,
which defaults to on, so the fill must be off before any DeepEP buffer exists. Outside the mode nothing
changes: the buffer stays the default one and the fill flag is left alone.

DeepEP is replaced by recorders, so the dispatch seam and both backends run their real sizing and arena
logic on a CPU box.

    python tests/cpu/parallelism/test_ep_deterministic_dispatch.py
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from tests.common.distributed import pin_deterministic_ep_dispatch

NUM_EXPERTS, HIDDEN, TOPK = 8, 256, 2


class _RecordingBuffer:
    """Stands in for both DeepEP buffers: records each build (with the fill flag it found) and free."""

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
    """The state ``DeepEPDispatcher._ensure_buffer`` touches, on a fake intra-node EP group."""

    ep_group = object()
    hidden_dim = HIDDEN
    num_experts = NUM_EXPERTS
    ep_size = 2
    is_inter_node = False
    ep_config = SimpleNamespace(expert_tp_size=1)

    def __init__(self, backend_cls):
        self.backend = backend_cls(self)

    def ensure_buffer(self):
        dispatcher_mod.DeepEPDispatcher._ensure_buffer(self, 64, TOPK)


@pytest.fixture(autouse=True)
def recorded_deep_ep(monkeypatch):
    _RecordingBuffer.built, _RecordingBuffer.destroyed = [], []
    monkeypatch.setattr(
        dispatcher_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=_RecordingBuffer, Buffer=_RecordingBuffer)
    )
    monkeypatch.setattr(dispatcher_mod._SharedArena, "_REGISTRY", {})
    monkeypatch.setattr(dispatcher_mod, "_CAPACITY_CACHE", {})
    monkeypatch.setattr(dispatcher_mod, "_LIVE_DISPATCHERS", weakref.WeakValueDictionary())
    real_tensor = torch.tensor
    monkeypatch.setattr(
        dispatcher_mod.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"})
    )
    monkeypatch.setattr(dispatcher_mod.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setenv("EP_DISABLE_GIN", "1")
    # pin_deterministic_ep_dispatch replaces the seam for the rest of a process; restored after each test.
    monkeypatch.setattr(dispatcher_mod, "_deterministic_dispatch", dispatcher_mod._deterministic_dispatch)


@pytest.fixture
def deterministic_mode():
    """Torch's deterministic-algorithms mode as ``enable_full_determinism`` leaves it: the fill still on."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def test_outside_the_mode_the_default_buffer_is_built_and_the_fill_left_alone():
    fill = torch.utils.deterministic.fill_uninitialized_memory
    _DispatcherShell(dispatcher_mod._ElasticBackend).ensure_buffer()
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False]
    assert torch.utils.deterministic.fill_uninitialized_memory is fill


@pytest.mark.parametrize(
    "backend_cls", [dispatcher_mod._ElasticBackend, dispatcher_mod._LegacyBackend], ids=["elastic", "legacy"]
)
def test_under_the_mode_the_fill_is_off_before_any_deepep_buffer_exists(backend_cls, deterministic_mode):
    _DispatcherShell(backend_cls).ensure_buffer()
    (kwargs,) = _RecordingBuffer.built
    assert kwargs["fill_at_build"] is False
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    if backend_cls is dispatcher_mod._ElasticBackend:
        assert kwargs["deterministic"] is True


def test_a_mode_change_moves_the_layer_to_a_buffer_of_the_new_mode():
    """A buffer built before the mode turned on (a forward ahead of ``Trainer.__init__``) must not keep
    serving the deterministic run, and a same-mode call reuses the buffer it has."""
    backend = _DispatcherShell(dispatcher_mod._ElasticBackend).backend
    backend.ensure(64, TOPK, deterministic=False)
    backend.ensure(64, TOPK, deterministic=True)
    backend.ensure(64, TOPK, deterministic=True)
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False, True]
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.destroyed] == [False]


def test_the_test_pin_builds_deterministic_buffers_through_the_same_seam():
    """A replay test pins the mode without turning torch's deterministic algorithms on."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    pin_deterministic_ep_dispatch()
    _DispatcherShell(dispatcher_mod._ElasticBackend).ensure_buffer()
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [True]
    assert torch.utils.deterministic.fill_uninitialized_memory is fill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
