#!/usr/bin/env python
"""The EP dispatcher follows torch's deterministic-algorithms mode into DeepEP.

DeepEP's default dispatch claims receive slots with atomics, so every expert weight gradient is summed
in a different order on every step. Under the mode HF's ``full_determinism`` turns on, the
``ElasticBuffer`` must therefore be built in DeepEP's deterministic mode, and a buffer built before the
mode turned on must not keep serving the run. DeepEP also asserts at every dispatch that the mode does
not run beside ``torch.utils.deterministic.fill_uninitialized_memory``, which the mode turns on by
default. Outside the mode nothing changes: the buffer stays the default one and the fill flag is left
alone.

The DeepEP extension is replaced by a recorder, so the backend runs its real sizing and arena logic on a
CPU box.

    python tests/cpu/parallelism/test_ep_deterministic_dispatch.py
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from src.distributed.expert_parallel import dispatcher as dispatcher_mod

NUM_EXPERTS, HIDDEN, TOPK = 8, 256, 2


class _RecordingBuffer:
    """Stands in for ``deep_ep.ElasticBuffer``: records how each buffer was built and freed."""

    built: list[dict] = []
    destroyed: list[dict] = []

    def __init__(self, group, **kwargs):
        self.kwargs = kwargs
        _RecordingBuffer.built.append(kwargs)

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return 24

    def destroy(self):
        _RecordingBuffer.destroyed.append(self.kwargs)


@pytest.fixture
def backend(monkeypatch):
    """An ``_ElasticBackend`` on a fake intra-node EP group whose DeepEP calls are recorded."""
    _RecordingBuffer.built, _RecordingBuffer.destroyed = [], []
    monkeypatch.setattr(dispatcher_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=_RecordingBuffer))
    monkeypatch.setattr(dispatcher_mod._SharedArena, "_REGISTRY", {})
    monkeypatch.setattr(dispatcher_mod, "_CAPACITY_CACHE", {})
    real_tensor = torch.tensor
    monkeypatch.setattr(
        dispatcher_mod.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"})
    )
    monkeypatch.setattr(dispatcher_mod.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setenv("EP_DISABLE_GIN", "1")
    dispatcher = SimpleNamespace(
        ep_group=object(),
        hidden_dim=HIDDEN,
        num_experts=NUM_EXPERTS,
        ep_size=2,
        is_inter_node=False,
        ep_config=SimpleNamespace(expert_tp_size=1),
    )
    return dispatcher_mod._ElasticBackend(dispatcher)


@pytest.fixture
def deterministic_mode():
    """Torch's deterministic-algorithms mode as ``enable_full_determinism`` leaves it (fill on)."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def test_the_default_mode_builds_the_default_buffer(backend):
    backend.ensure(64, TOPK, deterministic=False)
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False]


def test_a_deterministic_request_builds_a_deterministic_buffer(backend):
    backend.ensure(64, TOPK, deterministic=True)
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [True]


def test_a_mode_change_moves_the_layer_to_a_buffer_of_the_new_mode(backend):
    """A buffer built before the mode turned on (a forward ahead of ``Trainer.__init__``) must not keep
    serving the deterministic run, and a same-mode call reuses the buffer it has."""
    backend.ensure(64, TOPK, deterministic=False)
    backend.ensure(64, TOPK, deterministic=True)
    backend.ensure(64, TOPK, deterministic=True)
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.built] == [False, True]
    assert [kwargs["deterministic"] for kwargs in _RecordingBuffer.destroyed] == [False]


class _RecordingBackend:
    """Records the mode each buffer sizing is asked for."""

    def __init__(self):
        self.modes: list[bool] = []

    def ensure(self, num_tokens, num_topk, *, deterministic):
        self.modes.append(deterministic)


class _DispatcherShell:
    """The state ``DeepEPDispatcher._ensure_buffer`` touches, without the EP process groups."""

    def __init__(self):
        self.backend = _RecordingBackend()


@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_every_dispatch_sizes_its_buffer_in_the_mode_torch_is_in(deterministic, monkeypatch, request):
    """The seam every dispatch goes through hands the backend torch's current mode."""
    if deterministic:
        request.getfixturevalue("deterministic_mode")
    monkeypatch.setattr(dispatcher_mod, "_LIVE_DISPATCHERS", weakref.WeakValueDictionary())
    shell = _DispatcherShell()
    dispatcher_mod.DeepEPDispatcher._ensure_buffer(shell, 64, TOPK)
    assert shell.backend.modes == [deterministic]


def test_the_mode_is_read_off_torch_and_the_fill_deepep_rejects_is_switched_off(deterministic_mode):
    assert dispatcher_mod._resolve_deterministic_dispatch() is True
    assert torch.utils.deterministic.fill_uninitialized_memory is False


def test_outside_the_mode_the_fill_flag_is_left_alone():
    fill = torch.utils.deterministic.fill_uninitialized_memory
    assert not torch.are_deterministic_algorithms_enabled()
    assert dispatcher_mod._resolve_deterministic_dispatch() is False
    assert torch.utils.deterministic.fill_uninitialized_memory is fill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
