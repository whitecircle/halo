#!/usr/bin/env python
"""The FSDP2 grad-accum window toggles per window, never pinned for the run.
Pinned off, the reshard leaves unsharded params registered and the reduce never delivers a gradient to the shards."""

import types

import pytest
import torch.nn as nn

from halo.distributed.fsdp import fsdp2_modules
from halo.trainers.parallelism import grad_sync
from halo.trainers.parallelism.grad_sync import GradientSyncMixin

GRAD_ACCUM = 3
RESHARD = "set_reshard_after_backward"
REDUCE = "set_requires_gradient_sync"


class RecordingFSDPModule:
    def __init__(self):
        self.writes: list[tuple[str, bool, bool]] = []

    def set_reshard_after_backward(self, reshard, *, recurse=True):
        self.writes.append((RESHARD, reshard, recurse))

    def set_requires_gradient_sync(self, requires_gradient_sync, *, recurse=True):
        self.writes.append((REDUCE, requires_gradient_sync, recurse))


class StubTrainer(GradientSyncMixin):
    def __init__(self, *, fsdp_wrapped=True, reshard_after_backward=True, defer_grad_sync=False):
        self.parallelism_config = types.SimpleNamespace(
            fsdp_reshard_after_backward=reshard_after_backward, fsdp_defer_grad_sync=defer_grad_sync
        )
        self._fsdp_wrapped = fsdp_wrapped
        self.model = nn.Linear(4, 4)
        self._window_modules = []
        self._window_end_armed = True


def _run_windows(trainer, n_windows=1):
    for _window in range(n_windows):
        for microstep in range(GRAD_ACCUM):
            trainer._set_window_end(microstep == GRAD_ACCUM - 1)


def test_unwrapped_module_tree_yields_no_fsdp_modules():
    model = nn.Sequential(nn.Linear(4, 4), nn.ModuleList([nn.Linear(4, 4)]))
    assert fsdp2_modules(model) == []


@pytest.mark.parametrize(
    "fsdp_wrapped, reshard_after_backward, defer_grad_sync, setters",
    [
        (True, True, False, ()),  # the defaults: torch reshards and reduces after every backward
        (True, False, False, (RESHARD,)),
        (True, True, True, (REDUCE,)),
        (True, False, True, (RESHARD, REDUCE)),
        (False, False, True, ()),  # accelerate or QLoRA: the mixin applied no wrap to toggle
    ],
)
def test_each_knob_drives_only_its_own_setter(
    monkeypatch, fsdp_wrapped, reshard_after_backward, defer_grad_sync, setters
):
    """Arming needs this mixin's own wrap and a knob off its default."""
    module = RecordingFSDPModule()
    monkeypatch.setattr(grad_sync, "fsdp2_modules", lambda model: [module])

    trainer = StubTrainer(
        fsdp_wrapped=fsdp_wrapped, reshard_after_backward=reshard_after_backward, defer_grad_sync=defer_grad_sync
    )
    trainer._install_grad_accum_window()
    assert trainer._window_modules == ([module] if setters else [])

    _run_windows(trainer)
    disarm = [(setter, False, False) for setter in setters]
    rearm = [(setter, True, False) for setter in setters]
    assert module.writes == disarm + rearm


def test_window_writes_off_for_microsteps_and_back_on_for_the_last():
    trainer = StubTrainer(defer_grad_sync=True)
    modules = [RecordingFSDPModule() for _ in range(3)]
    trainer._window_modules = modules

    _run_windows(trainer)

    for module in modules:
        # Two writes, not one per microstep: the armed latch makes the repeats free.
        assert module.writes == [(REDUCE, False, False), (REDUCE, True, False)]


def test_two_windows_each_end_armed():
    """Every window must end armed, or the optimizer step reads unsharded params or unreduced grads."""
    trainer = StubTrainer(reshard_after_backward=False, defer_grad_sync=True)
    module = RecordingFSDPModule()
    trainer._window_modules = [module]

    _run_windows(trainer, n_windows=2)

    assert trainer._window_end_armed is True
    one_window = [(RESHARD, False, False), (REDUCE, False, False), (RESHARD, True, False), (REDUCE, True, False)]
    assert module.writes == one_window * 2


def test_gradient_accumulation_one_never_disarms():
    """At GA=1 every microstep is the window's last, so neither knob may skip anything."""
    trainer = StubTrainer(reshard_after_backward=False, defer_grad_sync=True)
    module = RecordingFSDPModule()
    trainer._window_modules = [module]

    for _step in range(4):
        trainer._set_window_end(True)

    assert module.writes == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
