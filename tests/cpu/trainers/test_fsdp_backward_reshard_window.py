#!/usr/bin/env python
"""``fsdp_reshard_after_backward=False`` toggles per grad-accum window, never pinned for the run.

torch's ``set_reshard_after_backward`` contract is per window: off for microbatches 1..n-1, back on
for the last, so the window's final backward reshards and the optimizer reads sharded params
carrying gradients. Pinning it off instead leaves the transient unsharded params registered forever
(0 grad norm, nothing clipped, the next forward blind to the optimizer's update). This pins the
trainer-side wiring: which runs arm the toggle at all, and the exact flag sequence a window writes.

The runtime invariants it protects are GPU-only —
``tests/gpu/trainers/sft/test_sft_fsdp_backward_reshard.py``.

    python tests/cpu/trainers/test_fsdp_backward_reshard_window.py
"""

import types

import pytest
import torch.nn as nn

from src.distributed.fsdp import fsdp2_modules
from src.trainers.mixins import grad_sync
from src.trainers.mixins.grad_sync import GradientSyncMixin

GRAD_ACCUM = 3


class RecordingFSDPModule:
    """Stands in for an ``FSDPModule``, recording every ``set_reshard_after_backward`` write."""

    def __init__(self):
        self.writes: list[tuple[bool, bool]] = []

    def set_reshard_after_backward(self, reshard, *, recurse=True):
        self.writes.append((reshard, recurse))


class StubTrainer(GradientSyncMixin):
    """The real mixin methods over the minimum state they read."""

    def __init__(self, *, fsdp_wrapped, reshard_after_backward, model=None):
        self.parallelism_config = types.SimpleNamespace(fsdp_reshard_after_backward=reshard_after_backward)
        self._fsdp_wrapped = fsdp_wrapped
        self.model = model if model is not None else nn.Linear(4, 4)
        self._backward_reshard_modules = []
        self._backward_reshard_armed = True


def test_unwrapped_module_tree_yields_no_fsdp_modules():
    """The toggle can only ever reach modules ``fully_shard`` actually wrapped."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ModuleList([nn.Linear(4, 4)]))
    assert fsdp2_modules(model) == []


@pytest.mark.parametrize(
    "fsdp_wrapped, reshard_after_backward, armed",
    [
        (True, False, True),  # the lever: this mixin's own wrap, flag off
        (True, True, False),  # default run — torch reshards after every backward already
        (False, False, False),  # accelerate/DDP owns the wrap; the knob is warned and ignored
        (False, True, False),
    ],
)
def test_window_armed_only_for_the_lever(monkeypatch, fsdp_wrapped, reshard_after_backward, armed):
    """Arming is exactly ``_fsdp_wrapped and not fsdp_reshard_after_backward`` — nothing else."""
    wrapped = [RecordingFSDPModule()]
    monkeypatch.setattr(grad_sync, "fsdp2_modules", lambda model: list(wrapped))

    trainer = StubTrainer(fsdp_wrapped=fsdp_wrapped, reshard_after_backward=reshard_after_backward)
    trainer._setup_backward_reshard_window()
    assert trainer._backward_reshard_modules == (wrapped if armed else [])

    # Unarmed, a microstep asking to unshard must leave torch's always-reshard default in place.
    trainer._set_backward_reshard(False)
    assert trainer._backward_reshard_armed is (not armed)


def test_window_writes_off_for_microsteps_and_back_on_for_the_last():
    """One window: disarm once at its first microstep, re-arm once at its last."""
    trainer = StubTrainer(fsdp_wrapped=True, reshard_after_backward=False)
    modules = [RecordingFSDPModule() for _ in range(3)]
    trainer._backward_reshard_modules = modules

    for microstep in range(GRAD_ACCUM):
        trainer._set_backward_reshard(microstep == GRAD_ACCUM - 1)

    for module in modules:
        # Two writes, not one per microstep: the armed latch makes the repeats free.
        assert module.writes == [(False, False), (True, False)]


def test_two_windows_each_end_resharded():
    """Every window must END armed, or the next optimizer step reads unsharded params again."""
    trainer = StubTrainer(fsdp_wrapped=True, reshard_after_backward=False)
    module = RecordingFSDPModule()
    trainer._backward_reshard_modules = [module]

    for _window in range(2):
        for microstep in range(GRAD_ACCUM):
            trainer._set_backward_reshard(microstep == GRAD_ACCUM - 1)
        assert trainer._backward_reshard_armed is True

    assert module.writes == [(False, False), (True, False), (False, False), (True, False)]


def test_gradient_accumulation_one_never_disarms():
    """At GA=1 every microstep is the window's last, so the lever must not unshard anything."""
    trainer = StubTrainer(fsdp_wrapped=True, reshard_after_backward=False)
    module = RecordingFSDPModule()
    trainer._backward_reshard_modules = [module]

    for _step in range(4):
        trainer._set_backward_reshard(True)

    assert module.writes == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
