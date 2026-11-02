#!/usr/bin/env python
"""``local_optimizer_state`` observes the optimizer it reads without stepping it.

The resume suites snapshot a trainer's optimizer at ``on_train_begin``, after the restore and before the first
resumed step. A restore that loaded no state leaves the optimizer empty, and ``get_optimizer_state_dict`` fills an
empty optimizer with a zero-LR step before reading it. That step runs the trainer's step-pre-hooks outside the
restore's mute: the deferred EP sweep claims the current step there, and the first resumed step's clip then raises
"ran twice" instead of the resume failing on its named checks. The snapshot must read an empty optimizer as empty,
leave it untouched, and still read a stepped one in full.

    python tests/cpu/conventions/test_local_optimizer_state.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from halo.trainers.parallelism.grad_sync import claim_optimizer_step, register_grad_sync_step_hook
from tests.common.optimizer_state import local_optimizer_state

RESUMED_STEP = 3
SWEEP_MARKER = "_deferred_sweep_last_step"


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _accelerate_wrapped(optimizer):
    """``optimizer`` behind accelerate's wrapper, which the trainers hand their callbacks; a step through it
    reaches ``optimizer`` as a synced, unscaled training step does."""
    wrapper = object.__new__(AcceleratedOptimizer)
    wrapper.optimizer, wrapper.scaler = optimizer, None
    wrapper.gradient_state = SimpleNamespace(sync_gradients=True, is_xla_gradients_synced=True)
    wrapper.accelerator_state = SimpleNamespace(distributed_type=DistributedType.NO)
    return wrapper


def _resumed_trainer(optimizer):
    """A trainer at the first resumed step, its optimizer carrying a claiming step-pre-hook as the deferred EP
    sweep's backstop does; returns the trainer and the list each hook call appends to."""
    calls = []
    trainer = SimpleNamespace(optimizer=optimizer, state=SimpleNamespace(global_step=RESUMED_STEP))

    def sweep():
        calls.append(trainer.state.global_step)
        claim_optimizer_step(trainer, SWEEP_MARKER)

    register_grad_sync_step_hook(trainer, "_sweep_hook_registered", lambda: True, sweep)
    assert trainer._sweep_hook_registered, "premise: the hook is registered"
    return trainer, calls


@pytest.mark.parametrize("wrap", [False, True], ids=["bare", "accelerate-wrapped"])
def test_an_optimizer_holding_no_state_reads_as_empty_and_is_not_stepped(wrap):
    model = _model()
    inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer, calls = _resumed_trainer(inner)
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}

    snapshot = local_optimizer_state(model, _accelerate_wrapped(inner) if wrap else inner)

    assert snapshot == {"state": {}}
    assert calls == [], "the snapshot ran the trainer's step-pre-hooks"
    assert len(inner.state) == 0, "the snapshot left state in the live optimizer"
    assert all(param.grad is None for param in model.parameters())
    assert all(torch.equal(param, weights[name]) for name, param in model.named_parameters())
    assert claim_optimizer_step(trainer, SWEEP_MARKER), "the first resumed step's clip would find its step claimed"


def test_a_stepped_optimizer_reads_every_param_state_by_fqn():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    optimizer.step()

    snapshot = local_optimizer_state(model, _accelerate_wrapped(optimizer))

    names = {param: name for name, param in model.named_parameters()}
    live = {names[param]: entry for param, entry in optimizer.state.items()}
    assert set(snapshot["state"]) == set(names.values()) == set(live)
    for fqn, entry in live.items():
        assert snapshot["state"][fqn].keys() == entry.keys(), fqn
        for key, value in entry.items():
            assert torch.equal(snapshot["state"][fqn][key], value), (fqn, key)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
