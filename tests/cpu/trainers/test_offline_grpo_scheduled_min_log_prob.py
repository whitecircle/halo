#!/usr/bin/env python
"""Test that offline GRPO's loss reads the SCHEDULED ``min_log_prob`` the callback actually set.

``VariableSchedulerCallback`` unwraps DDP/accelerate (``target_obj.module``) before setting the
attribute, so the scheduled value lands on the trainer's unwrapped ``self.model``. Reading it off
the wrapped ``model`` handed to ``compute_loss`` finds nothing and silently falls back to the
config's start value for the whole run, while ``log()`` (which reads ``self.model``) reports the
ramp — the loss and the logged metric disagree.

Run: python tests/cpu/trainers/test_offline_grpo_scheduled_min_log_prob.py
"""

import contextlib
import types

import pytest
import torch

from src.callbacks.variable_scheduler import VariableSchedulerCallback
from src.trainers.grpo.offline import OfflineGRPOTrainer

_CONFIG_START = -2.3
_SCHEDULED = -0.5


class _Sentinel(Exception):
    """Raised from the stubbed log-prob call to end compute_loss at the point of interest."""


def _inputs(batch=2, prompt_len=3, completion_len=4):
    return {
        "prompt_input_ids": torch.zeros(batch, prompt_len, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
        "completion_input_ids": torch.zeros(batch, completion_len, dtype=torch.long),
        "completion_attention_mask": torch.ones(batch, completion_len, dtype=torch.long),
        "advantage": torch.zeros(batch),
    }


def _trainer_and_wrapped(*, scheduled_on_module):
    """A stub trainer plus the DDP-style wrapper compute_loss receives.

    ``self.model`` is the unwrapped module (what the callback writes to); ``wrapped.module`` is it,
    mirroring DDP, and the wrapper itself carries no ``min_log_prob``.
    """
    unwrapped = types.SimpleNamespace()
    wrapped = types.SimpleNamespace(module=unwrapped)
    me = types.SimpleNamespace(
        model=unwrapped,
        min_log_prob=_CONFIG_START,
        # compute_loss opens the (inert here) QLoRA bf16 autocast region around the forwards.
        _peft_has_been_casted_to_bf16=False,
        accelerator=types.SimpleNamespace(device=torch.device("cpu")),
    )

    if scheduled_on_module:
        callback = VariableSchedulerCallback(
            attribute_name="min_log_prob",
            initial_value=_SCHEDULED,
            final_value=_SCHEDULED,
        )
        callback.total_steps = 10
        callback.on_train_begin(None, types.SimpleNamespace(max_steps=10), None, model=wrapped)

    seen = {}

    def _fake_logps(_model, *_args, min_log_prob=None, **_kwargs):
        seen["min_log_prob"] = min_log_prob
        raise _Sentinel

    me._get_per_token_logps = _fake_logps
    # compute_loss is the autocast wrapper; the body under test is the inner half.
    me._compute_loss_inner = types.MethodType(OfflineGRPOTrainer._compute_loss_inner, me)
    return me, wrapped, seen


def _min_log_prob_used(*, scheduled_on_module):
    me, wrapped, seen = _trainer_and_wrapped(scheduled_on_module=scheduled_on_module)
    with contextlib.suppress(_Sentinel):
        OfflineGRPOTrainer.compute_loss(me, wrapped, _inputs())
    return seen["min_log_prob"]


def test_callback_writes_to_the_unwrapped_module():
    # Guards the premise: the scheduler unwraps, so the attribute is NOT on the wrapper.
    _, wrapped, _ = _trainer_and_wrapped(scheduled_on_module=True)
    assert wrapped.module.min_log_prob == _SCHEDULED
    assert not hasattr(wrapped, "min_log_prob")


def test_loss_uses_the_scheduled_value():
    assert _min_log_prob_used(scheduled_on_module=True) == _SCHEDULED


def test_loss_falls_back_to_config_when_unscheduled():
    assert _min_log_prob_used(scheduled_on_module=False) == _CONFIG_START


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
