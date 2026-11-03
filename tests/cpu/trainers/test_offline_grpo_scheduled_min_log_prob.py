#!/usr/bin/env python
"""Offline GRPO's loss reads the scheduled ``min_log_prob`` the callback set on the unwrapped model.
Read off the wrapped ``model``, it would silently stay at the config's start value while ``log()`` reports the ramp."""

import contextlib
import types

import pytest
import torch

from halo.callbacks.variable_scheduler import VariableSchedulerCallback
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer

_CONFIG_START = -2.3
_SCHEDULED = -0.5


class _Sentinel(Exception):
    """Raised from the stubbed objective to end compute_loss at the point of interest."""


def _inputs(batch=2, prompt_len=3, completion_len=4):
    return {
        "prompt_input_ids": torch.zeros(batch, prompt_len, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
        "completion_input_ids": torch.zeros(batch, completion_len, dtype=torch.long),
        "completion_attention_mask": torch.ones(batch, completion_len, dtype=torch.long),
        "advantage": torch.zeros(batch),
    }


def _trainer_and_wrapped(*, scheduled_on_module):
    """``self.model`` is the unwrapped module the callback writes to; the DDP-style wrapper has no ``min_log_prob``."""
    unwrapped = types.SimpleNamespace()
    wrapped = types.SimpleNamespace(module=unwrapped)
    me = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    me.model = unwrapped
    me.min_log_prob = _CONFIG_START
    me.beta = 0.0
    # compute_loss opens the (inert here) QLoRA bf16 autocast region around the forwards.
    me._peft_has_been_casted_to_bf16 = False
    me.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    me.parallelism_config = types.SimpleNamespace(is_cp_mode=False)

    if scheduled_on_module:
        callback = VariableSchedulerCallback(
            attribute_name="min_log_prob",
            initial_value=_SCHEDULED,
            final_value=_SCHEDULED,
        )
        callback.total_steps = 10
        callback.on_train_begin(None, types.SimpleNamespace(max_steps=10), None, model=wrapped)

    seen = {}

    def _fake_objective(_logps, _advantages, _ref_logps, min_log_prob):
        seen["min_log_prob"] = min_log_prob
        raise _Sentinel

    me._get_per_token_logps = lambda _model, input_ids, _mask, logits_to_keep: torch.zeros(
        input_ids.size(0), logits_to_keep
    )
    me._token_objective = _fake_objective
    return me, wrapped, seen


def _min_log_prob_used(*, scheduled_on_module):
    me, wrapped, seen = _trainer_and_wrapped(scheduled_on_module=scheduled_on_module)
    with contextlib.suppress(_Sentinel):
        OfflineGRPOTrainer.compute_loss(me, wrapped, _inputs())
    return seen["min_log_prob"]


def test_callback_writes_to_the_unwrapped_module():
    _, wrapped, _ = _trainer_and_wrapped(scheduled_on_module=True)
    assert wrapped.module.min_log_prob == _SCHEDULED
    assert not hasattr(wrapped, "min_log_prob")


def test_loss_uses_the_scheduled_value():
    assert _min_log_prob_used(scheduled_on_module=True) == _SCHEDULED


def test_loss_falls_back_to_config_when_unscheduled():
    assert _min_log_prob_used(scheduled_on_module=False) == _CONFIG_START


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
