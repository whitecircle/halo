#!/usr/bin/env python
"""``DistributedSFTTrainer`` forwards exactly the inputs TRL's ``SFTTrainer.compute_loss`` would.

The trainer calls the base ``Trainer.compute_loss`` itself, so its metrics stay on the device, and therefore
repeats the input preparation TRL's ``compute_loss`` does before it: the cache off, the eval-intent key popped,
the router logits where TRL weighs the aux loss and Liger's fused-loss keys. Run against the installed TRL across
every switch that preparation reads, so a TRL release that adds or changes a key fails here instead of drifting.

Run: python tests/cpu/trainers/test_sft_trl_loss_inputs.py
"""

import itertools
import types
from unittest import mock

import pytest
import torch
from transformers import Trainer
from trl import SFTTrainer

from halo.trainers.sft import DistributedSFTTrainer


class _ReachedTheForward(Exception):
    """Raised by the stubbed base loss: everything before it ran."""


def _host(*, training, liger, loss_type, aux_loss, compute_metrics):
    trainer = object.__new__(DistributedSFTTrainer)
    trainer.model = types.SimpleNamespace(training=training)
    trainer.args = types.SimpleNamespace(use_liger_kernel=liger, prediction_loss_only=False, loss_type=loss_type)
    trainer.aux_loss_enabled = aux_loss
    trainer.compute_metrics = compute_metrics
    trainer.preprocess_logits_for_metrics = None
    trainer.parallelism_config = types.SimpleNamespace(is_cp_mode=False)
    trainer._warned_empty_labels = True
    return trainer


def _forwarded(compute_loss, trainer, prediction_loss_only):
    """The inputs ``compute_loss`` hands the base ``Trainer.compute_loss``."""
    ids = torch.arange(6).reshape(1, 6)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    if prediction_loss_only is not None:
        inputs["_prediction_loss_only"] = prediction_loss_only
    seen = {}

    def base_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        seen.update(inputs)
        raise _ReachedTheForward

    with mock.patch.object(Trainer, "compute_loss", base_loss), pytest.raises(_ReachedTheForward):
        compute_loss(trainer, None, inputs)
    return seen


_SWITCHES = {
    "training": [True, False],
    "liger": [True, False],
    "loss_type": ["nll", "dft"],
    "aux_loss": [True, False],
    "compute_metrics": [None, len],
}


@pytest.mark.parametrize("prediction_loss_only", [None, True, False])
@pytest.mark.parametrize(
    "switches", [dict(zip(_SWITCHES, values, strict=True)) for values in itertools.product(*_SWITCHES.values())]
)
def test_the_forward_sees_trls_loss_inputs(switches, prediction_loss_only):
    trl = _forwarded(SFTTrainer.compute_loss, _host(**switches), prediction_loss_only)
    halo = _forwarded(DistributedSFTTrainer.compute_loss, _host(**switches), prediction_loss_only)
    assert trl.keys() == halo.keys()
    for key, value in trl.items():
        assert torch.equal(halo[key], value) if torch.is_tensor(value) else halo[key] == value, key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
