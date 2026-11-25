#!/usr/bin/env python
"""``DistributedSFTTrainer`` forwards exactly the inputs TRL's ``SFTTrainer.compute_loss`` would.
The trainer repeats TRL's input preparation, so a TRL release that adds or changes a key fails here.
"""

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


_BASE = {"training": True, "liger": True, "loss_type": "nll", "aux_loss": True, "compute_metrics": None}
_NO_AUX = {**_BASE, "aux_loss": False}
_NO_AUX_METRICS = {**_NO_AUX, "compute_metrics": len}
_EVAL_NO_AUX = {**_NO_AUX, "training": False}
_EVAL_NO_AUX_METRICS = {**_EVAL_NO_AUX, "compute_metrics": len}


@pytest.mark.parametrize(
    ("switches", "prediction_loss_only"),
    [
        pytest.param(_NO_AUX, None, id="switches2-None"),
        pytest.param(_NO_AUX, True, id="switches2-True"),
        pytest.param(_NO_AUX, False, id="switches2-False"),
        pytest.param(_NO_AUX_METRICS, None, id="switches3-None"),
        pytest.param(_NO_AUX_METRICS, True, id="switches3-True"),
        pytest.param(_NO_AUX_METRICS, False, id="switches3-False"),
        pytest.param(_EVAL_NO_AUX, None, id="switches18-None"),
        pytest.param(_EVAL_NO_AUX, True, id="switches18-True"),
        pytest.param(_EVAL_NO_AUX, False, id="switches18-False"),
        pytest.param(_EVAL_NO_AUX_METRICS, None, id="switches19-None"),
        pytest.param(_EVAL_NO_AUX_METRICS, True, id="switches19-True"),
        pytest.param(_EVAL_NO_AUX_METRICS, False, id="switches19-False"),
        pytest.param({**_NO_AUX, "loss_type": "dft"}, None, id="switches6-None"),
        pytest.param(_BASE, None, id="switches0-None"),
        pytest.param({**_BASE, "liger": False}, None, id="switches8-None"),
        pytest.param({**_NO_AUX, "liger": False}, None, id="switches10-None"),
    ],
)
def test_the_forward_sees_trls_loss_inputs(switches, prediction_loss_only):
    trl = _forwarded(SFTTrainer.compute_loss, _host(**switches), prediction_loss_only)
    halo = _forwarded(DistributedSFTTrainer.compute_loss, _host(**switches), prediction_loss_only)
    assert trl.keys() == halo.keys()
    for key, value in trl.items():
        assert torch.equal(halo[key], value) if torch.is_tensor(value) else halo[key] == value, key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
