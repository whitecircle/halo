#!/usr/bin/env python
"""Per-microbatch metrics must be stored as detached tensors, not ``.item()`` floats.
An ``.item()`` per metric per microbatch is a host sync per value; ``StoredMetricsMixin`` drains once per log."""

from types import SimpleNamespace

import pytest
import torch

from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.stored_metrics import StoredMetricsMixin
from tests.common.smpo_trainer import smpo_trainer


def _bare_smpo() -> SMPOTrainer:
    t = smpo_trainer(target_margin=0.5)
    t.concatenated_forward = lambda model, batch, real_pairs=None: {
        "chosen_logps": torch.tensor([-1.0, -2.0]),
        "rejected_logps": torch.tensor([-3.0, -4.0]),
        "chosen_sft_loss": torch.tensor(0.7),
        "rejected_sft_loss": torch.tensor(0.9),
        "mean_chosen_logits": torch.tensor(0.1),
        "mean_rejected_logits": torch.tensor(-0.1),
    }
    t.objective = SimpleNamespace(
        chosen_sft_ratio=0.8,
        margin_loss=lambda chosen, rejected, target_margin: (
            torch.tensor([0.2, 0.3]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 0.6]),
        ),
    )
    return t


def test_smpo_metrics_are_detached_tensors():
    trainer = _bare_smpo()
    loss, metrics, _ = trainer.get_batch_loss_metrics(model=None, batch={}, train_eval="train")
    assert torch.is_tensor(loss)
    assert metrics, "no metrics produced"
    for name, value in metrics.items():
        assert torch.is_tensor(value), f"metric {name!r} is {type(value).__name__}, not a tensor (host sync)"
        assert value.dim() == 0, f"metric {name!r} is not 0-dim"
        assert not value.requires_grad, f"metric {name!r} is not detached"


def test_smpo_eval_metrics_are_tensors_and_prefixed():
    trainer = _bare_smpo()
    trainer.eval_split_rows = lambda num_rows: num_rows
    _, metrics, _ = trainer.get_batch_loss_metrics(
        model=None, batch={"prompt_input_ids": torch.zeros(2, 1)}, train_eval="eval"
    )
    assert all(name.startswith("eval_") for name in metrics)
    assert all(torch.is_tensor(v) for v in metrics.values())


def test_sdpg_routes_through_stored_metrics_mixin():
    assert issubclass(DistributedSDPGTrainer, StoredMetricsMixin)
    t = DistributedSDPGTrainer.__new__(DistributedSDPGTrainer)
    t.store_metrics({"opd_loss": torch.tensor(1.5)}, train_eval="train")
    assert t._stored_metrics["train"].flush() == {"opd_loss": 1.5}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
