"""SMPO's scheduled margin is read off the model without a fallback to the static config value: a skipped seeding path
would otherwise silently train the static margin for the whole schedule."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from halo.trainers.preference.smpo import SMPOTrainer


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _trainer(model: nn.Module, *, scheduled: bool = True) -> SMPOTrainer:
    trainer = object.__new__(SMPOTrainer)
    # The static config value a fallback would silently return.
    trainer.args = SimpleNamespace(use_margin_schedule=scheduled, target_margin=0.9)
    trainer.model = model
    return trainer


def test_smpo_scheduled_margin_reads_the_model_and_never_falls_back():
    model = _Tiny()
    model.target_margin = 0.25
    trainer = _trainer(model)
    assert trainer._get_current_margin() == 0.25

    del model.target_margin
    with pytest.raises(AttributeError):
        trainer._get_current_margin()


def test_smpo_scheduled_margin_does_not_unwrap_per_read():
    """The read runs once per microbatch, so the unwrap walk must stay off the path when the model carries the value."""
    model = _Tiny()
    model.target_margin = 0.25
    trainer = _trainer(model)

    def _unwrap_forbidden():
        raise AssertionError("per-microbatch margin read walked the wrapper chain")

    trainer._top_level_model = _unwrap_forbidden
    assert trainer._get_current_margin() == 0.25


def test_smpo_scheduled_margin_survives_a_wrapper_that_hides_the_attribute():
    """A wrapper that does not delegate attribute access (DataParallel stands in for DDP) hides the value the callback
    wrote on the module beneath it, so the unwrap seam must still be consulted on that miss."""
    inner = _Tiny()
    inner.target_margin = 0.25
    trainer = _trainer(nn.DataParallel(inner))

    assert getattr(trainer.model, "target_margin", None) is None, "wrapper must not expose it"
    assert trainer._get_current_margin() == 0.25


def test_smpo_unscheduled_margin_is_the_config_value():
    assert _trainer(_Tiny(), scheduled=False)._get_current_margin() == 0.9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
