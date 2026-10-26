#!/usr/bin/env python
"""Test the rename of TRL's cumulative ``num_tokens`` metric to ``num_attended_tokens_seen`` in single-process isolation.

Drives ``DistributedSFTTrainer.compute_loss`` over a stubbed TRL loss — no GPU or distributed init.

Run: python tests/cpu/trainers/test_attended_tokens_metric.py
"""

import types
from collections import defaultdict
from unittest import mock

import pytest
import torch
from trl import SFTTrainer

from src.trainers.sft import DistributedSFTTrainer


def _fake_self(metrics):
    """A bare trainer whose TRL loss left ``metrics``, run through the real non-CP ``compute_loss``."""
    trainer = object.__new__(DistributedSFTTrainer)
    trainer._metrics = metrics
    trainer._total_train_tokens = 0
    trainer._warned_empty_labels = True
    trainer.model = types.SimpleNamespace(training=True)
    trainer.parallelism_config = types.SimpleNamespace(is_cp_mode=False, non_dp_replication_factor=1)
    return trainer


def _relabel(trainer):
    with mock.patch.object(SFTTrainer, "compute_loss", return_value=(torch.zeros(()), {})):
        DistributedSFTTrainer.compute_loss(trainer, None, {})


def _mode_dict(**kwargs):
    d = defaultdict(list)
    d.update(kwargs)
    return d


def test_renames_train_num_tokens_preserving_value():
    me = _fake_self({"train": _mode_dict(num_tokens=[1234], loss=[0.5]), "eval": _mode_dict()})
    _relabel(me)
    assert "num_tokens" not in me._metrics["train"], "stale num_tokens must be popped"
    assert me._metrics["train"]["num_attended_tokens_seen"] == [1234], "value must be preserved"
    assert me._metrics["train"]["loss"] == [0.5], "other metrics must be untouched"


def test_renames_in_every_mode():
    me = _fake_self({"train": _mode_dict(num_tokens=[10]), "eval": _mode_dict(num_tokens=[7])})
    _relabel(me)
    assert me._metrics["train"]["num_attended_tokens_seen"] == [10]
    assert me._metrics["eval"]["num_attended_tokens_seen"] == [7]
    assert "num_tokens" not in me._metrics["train"]
    assert "num_tokens" not in me._metrics["eval"]


def test_noop_when_num_tokens_absent():
    me = _fake_self({"train": _mode_dict(loss=[0.1]), "eval": _mode_dict()})
    _relabel(me)
    assert "num_attended_tokens_seen" not in me._metrics["train"], "must not invent the key"
    assert "num_tokens" not in me._metrics["train"]


def test_noop_when_metrics_empty():
    # TRL's SFTTrainer.__init__ always installs _metrics; an empty one must not raise.
    _relabel(_fake_self({}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
