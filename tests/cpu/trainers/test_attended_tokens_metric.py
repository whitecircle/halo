#!/usr/bin/env python
"""Test DistributedSFTTrainer._relabel_attended_tokens in single-process isolation.

Exercises the rename of TRL's cumulative ``num_tokens`` metric to the disambiguated
``num_attended_tokens_seen`` against a fake trainer ``self`` — no GPU or distributed init.

Run: python tests/cpu/trainers/test_attended_tokens_metric.py
"""

import types
from collections import defaultdict

import pytest

from src.trainers.sft import DistributedSFTTrainer


def _fake_self(metrics):
    return types.SimpleNamespace(_metrics=metrics)


def _mode_dict(**kwargs):
    d = defaultdict(list)
    d.update(kwargs)
    return d


def test_renames_train_num_tokens_preserving_value():
    me = _fake_self({"train": _mode_dict(num_tokens=[1234], loss=[0.5]), "eval": _mode_dict()})
    DistributedSFTTrainer._relabel_attended_tokens(me)
    assert "num_tokens" not in me._metrics["train"], "stale num_tokens must be popped"
    assert me._metrics["train"]["num_attended_tokens_seen"] == [1234], "value must be preserved"
    assert me._metrics["train"]["loss"] == [0.5], "other metrics must be untouched"


def test_renames_in_every_mode():
    me = _fake_self({"train": _mode_dict(num_tokens=[10]), "eval": _mode_dict(num_tokens=[7])})
    DistributedSFTTrainer._relabel_attended_tokens(me)
    assert me._metrics["train"]["num_attended_tokens_seen"] == [10]
    assert me._metrics["eval"]["num_attended_tokens_seen"] == [7]
    assert "num_tokens" not in me._metrics["train"]
    assert "num_tokens" not in me._metrics["eval"]


def test_noop_when_num_tokens_absent():
    me = _fake_self({"train": _mode_dict(loss=[0.1]), "eval": _mode_dict()})
    DistributedSFTTrainer._relabel_attended_tokens(me)
    assert "num_attended_tokens_seen" not in me._metrics["train"], "must not invent the key"
    assert "num_tokens" not in me._metrics["train"]


def test_noop_when_metrics_empty():
    # TRL's SFTTrainer.__init__ always installs _metrics; an empty one must not raise.
    DistributedSFTTrainer._relabel_attended_tokens(types.SimpleNamespace(_metrics={}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
