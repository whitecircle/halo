#!/usr/bin/env python
"""StoredMetricsMixin: accumulation + log-time drain (floats and detached 0-dim tensors).

The mixin buffers per-microbatch metrics and folds their mean into ``logs`` once per ``log`` call.
Values may be plain floats or detached 0-dim tensors (the buffered-drain path that avoids a host
sync per microbatch — teacher distillation stores tensors); both must produce identical logged
values and the buffer must clear after the flush.

    python tests/cpu/trainers/test_stored_metrics.py
"""

import pytest
import torch

from src.trainers.mixins.stored_metrics import StoredMetricsMixin


class _Recorder:
    """MRO terminal capturing what the mixin passes up."""

    def log(self, logs, start_time=None):
        self.last_logs = dict(logs)
        return logs


class _Trainer(StoredMetricsMixin, _Recorder):
    pass


def test_float_metrics_mean_flushed_and_cleared():
    t = _Trainer()
    t.store_metrics({"sft_loss": 1.0, "kd": 3.0})
    t.store_metrics({"sft_loss": 2.0, "kd": 5.0})
    logs = {"loss": 0.1}
    t.log(logs)
    assert logs["sft_loss"] == pytest.approx(1.5)
    assert logs["kd"] == pytest.approx(4.0)
    # Buffer cleared: a second log window starts fresh (no carry-over mean).
    t.store_metrics({"sft_loss": 4.0})
    logs2 = {"loss": 0.1}
    t.log(logs2)
    assert logs2["sft_loss"] == pytest.approx(4.0)


def test_tensor_metrics_drain_identically_to_item():
    # Detached 0-dim tensors (incl. bf16, as the losses come off a bf16 forward) must log the same
    # value the eager `.item()` path produced.
    t = _Trainer()
    values = [torch.tensor(1.25, dtype=torch.bfloat16), torch.tensor(2.75, dtype=torch.bfloat16)]
    for v in values:
        t.store_metrics({"distillation_loss": v})
    logs = {"loss": 0.1}
    t.log(logs)
    expected = torch.tensor([v.item() for v in values]).mean().item()
    assert logs["distillation_loss"] == pytest.approx(expected)


def test_train_eval_buffers_are_separate():
    t = _Trainer()
    t.store_metrics({"m": 1.0}, train_eval="train")
    t.store_metrics({"m": 9.0}, train_eval="eval")
    train_logs = {"loss": 0.1}  # "loss" key routes to the train buffer
    t.log(train_logs)
    assert train_logs["m"] == pytest.approx(1.0)
    eval_logs = {"eval_loss": 0.1}
    t.log(eval_logs)
    # Eval flushes gain the eval_ prefix (TRL convention) so they never land on the train series.
    assert eval_logs["eval_m"] == pytest.approx(9.0)
    assert "m" not in eval_logs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
