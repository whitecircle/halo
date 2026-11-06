#!/usr/bin/env python
"""Seconds blocked in the train loader's ``next`` reach the efficiency callback per optimizer step: a sleeping
collator raises the metric by about the sleep, a free one leaves it near zero, and the world reduction rides the
step's one existing all-reduce."""

import time
import types

import pytest
import torch
from torch.utils.data import DataLoader

from halo.callbacks import efficiency
from halo.callbacks.efficiency import EfficiencyCallback
from halo.trainers import token_metrics
from halo.trainers.step_guard import NonFiniteStepGuardMixin
from halo.trainers.token_metrics import TokenMetricsMixin
from tests.common.parallelism import make_parallelism_config

SLEEP = 0.05
MICRO_BATCHES = 3


class _Base:
    """Stands in for ``Trainer.get_batch_samples``: draws ``num_batches`` from the iterator."""

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        return [next(epoch_iterator) for _ in range(num_batches)], None


class _Host(NonFiniteStepGuardMixin, TokenMetricsMixin, _Base):
    def __init__(self):
        self.args = types.SimpleNamespace(include_num_input_tokens_seen="no")
        self._tr_loss = torch.zeros(())


def _loader(delay: float) -> DataLoader:
    def collate(rows):
        time.sleep(delay)
        return torch.tensor(rows)

    return DataLoader(list(range(MICRO_BATCHES * 4)), batch_size=1, collate_fn=collate)


def _wait_of_one_step(delay: float) -> float:
    host = _Host()
    host.get_batch_samples(iter(_loader(delay)), MICRO_BATCHES, "cpu")
    return host.drain_step_work()[2]


def test_a_sleeping_collator_raises_the_wait_by_about_the_sleep():
    assert _wait_of_one_step(SLEEP) == pytest.approx(MICRO_BATCHES * SLEEP, abs=0.5 * SLEEP)


def test_a_free_collator_leaves_the_wait_near_zero():
    assert _wait_of_one_step(0.0) < 0.5 * SLEEP


def test_the_drain_resets_the_window():
    host = _Host()
    host.get_batch_samples(iter(_loader(SLEEP)), 1, "cpu")
    assert host.drain_step_work()[2] > 0
    assert host.drain_step_work()[2] == 0.0


def test_the_wait_rides_the_input_token_all_reduce_and_averages_over_ranks(monkeypatch):
    sent = []

    def world_sums(values):
        sent.append(list(values))
        return [2 * value for value in values]

    monkeypatch.setattr(token_metrics, "world_sums", world_sums)
    monkeypatch.setattr(token_metrics, "get_global_world_size", lambda: 4)
    host = _Host()
    host.args.include_num_input_tokens_seen = "all"
    host.state = types.SimpleNamespace(num_input_tokens_seen=0)
    host._unsummed_input_tokens = 10
    host._data_wait_seconds = 1.5
    assert host.drain_step_work()[2] == pytest.approx(2 * 1.5 / 4)
    assert sent == [[10, 1.5]], "one all-reduce carries the tokens and the wait"


def test_the_callback_logs_the_step_and_running_mean_wait():
    class Source:
        waits = iter([0.5, 1.5])

        def drain_step_work(self):
            return 0.0, 0, next(self.waits)

    config = make_parallelism_config(world_size=1, gpus_per_node=1)
    callback = EfficiencyCallback(config, n_warmup_steps=0)
    callback._attention_source = lambda: Source()
    for _ in range(2):
        callback._pending_steps.append(
            efficiency._PendingStep(0.0, 1.0, 1, 1, None, callback._drain_step_work().data_wait)
        )
    callback._record_pending_steps()
    logs = {}
    callback.on_log(None, types.SimpleNamespace(global_step=5), None, logs)
    assert logs["step_data_wait_seconds"] == 1.5 and logs["avg_data_wait_seconds"] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
