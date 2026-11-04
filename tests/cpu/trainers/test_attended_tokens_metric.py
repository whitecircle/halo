#!/usr/bin/env python
"""TRL's attended-token counter counts a TP/ETP/CP batch once and logs as ``num_attended_tokens_seen``.
TRL divides its world gather by a TP width the toolkit's accelerator never carries, so the spine sets that divisor."""

import inspect
import types
from collections import defaultdict

import pytest
import torch
from trl import DPOTrainer, KTOTrainer, RewardTrainer

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer


def _reward_trainer(replication: int) -> DistributedRewardTrainer:
    trainer = object.__new__(DistributedRewardTrainer)
    trainer.parallelism_config = types.SimpleNamespace(non_dp_replication_factor=replication)
    trainer._tp_size = 1  # what TRL reads off an accelerator without a parallelism config
    trainer._deferred_liger_kernel = False
    trainer._router_aux_fields = []
    trainer._pristine_special_token_ids = [0]
    DistributedTrainerMixin._restore_trl_overrides(trainer)
    trainer.model = types.SimpleNamespace(training=True)
    trainer.args = types.SimpleNamespace(center_rewards_coefficient=None)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._total_train_tokens = 0
    # Every TP rank holds this batch, so the world gather returns it once per rank.
    trainer.accelerator = types.SimpleNamespace(
        gather_for_metrics=lambda tensor: torch.stack([tensor] * replication),
        gather=lambda tensor: torch.cat([tensor] * replication),
    )
    return trainer


def _pair_scorer(input_ids, attention_mask, use_cache):
    return types.SimpleNamespace(logits=input_ids[:, :1].float())


@pytest.mark.parametrize("replication", [1, 2, 4], ids=["data-parallel", "tp2", "etp4"])
def test_a_batch_the_tp_group_shares_counts_once(replication):
    trainer = _reward_trainer(replication)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    inputs = {"input_ids": torch.arange(16).reshape(4, 4), "attention_mask": attention_mask}
    trainer.compute_loss(_pair_scorer, inputs)
    assert trainer._total_train_tokens == int(attention_mask.sum())


@pytest.mark.parametrize("trl_loss", [DPOTrainer._compute_loss, KTOTrainer._compute_loss, RewardTrainer.compute_loss])
def test_every_trl_counter_divides_by_the_width_the_spine_sets(trl_loss):
    """A TRL release that counts another way must fail here rather than over-count."""
    assert "// self._tp_size" in inspect.getsource(trl_loss)


class _LoggingBase:
    def log(self, logs, start_time=None):
        self.logged = dict(logs)


class _Host(DistributedTrainerMixin, _LoggingBase):
    pass


def _host(metrics, *, trl_counter: bool) -> _Host:
    host = object.__new__(_Host)
    host._metrics = metrics
    host.state = types.SimpleNamespace(global_step=0)
    if trl_counter:
        host._tp_size = 1
    return host


def test_the_counter_logs_as_num_attended_tokens_seen_in_every_mode():
    metrics = {"train": defaultdict(list, num_tokens=[1234], loss=[0.5]), "eval": defaultdict(list, num_tokens=[7])}
    _host(metrics, trl_counter=True).log({})
    assert metrics["train"]["num_attended_tokens_seen"] == [1234] and "num_tokens" not in metrics["train"]
    assert metrics["eval"]["num_attended_tokens_seen"] == [7] and "num_tokens" not in metrics["eval"]
    assert metrics["train"]["loss"] == [0.5], "other metrics must be untouched"


def test_a_trainer_without_trls_counter_keeps_its_num_tokens():
    """GRPO's ``num_tokens`` is the generation's input-token count, not this counter."""
    metrics = {"train": defaultdict(list, num_tokens=[10])}
    _host(metrics, trl_counter=False).log({})
    assert metrics["train"]["num_tokens"] == [10] and "num_attended_tokens_seen" not in metrics["train"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
