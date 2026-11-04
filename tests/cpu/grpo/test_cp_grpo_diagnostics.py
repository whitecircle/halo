"""Offline GRPO's sign diagnostics: the logged keys and, under CP, full logical rows with zero-target shards."""

import datetime
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from halo.trainers.grpo.objective.offline_loss import offline_token_objective
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.world_metrics import WorldMetrics
from tests.common.gloo import run_gloo_ranks


def _trainer(*, cp_size=1, cp_rank=0, training=True):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = nn.Linear(1, 1).train(training)
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=cp_size > 1)
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, process_group=dist.group.WORLD)
    trainer._pp_runner = None
    trainer._sign_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return trainer


def _diagnostics_worker(rank, cp_size, training):
    advantages = torch.tensor([1.2, -0.6, -0.2, 0.0])
    mask = torch.zeros(4, 12)
    mask[0, 3:11] = 1
    mask[1, 7:10] = 1
    mask[2, 5:7] = 1
    values = {
        key: torch.arange(48).reshape(4, 12).float() * (index + 1) / 7 - index * 2
        for index, key in enumerate(OfflineGRPOTrainer._SIGN_METRIC_KEYS)
    }
    start, end = rank * (12 // cp_size), (rank + 1) * (12 // cp_size)
    mode = "train" if training else "eval"
    shards = {key: value[:, start:end] for key, value in values.items()}
    expected = _trainer(training=training)
    expected._record_sign_metrics(values, advantages, mask)
    actual = _trainer(cp_size=cp_size, cp_rank=rank, training=training)
    actual._record_sign_metrics(shards, advantages, mask[:, start:end])
    broken = _trainer(cp_size=cp_size, cp_rank=rank, training=training)
    with patch("halo.trainers.grpo.offline.trainer.cp_sum_rows", side_effect=lambda totals, _config: totals):
        broken._record_sign_metrics(shards, advantages, mask[:, start:end])
    for trainer in (expected, actual, broken):
        trainer._sign_metrics[mode].flush(trainer._metrics[mode])

    assert actual._metrics[mode].keys() == expected._metrics[mode].keys()
    for key, summaries in expected._metrics[mode].items():
        assert actual._metrics[mode][key] == pytest.approx(summaries, abs=1e-5, rel=1e-6), key
    assert not actual._sign_metrics[mode]
    assert broken._metrics[mode]["positive/logps_mean"] != pytest.approx(
        expected._metrics[mode]["positive/logps_mean"], abs=1e-3
    ), "fixture lost reduction sensitivity"


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_cp_sign_metrics_match_cp1_and_detect_dropped_reduction(cp_size, training):
    run_gloo_ranks(_diagnostics_worker, cp_size, cp_size, training, pg_timeout=datetime.timedelta(seconds=30))


def test_drained_metrics_name_the_policy_gradient_term_not_a_reward():
    trainer = _trainer()
    logps = torch.tensor([[-0.5, -1.0], [-2.0, -0.25]])
    advantages = torch.tensor([1.0, -1.0])
    _, sample_values = offline_token_objective(
        logps,
        logps,
        advantages,
        policy_gradient_formulation="reinforce",
        beta=0.1,
        ref_logps=logps - 0.1,
        ref_logps_unclamped=logps - 0.1,
    )
    trainer._record_sign_metrics(sample_values, advantages, torch.ones_like(logps))
    trainer._sign_metrics["train"].flush(trainer._metrics["train"])
    metrics = trainer._metrics["train"]
    assert not any("rewards" in key for key in metrics), sorted(metrics)
    # Pre-KL log π · A, averaged over each row's tokens.
    assert metrics["positive/pg_objective_mean"] == [pytest.approx(-0.75)]
    assert metrics["negative/pg_objective_mean"] == [pytest.approx(1.125)]


def test_a_pipeline_batchs_filler_rows_are_no_samples():
    """PP's all-ignore filler rows have zero advantage; read as real they pull the positive mean and minimum to 0."""
    values = {key: torch.tensor([[-2.0, -1.0], [-3.0, -0.5], [0.0, 0.0]]) for key in ("logps", "pg_objective")}
    advantages, mask = torch.tensor([1.0, 0.5, 0.0]), torch.tensor([[1, 1], [1, 1], [0, 0]])
    padded, alone = _trainer(), _trainer()
    padded._record_sign_metrics(values, advantages, mask, rows=torch.tensor([True, True, False]))
    alone._record_sign_metrics({key: value[:2] for key, value in values.items()}, advantages[:2], mask[:2])
    for trainer in (padded, alone):
        trainer._sign_metrics["train"].flush(trainer._metrics["train"])
    assert padded._metrics["train"] == alone._metrics["train"]
    assert padded._metrics["train"]["positive/logps_max"] == [-1.5]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
