"""CP sign diagnostics use full logical rows, including zero-target shards."""

import datetime
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.gloo import run_gloo_ranks


def _trainer(*, cp_size=1, cp_rank=0, training=True):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = nn.Linear(1, 1).train(training)
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=cp_size > 1)
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, process_group=dist.group.WORLD)
    trainer._pp_runtime = None
    trainer._sign_metric_buffer = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.accelerator = SimpleNamespace(gather=lambda values: values)
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
    expected = _trainer(training=training)
    expected._buffer_sign_metrics(values, advantages, mask)
    actual = _trainer(cp_size=cp_size, cp_rank=rank, training=training)
    actual._buffer_sign_metrics(
        {key: value[:, start:end] for key, value in values.items()}, advantages, mask[:, start:end]
    )
    for key in expected._sign_metric_buffer[mode]:
        torch.testing.assert_close(
            actual._sign_metric_buffer[mode][key][0], expected._sign_metric_buffer[mode][key][0], atol=1e-5, rtol=1e-6
        )
    expected._drain_sign_metrics(mode)
    actual._drain_sign_metrics(mode)
    assert actual._metrics[mode].keys() == expected._metrics[mode].keys()
    for key, summaries in expected._metrics[mode].items():
        assert actual._metrics[mode][key] == pytest.approx(summaries, abs=1e-5, rel=1e-6)
    assert not actual._sign_metric_buffer[mode]

    broken = _trainer(cp_size=cp_size, cp_rank=rank, training=training)
    with patch("src.trainers.grpo.offline.cp_sum_rows", side_effect=lambda totals, _config: totals):
        broken._buffer_sign_metrics(
            {key: value[:, start:end] for key, value in values.items()}, advantages, mask[:, start:end]
        )
    full_means = (values["logps"] * mask).sum(1) / mask.sum(1).clamp(min=1)
    assert not torch.allclose(broken._sign_metric_buffer[mode]["logps"][0], full_means), (
        "fixture lost reduction sensitivity"
    )


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_cp_sign_metrics_match_cp1_and_detect_dropped_reduction(cp_size, training):
    run_gloo_ranks(_diagnostics_worker, cp_size, cp_size, training, pg_timeout=datetime.timedelta(seconds=30))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
