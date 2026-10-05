"""CP reference scoring reconstructs ordered ragged rows."""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from src.data.collators.offline_grpo import OfflineGRPOCPDataCollatorWithPadding
from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.gloo import run_gloo_ranks


def _batch(cp_size):
    return OfflineGRPOCPDataCollatorWithPadding(cp_size=cp_size)(
        [
            {
                "prompt_input_ids": [1, 2, 3, 4],
                "completion_input_ids": [5, 6, 7, 8, 9, 10, 11],
                "group_id": 0,
                "group_size": 2,
                "advantage": 1.0,
            },
            {
                "prompt_input_ids": [12],
                "completion_input_ids": [13, 14, 15],
                "group_id": 0,
                "group_size": 2,
                "advantage": -1.0,
            },
            {
                "prompt_input_ids": [16, 17],
                "completion_input_ids": [],
                "group_id": 1,
                "group_size": 1,
                "advantage": 0.0,
            },
        ]
    )


def _reassembly_worker(rank, cp_size):
    batch = _batch(cp_size)
    width = batch["input_ids"].size(1)
    full = -0.25 - torch.arange(3 * (width - 1)).reshape(3, width - 1).float() * 0.13
    chunk = width // cp_size
    start, end = rank * chunk, min((rank + 1) * chunk, width - 1)
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = object()
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=rank, process_group=dist.group.WORLD)
    trainer._cp_chunked_logps = lambda model, ids, mask, labels: (full[:, start:end], labels[:, start + 1 : end + 1])
    expected = [
        scores[labels != LABEL_IGNORE_INDEX] for scores, labels in zip(full, batch["labels"][:, 1:], strict=True)
    ]
    actual = trainer._cp_score_reference_batch(batch)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert actual[2].numel() == 0

    gather = dist.all_gather

    def reverse_shards(shards, local, **kwargs):
        gather(shards, local, **kwargs)
        shards.reverse()

    with patch("src.trainers.grpo.offline.dist.all_gather", side_effect=reverse_shards):
        broken = trainer._cp_score_reference_batch(batch)
    assert all(got.shape == want.shape for got, want in zip(broken, expected, strict=True))
    assert any(not torch.equal(got, want) for got, want in zip(broken, expected, strict=True)), (
        "fixture lost shard-order sensitivity"
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_reference_scores_reassemble_in_token_order_and_reject_reversed_shards(cp_size):
    run_gloo_ranks(_reassembly_worker, cp_size, cp_size, pg_timeout=datetime.timedelta(seconds=30))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
