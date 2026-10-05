"""CP reference scoring reconstructs ordered ragged rows."""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.data.collators.offline_grpo import OfflineGRPOCPDataCollatorWithPadding
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.config import cp_shift_against_full_labels
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


def _ragged_batch(cp_size):
    generator = torch.Generator().manual_seed(cp_size)
    rows = []
    for index in range(6):
        prompt = torch.randint(1, 50, (int(torch.randint(1, 9, (), generator=generator)),), generator=generator)
        completion = torch.randint(1, 50, (int(torch.randint(0, 9, (), generator=generator)),), generator=generator)
        rows.append(
            {
                "prompt_input_ids": prompt.tolist(),
                "completion_input_ids": completion.tolist(),
                "group_id": index // 2,
                "group_size": 2,
                "advantage": 1.0,
            }
        )
    return OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=cp_size)(rows)


def _local_mask_worker(rank, cp_size):
    batch = _ragged_batch(cp_size)
    labels = batch["labels"]
    width = labels.size(1)
    chunk = width // cp_size
    grid = -0.25 - torch.arange(labels.numel()).reshape(labels.shape).float() * 0.01
    _, local_labels = cp_shift_against_full_labels(torch.zeros(labels.size(0), chunk, 1), labels, rank, cp_size)
    # Each shard's shifted targets padded to its chunk and gathered: the grid the sweep's mask must equal.
    local_valid = F.pad(local_labels != LABEL_IGNORE_INDEX, (0, chunk - local_labels.size(1)))
    shards = [torch.empty_like(local_valid) for _ in range(cp_size)]
    dist.all_gather(shards, local_valid)
    gathered = torch.cat(shards, dim=1)
    assert gathered.any(dim=1).sum() > 1, "fixture lost supervised rows"

    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = object()
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=rank, process_group=dist.group.WORLD)
    start = rank * chunk
    trainer._cp_chunked_logps = lambda model, ids, mask, labels: (
        grid[:, start : start + local_labels.size(1)],
        local_labels,
    )
    with patch("src.trainers.grpo.offline.dist.all_gather", wraps=dist.all_gather) as gathers:
        actual = trainer._cp_score_reference_batch(batch)
    assert gathers.call_count == 1, "only the log-probs need gathering; every rank holds the full labels"
    for got, scores, valid in zip(actual, grid, gathered, strict=True):
        assert torch.equal(got, scores[valid])


@pytest.mark.parametrize("cp_size", [2, 4])
def test_reference_sweep_mask_matches_the_gathered_shard_targets(cp_size):
    run_gloo_ranks(_local_mask_worker, cp_size, cp_size, pg_timeout=datetime.timedelta(seconds=30))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
