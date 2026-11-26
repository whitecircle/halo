#!/usr/bin/env python
"""The environmental GRPO step's padded batch masks, built without a host read per row. A per-row ``.any()`` on device
tensors would synchronize the host once per row per round."""

import types

import pytest
import torch

from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow


def _row(prompt: list[int], loss_mask: list[int]) -> TurnRow:
    return TurnRow(
        prompt_ids=torch.tensor(prompt),
        completion_ids=torch.arange(10, 10 + len(loss_mask)),
        loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
        sampling_logps=None,
        turn_routing=None,
    )


def test_step_masks_attend_trained_rows_whole_and_hide_masked_ones(monkeypatch):
    host = types.SimpleNamespace(
        pad_token_id=0, _is_correction=False, async_config=types.SimpleNamespace(score_centering_top_k=None)
    )
    rows = [_row([1, 2], [1, 0, 1]), _row([3], [0, 0]), _row([4, 5, 6], [0, 1, 0, 0, 1])]
    reads = []
    real_bool = torch.Tensor.__bool__

    def counting_bool(tensor):
        reads.append(tuple(tensor.shape))
        return real_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", counting_bool)
    step = DistributedAsyncEnvironmentalGRPOTrainer._stack_rows(host, rows, torch.device("cpu"))
    monkeypatch.undo()

    assert not reads, f"host reads while stacking {len(rows)} rows: {reads}"
    assert step.tool_mask.tolist() == [[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1]]
    assert step.completion_mask.tolist() == [[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert step.attention_mask.tolist()[1] == [0, 0, 1, 0, 0, 0, 0, 0], "a fully masked row attends its prompt alone"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
