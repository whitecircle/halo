"""SFT's token accuracy under CP's fused head loss, which returns each rank's chunk accuracy in place of logits: the
world value must pool every rank's correct targets over every rank's supervised targets, so each chunk weighs in by
its own target count, not by the whole row's.

Run: python tests/cpu/trainers/test_sft_cp_token_accuracy.py
"""

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.sft import _record_loss_metrics
from halo.trainers.world_metrics import WorldMetrics
from tests.common.world_metrics import fake_world

CP_SIZE = 2
# Rank 0's chunk (positions 0-3) supervises only its boundary target, the label at position 4; rank 1's supervises the
# labels at positions 5-7.
LABELS = torch.tensor([[LABEL_IGNORE_INDEX] * 4 + [5, 6, 7, 8]])
# Rank 0 predicts its one target, rank 1 none of its three: one correct target of four.
CHUNK_ACCURACY = (1.0, 0.0)


def test_the_world_accuracy_pools_each_chunks_supervised_targets():
    ranks = [WorldMetrics() for _ in range(CP_SIZE)]
    inputs = {"input_ids": torch.zeros_like(LABELS), "attention_mask": torch.ones_like(LABELS)}
    for rank, (metrics, accuracy) in enumerate(zip(ranks, CHUNK_ACCURACY, strict=True)):
        outputs = CausalLMOutputWithPast(logits=None)
        outputs["token_accuracy"] = torch.tensor(accuracy)
        _record_loss_metrics(metrics, outputs, inputs, LABELS, (rank, CP_SIZE), training=True)
    with fake_world(*ranks):
        world = ranks[0].flush()
    assert world["mean_token_accuracy"] == pytest.approx(1 / 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
