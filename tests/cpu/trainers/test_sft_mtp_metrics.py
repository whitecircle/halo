#!/usr/bin/env python
"""``ntp_ce``, ``mtp_ce`` and a held-out acceptance through the trainer's once-per-log fold: per target across the world,
so a rank with more targets weighs more, and absent on a run with no head."""

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.mtp.acceptance import MtpAcceptance
from halo.models.mtp.head import MtpLossSplit
from halo.trainers.sft import _record_loss_metrics
from halo.trainers.world_metrics import WorldMetrics
from tests.common.world_metrics import fake_world

_LABELS = torch.tensor([[LABEL_IGNORE_INDEX, 1, 2, 3]])
_INPUTS = {"input_ids": torch.zeros_like(_LABELS), "attention_mask": torch.ones_like(_LABELS)}


def _output(split: MtpLossSplit | None) -> CausalLMOutputWithPast:
    output = CausalLMOutputWithPast(logits=None)
    if split is not None:
        output.mtp_loss_split = split
    return output


def _split(next_sum: float, next_targets: int, head_sum: float, head_targets: int) -> MtpLossSplit:
    return MtpLossSplit(
        torch.tensor(next_sum), torch.tensor(next_targets), torch.tensor(head_sum), torch.tensor(head_targets)
    )


def _flush(*outputs) -> dict[str, float]:
    ranks = [WorldMetrics() for _ in outputs]
    for metrics, output in zip(ranks, outputs, strict=True):
        _record_loss_metrics(metrics, output, _INPUTS, _LABELS, None, training=True)
    with fake_world(*ranks):
        return ranks[0].flush()


def test_each_cross_entropy_pools_over_its_own_targets_across_the_ranks():
    world = _flush(_output(_split(30.0, 3, 8.0, 2)), _output(_split(10.0, 1, 12.0, 6)))
    assert world["ntp_ce"] == pytest.approx(40.0 / 4)
    assert world["mtp_ce"] == pytest.approx(20.0 / 8)


def test_a_rank_with_no_head_target_adds_nothing_to_the_mean():
    world = _flush(_output(_split(10.0, 2, 0.0, 0)), _output(_split(10.0, 2, 9.0, 3)))
    assert world["mtp_ce"] == pytest.approx(3.0)


def test_a_run_with_no_head_records_neither():
    world = _flush(_output(None))
    assert not {"ntp_ce", "mtp_ce"} & set(world)


def _scored(overlap: float, greedy: float, top1: int, positions: int) -> CausalLMOutputWithPast:
    output = _output(None)
    output.mtp_acceptance = MtpAcceptance(
        torch.tensor(overlap), torch.tensor(greedy), torch.tensor(top1), torch.tensor(positions)
    )
    return output


def test_each_acceptance_pools_over_the_positions_of_the_world():
    world = _flush(_scored(3.0, 2.0, 1, 4), _scored(1.0, 4.0, 3, 6))
    assert world["mtp_acceptance_tv"] == pytest.approx(4.0 / 10)
    assert world["mtp_acceptance_greedy"] == pytest.approx(6.0 / 10)
    assert world["mtp_top1"] == pytest.approx(4 / 10)
    assert world["mtp_scored_positions"] == 10


def test_a_run_with_no_head_records_no_acceptance():
    assert not {key for key in _flush(_output(None)) if "mtp_" in key}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
