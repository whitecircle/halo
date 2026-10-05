#!/usr/bin/env python
"""Fixed-batch checkpoint probes read stepped FSDP2 shards, not a cached eval forward."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers.modeling_outputs import CausalLMOutput

from tests.common.checkpoint_io import fixed_batch_loss


class _RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 1, bias=False)
        self.fail_forward = False

    def forward(self, input_ids, labels, use_cache=False):
        if self.fail_forward:
            raise ValueError("probe forward failed")
        return CausalLMOutput(loss=(self.projection(input_ids) - labels).square().mean())


@pytest.fixture
def single_rank_mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_probe_reads_stepped_shards_after_an_eval_forward(single_rank_mesh):
    torch.manual_seed(0)
    model = _RegressionModel()
    oracle = copy.deepcopy(model)
    fully_shard(model, mesh=single_rank_mesh, reshard_after_forward=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ids = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    labels = torch.zeros(1, 1)
    model(input_ids=ids, labels=labels).loss.backward()
    with torch.no_grad():
        before_step = model(input_ids=ids, labels=labels).loss.item()
    optimizer.step()
    for saved, live in zip(oracle.parameters(), optimizer.param_groups[0]["params"], strict=True):
        saved.data.copy_(live.full_tensor())

    expected = fixed_batch_loss(oracle, ids, labels)
    assert expected != pytest.approx(before_step), "the optimizer must change the probe loss"
    with torch.no_grad():
        stale = model(input_ids=ids, labels=labels).loss.item()
    assert stale == pytest.approx(before_step), "the eval-only forward must leave a stale gathered copy"
    assert fixed_batch_loss(model, ids, labels) == pytest.approx(expected)
    assert model.training
    assert all(isinstance(param, DTensor) for param in model.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_probe_restores_mode_when_forward_fails(training):
    model = _RegressionModel().train(training)
    model.fail_forward = True
    ids = torch.ones(1, 4)
    labels = torch.zeros(1, 1)

    with pytest.raises(ValueError, match="probe forward failed"):
        fixed_batch_loss(model, ids, labels)

    assert model.training is training


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
