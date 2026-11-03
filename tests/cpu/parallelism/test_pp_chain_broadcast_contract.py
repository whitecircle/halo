#!/usr/bin/env python
"""The chain scalar broadcast agrees with the buffer receivers allocate blind.
NCCL validates neither dtype nor shape, so a mismatch corrupts ``train/loss`` with no error."""

import pytest
import torch

import halo.distributed.pipeline_parallel.groups as groups
from halo.distributed.pipeline_parallel.groups import broadcast_scalar_from_last_stage

LAST_STAGE_RANK = 1


@pytest.fixture
def sent(monkeypatch):
    """CPU buffers, so no CUDA context is created."""
    buffers: list[torch.Tensor] = []
    monkeypatch.setattr(groups, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(groups.dist, "broadcast", lambda tensor, src, group: buffers.append(tensor))
    return buffers


@pytest.mark.parametrize("loss_dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_source_buffer_matches_the_receiver_buffer(sent, loss_dtype):
    """Receivers always allocate fp32, so the source is converted."""
    received = broadcast_scalar_from_last_stage(None, LAST_STAGE_RANK, group=None)
    source = broadcast_scalar_from_last_stage(torch.tensor(1.25, dtype=loss_dtype), LAST_STAGE_RANK, group=None)

    assert [buffer.dtype for buffer in sent] == [received.dtype] * 2, "source and receiver dtypes must match"
    assert [buffer.shape for buffer in sent] == [received.shape] * 2, "source and receiver shapes must match"
    assert float(source) == 1.25


def test_a_non_scalar_source_is_refused(sent):
    """The receivers' shape is (), so more elements would be truncated silently."""
    with pytest.raises(RuntimeError):
        broadcast_scalar_from_last_stage(torch.zeros(2), LAST_STAGE_RANK, group=None)


def test_the_source_is_detached(sent):
    """Its graph would keep the whole stage's activations alive on every rank."""
    loss = (torch.tensor(2.0, requires_grad=True) * 3).sum()
    assert broadcast_scalar_from_last_stage(loss, LAST_STAGE_RANK, group=None).requires_grad is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
