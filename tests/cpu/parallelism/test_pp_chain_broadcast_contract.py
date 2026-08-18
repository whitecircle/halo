#!/usr/bin/env python
"""CPU test: the chain scalar broadcast agrees with the buffer its receivers allocate.

Only the last stage computes a loss; every other stage receives it into a buffer it allocates blind,
before any byte arrives. NCCL validates neither dtype nor shape across ranks, so a source that sends
anything else corrupts ``train/loss`` on every stage that did not compute it — and feeds the
nan/inf filter garbage — without an error anywhere.

    python tests/cpu/parallelism/test_pp_chain_broadcast_contract.py
"""

import pytest
import torch

import src.distributed.pipeline_parallel.groups as groups
from src.distributed.pipeline_parallel.groups import broadcast_scalar_from_last_stage

LAST_STAGE_RANK = 1


@pytest.fixture
def sent(monkeypatch):
    """Capture what each rank hands to the collective; CPU buffers so no CUDA context is created."""
    buffers: list[torch.Tensor] = []
    monkeypatch.setattr(groups, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(groups.dist, "broadcast", lambda tensor, src, group: buffers.append(tensor))
    return buffers


@pytest.mark.parametrize("loss_dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_source_buffer_matches_the_receiver_buffer(sent, loss_dtype):
    """A stage's ``token_loss_fn`` returns whatever dtype its adapter computes in; the receivers
    always allocate fp32, so the source must be converted here."""
    received = broadcast_scalar_from_last_stage(None, LAST_STAGE_RANK, group=None)
    source = broadcast_scalar_from_last_stage(torch.tensor(1.25, dtype=loss_dtype), LAST_STAGE_RANK, group=None)

    assert [buffer.dtype for buffer in sent] == [received.dtype] * 2, "source and receiver dtypes must match"
    assert [buffer.shape for buffer in sent] == [received.shape] * 2, "source and receiver shapes must match"
    assert float(source) == 1.25


def test_a_non_scalar_source_is_refused(sent):
    """The receivers' shape is hard-coded to (), so a source with more than one element would
    otherwise be truncated into it silently."""
    with pytest.raises(RuntimeError):
        broadcast_scalar_from_last_stage(torch.zeros(2), LAST_STAGE_RANK, group=None)


def test_the_source_is_detached(sent):
    """The loss is broadcast after backward; carrying its graph into the chain would keep the whole
    stage's activations alive on every rank."""
    loss = (torch.tensor(2.0, requires_grad=True) * 3).sum()
    assert broadcast_scalar_from_last_stage(loss, LAST_STAGE_RANK, group=None).requires_grad is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
