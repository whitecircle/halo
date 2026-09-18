#!/usr/bin/env python
"""A native expert adapter's rank has to satisfy the grouped GEMM's stride contract, at construction.

``torch._grouped_mm`` contracts the expert delta over the adapter's rank dimension and reads that
stride as a multiple of 16 bytes; a rank below it (``lora_r: 4`` in bf16) fails at the first expert
matmul with ``strides should be multiple of 16 bytes``, after the model loaded and the run started.
The EP layer refuses the rank where the adapters are built, naming the multiple the dtype needs.

Run: pytest tests/cpu/peft/test_expert_lora_rank_alignment.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.kernels.grouped_mm_autograd import GROUPED_MM_STRIDE_ALIGNMENT_BYTES
from tests.common.ep_stubs import StubEPLayerBase

NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE = 2, 8, 16


class _StubEPLayer(StubEPLayerBase):
    """One adapted 3-D expert weight, at the dtype under test."""

    def __init__(self, spec: ExpertLoraSpec, dtype: torch.dtype):
        super().__init__()
        self.ep_config = SimpleNamespace(expert_lora=spec)
        self.gate_up_proj = nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype))

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_rank_below_the_stride_contract_is_refused_at_construction(dtype):
    spec = ExpertLoraSpec(r=GROUPED_MM_STRIDE_ALIGNMENT_BYTES // dtype.itemsize // 2, alpha=8)

    with pytest.raises(ValueError, match="multiple of"):
        _StubEPLayer(spec, dtype)._init_expert_lora()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_smallest_aligned_rank_builds_its_adapters(dtype):
    rank = GROUPED_MM_STRIDE_ALIGNMENT_BYTES // dtype.itemsize
    layer = _StubEPLayer(ExpertLoraSpec(r=rank, alpha=2 * rank), dtype)

    layer._init_expert_lora()

    assert layer.gate_up_proj_lora_A.shape == (NUM_LOCAL_EXPERTS, HIDDEN, rank)
    assert layer.gate_up_proj_lora_B.shape == (NUM_LOCAL_EXPERTS, rank, INTERMEDIATE)
    assert layer.gate_up_proj.requires_grad is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
