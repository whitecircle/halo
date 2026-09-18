#!/usr/bin/env python
"""A native expert adapter's rank has to satisfy the grouped GEMM's stride contract, at construction.

``torch._grouped_mm`` contracts the expert delta over the adapter's rank dimension and reads that
stride as a multiple of 16 bytes; a rank off it (``lora_r: 4`` in bf16) fails at the first expert
matmul with ``strides should be multiple of 16 bytes``, after the model loaded and the run started.
The adapter GEMMs run at the activation dtype, bf16 at the narrowest, so the multiple is 8 whatever
the experts are stored in; the EP layer refuses any other rank where the adapters are built.

Run: pytest tests/cpu/peft/test_expert_lora_rank_alignment.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EXPERT_LORA_RANK_MULTIPLE
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


def test_the_multiple_is_the_stride_at_bf16():
    assert EXPERT_LORA_RANK_MULTIPLE == GROUPED_MM_STRIDE_ALIGNMENT_BYTES // torch.bfloat16.itemsize == 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_rank_off_the_stride_contract_is_refused_at_construction(dtype):
    """fp32 experts too: at the bf16 the adapter GEMM runs in, 4 × 2 bytes is half a stride."""
    spec = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE // 2, alpha=8)

    with pytest.raises(ValueError, match=f"multiple of {EXPERT_LORA_RANK_MULTIPLE}"):
        _StubEPLayer(spec, dtype)._init_expert_lora()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_smallest_aligned_rank_builds_its_adapters(dtype):
    rank = EXPERT_LORA_RANK_MULTIPLE
    layer = _StubEPLayer(ExpertLoraSpec(r=rank, alpha=2 * rank), dtype)

    layer._init_expert_lora()

    assert layer.gate_up_proj_lora_A.shape == (NUM_LOCAL_EXPERTS, HIDDEN, rank)
    assert layer.gate_up_proj_lora_B.shape == (NUM_LOCAL_EXPERTS, rank, INTERMEDIATE)
    assert layer.gate_up_proj.requires_grad is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
