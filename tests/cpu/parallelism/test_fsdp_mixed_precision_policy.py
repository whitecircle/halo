#!/usr/bin/env python
"""The FSDP2 mixed-precision policy follows the run's precision flags through one dtype resolver.

``create_mixed_precision_policy_v2`` computes in the dtype ``resolve_training_dtype`` reads off the
training args (``bf16`` wins over ``fp16``, neither means fp32) and reduces in fp32 whenever fp32
master weights or ``fp32_grad_reduce`` ask for it. The full table is pinned here.

    python tests/cpu/parallelism/test_fsdp_mixed_precision_policy.py
"""

from types import SimpleNamespace

import pytest
import torch

from src.distributed.fsdp import create_mixed_precision_policy_v2


@pytest.mark.parametrize(
    ("bf16", "fp16", "fp32_grad_reduce", "fp32_master_weights", "param_dtype", "reduce_dtype"),
    [
        (True, False, False, False, torch.bfloat16, torch.bfloat16),
        (True, False, True, False, torch.bfloat16, torch.float32),
        (True, False, False, True, torch.bfloat16, torch.float32),
        (False, True, False, False, torch.float16, torch.float16),
        (False, True, True, False, torch.float16, torch.float32),
        (False, False, False, False, torch.float32, torch.float32),
        (False, False, True, True, torch.float32, torch.float32),
    ],
)
def test_policy_dtypes_follow_the_precision_flags(
    bf16, fp16, fp32_grad_reduce, fp32_master_weights, param_dtype, reduce_dtype
):
    args = SimpleNamespace(bf16=bf16, fp16=fp16, fp32_grad_reduce=fp32_grad_reduce)
    policy = create_mixed_precision_policy_v2(args, fp32_master_weights=fp32_master_weights, cast_forward_inputs=False)
    assert (policy.param_dtype, policy.reduce_dtype) == (param_dtype, reduce_dtype)
    assert policy.cast_forward_inputs is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
