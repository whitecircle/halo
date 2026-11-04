#!/usr/bin/env python
"""The FSDP2 mixed-precision policy computes in the run's dtype and reduces in fp32 on request."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import halo.distributed.fsdp as fsdp
from halo.distributed.fsdp import mixed_precision_policy
from halo.distributed.parallelism_config import ParallelismConfig


@pytest.mark.parametrize(
    ("compute_dtype", "fp32_reduce", "reduce_dtype"),
    [
        (torch.bfloat16, False, torch.bfloat16),
        (torch.bfloat16, True, torch.float32),
        (torch.float16, False, torch.float16),
        (torch.float16, True, torch.float32),
        (torch.float32, False, torch.float32),
        (torch.float32, True, torch.float32),
    ],
)
def test_policy_dtypes_follow_the_precision_flags(compute_dtype, fp32_reduce, reduce_dtype):
    policy = mixed_precision_policy(compute_dtype, fp32_reduce=fp32_reduce, cast_forward_inputs=False)
    assert (policy.param_dtype, policy.reduce_dtype) == (compute_dtype, reduce_dtype)
    assert policy.cast_forward_inputs is False


@pytest.mark.parametrize(
    ("fp32_reduce", "reduce_dtype"),
    [(False, torch.bfloat16), (True, torch.float32)],
)
def test_the_wrap_reduces_in_the_dtype_it_is_handed(fp32_reduce, reduce_dtype):
    policies = []

    def record(_model, _mesh, policy, *_):
        policies.append(policy)
        return 1

    with (
        patch.object(fsdp, "apply_fsdp2_per_layer", side_effect=record),
        patch.object(fsdp, "is_global_main_process", return_value=False),
    ):
        fsdp._apply_fsdp2(
            nn.Linear(2, 2),
            object(),
            compute_dtype=torch.bfloat16,
            fp32_reduce=fp32_reduce,
            topology="1D full-shard",
        )
    assert [(p.param_dtype, p.reduce_dtype) for p in policies] == [(torch.bfloat16, reduce_dtype)]


@pytest.mark.parametrize(
    ("fp32_grad_reduce", "fp32_non_ep_params", "reduces_in_fp32"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_fsdp_reduces_in_fp32_under_either_knob(fp32_grad_reduce, fp32_non_ep_params, reduces_in_fp32):
    config = ParallelismConfig(
        world_size=2,
        gpus_per_node=2,
        fp32_grad_reduce=fp32_grad_reduce,
        fp32_non_ep_params=fp32_non_ep_params,
        fsdp_shard_ep1_experts=False,
    )
    assert config.fsdp_reduces_in_fp32 is reduces_in_fp32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
