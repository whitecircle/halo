#!/usr/bin/env python
"""``build_tensor_type_grouped_optimizer`` splits (weight decay, dtype, tensor type) groups in
single-tensor mode — FSDP2 DTensors and FSDP-ignored plain EP tensors cannot share a foreach/fused
update, so the builder must partition them and disable both paths.

    python tests/cpu/optimizers/test_tensor_type_grouping.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from src.optimizers.param_groups import build_tensor_type_grouped_optimizer


def test_groups_split_by_tensor_type_and_dtype(tmp_path):
    # file:// rendezvous — a fixed MASTER_PORT collides with any concurrently running suite.
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = nn.Module()
        model.register_parameter("w_dtensor", nn.Parameter(distribute_tensor(torch.randn(4, 4), mesh, [Replicate()])))
        model.register_parameter("b_plain_bf16", nn.Parameter(torch.randn(4, dtype=torch.bfloat16)))
        model.register_parameter("w_plain_fp32", nn.Parameter(torch.randn(4, 4)))
        assert isinstance(model.w_dtensor, DTensor)

        args = SimpleNamespace(weight_decay=0.1)
        optimizer = build_tensor_type_grouped_optimizer(
            model,
            args,
            ["w_dtensor", "w_plain_fp32"],
            lambda a, m: (torch.optim.SGD, {"lr": 0.1, "foreach": True, "fused": False}),
        )

        groups = optimizer.param_groups
        # (wd=0.1, fp32, DTensor), (wd=0.0, bf16, plain), (wd=0.1, fp32, plain) — never merged.
        assert len(groups) == 3
        for group in groups:
            assert group["foreach"] is False
            assert group["fused"] is False
            kinds = {isinstance(p, DTensor) for p in group["params"]}
            assert len(kinds) == 1, "a group mixes DTensor and plain tensors"
        assert sorted(g["weight_decay"] for g in groups) == [0.0, 0.1, 0.1]
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
