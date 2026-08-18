"""``LowPrecisionLinear`` under transformers' tensor parallelism: compose with the instance-level
``forward`` TP installs, or refuse.

transformers ≥ 5.16 shards a planned projection into DTensor params and replaces ``module.forward``
with an INSTANCE attribute (``tp_forward``) closed over the bound ``nn.Linear.forward`` it found.
``LowPrecisionLinear.convert_`` retypes the module in place, and an instance attribute shadows the
class ``forward``: the wrapper keeps calling the captured bf16 forward while
``apply_mixed_precision_compute`` reports the module converted — a run under ``lowp_precision: fp8``
that trains in bf16 with no other symptom.

The TP machinery in play is transformers' own (``apply_tensor_parallelism`` on a fake process group,
which slices shards locally and turns the collectives into no-ops), so what is pinned is the real
composition, not a stand-in.

    python tests/cpu/kernels/test_lowp_linear_tp_forward.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers.distributed.tensor_parallel import apply_tensor_parallelism

from src.distributed.mesh import MeshDim
from src.kernels.lowp.linear import PRECISION_TO_FORMAT, LinearPrecision, LowPrecisionLinear, lowp_linear
from src.kernels.lowp.mixed_precision import apply_mixed_precision_compute
from src.kernels.lowp.quantization import fake_quant
from tests.common.distributed import fake_process_group_mesh

WIDTH = 64  # a multiple of every block size (32 mx, 16 nvfp4)
TP_PLAN = {
    "layers.*.gate_proj": "colwise",
    "layers.*.up_proj": "colwise",
    "layers.*.down_proj": "rowwise",
}


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.up_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.down_proj = nn.Linear(WIDTH, WIDTH, bias=False)


class _Model(nn.Module):
    """The two plan attributes transformers' apply (``tp_plan``) and the toolkit (``_tp_plan``) read."""

    tp_plan = TP_PLAN
    _tp_plan = TP_PLAN

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MLP()])


def _shadow_like_transformers(linear: nn.Linear) -> None:
    """What ``TensorParallelLayer.install_forward`` does: an instance ``forward`` over the bound original."""
    original_forward = linear.forward

    def tp_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    linear.forward = tp_forward


def test_convert_refuses_a_shadowing_instance_forward():
    linear = nn.Linear(WIDTH, WIDTH, bias=False)
    _shadow_like_transformers(linear)
    x = torch.randn(4, WIDTH)

    with pytest.raises(TypeError, match="instance-level forward"):
        LowPrecisionLinear.convert_(linear, LinearPrecision.FP8)
    assert type(linear) is nn.Linear, "a refused conversion must not half-retype the module"

    # The mechanism the refusal exists for: a bare retype under the shadow computes bf16.
    linear.__class__ = LowPrecisionLinear
    linear.precision = LinearPrecision.FP8
    assert torch.equal(linear(x), F.linear(x, linear.weight))
    assert not torch.equal(linear(x), lowp_linear(x, linear.weight, precision=LinearPrecision.FP8))


def test_apply_refuses_a_shadow_it_cannot_reinstall():
    """An instance forward the plan does not explain has no style to re-install; converting under
    it would report ``dense_linears`` while computing bf16."""
    model = _Model()
    model._tp_plan = {}
    _shadow_like_transformers(model.layers[0].down_proj)

    with pytest.raises(RuntimeError, match="instance-level forward"):
        apply_mixed_precision_compute(model, precision="fp8")


@pytest.mark.parametrize("precision", [p.value for p in LinearPrecision])
def test_apply_reinstalls_the_tp_transforms_around_lowp_linear(precision):
    """After transformers' TP load, every converted projection must compute ``lowp_linear`` through
    the plan's own transforms — in the training (DTensor) path and the no-grad local path alike."""
    torch.manual_seed(0)
    with fake_process_group_mesh(0, 1):
        # Named like the loader's mesh: the conversion finds the TP mesh through its ``tp`` dim.
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=(MeshDim.TP,))
        model = _Model()
        apply_tensor_parallelism(model, mesh)
        model._device_mesh = mesh
        for proj in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(model.layers[0], proj)
            assert "forward" in vars(module), "premise: transformers installs an instance-level forward"
            assert isinstance(module.weight, DTensor), "premise: transformers shards into DTensor params"

        summary = apply_mixed_precision_compute(model, precision=precision)
        assert summary["dense_linears"] == 3

        x = torch.randn(4, WIDTH)
        lin_precision = LinearPrecision(precision)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(model.layers[0], proj)
            assert isinstance(module, LowPrecisionLinear)
            # Detached: ``to_local()`` is differentiable, so a plain clone of it would stay attached
            # to the module's graph — never a leaf (no ``.grad``) and accumulating a second time
            # into ``module.weight``. On a 1-rank mesh the local shard IS the full weight.
            weight = module.weight.to_local().detach()
            expected = lowp_linear(x, weight, precision=lin_precision)
            assert not torch.equal(expected, F.linear(x, weight)), "premise: fake-quant changes the numerics"

            out = module(x)
            assert not isinstance(out, DTensor), "the plan's transform must still unwrap the output"
            assert torch.equal(out, expected), f"{proj}: training path bypassed lowp_linear"
            with torch.no_grad():
                assert torch.equal(module(x), expected), f"{proj}: no-grad local path bypassed lowp_linear"

            # Straight-through gradient still reaches the DTensor master through the re-installed transforms.
            module(x).sum().backward()
            reference = weight.clone().requires_grad_()
            lowp_linear(x, reference, precision=lin_precision).sum().backward()
            assert isinstance(module.weight.grad, DTensor)
            assert torch.allclose(module.weight.grad.to_local(), reference.grad)


@pytest.mark.parametrize("precision", [p.value for p in LinearPrecision])
def test_lowp_linear_quantizes_dtensor_operands_shard_locally(precision):
    """Under TP the operands are DTensors; quantizing each rank's shard must equal the full-tensor
    quantization (block boundaries tile the contraction shards) — and must run at all for the fp4
    formats, whose ``bucketize`` DTensor dispatch has no sharding rule."""
    torch.manual_seed(0)
    fmt = PRECISION_TO_FORMAT[LinearPrecision(precision)]
    x_full = torch.randn(8, WIDTH)
    w_full = torch.randn(WIDTH, WIDTH)
    tp_size = 2
    for rank in range(tp_size):
        with fake_process_group_mesh(rank, tp_size) as mesh:
            # Rowwise: input and weight both sharded along the contraction axis; output is a partial sum.
            x_shard = distribute_tensor(x_full, mesh, [Shard(-1)], src_data_rank=None)
            w_shard = distribute_tensor(w_full, mesh, [Shard(-1)], src_data_rank=None)
            partial = lowp_linear(x_shard, w_shard, precision=LinearPrecision(precision)).to_local()
            expected = F.linear(
                fake_quant(x_full, fmt, axis=-1).chunk(tp_size, dim=-1)[rank],
                fake_quant(w_full, fmt, axis=-1).chunk(tp_size, dim=-1)[rank],
            )
            assert torch.equal(partial, expected), f"rowwise rank {rank}"

            # Colwise: replicated input, weight sharded along its output rows.
            x_rep = distribute_tensor(x_full, mesh, [Replicate()], src_data_rank=None)
            w_rows = distribute_tensor(w_full, mesh, [Shard(0)], src_data_rank=None)
            local = lowp_linear(x_rep, w_rows, precision=LinearPrecision(precision)).to_local()
            expected = F.linear(
                fake_quant(x_full, fmt, axis=-1),
                fake_quant(w_full, fmt, axis=-1).chunk(tp_size, dim=0)[rank],
            )
            assert torch.equal(local, expected), f"colwise rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
