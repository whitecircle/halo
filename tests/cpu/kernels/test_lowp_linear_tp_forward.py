"""``LowPrecisionLinear`` under transformers' TP: compose with the instance-level ``forward``, or refuse. An instance
attribute shadows the class ``forward``, so a bare retype trains in bf16 while reporting the module converted."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers.distributed.tensor_parallel import apply_tensor_parallelism

from halo.distributed.mesh import MeshDim
from halo.distributed.tensor_parallel import local_linears
from halo.distributed.tensor_parallel.local_linears import install_planned_local_forwards
from halo.kernels.lowp.linear import LowPrecisionLinear, lowp_linear
from halo.kernels.lowp.quantization import QUANTIZERS, fake_quant
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from tests.common.distributed import fake_process_group, fake_process_group_mesh

WIDTH = 64  # A multiple of every block size (32 mx, 16 nvfp4).
# The two-way rowwise shard is 48: the 16-block format divides it, the 32-block ones do not.
MISALIGNED_INTERMEDIATE = 96
MISALIGNED_HIDDEN = 48  # The input colwise TP leaves whole, at a width only the 16-block format divides.
TP_PLAN = {
    "layers.*.gate_proj": "colwise",
    "layers.*.up_proj": "colwise",
    "layers.*.down_proj": "rowwise",
}


class _MLP(nn.Module):
    def __init__(self, intermediate: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _Model(nn.Module):
    tp_plan = TP_PLAN
    _tp_plan = TP_PLAN

    def __init__(self, intermediate: int = WIDTH, hidden: int = WIDTH):
        super().__init__()
        self.layers = nn.ModuleList([_MLP(intermediate, hidden)])


def _tp_model(tp_size: int, intermediate: int, hidden: int = WIDTH) -> _Model:
    """Called inside ``fake_process_group``, which sets the rank."""
    torch.manual_seed(0)
    # The conversion finds the TP mesh through its ``tp`` dim.
    mesh = init_device_mesh("cpu", (tp_size,), mesh_dim_names=(MeshDim.TP,))
    model = _Model(intermediate, hidden)
    apply_tensor_parallelism(model, mesh)
    install_planned_local_forwards(model, TP_PLAN)
    model._device_mesh = mesh
    return model


def _shadow_like_transformers(linear: nn.Linear) -> None:
    """What ``TensorParallelLayer.install_forward`` does."""
    original_forward = linear.forward

    def tp_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    linear.forward = tp_forward


def test_convert_refuses_a_shadowing_instance_forward():
    linear = nn.Linear(WIDTH, WIDTH, bias=False)
    _shadow_like_transformers(linear)
    x = torch.randn(4, WIDTH)

    with pytest.raises(TypeError, match="instance-level forward"):
        LowPrecisionLinear.convert_(linear, "mxfp8")
    assert type(linear) is nn.Linear, "a refused conversion must not half-retype the module"

    # Under the shadow a bare retype computes bf16.
    linear.__class__ = LowPrecisionLinear
    linear.quant_format = "mxfp8"
    assert torch.equal(linear(x), F.linear(x, linear.weight))
    assert not torch.equal(linear(x), lowp_linear(x, linear.weight, quant_format="mxfp8"))


def test_apply_refuses_a_shadow_it_cannot_reinstall():
    """Converting under it would report ``dense_linears`` while computing bf16."""
    model = _Model()
    model._tp_plan = {}
    _shadow_like_transformers(model.layers[0].down_proj)

    with pytest.raises(RuntimeError, match="instance-level forward"):
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=True)


@pytest.mark.parametrize("precision", sorted(LOWP_RECIPES))
def test_apply_reinstalls_the_tp_transforms_around_lowp_linear(precision):
    """In the training (DTensor) path and the no-grad local path alike."""
    with fake_process_group(0, 1):
        model = _tp_model(1, WIDTH)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(model.layers[0], proj)
            assert "forward" in vars(module), "premise: transformers installs an instance-level forward"
            assert isinstance(module.weight, DTensor), "premise: transformers shards into DTensor params"

        summary = apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=True)
        assert summary["dense_linears"] == 3

        x = torch.randn(4, WIDTH)
        quant_format = LOWP_RECIPES[precision].quant_format
        for proj in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(model.layers[0], proj)
            assert isinstance(module, LowPrecisionLinear)
            assert vars(module)["forward"].__module__ == local_linears.__name__, "not the local-shard forward"
            # ``to_local()`` is differentiable, so a plain clone stays attached and accumulates a second time. On a
            # 1-rank mesh the shard is the full weight.
            weight = module.weight.to_local().detach()
            expected = lowp_linear(x, weight, quant_format=quant_format)
            assert not torch.equal(expected, F.linear(x, weight)), "premise: fake-quant changes the numerics"

            out = module(x)
            assert not isinstance(out, DTensor), "the plan's transform must still unwrap the output"
            assert torch.equal(out, expected), f"{proj}: training path bypassed lowp_linear"
            with torch.no_grad():
                assert torch.equal(module(x), expected), f"{proj}: no-grad local path bypassed lowp_linear"

            module(x).sum().backward()
            reference = weight.clone().requires_grad_()
            lowp_linear(x, reference, quant_format=quant_format).sum().backward()
            assert isinstance(module.weight.grad, DTensor)
            assert torch.allclose(module.weight.grad.to_local(), reference.grad)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_apply_refuses_a_rowwise_shard_its_format_cannot_block(precision, rank):
    """96-wide at TP 2 is a 48-wide shard, which a 32-block format cannot split."""
    with fake_process_group(rank, 2):
        model = _tp_model(2, MISALIGNED_INTERMEDIATE)
        with pytest.raises(ValueError) as raised:
            apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=True)
    message = str(raised.value)
    for fragment in (
        "layers.0.down_proj",
        "over 48",
        "blocks of 32",
        "lower tensor_parallel_size",
        "['fp4']",
        "lowp_apply_dense_mlp",
    ):
        assert fragment in message, f"{fragment!r} missing from: {message}"


@pytest.mark.parametrize("rank", [0, 1])
def test_the_refusal_guards_a_rowwise_matmul_that_fails(rank):
    """The premise: past the load, that shard's low-precision matmul fails inside the quantizer."""
    with fake_process_group(rank, 2):
        weight = _tp_model(2, MISALIGNED_INTERMEDIATE).layers[0].down_proj.weight
        x = distribute_tensor(
            torch.randn(4, MISALIGNED_INTERMEDIATE), weight.device_mesh, [Shard(-1)], src_data_rank=None
        )
        with pytest.raises(ValueError, match="not divisible by block_size 32"):
            lowp_linear(x, weight, quant_format="mxfp8")


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    ("intermediate", "precision"), [(MISALIGNED_INTERMEDIATE, "fp4"), (2 * WIDTH, "fp8"), (2 * WIDTH, "mxfp4")]
)
def test_apply_converts_rowwise_shards_its_format_blocks(intermediate, precision, rank):
    """The fake group's all-reduce returns the shard unchanged."""
    with fake_process_group(rank, 2):
        model = _tp_model(2, intermediate)
        summary = apply_lowp_compute(model, LowpSpec(precision=precision), expert_weight_cacheable=True)
        assert summary["dense_linears"] == 3
        down_proj = model.layers[0].down_proj
        weight = down_proj.weight.to_local().detach()
        x = torch.randn(4, intermediate // 2)
        with torch.no_grad():
            out = down_proj(x)
        assert torch.equal(out, lowp_linear(x, weight, quant_format=LOWP_RECIPES[precision].quant_format))


def test_apply_refuses_a_whole_axis_without_the_tp_remedy():
    """A smaller ``tensor_parallel_size`` would not shard this input, so the error does not offer it."""
    model = _Model(MISALIGNED_INTERMEDIATE // 2)
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=True)
    message = str(raised.value)
    for fragment in ("layers.0.down_proj", "over 48", "['fp4']", "lowp_apply_dense_mlp"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "tensor_parallel_size" not in message, message


@pytest.mark.parametrize("rank", [0, 1])
def test_apply_refuses_a_colwise_input_without_the_tp_remedy(rank):
    """A smaller ``tensor_parallel_size`` leaves that axis whole, so the refusal does not offer it."""
    with fake_process_group(rank, 2):
        model = _tp_model(2, 2 * WIDTH, hidden=MISALIGNED_HIDDEN)
        assert isinstance(model.layers[0].gate_proj.weight, DTensor), "premise: the projection is TP-sharded"
        with pytest.raises(ValueError) as raised:
            apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=True)
    message = str(raised.value)
    for fragment in ("layers.0.gate_proj", "over 48", "['fp4']", "lowp_apply_dense_mlp"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "tensor_parallel_size" not in message, message


@pytest.mark.parametrize("quant_format", sorted(QUANTIZERS))
def test_lowp_linear_quantizes_dtensor_operands_shard_locally(quant_format):
    """Block boundaries tile the contraction shards; the fp4 formats' ``bucketize`` has no DTensor sharding rule."""
    torch.manual_seed(0)
    x_full = torch.randn(8, WIDTH)
    w_full = torch.randn(WIDTH, WIDTH)
    tp_size = 2
    for rank in range(tp_size):
        with fake_process_group_mesh(rank, tp_size) as mesh:
            x_shard = distribute_tensor(x_full, mesh, [Shard(-1)], src_data_rank=None)
            w_shard = distribute_tensor(w_full, mesh, [Shard(-1)], src_data_rank=None)
            partial = lowp_linear(x_shard, w_shard, quant_format=quant_format).to_local()
            expected = F.linear(
                fake_quant(x_full, quant_format, axis=-1).chunk(tp_size, dim=-1)[rank],
                fake_quant(w_full, quant_format, axis=-1).chunk(tp_size, dim=-1)[rank],
            )
            assert torch.equal(partial, expected), f"rowwise rank {rank}"

            x_rep = distribute_tensor(x_full, mesh, [Replicate()], src_data_rank=None)
            w_rows = distribute_tensor(w_full, mesh, [Shard(0)], src_data_rank=None)
            local = lowp_linear(x_rep, w_rows, quant_format=quant_format).to_local()
            expected = F.linear(
                fake_quant(x_full, quant_format, axis=-1),
                fake_quant(w_full, quant_format, axis=-1).chunk(tp_size, dim=0)[rank],
            )
            assert torch.equal(local, expected), f"colwise rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
