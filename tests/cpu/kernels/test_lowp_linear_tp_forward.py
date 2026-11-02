"""``LowPrecisionLinear`` under transformers' tensor parallelism: compose with the instance-level
``forward`` TP installs, or refuse.

transformers ≥ 5.16 shards a planned projection into DTensor params and replaces ``module.forward``
with an INSTANCE attribute (``tp_forward``) closed over the bound ``nn.Linear.forward`` it found, and the
toolkit's load swaps in its local-shard forward the same way (``install_planned_local_forwards``).
``LowPrecisionLinear.convert_`` retypes the module in place, and an instance attribute shadows the
class ``forward``: the wrapper keeps computing bf16 while
``apply_lowp_compute`` reports the module converted — a run under ``lowp_precision: fp8``
that trains in bf16 with no other symptom. The fake-quant blocks each rank's shard, so a rowwise
shard the format cannot split into blocks is refused at load on every rank.

The TP machinery in play is the toolkit's load (``apply_tensor_parallelism`` then
``install_planned_local_forwards``, on a fake process group, which slices shards locally and turns the
collectives into no-ops), so what is pinned is the real composition, not a stand-in.

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

from halo.distributed.mesh import MeshDim
from halo.distributed.tensor_parallel import local_linears
from halo.distributed.tensor_parallel.local_linears import install_planned_local_forwards
from halo.kernels.lowp.linear import LowPrecisionLinear, lowp_linear
from halo.kernels.lowp.quantization import QUANTIZERS, fake_quant
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from tests.common.distributed import fake_process_group, fake_process_group_mesh

WIDTH = 64  # a multiple of every block size (32 mx, 16 nvfp4)
# An intermediate whose two-way rowwise shard is 48: the 16-block format divides it, the 32-block ones do not.
MISALIGNED_INTERMEDIATE = 96
MISALIGNED_HIDDEN = 48  # the input colwise TP leaves whole, at a width only the 16-block format divides
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
    """The two plan attributes transformers' apply (``tp_plan``) and the toolkit (``_tp_plan``) read."""

    tp_plan = TP_PLAN
    _tp_plan = TP_PLAN

    def __init__(self, intermediate: int = WIDTH, hidden: int = WIDTH):
        super().__init__()
        self.layers = nn.ModuleList([_MLP(intermediate, hidden)])


def _tp_model(tp_size: int, intermediate: int, hidden: int = WIDTH) -> _Model:
    """This rank's shards of a ``_Model`` after the toolkit's TP load on a ``tp_size``-way mesh (transformers'
    shards, the local-shard forward installed over them); called inside ``fake_process_group``, which sets the
    rank."""
    torch.manual_seed(0)
    # Named like the loader's mesh: the conversion finds the TP mesh through its ``tp`` dim.
    mesh = init_device_mesh("cpu", (tp_size,), mesh_dim_names=(MeshDim.TP,))
    model = _Model(intermediate, hidden)
    apply_tensor_parallelism(model, mesh)
    install_planned_local_forwards(model, TP_PLAN)
    model._device_mesh = mesh
    return model


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
        LowPrecisionLinear.convert_(linear, "mxfp8")
    assert type(linear) is nn.Linear, "a refused conversion must not half-retype the module"

    # The mechanism the refusal exists for: a bare retype under the shadow computes bf16.
    linear.__class__ = LowPrecisionLinear
    linear.quant_format = "mxfp8"
    assert torch.equal(linear(x), F.linear(x, linear.weight))
    assert not torch.equal(linear(x), lowp_linear(x, linear.weight, quant_format="mxfp8"))


def test_apply_refuses_a_shadow_it_cannot_reinstall():
    """An instance forward the plan does not explain has no style to re-install; converting under
    it would report ``dense_linears`` while computing bf16."""
    model = _Model()
    model._tp_plan = {}
    _shadow_like_transformers(model.layers[0].down_proj)

    with pytest.raises(RuntimeError, match="instance-level forward"):
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=True)


@pytest.mark.parametrize("precision", sorted(LOWP_RECIPES))
def test_apply_reinstalls_the_tp_transforms_around_lowp_linear(precision):
    """After transformers' TP load, every converted projection must compute ``lowp_linear`` through
    the plan's own transforms — in the training (DTensor) path and the no-grad local path alike."""
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
            # Detached: ``to_local()`` is differentiable, so a plain clone of it would stay attached
            # to the module's graph — never a leaf (no ``.grad``) and accumulating a second time
            # into ``module.weight``. On a 1-rank mesh the local shard IS the full weight.
            weight = module.weight.to_local().detach()
            expected = lowp_linear(x, weight, quant_format=quant_format)
            assert not torch.equal(expected, F.linear(x, weight)), "premise: fake-quant changes the numerics"

            out = module(x)
            assert not isinstance(out, DTensor), "the plan's transform must still unwrap the output"
            assert torch.equal(out, expected), f"{proj}: training path bypassed lowp_linear"
            with torch.no_grad():
                assert torch.equal(module(x), expected), f"{proj}: no-grad local path bypassed lowp_linear"

            # Straight-through gradient still reaches the DTensor master through the re-installed transforms.
            module(x).sum().backward()
            reference = weight.clone().requires_grad_()
            lowp_linear(x, reference, quant_format=quant_format).sum().backward()
            assert isinstance(module.weight.grad, DTensor)
            assert torch.allclose(module.weight.grad.to_local(), reference.grad)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_apply_refuses_a_rowwise_shard_its_format_cannot_block(precision, rank):
    """``down_proj`` rowwise at TP 2 over a 96-wide intermediate contracts over a 48-wide shard on each rank,
    which a 32-block format cannot split: every rank refuses at load, naming the projection, the shard, the
    block, the TP remedy, the precision whose block divides it and the opt-out."""
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
    """Shards a multiple of the block: every projection converts, and the rowwise one computes ``lowp_linear`` on
    this rank's shard (the fake group's all-reduce returns it unchanged)."""
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
    """No TP: a 48-wide ``down_proj`` input is refused at load, and the error does not offer a smaller
    ``tensor_parallel_size``, which would not shard it."""
    model = _Model(MISALIGNED_INTERMEDIATE // 2)
    with pytest.raises(ValueError) as raised:
        apply_lowp_compute(model, LowpSpec(precision="fp8"), expert_weight_cacheable=True)
    message = str(raised.value)
    for fragment in ("layers.0.down_proj", "over 48", "['fp4']", "lowp_apply_dense_mlp"):
        assert fragment in message, f"{fragment!r} missing from: {message}"
    assert "tensor_parallel_size" not in message, message


@pytest.mark.parametrize("rank", [0, 1])
def test_apply_refuses_a_colwise_input_without_the_tp_remedy(rank):
    """TP 2 over a 48-wide hidden size: colwise ``gate_proj`` contracts over the whole 48 on every rank, so the
    refusal does not offer a smaller ``tensor_parallel_size``, which leaves that axis whole."""
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
    """Under TP the operands are DTensors; quantizing each rank's shard must equal the full-tensor
    quantization (block boundaries tile the contraction shards) — and must run at all for the fp4
    formats, whose ``bucketize`` DTensor dispatch has no sharding rule."""
    torch.manual_seed(0)
    x_full = torch.randn(8, WIDTH)
    w_full = torch.randn(WIDTH, WIDTH)
    tp_size = 2
    for rank in range(tp_size):
        with fake_process_group_mesh(rank, tp_size) as mesh:
            # Rowwise: input and weight both sharded along the contraction axis; output is a partial sum.
            x_shard = distribute_tensor(x_full, mesh, [Shard(-1)], src_data_rank=None)
            w_shard = distribute_tensor(w_full, mesh, [Shard(-1)], src_data_rank=None)
            partial = lowp_linear(x_shard, w_shard, quant_format=quant_format).to_local()
            expected = F.linear(
                fake_quant(x_full, quant_format, axis=-1).chunk(tp_size, dim=-1)[rank],
                fake_quant(w_full, quant_format, axis=-1).chunk(tp_size, dim=-1)[rank],
            )
            assert torch.equal(partial, expected), f"rowwise rank {rank}"

            # Colwise: replicated input, weight sharded along its output rows.
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
