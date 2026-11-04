"""TP linears computed on their local shards: ``F.linear`` on ``weight.to_local()`` between explicit c10d collectives,
in place of DTensor dispatch (about ten dispatches per linear and a copy inside every functional all-reduce). The
parameters keep their DTensor placements."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor
from transformers.distributed.tensor_parallel import _get_parameter_plan

from halo.distributed.grad_reduce import SumAcrossGroup, SumGradAcrossGroup
from halo.kernels.lowp.linear import LowPrecisionLinear, lowp_linear


@dataclass(frozen=True)
class LinearTPStyle:
    """Where a TP linear splits its weight. Colwise shards output features (weight and bias ``Shard(0)``) with a
    replicated input; rowwise shards input features (weight ``Shard(1)``, bias replicated and added once after the
    SUM). ``full_width`` gathers a colwise output, or splits a replicated rowwise input here."""

    colwise: bool
    full_width: bool

    @property
    def placements(self) -> dict[str, Placement]:
        return {"weight": Shard(0), "bias": Shard(0)} if self.colwise else {"weight": Shard(1), "bias": Replicate()}


# Keyed by transformers' plan style names, which the selective attention plan uses too.
LINEAR_TP_STYLES: dict[str, LinearTPStyle] = {
    "colwise": LinearTPStyle(colwise=True, full_width=False),
    "colwise_gather_output": LinearTPStyle(colwise=True, full_width=True),
    "rowwise": LinearTPStyle(colwise=False, full_width=False),
    "rowwise_split_input": LinearTPStyle(colwise=False, full_width=True),
}


def _gather_last_dim(x: torch.Tensor, group, world: int) -> torch.Tensor:
    gathered = x.new_empty((world * x.shape[0], *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
    return torch.cat(gathered.chunk(world, dim=0), dim=-1)


def _last_dim_shard(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    # Contiguous, as DTensor's Replicate -> Shard(-1) hands it on: the matmul consuming it sees the same strides.
    return x.chunk(world, dim=-1)[rank].contiguous()


class _GatherLastDim(torch.autograd.Function):
    """All-gather of a colwise output's last dim forward; this rank's slice of the gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, rank: int, world: int):
        ctx.rank, ctx.world = rank, world
        return _gather_last_dim(x, group, world)

    @staticmethod
    def backward(ctx, grad):
        return _last_dim_shard(grad, ctx.rank, ctx.world), None, None, None


class _SplitLastDim(torch.autograd.Function):
    """This rank's slice of a replicated input's last dim forward; all-gather of the gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, rank: int, world: int):
        ctx.group, ctx.world = group, world
        return _last_dim_shard(x, rank, world)

    @staticmethod
    def backward(ctx, grad):
        return _gather_last_dim(grad, ctx.group, ctx.world), None, None, None


def _tp_shard(param: DTensor, name: str) -> torch.Tensor:
    """This rank's TP shard of ``param``; raises on FSDP2's (dp, tp) shard, which a call bypassing the FSDP unit's
    forward leaves registered and whose slice would yield part of the output with no error."""
    if param.device_mesh.ndim != 1:
        raise RuntimeError(
            f"{name} ran with its parameters FSDP2-sharded over {param.device_mesh.mesh_dim_names}: call it "
            f"through the FSDP-wrapped module, whose forward unshards them onto the TP mesh."
        )
    return param.to_local()


def _colwise_forward(name: str, matmul, group, rank: int, world: int, gather: bool):
    def forward(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        bias = _tp_shard(linear.bias, name) if linear.bias is not None else None
        out = matmul(SumGradAcrossGroup.apply(x, group), _tp_shard(linear.weight, name), bias)
        return _GatherLastDim.apply(out, group, rank, world) if gather else out

    return forward


def _rowwise_forward(name: str, matmul, group, rank: int, world: int, split: bool):
    def forward(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if split:
            x = _SplitLastDim.apply(x, group, rank, world)
        out = SumAcrossGroup.apply(matmul(x, _tp_shard(linear.weight, name), None), group)
        # In the matmul's dtype, as a biased F.linear returns it under autocast.
        return out if linear.bias is None else out + _tp_shard(linear.bias, name).to(out.dtype)

    return forward


def _local_matmul(linear: nn.Linear, name: str):
    """``linear``'s own ``x @ weight.T + bias`` over local operands: the fake-quant one for a low-precision
    projection, which quantizes each rank's shard as it quantized the DTensor's."""
    if isinstance(linear, LowPrecisionLinear):
        return partial(lowp_linear, quant_format=linear.quant_format)
    if type(linear).forward is not nn.Linear.forward:
        raise TypeError(
            f"{name} is a {type(linear).__name__} with a forward of its own; the local-shard TP forward "
            f"computes F.linear and would bypass it."
        )
    return F.linear


def _require_tp_mesh(linear: nn.Linear, name: str, style_name: str) -> DeviceMesh:
    """The 1-D TP mesh ``linear``'s parameters live on; raises unless each holds ``style_name``'s placement."""
    expected = LINEAR_TP_STYLES[style_name].placements
    meshes = set()
    for pname, param in linear.named_parameters(recurse=False):
        if not isinstance(param, DTensor) or param.placements != (expected[pname],):
            layout = param.placements if isinstance(param, DTensor) else "a plain tensor"
            raise ValueError(
                f"{name}.{pname} under TP style {style_name!r} is {layout}, not ({expected[pname]},): the "
                f"local-shard forward would compute on a layout it was not written for."
            )
        meshes.add(param.device_mesh)
    if len(meshes) != 1 or next(iter(meshes)).ndim != 1:
        raise ValueError(
            f"{name}: the TP linear's parameters must share one 1-D TP mesh, not {sorted(map(str, meshes))}"
        )
    return meshes.pop()


def install_local_forward(linear: nn.Linear, style_name: str, *, name: str = "linear") -> None:
    """Compute ``linear`` on its local shards under ``style_name``; raises off that style's placements."""
    matmul = _local_matmul(linear, name)
    style = LINEAR_TP_STYLES[style_name]
    mesh = _require_tp_mesh(linear, name, style_name)
    group, rank, world = mesh.get_group(), mesh.get_local_rank(), mesh.size()
    if style.full_width:
        full = linear.out_features if style.colwise else linear.in_features
        if full % world:
            raise ValueError(
                f"{name}: TP style {style_name!r} keeps a {full}-wide activation full width, which "
                f"tensor_parallel_size={world} does not divide."
            )
    build = _colwise_forward if style.colwise else _rowwise_forward
    # Bound, not closed over ``linear``: a deepcopy rebinds a method, so the copy computes with its own parameters.
    linear.forward = MethodType(build(name, matmul, group, rank, world, style.full_width), linear)


def shard_linear(linear: nn.Linear, style_name: str, tp_mesh: DeviceMesh, *, name: str = "linear") -> None:
    """COLLECTIVE over ``tp_mesh``: distribute ``linear``'s parameters into ``style_name``'s placements, rank 0's
    values as torch's own TP styles scatter them, then install the local-shard forward."""
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{name} is a {type(linear).__name__}; TP shards only nn.Linear projections")
    placements = LINEAR_TP_STYLES[style_name].placements
    for pname, param in list(linear.named_parameters(recurse=False)):
        linear.register_parameter(
            pname,
            nn.Parameter(distribute_tensor(param, tp_mesh, [placements[pname]]), requires_grad=param.requires_grad),
        )
    install_local_forward(linear, style_name, name=name)


def runs_local_forward(module: nn.Module) -> bool:
    forward = vars(module).get("forward")
    return isinstance(forward, MethodType) and forward.__func__.__module__ == __name__


def install_planned_local_forwards(model: nn.Module, applied_plan: dict[str, str]) -> int:
    """Install the local-shard forward on every plain ``nn.Linear`` an HF-native load sharded under
    :data:`LINEAR_TP_STYLES`; returns how many. Embeddings, MoE and activation styles, and Linear subclasses with their
    own forward keep transformers' forward."""
    installed = 0
    for name, module in model.named_modules():
        style_name = _get_parameter_plan(name, applied_plan, is_weight=False)
        if (
            style_name in LINEAR_TP_STYLES
            and isinstance(module, nn.Linear)
            and type(module).forward is nn.Linear.forward
        ):
            install_local_forward(module, style_name, name=name)
            installed += 1
    return installed
