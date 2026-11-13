#!/usr/bin/env python
"""One expert GEMM per arm at each shape, on one GPU and without a model: what a low-precision kernel costs before any
EP dispatch, against bf16 ``F.grouped_mm``.

Rows per expert stay multiples of 128, the padded kernels' segment, so every arm multiplies the same tokens. A row holds
the arm's median seconds, ``vs_bf16`` (above 1 is faster than the bf16 arm of the same pass) and, for an arm computing a
forward, ``rel_err`` against bf16, which flags an operand layout it gets wrong before its speed is read. A kernel that
refuses a shape or a platform is recorded as that arm's result, since whether it runs at all is part of the question.

    CUDA_VISIBLE_DEVICES=0 python tests/gpu/profiling/benchmark_lowp_kernels.py --groups gpt-oss-20b --output <file>
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.grouped_mm import grouped_mm, grouped_mm_grads
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.quantization import BlockScaledTensor, quantize_mxfp4, quantize_mxfp8, quantize_nvfp4
from tests.common.numerics import fro_rel_err
from tests.gpu.profiling.benchmark_grouped_mm import _median_per_iter

SEGMENT_ROWS = 128
SEED = 0
DENSE_ROWS = 16384
# The fused and down GEMMs of each model, per its expert widths.
GPT_OSS_20B = ((2880, 2880),)
QWEN3_30B = ((2048, 1536), (768, 2048))
QWEN3_5_35B = ((2048, 1024), (512, 2048))
WIDE = ((4096, 4096), (6144, 6144), (3072, 6144))


@dataclass(frozen=True)
class Shape:
    group: str
    experts: int
    k: int
    n: int
    rows_per_expert: int


def _grid(group: str, experts: tuple, gemms: tuple, rows: tuple) -> list[Shape]:
    return [Shape(group, e, k, n, r) for e in experts for k, n in gemms for r in rows]


SHAPE_GROUPS: dict[str, list[Shape]] = {
    "gpt-oss-20b": _grid("gpt-oss-20b", (4, 16, 32), GPT_OSS_20B, (512, 1024, 2048, 4096, 16384)),
    "qwen3-30b": _grid("qwen3-30b", (16, 64), QWEN3_30B, (512, 1024, 2048, 16384)),
    "qwen3.5-35b": _grid("qwen3.5-35b", (32, 128), QWEN3_5_35B, (256, 1024, 4096, 8192)),
    # The Qwen3-30B pair at 32 local experts: the shape of TorchTitan #5465's published ratios.
    "published-anchor": _grid("published-anchor", (32,), QWEN3_30B, (256, 512, 2048)),
    "wide": _grid("wide", (16,), WIDE, (1024, 4096)),
    # The crossover table in low-precision-moe-kernels.md.
    "crossover": [
        *_grid("crossover", (8,), ((4096, 8192),), (256, 512, 1024, 2048, 4096, 8192, 16384)),
        *_grid("crossover", (8,), ((4096, 16384),), (4096, 8192, 16384)),
        *_grid("crossover", (8,), GPT_OSS_20B, (256, 512, 1024, 2048, 4096)),
    ],
    "dense": _grid("dense", (1,), ((2880, 2880), (2048, 2048)), (DENSE_ROWS,)),
}


@dataclass(frozen=True)
class Operands:
    x: torch.Tensor
    w: nn.Parameter
    grad_output: torch.Tensor
    offs: torch.Tensor
    counts: torch.Tensor
    reference: torch.Tensor


def _expert_counts(shape: Shape, routing: str) -> torch.Tensor:
    """``uniform``, or Zipf over the experts at about the same total; a light expert may round to no rows at all."""
    if routing == "uniform":
        return torch.full((shape.experts,), shape.rows_per_expert, dtype=torch.int32)
    weights = 1.0 / torch.arange(1, shape.experts + 1, dtype=torch.float64)
    rows = (weights / weights.sum() * shape.experts * shape.rows_per_expert / SEGMENT_ROWS).round()
    return rows.to(torch.int32) * SEGMENT_ROWS


def _operands(shape: Shape, routing: str, device: torch.device) -> Operands:
    torch.manual_seed(SEED)
    counts = _expert_counts(shape, routing).to(device)
    rows = int(counts.sum())
    x = torch.randn(rows, shape.k, device=device, dtype=torch.bfloat16) * 0.3
    w = nn.Parameter(torch.randn(shape.experts, shape.k, shape.n, device=device, dtype=torch.bfloat16) / shape.k**0.5)
    offs = counts.cumsum(0).to(torch.int32)
    grad_output = torch.randn(rows, shape.n, device=device, dtype=torch.bfloat16)
    return Operands(x, w, grad_output, offs, counts, F.grouped_mm(x, w.detach(), offs=offs).clone())


def _with_backward(forward: Callable, ops: Operands) -> Callable:
    """Forward plus both gradients, as a training step runs them; the ``w.grad`` write is cleared so nothing grows."""

    def step() -> None:
        x = ops.x.detach().requires_grad_(True)
        forward(x, ops.w).backward(ops.grad_output)
        ops.w.grad = None

    return step


def _no_grad_forward(forward: Callable, ops: Operands) -> Callable:
    def step() -> torch.Tensor:
        with torch.no_grad():
            return forward(ops.x, ops.w)

    return step


def _forward_and_step(name: str, forward_of: Callable[[Operands], Callable]) -> dict[str, Callable]:
    return {
        f"{name}_forward": lambda ops: _no_grad_forward(forward_of(ops), ops),
        f"{name}_step": lambda ops: _with_backward(forward_of(ops), ops),
    }


def _simulated(quant_format: str, *, cacheable: bool = True) -> Callable:
    return lambda ops: lambda x, w: expert_gemm(
        x, w, offs=ops.offs, quant_format=quant_format, weight_cacheable=cacheable
    )


def _adapter(quant_format: str, *, fp8_backward: bool = False) -> Callable:
    return lambda ops: lambda x, w: deepgemm.deepgemm_grouped_gemm(
        x, w, offs=ops.offs, quant_format=quant_format, fp8_backward=fp8_backward
    )


def _torchao():
    """AO comes with the base image, not the lock, so an absent one records an arm result."""
    from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm  # noqa: PLC0415
    from torchao.prototype.moe_training.kernels.mxfp8 import (  # noqa: PLC0415
        mx_block_rearrange_2d_M_groups_cuda,
        triton_mx_block_rearrange_per_group_3d,
    )
    from torchao.prototype.mx_formats.utils import to_blocked  # noqa: PLC0415

    return (
        _to_mxfp8_then_scaled_grouped_mm,
        mx_block_rearrange_2d_M_groups_cuda,
        triton_mx_block_rearrange_per_group_3d,
        to_blocked,
    )


def _ao_mxfp8(*, wgrad_with_hp: bool) -> Callable:
    def forward_of(ops: Operands) -> Callable:
        grouped = _torchao()[0]
        return lambda x, w: grouped(
            x, w.transpose(-2, -1).contiguous().transpose(-2, -1), ops.offs, wgrad_with_hp=wgrad_with_hp
        )

    return forward_of


def _dg_operands(ops: Operands, quant_format: str) -> tuple:
    """The padded activation cast and the weight cast, for a kernel timed alone."""
    _, utils = deepgemm._deep_gemm()
    k = ops.w.shape[1]
    k_pad = utils.align(k, deepgemm._FP4_K_ALIGN) if quant_format == "mxfp4" else k
    x = F.pad(ops.x, (0, k_pad - k)) if k_pad != k else ops.x
    with torch.no_grad():
        return (
            utils.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=deepgemm._ALIGN),
            deepgemm._quant_weight(ops.w, quant_format, k_pad, cacheable=False),
            x,
            k_pad,
        )


def _dg_m_indices(ops: Operands) -> torch.Tensor:
    experts = torch.arange(ops.counts.numel(), device=ops.counts.device, dtype=torch.int32)
    return experts.repeat_interleave(ops.counts)


def _dg_kernel(quant_format: str) -> Callable:
    def build(ops: Operands) -> Callable:
        kernels, _ = deepgemm._deep_gemm()
        a_op, b_op, _, _ = _dg_operands(ops, quant_format)
        d = ops.x.new_empty(ops.x.shape[0], ops.w.shape[2], dtype=torch.bfloat16)
        m_indices = _dg_m_indices(ops)

        def step() -> torch.Tensor:
            if quant_format == "mxfp4":
                kernels.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a_op, b_op, d, m_indices, recipe=None, recipe_a=(1, 128), recipe_b=(1, 32)
                )
            else:
                kernels.m_grouped_fp8_gemm_nt_contiguous(a_op, b_op, d, m_indices, recipe=None)
            return d

        return step

    return build


def _dg_activation_cast(ops: Operands) -> Callable:
    _, utils = deepgemm._deep_gemm()
    return lambda: utils.per_token_cast_to_fp8(ops.x, use_ue8m0=True, gran_k=deepgemm._ALIGN)


def _dg_weight_cast(quant_format: str) -> Callable:
    def build(ops: Operands) -> Callable:
        _, _, _, k_pad = _dg_operands(ops, quant_format)

        @torch.no_grad()
        def step() -> None:
            deepgemm._quant_weight(ops.w, quant_format, k_pad, cacheable=False)

        return step

    return build


def _dg_m_grouped_bf16(ops: Operands) -> Callable:
    kernels, _ = deepgemm._deep_gemm()
    weight = ops.w.detach().transpose(-2, -1).contiguous()
    d = ops.x.new_empty(ops.x.shape[0], ops.w.shape[2], dtype=torch.bfloat16)
    m_indices = _dg_m_indices(ops)

    def step() -> torch.Tensor:
        kernels.m_grouped_bf16_gemm_nt_contiguous(ops.x, weight, d, m_indices)
        return d

    return step


def _dg_k_grouped_bf16(ops: Operands) -> Callable:
    """The weight gradient; the kernel adds into ``c``, so this keeps the fp32 accumulation it validates with."""
    kernels, _ = deepgemm._deep_gemm()
    accumulated = torch.zeros(ops.w.shape, device=ops.x.device, dtype=torch.float32)
    return lambda: kernels.k_grouped_bf16_gemm_tn_contiguous(
        ops.x, ops.grad_output, accumulated, ops.counts.tolist(), ops.counts, accumulated
    )


def _blocked(q: BlockScaledTensor, scale_dtype: torch.dtype, blocked_of: Callable) -> torch.Tensor:
    return blocked_of(q.scales.view(torch.uint8)).view(scale_dtype)


def _torch_scaled(quantize: Callable, scale_dtype: torch.dtype, recipe: str, *, global_scale: bool) -> Callable:
    """``F.scaled_grouped_mm`` on operands quantized once by Halo's own quantizers."""

    def build(ops: Operands) -> Callable:
        _, blocked_m_groups, blocked_3d, _ = _torchao()
        a = quantize(ops.x, axis=-1)
        b = quantize(ops.w.detach().transpose(-2, -1).contiguous(), axis=-1)
        scale_a = _blocked(a, scale_dtype, lambda s: blocked_m_groups(s, ops.offs))
        scale_b = _blocked(b, scale_dtype, blocked_3d)
        recipes = [getattr(F.ScalingType, recipe)]
        swizzles = [F.SwizzleType.SWIZZLE_32_4_4]
        if global_scale:
            recipes.append(F.ScalingType.TensorWise)
            swizzles.append(F.SwizzleType.NO_SWIZZLE)
            scale_a = [scale_a, a.global_scale.reshape(1)]
            scale_b = [scale_b, b.global_scale.reshape(1).expand(ops.w.shape[0]).contiguous()]
        data_a = a.data.view(torch.float4_e2m1fn_x2)
        data_b = b.data.view(torch.float4_e2m1fn_x2).transpose(-2, -1)
        return lambda: F.scaled_grouped_mm(
            data_a,
            data_b,
            scale_a=scale_a,
            scale_recipe_a=recipes,
            scale_b=scale_b,
            scale_recipe_b=recipes,
            swizzle_a=swizzles,
            swizzle_b=swizzles,
            offs=ops.offs,
            output_dtype=torch.bfloat16,
        )

    return build


def _dense_bf16(ops: Operands) -> Callable:
    x, weight = ops.x, ops.w.detach()[0]
    return lambda: x @ weight


def _dense_scaled_mxfp8(ops: Operands) -> Callable:
    to_blocked = _torchao()[3]
    a = quantize_mxfp8(ops.x, axis=-1)
    b = quantize_mxfp8(ops.w.detach()[0].t().contiguous(), axis=-1)
    scale_a, scale_b = (to_blocked(q.scales.view(torch.float8_e8m0fnu)) for q in (a, b))
    data_b = b.data.t()
    return lambda: F.scaled_mm(
        a.data,
        data_b,
        scale_a=scale_a,
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=scale_b,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )


BASELINE_PREFIX = "bf16_"
# An arm reads against the bf16 arm of the pass its name ends in; a cast has none.
PASSES = ("forward", "dgrad", "wgrad", "step")
GROUPED_ARMS: dict[str, Callable[[Operands], Callable]] = {
    "bf16_forward": lambda ops: lambda: F.grouped_mm(ops.x, ops.w.detach(), offs=ops.offs),
    "bf16_dgrad": lambda ops: lambda: grouped_mm_grads(ops.x, ops.w.detach(), ops.grad_output, ops.offs, True, False),
    "bf16_wgrad": lambda ops: lambda: grouped_mm_grads(ops.x, ops.w.detach(), ops.grad_output, ops.offs, False, True),
    "bf16_step": lambda ops: _with_backward(lambda x, w: grouped_mm(x, w, offs=ops.offs), ops),
    **_forward_and_step("simulated_mxfp8", _simulated("mxfp8")),
    **_forward_and_step("simulated_mxfp4", _simulated("mxfp4")),
    **_forward_and_step("simulated_nvfp4", _simulated("nvfp4")),
    "simulated_mxfp8_uncached_forward": lambda ops: _no_grad_forward(_simulated("mxfp8", cacheable=False)(ops), ops),
    **_forward_and_step("adapter_mxfp8", _adapter("mxfp8")),
    **_forward_and_step("adapter_mxfp4", _adapter("mxfp4")),
    "adapter_mxfp8_fp8_backward_step": lambda ops: _with_backward(_adapter("mxfp8", fp8_backward=True)(ops), ops),
    "adapter_mxfp4_fp8_backward_step": lambda ops: _with_backward(_adapter("mxfp4", fp8_backward=True)(ops), ops),
    **_forward_and_step("ao_mxfp8", _ao_mxfp8(wgrad_with_hp=False)),
    "ao_mxfp8_wgrad_with_hp_step": lambda ops: _with_backward(_ao_mxfp8(wgrad_with_hp=True)(ops), ops),
    "torch_scaled_mxfp4_forward": _torch_scaled(
        quantize_mxfp4, torch.float8_e8m0fnu, "BlockWise1x32", global_scale=False
    ),
    "torch_scaled_nvfp4_forward": _torch_scaled(
        quantize_nvfp4, torch.float8_e4m3fn, "BlockWise1x16", global_scale=True
    ),
    "deepgemm_mxfp8_kernel_forward": _dg_kernel("mxfp8"),
    "deepgemm_mxfp4_kernel_forward": _dg_kernel("mxfp4"),
    "deepgemm_activation_cast": _dg_activation_cast,
    "deepgemm_mxfp8_weight_cast": _dg_weight_cast("mxfp8"),
    "deepgemm_mxfp4_weight_cast": _dg_weight_cast("mxfp4"),
    "deepgemm_bf16_m_grouped_forward": _dg_m_grouped_bf16,
    "deepgemm_bf16_k_grouped_wgrad": _dg_k_grouped_bf16,
}
DENSE_ARMS: dict[str, Callable[[Operands], Callable]] = {
    "bf16_forward": _dense_bf16,
    "torch_scaled_mxfp8_forward": _dense_scaled_mxfp8,
}


def _measure(arm: str, build: Callable, ops: Operands, baselines: dict[str, float], args) -> dict:
    try:
        step = build(ops)
        seconds = _median_per_iter(step, args.iters, args.warmup)
        out = step()
    except Exception as exc:
        return {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
    row: dict = {"arm": arm, "median_s": seconds}
    if arm.startswith(BASELINE_PREFIX):
        baselines[arm] = seconds
    suffix = arm.rsplit("_", 1)[-1]
    if suffix in PASSES and f"{BASELINE_PREFIX}{suffix}" in baselines:
        row["vs_bf16"] = baselines[f"{BASELINE_PREFIX}{suffix}"] / seconds
    if isinstance(out, torch.Tensor) and out.shape == ops.reference.shape:
        row["rel_err"] = fro_rel_err(out, ops.reference)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=tuple(SHAPE_GROUPS), default=tuple(SHAPE_GROUPS))
    parser.add_argument("--routing", nargs="+", choices=("uniform", "skewed"), default=("uniform", "skewed"))
    parser.add_argument("--arms", nargs="+", help="Run only these arms; the bf16 baselines of each pass always run")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations per arm")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed iterations per arm, past any JIT compile")
    parser.add_argument("--output", required=True, help="JSON-lines file, appended to as each arm finishes")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with open(args.output, "a", encoding="utf-8") as output:
        for shape in (shape for group in args.groups for shape in SHAPE_GROUPS[group]):
            arms = DENSE_ARMS if shape.group == "dense" else GROUPED_ARMS
            for routing in ("uniform",) if shape.group == "dense" else args.routing:
                ops = _operands(shape, routing, device)
                baselines: dict[str, float] = {}
                for arm, build in arms.items():
                    if args.arms and arm not in args.arms and not arm.startswith(BASELINE_PREFIX):
                        continue
                    row = {**asdict(shape), "routing": routing, **_measure(arm, build, ops, baselines, args)}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(row, flush=True)
                del ops
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
