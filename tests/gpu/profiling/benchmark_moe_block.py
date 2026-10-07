#!/usr/bin/env python
"""Gemma 4 26B-A4B routed-expert block, forward + backward, on one GPU with all 128 experts local.

Compares the implementations a user can pick for this block at the checkpoint's real shapes (hidden 2816,
expert intermediate 704, 128 experts, top-8):

- ``hf_eager`` / ``hf_grouped_mm``: transformers' ``Gemma4TextExperts`` under ``experts_implementation``
  ``eager`` (per-expert loop) and ``grouped_mm``.
- ``halo``: this tree's grouped expert path (sort, grouped GEMM, fused GLU, atomic-free weighted unpermute),
  driven through ``EPMoELayerBase``'s own methods.
- ``halo+<combine>``: the same path with the GeGLU combine swapped for eager PyTorch, ``torch.compile`` or
  Liger, isolating the activation kernel.
- ``halo_padded_gather``: the same path, its sort included, with a padded-gather permute: the routing
  weights multiplied separately, the unpermute a padded ``[N, top_k, H]`` gather-sum, and the permute's
  backward the same.

Usage (inside the Halo image, one GPU):
    python tests/gpu/profiling/benchmark_moe_block.py --out results.json [--tokens 2048 8192 32768]
"""

import argparse
import contextlib
import json
import statistics
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import transformers
from liger_kernel.ops.geglu import LigerGELUMulFunction
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from src.distributed.expert_parallel import base_layer
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.fused_glu import PACKED_GLU_MULS, fused_gelu_tanh_mul
from src.kernels.grouped_gemm import grouped_gemm
from src.kernels.moe_permute import MoEWeightedUnpermute, padded_gather_reduce

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 2816, 704, 128, 8

# What ``_sort_tokens_for_grouped_mm`` reads off the layer: one EP rank holding every expert.
_LAYER = SimpleNamespace(ep_size=1, experts_per_rank=EXPERTS)


def time_ms(fn, warmup: int = 5, iters: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def peak_extra_mib(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def _combines():
    return {
        "eager": lambda gu: F.gelu(gu[:, :INTERMEDIATE], approximate="tanh") * gu[:, INTERMEDIATE:],
        "torch.compile": torch.compile(
            lambda gu: F.gelu(gu[:, :INTERMEDIATE], approximate="tanh") * gu[:, INTERMEDIATE:], dynamic=False
        ),
        "liger": lambda gu: LigerGELUMulFunction.apply(gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:]),
    }


class _PaddedGatherPermute(torch.autograd.Function):
    """``tokens[sorted_token_idx]``, with the padded gather-sum as its backward."""

    @staticmethod
    def forward(ctx, tokens, sorted_token_idx, inv_map):
        ctx.save_for_backward(inv_map)
        return tokens.index_select(0, sorted_token_idx)

    @staticmethod
    def backward(ctx, grad_sorted):
        (inv_map,) = ctx.saved_tensors
        return padded_gather_reduce(grad_sorted, inv_map), None, None


class _PaddedGatherUnpermute(torch.autograd.Function):
    """The padded ``[N, top_k, H]`` gather-sum of already-weighted rows, a gather as its backward."""

    @staticmethod
    def forward(ctx, expert_out, sorted_token_idx, inv_map):
        ctx.save_for_backward(sorted_token_idx)
        return padded_gather_reduce(expert_out, inv_map)

    @staticmethod
    def backward(ctx, grad_out):
        (sorted_token_idx,) = ctx.saved_tensors
        return grad_out.index_select(0, sorted_token_idx), None, None


def _sort(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor):
    """``_sort_tokens_for_grouped_mm`` at ``ep_size`` 1: the expert-sorted tokens and routing."""
    return EPMoELayerBase._sort_tokens_for_grouped_mm(_LAYER, x, idx, w)


# The sort's own permute swapped for the padded one, so the baseline differs from ``halo`` in the permute alone.
padded_permute = partial(patch.object, base_layer, "MoEGatherPermute", _PaddedGatherPermute)


def padded_gather_block():
    """:func:`halo_block` with the padded-gather permute (run under :data:`padded_permute`) and a separate
    routing-weight multiply."""
    glu = PACKED_GLU_MULS[fused_gelu_tanh_mul]

    def run(x, idx, w, gate_up_w, down_w):
        tokens, offs, token_idx, weights, _, inv_map = _sort(x, idx, w)
        y = grouped_gemm(glu(grouped_gemm(tokens, gate_up_w, offs=offs)), down_w, offs=offs)
        return _PaddedGatherUnpermute.apply(y * weights.unsqueeze(-1).to(y.dtype), token_idx, inv_map)

    return run


def halo_block(combine=None):
    """This tree's grouped expert compute (``_compute_experts_with_grouped_mm`` at ``ep_size`` 1)."""
    glu = combine or PACKED_GLU_MULS[fused_gelu_tanh_mul]

    def run(x, idx, w, gate_up_w, down_w):
        tokens, offs, token_idx, weights, _, inv_map = _sort(x, idx, w)
        y = grouped_gemm(glu(grouped_gemm(tokens, gate_up_w, offs=offs)), down_w, offs=offs)
        return MoEWeightedUnpermute.apply(y.contiguous(), weights.to(y.dtype), token_idx, inv_map)

    return run


def bench(tokens: int) -> dict:
    torch.manual_seed(0)
    config = Gemma4TextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        top_k_experts=TOP_K,
        hidden_activation="gelu_pytorch_tanh",
        num_hidden_layers=1,
    )
    hf = Gemma4TextExperts(config).to("cuda", torch.bfloat16)
    with torch.no_grad():
        hf.gate_up_proj.normal_(0, 0.02)
        hf.down_proj.normal_(0, 0.02)
    gate_up_w = hf.gate_up_proj.detach().transpose(1, 2).contiguous().requires_grad_()
    down_w = hf.down_proj.detach().transpose(1, 2).contiguous().requires_grad_()
    x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs, idx = torch.topk(torch.softmax(torch.randn(tokens, EXPERTS, device="cuda"), -1), TOP_K, dim=-1)
    w = (probs / probs.sum(-1, keepdim=True)).requires_grad_()
    grad = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

    with padded_permute():
        swapped = _sort(x, idx, w)[0]
    assert swapped.grad_fn._forward_cls is _PaddedGatherPermute, "the baseline's sort kept the fused permute"

    halo_params = [x, w, gate_up_w, down_w]
    variants = {}
    for impl in ("eager", "grouped_mm"):

        def hf_run(impl=impl):
            config._experts_implementation = impl
            return hf(x, idx, w.to(torch.bfloat16))

        variants[f"hf_{impl}"] = (hf_run, [x, w, hf.gate_up_proj, hf.down_proj], contextlib.nullcontext)
    variants["halo"] = (lambda: halo_block()(x, idx, w, gate_up_w, down_w), halo_params, contextlib.nullcontext)
    variants["halo_padded_gather"] = (
        lambda: padded_gather_block()(x, idx, w, gate_up_w, down_w),
        halo_params,
        padded_permute,
    )
    for name, combine in _combines().items():
        run = halo_block(combine)
        variants[f"halo+{name}"] = (
            (lambda run=run: run(x, idx, w, gate_up_w, down_w)),
            halo_params,
            contextlib.nullcontext,
        )

    reference, results = None, {}
    for name, (fwd, params, context) in variants.items():

        def fwd_bwd(fwd=fwd, params=params):
            for p in params:
                p.grad = None
            fwd().backward(grad)

        iters = 10 if name == "hf_eager" else 25
        with context():
            with torch.no_grad():
                fwd_ms = time_ms(fwd, iters=iters)
                out = fwd().float()
            reference = out if reference is None else reference
            results[name] = {
                "fwd_ms": fwd_ms,
                "fwd_bwd_ms": time_ms(fwd_bwd, iters=iters),
                "peak_extra_mib": peak_extra_mib(fwd_bwd),
                "max_rel_err_vs_hf_eager": ((out - reference).abs().max() / reference.abs().max()).item(),
            }
        print(tokens, name, {k: round(v, 4) for k, v in results[name].items()}, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    args = parser.parse_args()
    meta = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "shapes": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "experts": EXPERTS, "top_k": TOP_K},
    }
    results = {t: bench(t) for t in args.tokens}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
