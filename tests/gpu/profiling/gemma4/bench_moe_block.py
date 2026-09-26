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

Usage: python tests/gpu/profiling/gemma4/bench_moe_block.py --out results.json [--tokens 2048 8192 32768]
"""

import argparse
import json
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.fused_glu import PACKED_GLU_MULS, fused_gelu_tanh_mul
from src.kernels.grouped_gemm import grouped_gemm
from src.kernels.moe_permute import MoEWeightedUnpermute

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 2816, 704, 128, 8


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
    combines = {
        "eager": lambda gu: F.gelu(gu[:, :INTERMEDIATE], approximate="tanh") * gu[:, INTERMEDIATE:],
        "torch.compile": torch.compile(
            lambda gu: F.gelu(gu[:, :INTERMEDIATE], approximate="tanh") * gu[:, INTERMEDIATE:], dynamic=False
        ),
    }
    try:
        from liger_kernel.ops.geglu import LigerGELUMulFunction

        combines["liger"] = lambda gu: LigerGELUMulFunction.apply(gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:])
    except ImportError:
        pass
    return combines


def halo_block(combine=None):
    """This tree's grouped expert compute (``_compute_experts_with_grouped_mm`` at ``ep_size`` 1)."""
    layer = SimpleNamespace(ep_size=1, experts_per_rank=EXPERTS, _build_inv_map=EPMoELayerBase._build_inv_map)
    glu = combine or PACKED_GLU_MULS[fused_gelu_tanh_mul]

    def run(x, idx, w, gate_up_w, down_w):
        tokens, offs, token_idx, weights, _, inv_map = EPMoELayerBase._sort_tokens_for_grouped_mm(layer, x, idx, w)
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

    variants = {}
    for impl in ("eager", "grouped_mm"):

        def hf_run(impl=impl):
            config._experts_implementation = impl
            return hf(x, idx, w.to(torch.bfloat16))

        variants[f"hf_{impl}"] = (hf_run, [x, w, hf.gate_up_proj, hf.down_proj])
    variants["halo"] = (lambda: halo_block()(x, idx, w, gate_up_w, down_w), [x, w, gate_up_w, down_w])
    for name, combine in _combines().items():
        run = halo_block(combine)
        variants[f"halo+{name}"] = ((lambda run=run: run(x, idx, w, gate_up_w, down_w)), [x, w, gate_up_w, down_w])

    reference, results = None, {}
    for name, (fwd, params) in variants.items():

        def fwd_bwd(fwd=fwd, params=params):
            for p in params:
                p.grad = None
            fwd().backward(grad)

        iters = 10 if name == "hf_eager" else 25
        with torch.no_grad():
            fwd_ms = time_ms(fwd, iters=iters)
        with torch.no_grad():
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
    import transformers

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
