#!/usr/bin/env python
"""Upstream's Flash-Next gated residual and block injection against the toolkit's kernels at the release's sizes, on one
GPU: ms per call forward and backward, the graph memory one call holds and its peak, for frozen and trained
hyper-connection weights.

    python tests/gpu/profiling/benchmark_gated_residual.py --tokens 8192 16384
"""

import argparse
import os
import time

import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from halo.models.families.qwen4_exp.gated_residual import inject_block, patch_qwen4_exp_gated_residual

KNOB = "HALO_GATED_RESIDUAL_KERNELS"
_MIB = 2**20


def _block(module, stream, out):
    mixed, hyper_input, gates = module(stream)
    return (inject_block(hyper_input, mixed + out, gates),)


def _measure(mode: str, module, stream, out, calls: int) -> dict[str, float]:
    os.environ[KNOB] = mode
    leaves = [stream, *(p for p in module.parameters() if p.requires_grad)]
    forward = lambda: _block(module, stream, out)  # noqa: E731
    cotangent = [torch.randn_like(forward()[0])]
    for _ in range(3):
        torch.autograd.grad(forward(), leaves, cotangent, allow_unused=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        forward()
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - start) / calls * 1e3
    backward = 0.0
    for _ in range(calls):
        outputs = forward()
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.autograd.grad(outputs, leaves, cotangent, allow_unused=True)
        torch.cuda.synchronize()
        backward += time.perf_counter() - start
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = forward()
    torch.cuda.synchronize()
    graph = torch.cuda.memory_allocated() - base
    torch.autograd.grad(outputs, leaves, cotangent, allow_unused=True)
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward / calls * 1e3,
        "graph_mib": graph / _MIB,
        "peak_mib": (torch.cuda.max_memory_allocated() - base) / _MIB,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--lowrank", type=int, default=320)
    parser.add_argument("--calls", type=int, default=96, help="the release runs two per layer on 48 layers")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    patch_qwen4_exp_gated_residual()
    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(hidden_size=args.hidden, hc_count=args.streams, hc_lowrank=args.lowrank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    module = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config).to(device="cuda", dtype=dtype)
    torch.nn.init.normal_(module.hc_norm.weight, std=0.1)
    for trained in (False, True):
        module.requires_grad_(trained)
        for tokens in args.tokens:
            stream = torch.randn(1, tokens, args.streams * args.hidden, device="cuda", dtype=dtype, requires_grad=True)
            out = torch.randn(1, tokens, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
            for name, mode in (("upstream", "0"), ("kernels", "1")):
                row = _measure(mode, module, stream, out, args.calls)
                print(
                    f"T={tokens} trained={trained} {name:9s} forward {row['forward_ms']:7.3f} ms  "
                    f"backward {row['backward_ms']:7.3f} ms  graph {row['graph_mib']:8.1f} MiB  "
                    f"peak {row['peak_mib']:8.1f} MiB",
                    flush=True,
                )


if __name__ == "__main__":
    main()
