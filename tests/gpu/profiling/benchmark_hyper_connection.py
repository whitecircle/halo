#!/usr/bin/env python
"""Upstream's hyper-connection against the toolkit's kernels at real sizes, on one GPU: ms per call forward and backward
for the weights and the mix, the graph memory one call holds, its peak, and the first call's cost, which includes the
compile on a cold ``TRITON_CACHE_DIR``.

    python tests/gpu/profiling/benchmark_hyper_connection.py --tokens 8192 16384
"""

import argparse
import time

import torch
from transformers import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

from halo.kernels.hyper_connection import mix_streams
from halo.models.hyper_connections import apply_hyper_connection_patches

_MIB = 2**20


def _stock_mix(block, residual, post, comb):
    return post.unsqueeze(-1) * block.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)


def _measure(forward, leaves, calls: int) -> dict[str, float]:
    """``forward()`` returns the tensors a backward starts from, each taking one fixed random cotangent."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = forward()
    cotangents = [torch.randn_like(output) for output in outputs]
    torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True)
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    del outputs

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
        torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True)
        torch.cuda.synchronize()
        backward += time.perf_counter() - start
    del outputs

    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = forward()
    torch.cuda.synchronize()
    graph = torch.cuda.memory_allocated() - base
    torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True)
    return {
        "first_s": first,
        "forward_ms": forward_ms,
        "backward_ms": backward / calls * 1e3,
        "graph_mib": graph / _MIB,
        "peak_mib": (torch.cuda.max_memory_allocated() - base) / _MIB,
    }


def _run(tokens: int, hc: Glm5NextTextHyperConnection, frozen: bool, args) -> None:
    dtype = next(hc.parameters()).dtype
    stream = torch.randn(1, tokens, args.hc_mult, args.hidden, device="cuda", dtype=dtype).requires_grad_(True)
    weight_leaves = [stream, *(p for p in hc.parameters() if p.requires_grad)]
    stock_forward = type(hc).forward.__wrapped__
    block = torch.randn(1, tokens, args.hidden, device="cuda", dtype=dtype).requires_grad_(True)
    post, comb, _ = (t.detach() for t in stock_forward(hc, stream))
    post, comb = post.to(dtype).requires_grad_(True), comb.to(dtype).requires_grad_(True)
    mix_leaves = [block, stream, post, comb]
    rows = {
        "weights stock": (lambda: stock_forward(hc, stream), weight_leaves),
        "weights kernel": (lambda: hc(stream), weight_leaves),
        "mix stock": (lambda: (_stock_mix(block, stream, post, comb),), mix_leaves),
        "mix kernel": (lambda: (mix_streams(block, stream, post, comb),), mix_leaves),
    }
    for name, (forward, leaves) in rows.items():
        row = _measure(forward, leaves, args.calls)
        print(
            f"T={tokens} frozen={frozen} {name:15s} forward {row['forward_ms']:7.3f} ms  "
            f"backward {row['backward_ms']:7.3f} ms  graph {row['graph_mib']:8.1f} MiB  "
            f"peak {row['peak_mib']:8.1f} MiB  first {row['first_s']:6.3f} s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc_mult", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20, help="GLM-5.3-Flash's hc_sinkhorn_iters")
    parser.add_argument("--calls", type=int, default=90, help="GLM-5.3-Flash runs two per layer on 45 layers")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    torch.manual_seed(0)
    config = Glm5NextTextConfig(hidden_size=args.hidden, hc_mult=args.hc_mult, hc_sinkhorn_iters=args.iters)
    apply_hyper_connection_patches(config)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    hc = Glm5NextTextHyperConnection(config).to(device="cuda", dtype=dtype)
    torch.nn.init.normal_(hc.fn, std=0.02)
    torch.nn.init.zeros_(hc.base)
    torch.nn.init.ones_(hc.scale)
    for frozen in (False, True):
        hc.requires_grad_(not frozen)
        for tokens in args.tokens:
            _run(tokens, hc, frozen, args)


if __name__ == "__main__":
    main()
