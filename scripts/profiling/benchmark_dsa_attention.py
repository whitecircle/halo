#!/usr/bin/env python
"""GLM-5.3-Flash DSA attention on one GPU, forward and backward: stock SDPA, on the backend it selects and pinned to its
mem-efficient kernel, against the masked matmul kernel on the same bf16 inputs and mask, each output and gradient against
an fp32 reference that runs on none of them.

    python scripts/profiling/benchmark_dsa_attention.py --mask /path/to/mask-L16384-layer3.pt
    python scripts/profiling/benchmark_dsa_attention.py --seq 8192 --heads 64 --head_dim 256

``--mask`` is a saved boolean ``[1, 1, T, T]`` from a real run and sets the length; without it each query selects random
whole pools at or before it, plus its own incomplete pool. Inputs keep the stock layouts: q and v strided views, k
contiguous. Peak memory is above what the inputs and mask hold.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.models.attention.masked_matmul import SCORES_PER_TILE, masked_matmul

POOL = 4
# Whole pools a synthetic query selects, in keys; the real masks hold about 1,800-2,050.
SYNTHETIC_KEYS = 2048
SYNTHETIC_QUERIES_PER_PASS = 1024
_GIB = 2**30


def _pool_mask(seq: int, device: torch.device) -> torch.Tensor:
    pools = seq // POOL
    ends = torch.arange(1, pools + 1, device=device) * POOL - 1
    positions = torch.arange(seq, device=device)
    mask = torch.zeros(1, 1, seq, seq, dtype=torch.bool, device=device)
    for start in range(0, seq, SYNTHETIC_QUERIES_PER_PASS):
        queries = positions[start : start + SYNTHETIC_QUERIES_PER_PASS]
        visible = ends <= queries[:, None]
        scores = torch.rand(queries.numel(), pools, device=device).masked_fill(~visible, -1.0)
        picked = scores.topk(min(SYNTHETIC_KEYS // POOL, pools), dim=-1).indices
        selected = torch.zeros_like(visible).scatter_(1, picked, visible.gather(1, picked))
        whole_pools = selected.repeat_interleave(POOL, dim=1)
        own_pool = (positions[None] // POOL == queries[:, None] // POOL) & (positions[None] <= queries[:, None])
        mask[0, 0, queries, : whole_pools.shape[1]] = whole_pools
        mask[0, 0, queries] |= own_pool
    return mask


def _load_mask(path: str, seq: int | None, device: torch.device) -> torch.Tensor:
    mask = torch.load(path, map_location=device, weights_only=True)
    if mask.dtype != torch.bool or mask.shape[:2] != (1, 1) or mask.ndim != 4 or mask.shape[2] != mask.shape[3]:
        raise ValueError(f"--mask must hold a boolean [1, 1, T, T] tensor, got {mask.dtype} {tuple(mask.shape)}")
    if seq is not None and seq != mask.shape[-1]:
        raise ValueError(f"--seq {seq} disagrees with the {mask.shape[-1]} tokens of --mask")
    return mask


def _stock_layout_inputs(seq: int, heads: int, dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """As ``Glm5NextTextAttention`` hands them over: ``view().transpose()`` for q, a split of ``kv_b_proj``'s transposed
    output for v, and a fresh contiguous k."""
    dtype = torch.bfloat16
    q = torch.randn(1, seq, heads, dim, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(1, heads, seq, dim, device=device, dtype=dtype).requires_grad_()
    kv = torch.randn(1, seq, heads, 2 * dim, device=device, dtype=dtype).transpose(1, 2)
    return q, k, kv[..., dim:].requires_grad_()


def _selected(q, k, v, mask, scale) -> torch.Tensor:
    """At the release's head width, mem-efficient on Blackwell and cuDNN on Hopper."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).transpose(1, 2).contiguous()


def _stock(q, k, v, mask, scale) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        return _selected(q, k, v, mask, scale)


KERNELS = {"stock_selected": _selected, "stock_mem_efficient": _stock, "masked_matmul": masked_matmul}


def _measure(kernel, q, k, v, mask, scale, grad, iters: int) -> tuple[dict, tuple[torch.Tensor, ...]]:
    """The first iteration is dropped from the times, as it grows the allocator's cache."""
    forward_ms, backward_ms = [], []
    forward_peak = total_peak = 0
    for _ in range(iters + 1):
        out = None  # held across iterations it would sit in the base and leave the peak short by its size
        for t in (q, k, v):
            t.grad = None
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = kernel(q, k, v, mask, scale)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        forward_peak = torch.cuda.max_memory_allocated() - base
        out.backward(grad)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_peak = torch.cuda.max_memory_allocated() - base
        forward_ms.append((middle - start) * 1e3)
        backward_ms.append((end - middle) * 1e3)
    timing = {
        "forward_ms": statistics.median(forward_ms[1:]),
        "backward_ms": statistics.median(backward_ms[1:]),
        "forward_peak_gib": forward_peak / _GIB,
        "peak_gib": total_peak / _GIB,
    }
    return timing, (out.detach(), q.grad, k.grad, v.grad)


def _fp32_reference(q, k, v, mask, scale, grad) -> tuple[torch.Tensor, ...]:
    """Plain autograd in true fp32, one tile of queries at a time, with a safe softmax for a query with no key."""
    _, heads, seq, _ = q.shape
    q32, k32, v32 = (t.detach().float().requires_grad_() for t in (q, k, v))
    grad32 = grad.float().transpose(1, 2)
    outputs = []
    step = max(1, SCORES_PER_TILE // (heads * seq))
    for start in range(0, seq, step):
        queries = slice(start, min(start + step, seq))
        allowed = mask[:, :, queries]
        any_key = allowed.any(dim=-1, keepdim=True)
        scores = torch.matmul(q32[:, :, queries], k32.transpose(-1, -2)) * scale
        probs = torch.softmax(scores.masked_fill(~allowed & any_key, float("-inf")), dim=-1) * any_key
        out = torch.matmul(probs, v32)
        out.backward(grad32[:, :, queries])
        outputs.append(out.detach())
    return torch.cat(outputs, dim=2).transpose(1, 2), q32.grad, k32.grad, v32.grad


def _relative_errors(measured: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]) -> dict:
    errors = {}
    for name, got, truth in zip(("out", "d_q", "d_k", "d_v"), measured, reference, strict=True):
        difference = torch.linalg.vector_norm((got.float() - truth).double())
        errors[f"{name}_relative_error"] = (difference / torch.linalg.vector_norm(truth.double())).item()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=int, default=None, help="tokens; required without --mask")
    parser.add_argument("--heads", type=int, default=64, help="attention heads of the synthetic inputs")
    parser.add_argument("--head_dim", type=int, default=256, help="query, key and value width per head")
    parser.add_argument("--mask", default=None, help="a saved boolean [1, 1, T, T] mask")
    parser.add_argument("--iters", type=int, default=5, help="timed iterations after the dropped first one")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random inputs")
    parser.add_argument("--out", default=None, help="append each JSON record to this file")
    args = parser.parse_args()
    if args.mask is None and args.seq is None:
        parser.error("give --seq or --mask")

    configure_float32_matmul_precision()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    mask = _load_mask(args.mask, args.seq, device) if args.mask else _pool_mask(args.seq, device)
    seq = mask.shape[-1]
    scale = args.head_dim**-0.5
    q, k, v = _stock_layout_inputs(seq, args.heads, args.head_dim, device)
    grad = torch.randn(1, seq, args.heads, args.head_dim, device=device, dtype=torch.bfloat16)

    common = {
        "seq": seq,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "mask": args.mask,
        "keys_per_query": mask.sum().item() / seq,
        "torch": torch.__version__,
    }
    runs = {name: _measure(kernel, q, k, v, mask, scale, grad, args.iters) for name, kernel in KERNELS.items()}
    reference = _fp32_reference(q, k, v, mask, scale, grad)
    for name, (timing, measured) in runs.items():
        record = {"kernel": name, **common, **timing, **_relative_errors(measured, reference)}
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
