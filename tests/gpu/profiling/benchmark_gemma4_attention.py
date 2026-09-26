#!/usr/bin/env python
"""Gemma 4 26B-A4B attention, forward + backward per layer type, on one GPU.

Sliding layers: 16 query / 8 KV heads, head_dim 256, window 1,024. Global layers: 16 / 2 heads,
head_dim 512. Each backend runs in its own process (``--backend``) because a kernel that rejects a
shape can leave the CUDA context unusable.

- ``sdpa``: mem-efficient SDPA with manual KV repeat and a dense boolean mask, the path the loader
  pins for Gemma 4 (``patch_sdpa_for_wide_heads``).
- ``eager``: matmul-softmax-matmul with fp32 softmax (transformers' eager attention).
- ``halo``: ``sdpa_flex_sliding``'s own attention function (:func:`flex_sliding_attention`) on the dense
  mask transformers builds: compiled FlexAttention under the tile block masks on a sliding layer, matmul
  attention (or mem-efficient SDPA past the budget) on a global one. The block masks come from its cache,
  as every sliding layer after the first in a forward gets them.
- ``fa2``: FlashAttention 2 (``flash_attn``) with its local window, where installed.
- ``fa4``: FlashAttention 4 (``flash_attn.cute``) where installed.

Usage (inside the Halo image, one GPU):
    python tests/gpu/profiling/benchmark_gemma4_attention.py --backend halo --layer sliding --seq 2048 --out r.jsonl
"""

import argparse
import json

import torch
import torch.nn.functional as F
import triton
from accelerate import PartialState
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.models.patches.attention import patch_sdpa_for_wide_heads
from src.models.patches.flex_sliding_attention import flex_sliding_attention

LAYERS = {"sliding": (256, 8, 1024), "global": (512, 2, None)}
HEADS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sdpa", "eager", "halo", "fa2", "fa4"), required=True)
    parser.add_argument("--layer", choices=tuple(LAYERS), required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    head_dim, kv_heads, window = LAYERS[args.layer]
    seq = args.seq
    q = torch.randn(1, HEADS, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(1, HEADS, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    params = [q, k, v]

    if args.backend == "sdpa":
        pos = torch.arange(seq, device="cuda")
        mask = (pos[None] <= pos[:, None]) & (pos[:, None] - pos[None] < window) if window else None

        def step():
            kk, vv = (t.repeat_interleave(HEADS // kv_heads, 1) for t in (k, v))
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=mask is None, scale=1.0)
            out.backward(grad)
    elif args.backend == "eager":
        pos = torch.arange(seq, device="cuda")
        allowed = pos[None] <= pos[:, None]
        if window:
            allowed &= pos[:, None] - pos[None] < window

        def step():
            kk, vv = (t.repeat_interleave(HEADS // kv_heads, 1) for t in (k, v))
            scores = (q @ kk.transpose(-1, -2)).masked_fill(~allowed, float("-inf"))
            (torch.softmax(scores.float(), -1).to(q.dtype) @ vv).backward(grad)
    elif args.backend == "halo":
        PartialState()  # the toolkit logs through accelerate's logger
        patch_sdpa_for_wide_heads()  # the Gemma 4 loader's process-wide SDPA pin
        module = torch.nn.Module()
        module.is_causal = True
        module.num_key_value_groups = HEADS // kv_heads
        # The masks the ``sdpa`` rows use: the dense window on a sliding layer, none (causal) on a global one.
        pos = torch.arange(seq, device="cuda")
        mask = ((pos[None] <= pos[:, None]) & (pos[:, None] - pos[None] < window))[None, None] if window else None
        grad_t = grad.transpose(1, 2)

        def step():
            out, _ = flex_sliding_attention(module, q, k, v, mask, scaling=1.0, sliding_window=window)
            out.backward(grad_t)
    elif args.backend == "fa2":
        from flash_attn import flash_attn_func  # noqa: PLC0415 - optional dependency

        params = [t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (q, k, v)]
        grad_t = grad.transpose(1, 2).contiguous()

        def step():
            out = flash_attn_func(
                *params, causal=True, window_size=(window - 1, 0) if window else (-1, -1), softmax_scale=1.0
            )
            out.backward(grad_t)
    else:
        from flash_attn.cute import flash_attn_func  # noqa: PLC0415 - optional dependency

        params = [t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (q, k, v)]
        grad_t = grad.transpose(1, 2).contiguous()

        def step():
            out = flash_attn_func(
                *params, causal=True, window_size=(window - 1, 0) if window else (None, None), softmax_scale=1.0
            )
            (out[0] if isinstance(out, tuple) else out).backward(grad_t)

    def fwd_bwd():
        for p in params:
            p.grad = None
        step()

    record = {"backend": args.backend, "layer": args.layer, "seq": seq, "torch": torch.__version__}
    try:
        record["fwd_bwd_ms"] = triton.testing.do_bench(fwd_bwd, warmup=5, rep=40)
    except Exception as exc:  # a backend that rejects this shape is a result, not a crash
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    print(json.dumps(record), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
