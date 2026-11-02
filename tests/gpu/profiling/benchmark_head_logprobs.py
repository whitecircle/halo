#!/usr/bin/env python
"""The chunked log-prob head (``halo.kernels.head_logprobs``) on one GPU, at a real model's head shape.

Times a scoring forward (no grad) and a forward+backward of ``chunked_head_logprobs`` with bf16 hidden states
and head weight, as GRPO calls it, and records each row count's peak allocated memory. The first
forward+backward and scoring pass are timed on their own: on a tree whose tiles run compiled they include the
compile, so run with fresh ``TORCHINDUCTOR_CACHE_DIR`` and ``TRITON_CACHE_DIR`` to read a cold rank's cost.

Only ``chunked_head_logprobs`` is called, so the same file times another tree's kernel put on ``PYTHONPATH``.
``--inductor_nondeterministic`` recompiles the tiles without inductor's ``deterministic`` mode, to price it.

Usage (inside the Halo image, one GPU):
    torchrun --nproc_per_node=1 tests/gpu/profiling/benchmark_head_logprobs.py --head qwen3-8b --out qwen3.json
"""

import argparse
import json
import statistics
import time

import torch

from halo.kernels import head_logprobs
from halo.kernels.head_logprobs import chunked_head_logprobs

# (hidden size, vocabulary, final logit softcap) of each model's output head.
HEADS = {
    "qwen3-8b": (4096, 151936, None),
    "gpt-oss-20b": (2880, 201088, None),
    "gemma4-26b-a4b": (2816, 262144, 30.0),
}
# Logit spread of a trained head: most mass on a few tokens.
LOGIT_STD = 6.0


def _time_ms(fn, warmup: int, iters: int) -> float:
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


def _recompile_tiles_nondeterministic() -> None:
    head_logprobs._TILE_COMPILE_OPTIONS = {**head_logprobs._TILE_COMPILE_OPTIONS, "deterministic": False}
    head_logprobs._COMPILED_TILES = tuple(
        head_logprobs._compile_tile(fn) for fn in (head_logprobs._forward_tile, head_logprobs._backward_tile)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=sorted(HEADS), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy", action="store_true", help="score with the fused entropy")
    parser.add_argument("--inductor_nondeterministic", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.inductor_nondeterministic:
        _recompile_tiles_nondeterministic()

    device = torch.device("cuda", torch.cuda.current_device())
    hidden_size, vocab, softcap = HEADS[args.head]
    generator = torch.Generator(device=device).manual_seed(0)
    weight = torch.randn(vocab, hidden_size, generator=generator, device=device) * (LOGIT_STD / hidden_size**0.5)
    weight = weight.bfloat16().requires_grad_(True)
    head = {"softcap": softcap}

    results = {}
    for tokens in args.tokens:
        hidden = torch.randn(tokens, hidden_size, generator=generator, device=device).bfloat16().requires_grad_(True)
        targets = torch.randint(0, vocab, (tokens,), generator=generator, device=device)
        upstream = torch.randn(tokens, generator=generator, device=device)

        def score():
            with torch.no_grad():
                chunked_head_logprobs(
                    hidden, weight, targets, None, args.temperature, compute_entropy=args.entropy, **head
                )

        def forward_backward():
            hidden.grad = weight.grad = None
            logps, _ = chunked_head_logprobs(hidden, weight, targets, None, args.temperature, **head)
            (logps * upstream).sum().backward()

        row = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        forward_backward()
        score()
        torch.cuda.synchronize()
        row["first_call_s"] = time.perf_counter() - start
        row["score_ms"] = _time_ms(score, args.warmup, args.iters)
        torch.cuda.reset_peak_memory_stats()
        row["forward_backward_ms"] = _time_ms(forward_backward, args.warmup, args.iters)
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        results[tokens] = row
        print(f"{args.head} {tokens} tokens: {row}", flush=True)

    meta = {**vars(args), "hidden_size": hidden_size, "vocab": vocab, "softcap": softcap}
    meta["device"] = torch.cuda.get_device_name(device)
    meta["compiled_tiles"] = hasattr(head_logprobs, "_COMPILED_TILES")
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
