#!/usr/bin/env python
"""Dispatch + combine time of a DeepEP V2 arena sized exactly to the tokens/rank in flight against the same arena
with the headroom a grow now takes (``arena_capacity``), at the wire shape of a shipped family. No model: random
routing over the whole EP group, one forward dispatch and its combine per iteration."""

import argparse
import statistics
import sys

import torch
import torch.distributed as dist

from halo.distributed.expert_parallel.deepep.backends import _ElasticArena, deep_ep, free_all_ep_buffers
from halo.distributed.expert_parallel.deepep.capacity import arena_capacity, ep_dispatch_capacity, padded_wire_hidden
from halo.distributed.expert_parallel.deepep.settings import DEEPEP_NUM_QPS
from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.rank_log import log

# name -> (hidden, top_k, experts)
SHAPES = {
    "qwen3_5_35b_a3b": (2048, 8, 256),
    "gpt_oss_20b": (2880, 4, 32),
}
WARMUP = 5
ITERS = 50


def time_dispatch_combine(shape: str, tokens: int, headroom: bool) -> tuple[float, int]:
    hidden, top_k, experts = SHAPES[shape]
    padded = padded_wire_hidden(hidden)
    needed = ep_dispatch_capacity(tokens)
    capacity = (
        arena_capacity(needed, num_topk=top_k, padded_hidden=padded, spans_domains=False) if headroom else needed
    )
    arena = _ElasticArena.acquire(dist.group.WORLD, padded, top_k, DEEPEP_NUM_QPS, False)
    arena.ensure(capacity)
    buffer = arena.buffer
    x = torch.randn(tokens, padded, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(tokens, experts, device="cuda")
    topk_weights, topk_idx = scores.topk(top_k, dim=-1)
    topk_idx = topk_idx.to(deep_ep().topk_idx_t)
    num_sms = buffer.get_theoretical_num_sms(experts, top_k)

    def step() -> None:
        recv_x, _, recv_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=experts,
            num_max_tokens_per_rank=arena.capacity,
            num_sms=num_sms,
        )
        buffer.combine(recv_x, handle=handle, topk_weights=recv_weights)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    dist.barrier()
    free_all_ep_buffers()
    return statistics.median(times), arena.capacity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=list(SHAPES), default="qwen3_5_35b_a3b")
    parser.add_argument("--tokens", type=int, default=8192, help="tokens/rank in flight")
    args = parser.parse_args()
    rank, _, _ = init_distributed()
    exact_ms, exact_capacity = time_dispatch_combine(args.shape, args.tokens, headroom=False)
    padded_ms, padded_capacity = time_dispatch_combine(args.shape, args.tokens, headroom=True)
    if rank == 0:
        log(f"\n{'=' * 60}\n  arena headroom, {args.shape}, {args.tokens} tokens/rank, ep={dist.get_world_size()}")
        log(f"  capacity {exact_capacity}: {exact_ms:.3f} ms dispatch+combine (median of {ITERS})")
        log(f"  capacity {padded_capacity}: {padded_ms:.3f} ms dispatch+combine (median of {ITERS})")
        log(f"  headroom/exact: {padded_ms / exact_ms:.3f}x")
    teardown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
