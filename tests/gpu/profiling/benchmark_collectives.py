#!/usr/bin/env python
"""NCCL collectives through ``torch.distributed`` across message sizes: seconds per call and bus bandwidth as
nccl-tests defines it. Message bytes are the full buffer (all-reduce's and broadcast's tensor, all-gather's output,
reduce-scatter's and all-to-all's input, one rank's send for send/recv); ``algbw = bytes / seconds`` and ``busbw`` is
``algbw`` times 2(n-1)/n for all-reduce, (n-1)/n for all-gather, reduce-scatter and all-to-all, and 1 for broadcast and
send/recv. ``--wait_probe`` instead holds every rank but the last inside one all-reduce for that many seconds.

    torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_collectives.py --output collectives-8.json
"""

import argparse
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from tests.common.distributed import init_distributed, teardown_distributed
from tests.gpu.profiling.measurement import calls_per_repeat, run_record, summarize, time_device, write_record

COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "send_recv")
DEFAULT_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
# 4 KiB to 4 GiB in powers of four.
DEFAULT_SIZES = tuple(4**exponent * 1024 for exponent in range(1, 12))
REPEAT_SECONDS = 0.05
# Rough per-call cost that sizes a repeat's call count; the measurement does not depend on it.
GUESS_LATENCY_S, GUESS_BUS_BYTES_PER_S = 20e-6, 3e11


def _bus_factor(collective: str, world: int) -> float:
    if collective == "all_reduce":
        return 2 * (world - 1) / world
    if collective in ("broadcast", "send_recv"):
        return 1.0
    return (world - 1) / world


def _call(collective: str, nbytes: int, rank: int, world: int) -> Callable[[], object]:
    """One call of ``collective`` on a message of ``nbytes`` total bytes, bf16; send/recv runs a ring, each rank
    sending to the next while it receives from the previous."""
    numel = nbytes // 2
    full = torch.ones(numel, device="cuda", dtype=torch.bfloat16)
    if collective == "all_reduce":
        return lambda: dist.all_reduce(full)
    if collective == "broadcast":
        return lambda: dist.broadcast(full, src=0)
    if collective == "all_to_all":
        received = torch.empty_like(full)
        return lambda: dist.all_to_all_single(received, full)
    if collective == "send_recv":
        received = torch.empty_like(full)
        ops = [
            dist.P2POp(dist.isend, full, (rank + 1) % world),
            dist.P2POp(dist.irecv, received, (rank - 1) % world),
        ]
        return lambda: [request.wait() for request in dist.batch_isend_irecv(ops)]
    shard = torch.ones(numel // world, device="cuda", dtype=torch.bfloat16)
    if collective == "all_gather":
        return lambda: dist.all_gather_into_tensor(full, shard)
    return lambda: dist.reduce_scatter_tensor(shard, full)


def _wait_probe(seconds: float, rank: int, world: int) -> dict:
    """COLLECTIVE: the last rank sleeps ``seconds`` before joining one small all-reduce that every other rank has
    already entered; returns the UNIX window in which the others wait inside NCCL."""
    buffer = torch.ones(1024, device="cuda")
    dist.all_reduce(buffer)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.time()
    if rank == world - 1:
        time.sleep(seconds)
    dist.all_reduce(buffer)
    torch.cuda.synchronize()
    return {"wait_window_unix": [start, time.time()], "sleeping_rank": world - 1}


def _per_call_seconds(call, iters: int, repeats: int, warmup: int) -> list[float]:
    """COLLECTIVE: every rank's seconds per call for each repeat, ranks aligned by a barrier before each; returns the
    slowest rank's sample per repeat."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        samples += time_device(call, iters=iters, repeats=1, warmup=0)
    slowest = torch.tensor(samples, device="cuda", dtype=torch.float64)
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    return slowest.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON file rank 0 rewrites after each point")
    parser.add_argument("--collectives", nargs="+", choices=COLLECTIVES, default=DEFAULT_COLLECTIVES)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="Message bytes")
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per point; each is one sample")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--wait_probe", type=float, help="Seconds every rank but the last waits in one all-reduce")
    args = parser.parse_args()

    rank, world, _ = init_distributed()
    if args.wait_probe and world < 2:
        teardown_distributed()
        parser.error(
            "--wait_probe needs one rank to sleep while another waits inside the all-reduce, and this run has one "
            "process: launch it with torchrun --nproc_per_node=2."
        )
    record = {
        **run_record(),
        "world_size": world,
        "nccl_version": ".".join(str(part) for part in torch.cuda.nccl.version()),
        "nccl_env": {key: value for key, value in os.environ.items() if key.startswith("NCCL_")},
        "dtype": "bf16",
        "timing": "CUDA events around back-to-back calls per rank; each sample is the slowest rank's repeat",
        "rows": [],
    }
    try:
        if args.wait_probe:
            record["wait_probe"] = _wait_probe(args.wait_probe, rank, world)
            if rank == 0:
                write_record(args.output, record)
            return
        for collective in args.collectives:
            for nbytes in args.sizes:
                call = _call(collective, nbytes, rank, world)
                guess = GUESS_LATENCY_S + nbytes * _bus_factor(collective, world) / GUESS_BUS_BYTES_PER_S
                iters = calls_per_repeat(guess, REPEAT_SECONDS, floor=5, cap=5000)
                samples = _per_call_seconds(call, iters, args.repeats, args.warmup)
                del call
                torch.cuda.empty_cache()
                if rank != 0:
                    continue
                stats = summarize(samples)
                algbw = nbytes / stats["median"]
                row = {
                    "collective": collective,
                    "bytes": nbytes,
                    "calls_per_repeat": iters,
                    "seconds": samples,
                    "seconds_summary": stats,
                    "algbw_bytes_per_s": algbw,
                    "busbw_bytes_per_s": algbw * _bus_factor(collective, world),
                    "busbw_bytes_per_s_samples": [nbytes / s * _bus_factor(collective, world) for s in samples],
                }
                record["rows"].append(row)
                write_record(args.output, record)
                print(
                    f"  {collective:>14} {nbytes:>12} B  {stats['median'] * 1e6:10.1f} us  "
                    f"busbw {row['busbw_bytes_per_s'] / 1e9:8.1f} GB/s  spread {stats['spread']:.3f}",
                    flush=True,
                )
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
