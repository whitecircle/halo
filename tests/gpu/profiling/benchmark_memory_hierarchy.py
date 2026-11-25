#!/usr/bin/env python
"""The memory side of one GPU: device bandwidth by access pattern and buffer size, the latency of a dependent random
load at each level of the hierarchy (a pointer chase), host-device copies from pinned and pageable memory, and, with a
second GPU visible, a one-way copy from GPU 0 to GPU 1 (``peer_copy``).

    python tests/gpu/profiling/benchmark_memory_hierarchy.py --output memory.json
"""

import argparse
import time

import torch
import triton
import triton.language as tl

from tests.gpu.profiling.measurement import (
    calls_per_repeat,
    gpu_state,
    run_record,
    summarize,
    time_device,
    write_record,
)

SECTIONS = ("bandwidth", "latency", "host_copy", "peer_copy")
DEFAULT_SECTIONS = ("bandwidth", "latency", "host_copy")
MIB = 2**20
REPEAT_SECONDS = 0.05
GUESS_BYTES_PER_S = 6e12
CHASE_STEPS = 200_000
STORE_BLOCK = 8192
MAX_CYCLE_ENTRIES = 2**30
PEER_COPY_WARMUP = 2
SEED = 0


@triton.jit
def _store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty), mask=offsets < numel)


@triton.jit
def _load_kernel(in_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(in_ptr + offsets, mask=offsets < numel, other=0.0)
    tl.store(out_ptr + tl.program_id(0), tl.sum(values.to(tl.float32), axis=0))


def _bandwidth_at(nbytes: int, repeats: int) -> list[dict]:
    """Copy (read plus write), an elementwise copy, read-only sums and write-only fills (PyTorch's and plain Triton load
    and store kernels) over ``nbytes`` buffers, bytes counted once per pass."""
    src = torch.ones(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    grid = (triton.cdiv(dst.numel(), STORE_BLOCK),)
    partials = torch.empty(grid[0], device="cuda", dtype=torch.float32)
    patterns = {
        "copy": (lambda: dst.copy_(src), 2 * nbytes),
        "elementwise_copy": (lambda: torch.mul(src, 1, out=dst), 2 * nbytes),
        "read_sum": (lambda: torch.sum(src, dtype=torch.float32), nbytes),
        "write_fill": (lambda: dst.fill_(1.0), nbytes),
        "write_zero": (dst.zero_, nbytes),
        "write_triton": (lambda: _store_kernel[grid](dst, dst.numel(), BLOCK=STORE_BLOCK, num_warps=8), nbytes),
        # One fp32 partial sum per 8192 elements: the write is 1/4096 of the read.
        "read_triton": (
            lambda: _load_kernel[grid](src, partials, src.numel(), BLOCK=STORE_BLOCK, num_warps=8),
            nbytes,
        ),
    }
    rows = []
    for pattern, (fn, moved) in patterns.items():
        iters = calls_per_repeat(moved / GUESS_BYTES_PER_S, REPEAT_SECONDS, floor=2)
        samples = time_device(fn, iters=iters, repeats=repeats, warmup=2)
        stats = summarize(samples)
        rows.append(
            {
                "pattern": pattern,
                "buffer_bytes": nbytes,
                "bytes_moved": moved,
                "calls_per_repeat": iters,
                "seconds": samples,
                "seconds_summary": stats,
                "tb_per_s_median": moved / stats["median"] / 1e12,
                "state": gpu_state(),
            }
        )
        print(f"  {pattern:>16} {nbytes / MIB:>8.0f} MiB  {moved / stats['median'] / 1e12:6.3f} TB/s", flush=True)
    return rows


@triton.jit
def _chase_kernel(next_ptr, out_ptr, steps):
    index = tl.load(next_ptr)
    for _ in range(steps):
        index = tl.load(next_ptr + index)
    tl.store(out_ptr, index)


def _random_cycle(slots: int) -> torch.Tensor:
    """A ``slots``-long int64 buffer holding one cycle, in random order, through evenly spaced entries (every slot up
    to ``MAX_CYCLE_ENTRIES``, the most ``randperm`` sorts), so no load can be predicted or skipped."""
    entries = min(slots, MAX_CYCLE_ENTRIES)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    positions = torch.randperm(entries, device="cuda", generator=generator) * (slots // entries)
    successor = torch.zeros(slots, dtype=torch.int64, device="cuda")
    successor[positions] = positions.roll(-1)
    return successor


def _latency_at(nbytes: int, repeats: int, out: torch.Tensor) -> dict:
    """Seconds per load of one thread following ``CHASE_STEPS`` dependent loads through a random cycle over ``nbytes``:
    the buffer's level of the hierarchy sets the time."""
    successor = _random_cycle(nbytes // 8)

    def chase() -> None:
        _chase_kernel[(1,)](successor, out, CHASE_STEPS, num_warps=1)

    samples = [s / CHASE_STEPS for s in time_device(chase, iters=1, repeats=repeats, warmup=2)]
    stats = summarize(samples)
    print(f"  chase {nbytes / MIB:>10.3f} MiB  {stats['median'] * 1e9:7.1f} ns/load", flush=True)
    return {
        "buffer_bytes": nbytes,
        "steps": CHASE_STEPS,
        "seconds_per_load": samples,
        "seconds_per_load_summary": stats,
        "state": gpu_state(),
    }


def _latency(args) -> list[dict]:
    out = torch.empty(1, dtype=torch.int64, device="cuda")
    rows = []
    for nbytes in args.latency_bytes:
        rows.append(_latency_at(nbytes, args.repeats, out))
        torch.cuda.empty_cache()
    return rows


def _host_copy_at(nbytes: int, repeats: int) -> list[dict]:
    """Host-to-device and device-to-host copies of ``nbytes`` from pinned and from pageable host memory."""
    device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    hosts = {
        "pinned": torch.ones(nbytes, dtype=torch.uint8, pin_memory=True),
        "pageable": torch.ones(nbytes, dtype=torch.uint8),
    }
    rows = []
    for memory, host in hosts.items():
        directions = {
            "host_to_device": lambda host=host: device.copy_(host, non_blocking=True),
            "device_to_host": lambda host=host: host.copy_(device, non_blocking=True),
        }
        for direction, fn in directions.items():
            samples = time_device(fn, iters=3, repeats=repeats, warmup=2)
            stats = summarize(samples)
            rows.append(
                {
                    "memory": memory,
                    "direction": direction,
                    "bytes": nbytes,
                    "seconds": samples,
                    "seconds_summary": stats,
                    "gb_per_s_median": nbytes / stats["median"] / 1e9,
                }
            )
            print(f"  {memory:>8} {direction:>14} {nbytes / MIB:.0f} MiB  {nbytes / stats['median'] / 1e9:7.1f} GB/s")
    return rows


def _host_copy(args) -> list[dict]:
    rows = []
    for mib in args.host_copy_mib:
        rows += _host_copy_at(mib * MIB, args.repeats)
        torch.cuda.empty_cache()
    return rows


def _peer_copy(args) -> list[dict]:
    """GPU 0 to GPU 1 copies of ``--host_copy_mib`` each, host-timed between full synchronizations of both devices."""
    if torch.cuda.device_count() < 2:
        raise SystemExit("peer_copy needs two visible GPUs")
    rows = []
    for mib in args.host_copy_mib:
        src = torch.ones(mib * MIB, dtype=torch.uint8, device="cuda:0")
        dst = torch.empty(mib * MIB, dtype=torch.uint8, device="cuda:1")
        samples = []
        for repeat in range(PEER_COPY_WARMUP + args.repeats):
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            start = time.perf_counter()
            dst.copy_(src)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            if repeat >= PEER_COPY_WARMUP:
                samples.append(time.perf_counter() - start)
        stats = summarize(samples)
        rows.append(
            {
                "bytes": mib * MIB,
                "peer_access": torch.cuda.can_device_access_peer(0, 1),
                "seconds": samples,
                "seconds_summary": stats,
                "gb_per_s_median": mib * MIB / stats["median"] / 1e9,
            }
        )
        print(f"  GPU0 -> GPU1 {mib} MiB  {mib * MIB / stats['median'] / 1e9:7.1f} GB/s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON file, rewritten after each section")
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=DEFAULT_SECTIONS)
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per point; each is one sample")
    parser.add_argument(
        "--bandwidth_mib", type=int, nargs="+", default=(16, 32, 64, 128, 256, 1024, 2048, 8192, 65536)
    )
    parser.add_argument(
        "--latency_bytes",
        type=int,
        nargs="+",
        default=tuple(2**exponent for exponent in (14, 17, 20, 23, 25, 26, 27, 28, 30, 32, 34)),
        help="Pointer-chase buffer sizes in bytes (default 16 KiB to 16 GiB)",
    )
    parser.add_argument("--host_copy_mib", type=int, nargs="+", default=(1024, 8192))
    args = parser.parse_args()

    torch.cuda.set_device(0)
    record = {**run_record(), "sections": {}}
    runners = {
        "bandwidth": lambda: [row for mib in args.bandwidth_mib for row in _bandwidth_at(mib * MIB, args.repeats)],
        "latency": lambda: _latency(args),
        "host_copy": lambda: _host_copy(args),
        "peer_copy": lambda: _peer_copy(args),
    }
    for section in args.sections:
        print(f"\n## {section}", flush=True)
        record["sections"][section] = runners[section]()
        write_record(args.output, record)


if __name__ == "__main__":
    main()
