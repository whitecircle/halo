#!/usr/bin/env python
"""Compute-side roofline anchors on one GPU: GEMM ceilings by precision, M sweeps, K/N/M alignment, the launch floor,
elementwise fusion and stream overlap; ``--sustain`` runs one regime steadily so power can be read beside it. The
memory side (bandwidth, latency, host copies) is ``benchmark_memory_hierarchy.py``.

Every section writes per-repeat samples to ``--output``. GEMM operands rotate through copies spanning several L2
caches (``--hot_l2`` keeps one copy), so each call streams its weight from HBM as a model's next layer reads its own.

    python tests/gpu/profiling/benchmark_roofline.py --output roofline.json
    python tests/gpu/profiling/benchmark_roofline.py --sections sweep --sweep_shapes 2880x2880 --output expert.json
    python tests/gpu/profiling/benchmark_roofline.py --sections sweep --sweep_m 8:2048:8 --kernel_shapes --output m.json
    python tests/gpu/profiling/benchmark_roofline.py --sustain gemm --seconds 20 --output gemm-power.json
"""

import argparse
import functools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.gpu.profiling.measurement import (
    GEMM_PRECISIONS,
    GUESS_BYTES_PER_S,
    ClockSampler,
    calls_per_repeat,
    capture_graph,
    gpu_state,
    profile_kernels,
    rotating,
    rotation_count,
    run_record,
    summarize,
    time_device,
    time_graph,
    write_record,
)

SECTIONS = ("peaks", "sweep", "alignment", "launch", "fusion", "streams", "triton_gemm", "memory_bound")
SUSTAINED = ("idle", "gemm", "elementwise", "silu_mul", "tiny", "small_m", "expert_loop")
# Rough FLOP rates that size a repeat's call count beside GUESS_BYTES_PER_S; the measurement does not depend on them.
GUESS_FLOPS = {"bf16": 2e15, "fp16": 2e15, "tf32": 1e15, "fp32": 1e14}
# Default length of one timed repeat.
REPEAT_SECONDS = 0.05
SUSTAINED_GEMM_DIM = 8192
GEMM_FORMS = ("forward", "wgrad", "fp32_out")
MAX_GRAPH_CALLS = 2000
SWEEP_POINTS_PER_OCTAVE = 4
SWEEP_M_RANGE = (16, 32768)
FUSION_DIM = 8192
FUSION_SIZE_EXPONENTS = range(16, 29, 2)
# Row width of the by-size tensors: Liger's SwiGLU launches one program per row.
FUSION_ROW = 4096
LAUNCH_OPS = 10000
SYNCED_OPS = 2000
STREAM_COPY_BYTES = 2 * 2**30
EXPERT_LOOP_EXPERTS, EXPERT_LOOP_ROWS = 128, 128
CLOCK_SAMPLE_SECONDS = 0.002
# About 1 ms at the top clock, short against a sustained chunk.
CLOCK_PROBE_CYCLES = 2_000_000


@dataclass(frozen=True)
class Timing:
    """How each GEMM point is timed: ``repeats`` samples of about ``repeat_seconds``, each after ``cooldown`` idle
    seconds when set (a burst from cool clocks) or back to back when not (sustained, at the power limit)."""

    repeats: int
    repeat_seconds: float
    cooldown: float


def _gemm_operands(m: int, k: int, n: int, dtype: torch.dtype, hot_l2: bool, form: str = "forward") -> list[tuple]:
    """Operand sets for ``form``: ``forward`` ``x [m, k]``, ``w [k, n]``, ``y [m, n]``; ``wgrad`` ``x [m, k]``,
    ``dy [m, n]``, ``dw [k, n]`` (the weight gradient, contracting over the m rows); ``fp32_out`` as ``forward`` with
    an fp32 ``y``."""
    element = torch.finfo(dtype).bits // 8
    count = 1 if hot_l2 else rotation_count(element * (m * k + k * n + m * n))
    second, out_shape = ((m, n), (k, n)) if form == "wgrad" else ((k, n), (m, n))
    out_dtype = torch.float32 if form == "fp32_out" else dtype
    return [
        (
            torch.randn(m, k, device="cuda", dtype=dtype),
            torch.randn(second, device="cuda", dtype=dtype),
            torch.empty(out_shape, device="cuda", dtype=out_dtype),
        )
        for _ in range(count)
    ]


def _gemm_point(
    m: int, k: int, n: int, dtype_name: str, *, hot_l2: bool, timing: Timing, graph: bool, form: str = "forward"
) -> dict:
    """One ``[m, k] @ [k, n]`` GEMM (or, for ``form="wgrad"``, ``x.T @ dy``; ``"fp32_out"``, an fp32 output): per-call
    seconds back to back (``eager``) and, with ``graph``, replayed as one CUDA graph (``graph``), with FLOP/s,
    arithmetic intensity and the bytes the call must move."""
    dtype, precision = GEMM_PRECISIONS[dtype_name]
    torch.set_float32_matmul_precision(precision)
    sets = _gemm_operands(m, k, n, dtype, hot_l2, form)
    next_set = rotating(sets)

    def call() -> None:
        x, w, y = next_set()
        if form == "wgrad":
            torch.mm(x.t(), w, out=y)
        elif form == "fp32_out":
            torch.mm(x, w, out_dtype=torch.float32, out=y)
        else:
            torch.mm(x, w, out=y)

    flops = 2 * m * k * n
    nbytes = sum(t.numel() * t.element_size() for t in sets[0])
    guess = flops / GUESS_FLOPS[dtype_name] + nbytes / GUESS_BYTES_PER_S + 5e-6
    iters = calls_per_repeat(guess, timing.repeat_seconds, floor=len(sets))
    row = {
        "m": m,
        "k": k,
        "n": n,
        "dtype": dtype_name,
        "form": form,
        "flops": flops,
        "bytes": nbytes,
        "arithmetic_intensity": flops / nbytes,
        "operand_copies": len(sets),
        "calls_per_repeat": iters,
        "cooldown_seconds": timing.cooldown,
    }
    warmup = max(3, len(sets))
    timings = {
        "eager": time_device(call, iters=iters, repeats=timing.repeats, warmup=warmup, cooldown=timing.cooldown)
    }
    if graph:
        calls = max(len(sets), min(iters, MAX_GRAPH_CALLS))
        timings["graph"] = time_graph(
            call, iters=calls, repeats=timing.repeats, warmup=warmup, cooldown=timing.cooldown
        )
    for timer, samples in timings.items():
        stats = summarize(samples)
        row[timer] = {
            "seconds": samples,
            "seconds_summary": stats,
            "tflops_median": flops / stats["median"] / 1e12,
            "bytes_per_s_median": nbytes / stats["median"],
        }
    row["state"] = gpu_state()
    del sets
    return row


def _record_kernels(row: dict) -> None:
    """Adds the kernel cuBLAS picks for ``row``'s GEMM and its launch grid, read from one profiled call."""
    dtype, precision = GEMM_PRECISIONS[row["dtype"]]
    torch.set_float32_matmul_precision(precision)
    x, w, y = _gemm_operands(row["m"], row["k"], row["n"], dtype, hot_l2=True)[0]
    row["kernels"] = profile_kernels(lambda: torch.mm(x, w, out=y), calls=1)["kernels"]


def _peaks(args, timing: Timing, rows: list[dict], flush: Callable[[], None]) -> None:
    for dim in args.peak_dims:
        for dtype_name in args.peak_dtypes:
            row = _gemm_point(dim, dim, dim, dtype_name, hot_l2=True, timing=timing, graph=False)
            row["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            row["float32_matmul_precision"] = torch.get_float32_matmul_precision()
            print(f"  {dtype_name:>4} {dim}^3  {row['eager']['tflops_median']:7.1f} TFLOP/s")
            rows.append(row)
            flush()
            torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("highest")


def _sweep_ms(spec: list[str] | None) -> list[int]:
    """M values from ``start:stop:step`` ranges (stop included) and single integers, or log-spaced by default."""
    if not spec:
        low, high = SWEEP_M_RANGE
        octaves = round(math.log2(high / low))
        points = octaves * SWEEP_POINTS_PER_OCTAVE + 1
        return sorted({round(low * 2 ** (i / SWEEP_POINTS_PER_OCTAVE)) for i in range(points)})
    ms = set()
    for part in spec:
        if ":" in part:
            start, stop, step = (int(value) for value in part.split(":"))
            ms.update(range(start, stop + 1, step))
        else:
            ms.add(int(part))
    return sorted(ms)


def _sweep(
    args, timing: Timing, rows: list[dict], flush: Callable[[], None], deferred: list[Callable[[], None]]
) -> None:
    for shape in args.sweep_shapes:
        k, n = (int(part) for part in shape.split("x"))
        for m in _sweep_ms(args.sweep_m):
            row = _gemm_point(m, k, n, "bf16", hot_l2=args.hot_l2, timing=timing, graph=True)
            row["hot_l2"] = args.hot_l2
            print(
                f"  K={k} N={n} M={m:>6}  AI {row['arithmetic_intensity']:7.1f}  eager "
                f"{row['eager']['tflops_median']:7.1f}  graph {row['graph']['tflops_median']:7.1f} TFLOP/s  "
                f"{row['state']['sm_clock_mhz']} MHz",
                flush=True,
            )
            rows.append(row)
            flush()
            if args.kernel_shapes:
                deferred.append(functools.partial(_record_kernels, row))
        torch.cuda.empty_cache()


def _alignment(args, timing: Timing, rows: list[dict], flush: Callable[[], None]) -> None:
    """Each of ``--alignment_axes`` stepped through ``base + 0..span`` with the other two at their base sizes, for
    every GEMM form of ``--alignment_forms``."""
    base = args.alignment_base
    for dtype_name in args.alignment_dtypes:
        for form in args.alignment_forms:
            for axis in args.alignment_axes:
                for step in range(args.alignment_span + 1):
                    dims = {"m": args.alignment_m, "k": base, "n": base}
                    dims[axis] += step
                    row = _gemm_point(
                        **dims, dtype_name=dtype_name, hot_l2=False, timing=timing, graph=False, form=form
                    )
                    row["axis"] = axis
                    print(
                        f"  {dtype_name:>4} {form} {axis}={dims[axis]}  {row['eager']['tflops_median']:7.1f} TFLOP/s"
                    )
                    rows.append(row)
                    flush()
    torch.set_float32_matmul_precision("highest")


def _eager_issue(op: Callable[[], object], repeats: int) -> tuple[list[float], list[float], list[float]]:
    """Per repeat, host seconds per op to issue ``LAUNCH_OPS`` calls, wall seconds per op including the drain, and
    seconds per op when every call is followed by a synchronize."""
    for _ in range(100):
        op()
    torch.cuda.synchronize()
    eager_host, eager_wall, synced = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(LAUNCH_OPS):
            op()
        issued = time.perf_counter()
        torch.cuda.synchronize()
        eager_host.append((issued - start) / LAUNCH_OPS)
        eager_wall.append((time.perf_counter() - start) / LAUNCH_OPS)
        start = time.perf_counter()
        for _ in range(SYNCED_OPS):
            op()
            torch.cuda.synchronize()
        synced.append((time.perf_counter() - start) / SYNCED_OPS)
    return eager_host, eager_wall, synced


def _launch_floor(op: Callable[[], object], repeats: int) -> dict:
    """Host seconds to issue ``op`` eagerly, the host cost of replaying ``LAUNCH_OPS`` of it from one CUDA graph, and
    its device time inside that graph and around one call."""
    eager_host, eager_wall, synced = _eager_issue(op, repeats)
    graph = capture_graph(op, LAUNCH_OPS, warmup=3)
    replay_host = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        graph.replay()
        replay_host.append(time.perf_counter() - start)
        torch.cuda.synchronize()
    graph_device = [s / LAUNCH_OPS for s in time_device(graph.replay, iters=1, repeats=repeats, warmup=1)]
    return {
        "ops_per_repeat": LAUNCH_OPS,
        "eager_host_seconds_per_op": eager_host,
        "eager_wall_seconds_per_op": eager_wall,
        "synchronized_each_seconds_per_op": synced,
        "graph_replay_host_seconds_first_then_warm": replay_host,
        "graph_device_seconds_per_op": graph_device,
        "single_op_event_seconds": time_device(op, iters=1, repeats=max(repeats, 50), warmup=10),
        "state": gpu_state(),
    }


def _launch(args, deferred: list[Callable[[], None]]) -> dict:
    """The launch floor for a trivial elementwise kernel and for the smallest matmul; the deferred pass adds each op's
    profiled kernel time and its eager cost after a profiling session (the ``*_after_profiler`` rows)."""
    x = torch.zeros(8, device="cuda")
    a = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    ops = {
        "add_8_fp32": lambda: x.add_(1.0),
        "mm_8x8x8_bf16": lambda: torch.mm(a, a, out=out),
    }
    record = {name: _launch_floor(op, args.repeats) for name, op in ops.items()}

    def profiled() -> None:
        for name, op in ops.items():
            record[name]["profiled_kernel_us"] = summarize(profile_kernels(op, calls=1000)["kernel_us"])
        for name, op in ops.items():
            host, _, synced = _eager_issue(op, args.repeats)
            record[name]["eager_host_seconds_per_op_after_profiler"] = host
            record[name]["synchronized_each_seconds_per_op_after_profiler"] = synced

    deferred.append(profiled)
    for name in ops:
        for key, value in record[name].items():
            if isinstance(value, list):
                print(f"  {name} {key:>50}: {summarize(value)['median'] * 1e6:10.3f} us")
    return record


def _fusion_by_size(args, expression: Callable) -> list[dict]:
    """Eager, Liger and compiled ``silu(a) * b`` from launch-bound to memory-bound sizes, back to back; each size
    compiles its own static-shape kernel, as a dynamic-shape recompile would generate a slower one."""
    rows = []
    for exponent in FUSION_SIZE_EXPONENTS:
        a = torch.randn(2**exponent // FUSION_ROW, FUSION_ROW, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        torch._dynamo.reset()  # Else the recompile limit, not the shape, decides when compile falls back to eager.
        compiled = torch.compile(expression, dynamic=False)
        variants = {
            "eager": (lambda: F.silu(a) * b, 5),
            "liger": (lambda: LigerSiLUMulFunction.apply(a, b), 3),
            "compiled": (lambda: compiled(a, b), 3),
        }
        for name, (fn, passes) in variants.items():
            moved = passes * a.numel() * a.element_size()
            iters = calls_per_repeat(moved / GUESS_BYTES_PER_S + 1e-5, REPEAT_SECONDS, floor=20)
            samples = time_device(fn, iters=iters, repeats=args.repeats, warmup=10)
            median = summarize(samples)["median"]
            rows.append(
                {
                    "variant": name,
                    "elements": a.numel(),
                    "tensor_passes": passes,
                    "seconds": samples,
                    "seconds_median": median,
                    "effective_tb_per_s": moved / median / 1e12,
                }
            )
            print(f"  2^{exponent:<2} {name:>8}  {median * 1e6:9.2f} us  {moved / median / 1e12:5.2f} TB/s")
    return rows


def _fusion(args, deferred: list[Callable[[], None]]) -> dict:
    """``silu(a) * b`` on ``[FUSION_DIM, FUSION_DIM]`` bf16 four ways, each with the tensor passes it makes over HBM
    (a read or write of one operand-sized tensor), against one ``FUSION_DIM``-cubed GEMM."""
    a = torch.randn(FUSION_DIM, FUSION_DIM, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    w = torch.randn_like(a)
    y = torch.empty_like(a)

    def expression(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.silu(a) * b

    compiled = torch.compile(expression)
    compiled_autotuned = torch.compile(expression, mode="max-autotune")
    variants = {
        # silu reads a, writes s; mul reads s and b, writes y.
        "eager": (lambda: expression(a, b), 5),
        # sigmoid reads a, writes g; mul reads a and g, writes s; mul reads s and b, writes y.
        "three_ops": (lambda: a * torch.sigmoid(a) * b, 8),
        "liger": (lambda: LigerSiLUMulFunction.apply(a, b), 3),
        "compiled": (lambda: compiled(a, b), 3),
        "compiled_max_autotune": (lambda: compiled_autotuned(a, b), 3),
        "gemm": (lambda: torch.mm(a, w, out=y), 3),
    }
    tensor_bytes = a.numel() * a.element_size()
    rows = []
    for name, (fn, passes) in variants.items():
        samples = time_device(fn, iters=50, repeats=args.repeats, warmup=10)
        stats = summarize(samples)
        moved = passes * tensor_bytes
        rows.append(
            {
                "variant": name,
                "tensor_passes": passes,
                "bytes_moved": moved,
                "seconds": samples,
                "seconds_summary": stats,
                "effective_tb_per_s": moved / stats["median"] / 1e12,
                "state": gpu_state(),
            }
        )
        print(
            f"  {name:>22}  {stats['median'] * 1e3:7.3f} ms  {passes} passes  {moved / stats['median'] / 1e12:5.2f} TB/s"
        )
    by_size = _fusion_by_size(args, expression)

    def profiled() -> None:
        for row, (fn, _) in zip(rows, variants.values(), strict=True):
            kernels = profile_kernels(fn)
            row.update(kernels_per_call=kernels["kernels_per_call"], kernels=kernels["kernels"])

    deferred.append(profiled)
    return {"dim": FUSION_DIM, "variants": rows, "by_size": by_size}


def _streams(args) -> list[dict]:
    """Two ``FUSION_DIM``-cubed bf16 GEMMs on one stream against two streams, and a GEMM with a device-to-device copy
    in sequence against concurrent; every variant starts after and ends before the timing stream's events."""
    x, w, y, x2, w2, y2 = (torch.randn(FUSION_DIM, FUSION_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(6))
    src = torch.ones(STREAM_COPY_BYTES // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    side = (torch.cuda.Stream(), torch.cuda.Stream())

    def gemm() -> None:
        torch.mm(x, w, out=y)

    def gemm2() -> None:
        torch.mm(x2, w2, out=y2)

    def copy() -> None:
        dst.copy_(src)

    def concurrent(*ops: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            current = torch.cuda.current_stream()
            for stream, op in zip(side, ops, strict=False):
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    op()
            for stream in side[: len(ops)]:
                current.wait_stream(stream)

        return run

    variants = {
        "gemm_alone": gemm,
        "copy_alone": copy,
        "two_gemms_one_stream": lambda: (gemm(), gemm2()),
        "two_gemms_two_streams": concurrent(gemm, gemm2),
        "gemm_then_copy_one_stream": lambda: (gemm(), copy()),
        "gemm_and_copy_two_streams": concurrent(gemm, copy),
    }
    rows = []
    for name, fn in variants.items():
        # A cooled repeat stays short (about 8 ms for the GEMM pairs) so it finishes before the power cap engages.
        iters = 5 if args.cooldown else 20
        samples = time_device(fn, iters=iters, repeats=args.repeats, warmup=5, cooldown=args.cooldown)
        rows.append({"variant": name, "seconds": samples, "seconds_summary": summarize(samples), "state": gpu_state()})
        print(f"  {name:>28}  {summarize(samples)['median'] * 1e3:8.3f} ms", flush=True)
    return rows


def _expert_loop() -> tuple[Callable[[], object], int, dict]:
    """A per-expert MoE loop at the Qwen3-30B-A3B expert shape: ``EXPERT_LOOP_EXPERTS`` experts, each a fused gate-up
    and a down projection on ``EXPERT_LOOP_ROWS`` rows, two matmuls per expert per call."""
    hidden, intermediate, rows = 2048, 768, EXPERT_LOOP_ROWS
    experts = [
        (
            torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(hidden, 2 * intermediate, device="cuda", dtype=torch.bfloat16),
            torch.randn(rows, intermediate, device="cuda", dtype=torch.bfloat16),
            torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(EXPERT_LOOP_EXPERTS)
    ]

    def loop() -> None:
        for x, gate_up, h, down in experts:
            torch.mm(x, gate_up)
            torch.mm(h, down)

    flops = EXPERT_LOOP_EXPERTS * 2 * rows * (hidden * 2 * intermediate + intermediate * hidden)
    return loop, 4, {"flops": flops}


def _record_profiled_kernels(row: dict, fn: Callable[[], object]) -> None:
    row["kernels"] = profile_kernels(fn, calls=1)["kernels"]


def _triton_gemm(args, deferred: list[Callable[[], None]]) -> list[dict]:
    """cuBLAS against Inductor's autotuned Triton GEMM templates on ``--peak_dims`` bf16 squares. Triton is the only
    GEMM backend autotuning may pick inside this section, where each compile runs on its first call."""
    rows = []
    with inductor_config.patch(max_autotune_gemm_backends="TRITON"):
        for dim in args.peak_dims:
            a, b = (torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
            triton_mm = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)
            # Bound per dim: the deferred profile of a row calls its own operands and its own compiled GEMM.
            variants = {"cublas": functools.partial(torch.mm, a, b), "triton": functools.partial(triton_mm, a, b)}
            for name, fn in variants.items():
                fn()
                samples = time_device(fn, iters=20, repeats=args.repeats, warmup=5, cooldown=args.cooldown)
                median = summarize(samples)["median"]
                row = {
                    "variant": name,
                    "dim": dim,
                    "seconds": samples,
                    "tflops_median": 2 * dim**3 / median / 1e12,
                    "state": gpu_state(),
                }
                rows.append(row)
                deferred.append(functools.partial(_record_profiled_kernels, row, fn))
                print(f"  {name:>7} {dim}^3  {row['tflops_median']:7.1f} TFLOP/s", flush=True)
    return rows


def _memory_bound(args) -> list[dict]:
    """Elementwise work on ``[FUSION_DIM, FUSION_DIM]`` bf16 placed on the roofline: FLOPs per element counted by hand
    (approximate), bytes from the tensor passes each op makes, one AdamWBF16 update included."""
    x, y, out, weight_grad = (
        torch.randn(FUSION_DIM, FUSION_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    norm_weight = torch.ones(FUSION_DIM, device="cuda", dtype=torch.bfloat16)
    param = torch.nn.Parameter(torch.randn(FUSION_DIM, FUSION_DIM, device="cuda", dtype=torch.bfloat16))
    param.grad = weight_grad
    optimizer = AdamWBF16([("weight", param)], lr=1e-4)
    # name: (call, FLOPs per element, tensor passes)
    ops = {
        "add": (lambda: torch.add(x, y, out=out), 1, 3),
        "rms_norm": (lambda: F.rms_norm(x, (FUSION_DIM,), norm_weight), 4, 2),
        "silu_mul_eager": (lambda: F.silu(x) * y, 5, 5),
        "silu_mul_liger": (lambda: LigerSiLUMulFunction.apply(x, y), 5, 3),
        # Reads weight, gradient and both moments; writes weight and both moments.
        "adamw_bf16_step": (optimizer.step, 16, 7),
    }
    elements, tensor_bytes = x.numel(), x.numel() * x.element_size()
    rows = []
    for name, (fn, flops_per_element, passes) in ops.items():
        samples = time_device(fn, iters=20, repeats=args.repeats, warmup=5)
        median = summarize(samples)["median"]
        flops, moved = flops_per_element * elements, passes * tensor_bytes
        rows.append(
            {
                "op": name,
                "flops_per_element": flops_per_element,
                "tensor_passes": passes,
                "arithmetic_intensity": flops / moved,
                "seconds": samples,
                "tflops_median": flops / median / 1e12,
                "tb_per_s_median": moved / median / 1e12,
                "state": gpu_state(),
            }
        )
        print(f"  {name:>16}  AI {flops / moved:5.2f}  {median * 1e3:7.3f} ms  {moved / median / 1e12:5.2f} TB/s")
    return rows


def _sustained_step(workload: str) -> tuple[Callable[[], object], int, dict]:
    """One call of a steady workload, the calls per synchronized chunk (about 10 ms of work), and what one call does,
    for converting the call rate into a throughput."""
    if workload == "idle":
        torch.zeros(1, device="cuda")  # Holds a CUDA context, as an idle training process does.
        return (lambda: time.sleep(0.01)), 1, {}
    if workload == "gemm":
        x, w, y = (
            torch.randn(SUSTAINED_GEMM_DIM, SUSTAINED_GEMM_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        return (lambda: torch.mm(x, w, out=y)), 16, {"flops": 2 * SUSTAINED_GEMM_DIM**3}
    if workload == "elementwise":
        numel = 2**31
        a, b, c, out = (torch.ones(numel, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        return (lambda: torch.addcmul(c, a, b, out=out)), 4, {"bytes": 4 * 2 * numel}
    if workload == "silu_mul":
        a, b = (torch.randn(FUSION_DIM, FUSION_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        return (lambda: F.silu(a) * b), 50, {"bytes": 5 * a.numel() * a.element_size()}
    if workload == "tiny":
        x = torch.zeros(1, device="cuda")
        return (lambda: x.add_(1.0)), 2000, {}
    if workload == "expert_loop":
        return _expert_loop()
    m, kn = 128, 2880
    next_set = rotating(_gemm_operands(m, kn, kn, torch.bfloat16, hot_l2=False))

    def small() -> None:
        x, w, y = next_set()
        torch.mm(x, w, out=y)

    return small, 1000, {"flops": 2 * m * kn * kn, "bytes": 2 * (m * kn + kn * kn + m * kn)}


def _sustain(workload: str, seconds: float, warmup_seconds: float, clock_probe: bool) -> dict:
    """Run ``workload`` for ``warmup_seconds`` then ``seconds``, synchronizing every chunk of calls. Every chunk's end
    time and length form the timeline from a cold start; the window's UNIX times bound the power samples taken beside
    it, and its rate counts the window alone. With ``clock_probe``, a one-block kernel spinning ``CLOCK_PROBE_CYCLES``
    SM cycles runs right after each chunk: cycles over its event-timed length is the SM clock the workload left set, which
    NVML reports only coarsely and late. Run beside the workload instead, the probe takes an SM the GEMM's grid needs and
    slows it by 10-20%."""
    step, chunk, per_call = _sustained_step(workload)
    timeline: list[tuple[float, float]] = []
    probes: list[tuple[float, torch.cuda.Event, torch.cuda.Event]] = []
    start = time.time()
    window_start = start + warmup_seconds
    calls = 0
    with ClockSampler(CLOCK_SAMPLE_SECONDS) as clocks:
        while (now := time.time()) - start < warmup_seconds + seconds:
            for _ in range(chunk):
                step()
            torch.cuda.synchronize()
            end = time.time()
            timeline.append((end - start, end - now))
            if now >= window_start:
                calls += chunk
            if clock_probe:
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                torch.cuda._sleep(CLOCK_PROBE_CYCLES)
                finish.record()
                finish.synchronize()
                probes.append((end - start, begin, finish))
    rate = calls / sum(length for at, length in timeline if at - length >= warmup_seconds)
    record = {
        "workload": workload,
        "window_unix": [window_start, start + timeline[-1][0]],
        "calls_per_chunk": chunk,
        "timeline_seconds_and_chunk_length": timeline,
        "nvml_seconds_sm_clock_power": [(at - start, clock, power) for at, clock, power in clocks.samples],
        "probe_seconds_and_sm_mhz": [
            (at, CLOCK_PROBE_CYCLES / (begin.elapsed_time(finish) * 1e3)) for at, begin, finish in probes
        ],
        "calls_in_window": calls,
        "calls_per_s": rate,
        "state_at_end": gpu_state(),
    }
    if "flops" in per_call:
        record["tflops"] = rate * per_call["flops"] / 1e12
    if "bytes" in per_call:
        record["tb_per_s"] = rate * per_call["bytes"] / 1e12
    print(f"  {workload}: {rate:,.0f} calls/s  {record.get('tflops', '')} TFLOP/s  {record.get('tb_per_s', '')} TB/s")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON file, rewritten after each section")
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=SECTIONS)
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per point; each is one sample")
    parser.add_argument("--hot_l2", action="store_true", help="One operand copy per GEMM, left hot in L2")
    parser.add_argument("--repeat_seconds", type=float, default=REPEAT_SECONDS, help="Target length of a GEMM repeat")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="Idle seconds before each GEMM repeat: a burst from cool clocks instead of a sustained, power-capped run",
    )
    parser.add_argument("--peak_dims", type=int, nargs="+", default=(8192, 16384, 32768))
    parser.add_argument("--peak_dtypes", nargs="+", choices=tuple(GEMM_PRECISIONS), default=tuple(GEMM_PRECISIONS))
    parser.add_argument(
        "--sweep_shapes",
        nargs="+",
        default=("2880x2880", "2048x768", "8192x8192"),
        help="KxN weight shapes swept over M (gpt-oss expert, Qwen3-30B-A3B expert projection, wide dense)",
    )
    parser.add_argument(
        "--sweep_m", nargs="+", help="M values: start:stop:step ranges or integers (default log-spaced)"
    )
    parser.add_argument(
        "--kernel_shapes", action="store_true", help="Record each sweep point's kernel and launch grid"
    )
    parser.add_argument("--alignment_base", type=int, default=2880, help="Base K and N of the alignment sweep")
    parser.add_argument("--alignment_m", type=int, default=4096, help="Base M of the alignment sweep")
    parser.add_argument("--alignment_span", type=int, default=16, help="Steps added to each dimension in turn")
    parser.add_argument(
        "--alignment_dtypes", nargs="+", choices=tuple(GEMM_PRECISIONS), default=("bf16", "fp32", "tf32")
    )
    parser.add_argument("--alignment_axes", nargs="+", choices=("k", "n", "m"), default=("k", "n", "m"))
    parser.add_argument("--alignment_forms", nargs="+", choices=GEMM_FORMS, default=("forward",))
    parser.add_argument("--sustain", choices=SUSTAINED, help="Run one steady workload instead of the sections")
    parser.add_argument("--seconds", type=float, default=20.0, help="Measured window of --sustain")
    parser.add_argument("--sustain_warmup", type=float, default=5.0, help="Unmeasured lead-in of --sustain")
    parser.add_argument("--clock_probe", action="store_true", help="Measure the SM clock after each --sustain chunk")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    torch.cuda.set_device(0)
    record = run_record()
    if args.sustain:
        record["sustain"] = _sustain(args.sustain, args.seconds, args.sustain_warmup, args.clock_probe)
        write_record(args.output, record)
        return
    timing = Timing(args.repeats, args.repeat_seconds, args.cooldown)
    deferred: list[Callable[[], None]] = []
    record["sections"] = {}

    def flush() -> None:
        write_record(args.output, record)

    # These append one row per point and rewrite the output after each.
    incremental = {
        "peaks": lambda rows: _peaks(args, timing, rows, flush),
        "sweep": lambda rows: _sweep(args, timing, rows, flush, deferred),
        "alignment": lambda rows: _alignment(args, timing, rows, flush),
    }
    whole = {
        "launch": lambda: _launch(args, deferred),
        "fusion": lambda: _fusion(args, deferred),
        "streams": lambda: _streams(args),
        "triton_gemm": lambda: _triton_gemm(args, deferred),
        "memory_bound": lambda: _memory_bound(args),
    }
    for section in args.sections:
        print(f"\n## {section}", flush=True)
        if section in incremental:
            incremental[section](record["sections"].setdefault(section, []))
        else:
            record["sections"][section] = whole[section]()
        flush()
    # Profiled after every timed section: once a profiling session has run, every later eager launch in the process
    # costs more host time.
    for job in deferred:
        job()
    flush()


if __name__ == "__main__":
    main()
