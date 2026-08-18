#!/usr/bin/env python
"""
Benchmark: F.grouped_mm vs naive expert loop for MoE compute.

Measures throughput, latency, and memory of grouped GEMM kernel vs naive
per-expert matmul loop. This isolates the expert compute kernel cost without
DeepEP dispatch/combine or model overhead.

SM90+ (H100/H200/B200) required for grouped GEMM; the naive loop path
always runs as a baseline.

Usage:
    python tests/gpu/profiling/benchmark_grouped_mm.py
    python tests/gpu/profiling/benchmark_grouped_mm.py --num_experts 32 --hidden 2048
    python tests/gpu/profiling/benchmark_grouped_mm.py --num_experts 128 --hidden 2048 --total_tokens 8192

    # fp32_experts=True scenario (weights stored in fp32, compute in bf16 via cast):
    python tests/gpu/profiling/benchmark_grouped_mm.py --expert_dtype fp32

    # GptOss dimensions (E=32, H=2880, I=2880, T=2048):
    python tests/gpu/profiling/benchmark_grouped_mm.py --gptoss
    python tests/gpu/profiling/benchmark_grouped_mm.py --gptoss --expert_dtype fp32

Configurations matching the docs benchmarks:
    python tests/gpu/profiling/benchmark_grouped_mm.py --num_experts 8 --hidden 2048 --intermediate 4096 --total_tokens 512
    python tests/gpu/profiling/benchmark_grouped_mm.py --num_experts 160 --hidden 1536 --intermediate 4096 --total_tokens 2048
    python tests/gpu/profiling/benchmark_grouped_mm.py --num_experts 128 --hidden 2048 --intermediate 4096 --total_tokens 4096
"""

import argparse
import statistics
import sys
import time

import torch
import torch.nn.functional as F

# Benchmark Functions


def _median_per_iter(step_fn, num_iters: int, warmup: int) -> float:
    """Time ``step_fn`` per iteration and return the MEDIAN seconds/iter.

    Mean-over-total (elapsed / num_iters) is sensitive to a single slow
    iteration — a one-time cuBLAS workspace alloc or a clock-boost transition
    inflates the whole average. The naive expert-loop path (many tiny kernel
    launches) swings 10× run-to-run under the mean estimator.
    Per-iter median is the standard robust microbenchmark estimator: each
    iteration is sync'd and timed individually, and the median rejects the
    cold-start / transient outliers.
    """
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def benchmark_naive_loop(expert_weights, token_lists, num_iters=100, warmup=10):
    """Naive: loop over experts, matmul each separately.

    This mirrors the non-grouped-GEMM path in EP MoE layers where each expert's
    tokens are processed with an individual matmul call (one cuBLAS launch per expert).

    Args:
        expert_weights: List of [H, I] weight tensors, one per expert.
        token_lists: List of [T_e, H] token tensors, one per expert.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds.
    """

    def step():
        for w, tokens in zip(expert_weights, token_lists, strict=False):
            if tokens.shape[0] > 0:
                _ = torch.matmul(tokens, w)

    return _median_per_iter(step, num_iters, warmup)


def benchmark_naive_loop_training(expert_weights, token_lists, num_iters=100, warmup=10):
    """Naive loop with forward + backward pass (training mode).

    Args:
        expert_weights: List of [H, I] weight tensors with requires_grad=True.
        token_lists: List of [T_e, H] token tensors with requires_grad=True.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds.
    """

    def step():
        for w, tokens in zip(expert_weights, token_lists, strict=False):
            if tokens.shape[0] > 0:
                out = torch.matmul(tokens, w)
                out.sum().backward()

    return _median_per_iter(step, num_iters, warmup)


def benchmark_grouped_mm(stacked_weights, all_tokens, offsets, num_iters=100, warmup=10):
    """Grouped MM: single F.grouped_mm kernel call.

    This mirrors the grouped GEMM path in EP MoE layers where all expert
    matmuls are batched into a single kernel launch using cumulative offsets.

    The `offs` parameter uses cumulative token counts (matching the EP MoE layer
    implementation in ep_moe_layer.py::_sort_tokens_for_grouped_mm).

    Args:
        stacked_weights: [E, H, I] stacked expert weights.
        all_tokens: [T_total, H] concatenated token tensor (sorted by expert).
        offsets: [E] int32 cumulative token counts per expert.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds, or None if grouped_mm unavailable.
    """
    if not hasattr(F, "grouped_mm"):
        return None

    def step():
        _ = F.grouped_mm(all_tokens, stacked_weights, offs=offsets)

    return _median_per_iter(step, num_iters, warmup)


def benchmark_grouped_mm_training(stacked_weights, all_tokens, offsets, num_iters=100, warmup=10):
    """Grouped MM with forward + backward pass (training mode).

    Args:
        stacked_weights: [E, H, I] stacked expert weights with requires_grad=True.
        all_tokens: [T_total, H] concatenated token tensor with requires_grad=True.
        offsets: [E] int32 cumulative token counts per expert.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds, or None if grouped_mm unavailable.
    """
    if not hasattr(F, "grouped_mm"):
        return None

    try:

        def step():
            out = F.grouped_mm(all_tokens, stacked_weights, offs=offsets)
            out.clone().sum().backward()

        return _median_per_iter(step, num_iters, warmup)
    except RuntimeError:
        # grouped_mm backward has stride issues in some PyTorch versions;
        # fall back to extrapolating from forward-only measurements.
        return None


def measure_memory(fn, *fn_args):
    """Run fn once and return peak memory allocated in GB.

    Resets peak memory stats before and after the call to isolate
    the memory footprint of the operation itself.

    Args:
        fn: Benchmark function to call.
        *fn_args: Arguments forwarded to fn (with num_iters=1).

    Returns:
        Peak memory allocated in GB during the call.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*fn_args, num_iters=1)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def benchmark_naive_loop_fp32_cast(expert_weights_fp32, token_lists, num_iters=100, warmup=10):
    """Naive loop where fp32 expert weights are cast to bf16 before each matmul.

    Mirrors the fp32_experts=True loop path in EPMoELayerBase._compute_experts_weighted:
    weights stored as fp32 (precision), compute runs in bf16 (via autocast or explicit cast).

    Args:
        expert_weights_fp32: List of [H, I] fp32 weight tensors (fp32_experts storage).
        token_lists: List of [T_e, H] bf16 token tensors.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds.
    """
    for _ in range(warmup):
        for w, tokens in zip(expert_weights_fp32, token_lists, strict=False):
            if tokens.shape[0] > 0:
                torch.matmul(tokens, w.to(torch.bfloat16))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        results = []
        for w, tokens in zip(expert_weights_fp32, token_lists, strict=False):
            if tokens.shape[0] > 0:
                results.append(torch.matmul(tokens, w.to(torch.bfloat16)))
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_grouped_mm_fp32_cast(stacked_weights_fp32, all_tokens_bf16, offsets, num_iters=100, warmup=10):
    """Grouped MM where fp32 expert weights are cast to bf16 before F.grouped_mm.

    Mirrors the fp32_experts=True grouped_mm path in _compute_experts_with_grouped_mm:
    weights are cast `.to(output_dtype)` before each grouped_mm call (bf16 tokens).

    F.grouped_mm requires matching dtypes — passing fp32 weights with bf16 tokens raises
    "expected mat1 and mat2 to have the same dtype". The cast is the mandated workaround.

    Args:
        stacked_weights_fp32: [E, H, I] fp32 stacked expert weights.
        all_tokens_bf16: [T_total, H] bf16 concatenated token tensor.
        offsets: [E] int32 cumulative token counts per expert.
        num_iters: Number of timed iterations.

    Returns:
        Average time per iteration in seconds, or None if grouped_mm unavailable.
    """
    if not hasattr(F, "grouped_mm"):
        return None

    weights_bf16 = stacked_weights_fp32.to(torch.bfloat16)  # cast once for warmup
    for _ in range(warmup):
        weights_bf16 = stacked_weights_fp32.to(torch.bfloat16)
        F.grouped_mm(all_tokens_bf16, weights_bf16, offs=offsets)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        # Re-cast each iteration to match the actual code path (no weight caching)
        weights_bf16 = stacked_weights_fp32.to(torch.bfloat16)
        _ = F.grouped_mm(all_tokens_bf16, weights_bf16, offs=offsets)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def compute_tflops(total_tokens, hidden, intermediate, elapsed_sec):
    """Compute achieved TFLOPS for a batched matmul benchmark.

    FLOPS per matmul = 2 * M * K * N (multiply-accumulate).
    Total FLOPS = 2 * total_tokens * H * I (total_tokens sums across all experts).

    Args:
        total_tokens: Total number of tokens across all experts.
        hidden: Hidden dimension (K).
        intermediate: Intermediate dimension (N).
        elapsed_sec: Time in seconds.

    Returns:
        Achieved TFLOPS (decimal).
    """
    flops = 2.0 * total_tokens * hidden * intermediate
    return flops / elapsed_sec / 1e12


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark F.grouped_mm vs naive expert loop for MoE compute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts (default: 32)")
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden dimension H (default: 2048)")
    parser.add_argument("--intermediate", type=int, default=4096, help="Intermediate dimension I (default: 4096)")
    parser.add_argument(
        "--total_tokens", type=int, default=4096, help="Total tokens distributed across experts (default: 4096)"
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument(
        "--expert_dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Expert weight dtype: bf16 (native) or fp32 (fp32_experts=True scenario). "
        "When fp32, benchmarks weight cast overhead and compares fp32-cast-to-bf16 "
        "vs native bf16 compute. (default: bf16)",
    )
    parser.add_argument(
        "--gptoss",
        action="store_true",
        help="Use GptOss dimensions (E=32, H=2880, I=2880, T=2048) overriding other args.",
    )
    args = parser.parse_args()

    if args.gptoss:
        args.num_experts = 32
        args.hidden = 2880
        args.intermediate = 2880
        args.total_tokens = 2048

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return 1

    device = "cuda"
    dtype = torch.bfloat16

    # --- GPU info ---
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    total_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("=" * 70)
    print("  Grouped GEMM Benchmark: F.grouped_mm vs Naive Expert Loop")
    print("=" * 70)
    print(f"  GPU:            {gpu_name} (SM{compute_cap[0]}{compute_cap[1]}, {total_gpu_mem_gb:.0f} GB)")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  Experts:        {args.num_experts}")
    print(f"  Hidden:         {args.hidden}")
    print(f"  Intermediate:   {args.intermediate}")
    print(f"  Total tokens:   {args.total_tokens}")
    print(f"  Tokens/expert:  {args.total_tokens // args.num_experts}")
    print(f"  Dtype:          {dtype}")
    print(f"  Expert dtype:   {args.expert_dtype}")
    print(f"  Iterations:     {args.iters}")

    if compute_cap[0] < 9:
        print("\n  WARNING: SM90+ (H100/H200/B200) required for torch.nn.functional.grouped_mm.")
        print(f"           Current GPU is SM{compute_cap[0]}{compute_cap[1]}. Grouped GEMM will be skipped.")

    has_grouped_mm = hasattr(F, "grouped_mm")
    print(f"  grouped_mm:     {'available' if has_grouped_mm else 'NOT available'}")
    print("=" * 70)

    # --- Create tensors ---
    # Expert weights: [E, H, I] in matmul convention (matches EP MoE layer layout)
    expert_weights_list = [
        torch.randn(args.hidden, args.intermediate, device=device, dtype=dtype) for _ in range(args.num_experts)
    ]
    stacked_weights = torch.stack(expert_weights_list)  # [E, H, I]

    # Distribute tokens uniformly across experts
    tokens_per_expert = args.total_tokens // args.num_experts
    remainder = args.total_tokens - tokens_per_expert * args.num_experts

    token_lists = []
    for i in range(args.num_experts):
        t = tokens_per_expert + (1 if i < remainder else 0)
        token_lists.append(torch.randn(t, args.hidden, device=device, dtype=dtype))

    all_tokens = torch.cat(token_lists, dim=0)  # [T, H]

    # Create cumulative offsets (matching ep_moe_layer.py::_sort_tokens_for_grouped_mm)
    expert_counts = torch.tensor([tl.shape[0] for tl in token_lists], device=device, dtype=torch.int32)
    offsets = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    # --- Forward pass benchmark ---
    print(f"\nBenchmarking forward pass ({args.iters} iterations)...")

    naive_time = benchmark_naive_loop(expert_weights_list, token_lists, args.iters, args.warmup)
    grouped_time = benchmark_grouped_mm(stacked_weights, all_tokens, offsets, args.iters, args.warmup)

    # --- Training (forward + backward) benchmark ---
    print(f"Benchmarking training (forward + backward, {args.iters} iterations)...")

    # Create grad-enabled copies for training benchmarks
    train_weights_list = [w.clone().requires_grad_(True) for w in expert_weights_list]
    train_token_lists = [t.clone().requires_grad_(True) for t in token_lists]
    train_stacked = stacked_weights.clone().requires_grad_(True)
    train_all_tokens = all_tokens.clone().requires_grad_(True)

    naive_train_time = benchmark_naive_loop_training(
        train_weights_list,
        train_token_lists,
        args.iters,
        args.warmup,
    )
    grouped_train_time = benchmark_grouped_mm_training(
        train_stacked,
        train_all_tokens,
        offsets,
        args.iters,
        args.warmup,
    )
    if grouped_train_time is None and grouped_time is not None:
        # Estimate training time from forward speedup ratio (backward scales similarly)
        fwd_ratio = grouped_time / naive_time
        grouped_train_time = naive_train_time * fwd_ratio
        grouped_train_estimated = True
        print("  (grouped_mm backward unavailable — training time estimated from forward ratio)")
    else:
        grouped_train_estimated = False

    # --- Benchmark memory (forward only) ---
    print("Benchmarking forward memory...")
    naive_mem = measure_memory(benchmark_naive_loop, expert_weights_list, token_lists)

    grouped_mem = None
    if grouped_time is not None:
        grouped_mem = measure_memory(benchmark_grouped_mm, stacked_weights, all_tokens, offsets)

    # --- Benchmark training memory ---
    print("Benchmarking training memory...")
    naive_train_mem = measure_memory(
        benchmark_naive_loop_training,
        train_weights_list,
        train_token_lists,
    )

    # Estimate training memory from forward memory ratio if backward is unavailable
    grouped_train_mem = None
    if grouped_train_time is not None and not grouped_train_estimated:
        grouped_train_mem = measure_memory(
            benchmark_grouped_mm_training,
            train_stacked,
            train_all_tokens,
            offsets,
        )
    elif grouped_mem is not None:
        mem_ratio = grouped_mem / naive_mem if naive_mem > 0 else 1.0
        grouped_train_mem = naive_train_mem * mem_ratio
        grouped_train_estimated = True

    # --- Compute TFLOPS ---
    naive_tflops = compute_tflops(args.total_tokens, args.hidden, args.intermediate, naive_time)
    grouped_tflops = None
    if grouped_time is not None:
        grouped_tflops = compute_tflops(args.total_tokens, args.hidden, args.intermediate, grouped_time)

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("  Results")
    print(f"{'=' * 70}")
    print(
        f"  Configuration: {args.num_experts} experts, hidden={args.hidden}, "
        f"intermediate={args.intermediate}, tokens={args.total_tokens}"
    )
    print()

    # Forward latency
    print("  Forward Pass Latency:")
    print(f"    Naive loop:     {naive_time * 1000:8.2f} ms")
    if grouped_time is not None:
        print(f"    Grouped MM:     {grouped_time * 1000:8.2f} ms")
        speedup = naive_time / grouped_time
        print(f"    Speedup:        {speedup:8.2f}x")
    else:
        print("    Grouped MM:     NOT AVAILABLE (F.grouped_mm not found)")

    # Training latency
    print()
    print("  Training (Forward + Backward) Latency:")
    print(f"    Naive loop:     {naive_train_time * 1000:8.2f} ms")
    if grouped_train_time is not None:
        print(f"    Grouped MM:     {grouped_train_time * 1000:8.2f} ms")
        train_speedup = naive_train_time / grouped_train_time
        print(f"    Speedup:        {train_speedup:8.2f}x")

    # TFLOPS
    print()
    print("  Achieved TFLOPS (forward):")
    print(f"    Naive loop:     {naive_tflops:8.2f} TFLOPS")
    if grouped_tflops is not None:
        print(f"    Grouped MM:     {grouped_tflops:8.2f} TFLOPS")

    # Memory
    print()
    print("  Peak Memory (forward):")
    print(f"    Naive loop:     {naive_mem * 1000:8.1f} MB")
    if grouped_mem is not None:
        print(f"    Grouped MM:     {grouped_mem * 1000:8.1f} MB")
        mem_savings = (1.0 - grouped_mem / naive_mem) * 100 if naive_mem > 0 else 0
        print(f"    Reduction:      {mem_savings:7.1f}%")

    print()
    print("  Peak Memory (training):")
    print(f"    Naive loop:     {naive_train_mem * 1000:8.1f} MB")
    if grouped_train_mem is not None:
        print(f"    Grouped MM:     {grouped_train_mem * 1000:8.1f} MB")
        train_mem_savings = (1.0 - grouped_train_mem / naive_train_mem) * 100 if naive_train_mem > 0 else 0
        print(f"    Reduction:      {train_mem_savings:7.1f}%")

    print(f"{'=' * 70}")

    # --- fp32_experts scenario ---
    if args.expert_dtype == "fp32":
        _run_fp32_experts_benchmark(args, device, offsets, all_tokens, token_lists)

    return 0


def _run_fp32_experts_benchmark(args, device, offsets, all_tokens_bf16, token_lists_bf16):
    """Benchmark the fp32_experts=True scenario for grouped GEMM.

    Tests three modes that arise when ep_config.fp32_experts=True:
    1. fp32 weights cast to bf16 per-expert in the loop path (baseline loop)
    2. fp32 weights cast to bf16 as a batch before grouped_mm (actual code path)
    3. fp32 full (tokens + weights both fp32) — for completeness / debugging

    F.grouped_mm requires matching dtypes: bf16×fp32 raises RuntimeError immediately.
    The cast-to-bf16 approach in _compute_experts_with_grouped_mm is the mandated
    workaround. This section quantifies the cast overhead vs native bf16 grouped_mm.
    """
    E = args.num_experts
    H = args.hidden
    I = args.intermediate

    print(f"\n{'=' * 70}")
    print("  fp32 Experts Scenario (fp32_experts=True in EPConfig)")
    print(f"{'=' * 70}")
    print("  Weights stored as fp32; compute uses bf16 via .to(output_dtype) cast")
    print()

    # fp32 storage tensors
    expert_weights_fp32 = [torch.randn(H, I, device=device, dtype=torch.float32) for _ in range(E)]
    stacked_fp32 = torch.stack(expert_weights_fp32)  # [E, H, I]

    # Native bf16 tensors for comparison (same values, different dtype)
    stacked_bf16 = stacked_fp32.to(torch.bfloat16)
    expert_weights_bf16 = [stacked_bf16[i] for i in range(E)]

    # Dtype mismatch check (the bug that use_grouped_gemm=False was introduced for)
    print("  Dtype safety check:")
    print("    F.grouped_mm(bf16_tokens, fp32_weights): ", end="", flush=True)
    try:
        F.grouped_mm(all_tokens_bf16, stacked_fp32, offs=offsets)
        print("OK (unexpected)")
    except RuntimeError:
        print("RuntimeError — dtype mismatch (expected, use .to(output_dtype))")

    print()
    print("  Cast overhead (fp32->[E,H,I] .to(bf16)):")
    for _ in range(args.warmup):
        stacked_fp32.to(torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        stacked_fp32.to(torch.bfloat16)
        torch.cuda.synchronize()
    cast_time = (time.perf_counter() - t0) / args.iters
    weight_mb = stacked_fp32.nbytes / 1e6
    print(f"    Weight tensor:  {weight_mb:.0f} MB (fp32) → {weight_mb / 2:.0f} MB (bf16 copy)")
    print(f"    Cast latency:   {cast_time * 1000:.3f} ms per forward pass")

    # Benchmark loop path: fp32 weights cast per-expert (loop) vs native bf16
    print()
    print("  Forward Pass Latency (loop path):")
    loop_fp32_cast_time = benchmark_naive_loop_fp32_cast(
        expert_weights_fp32,
        token_lists_bf16,
        args.iters,
        args.warmup,
    )
    loop_bf16_time = benchmark_naive_loop(
        expert_weights_bf16,
        token_lists_bf16,
        args.iters,
        args.warmup,
    )
    print(f"    bf16 weights (native):      {loop_bf16_time * 1000:8.3f} ms")
    print(
        f"    fp32 weights (cast+matmul): {loop_fp32_cast_time * 1000:8.3f} ms  "
        f"(+{(loop_fp32_cast_time - loop_bf16_time) * 1000:.3f} ms cast overhead)"
    )

    # Benchmark grouped_mm path: fp32 cast to bf16 once vs native bf16
    if hasattr(F, "grouped_mm"):
        print()
        print("  Forward Pass Latency (grouped MM path):")
        gmm_bf16_time = benchmark_grouped_mm(
            stacked_bf16,
            all_tokens_bf16,
            offsets,
            args.iters,
            args.warmup,
        )
        gmm_fp32_cast_time = benchmark_grouped_mm_fp32_cast(
            stacked_fp32,
            all_tokens_bf16,
            offsets,
            args.iters,
            args.warmup,
        )
        gmm_overhead = (gmm_fp32_cast_time - gmm_bf16_time) * 1000
        print(f"    bf16 weights (native):      {gmm_bf16_time * 1000:8.3f} ms")
        print(
            f"    fp32 weights (cast+gmm):    {gmm_fp32_cast_time * 1000:8.3f} ms  "
            f"(+{gmm_overhead:.3f} ms cast overhead)"
        )
        print(f"    Cast dominates kernel:      {'YES' if cast_time > gmm_bf16_time else 'no'}")

    # fp32 full: both tokens and weights in fp32
    print()
    print("  fp32 Full (tokens + weights fp32) — reference only:")
    tokens_fp32 = all_tokens_bf16.float()
    tok_lists_fp32 = [t.float() for t in token_lists_bf16]
    loop_fp32_full = benchmark_naive_loop(
        expert_weights_fp32,
        tok_lists_fp32,
        args.iters,
        args.warmup,
    )
    print(f"    Loop (fp32 full):           {loop_fp32_full * 1000:8.3f} ms")
    if hasattr(F, "grouped_mm"):
        gmm_fp32_full = benchmark_grouped_mm(
            stacked_fp32,
            tokens_fp32,
            offsets,
            args.iters,
            args.warmup,
        )
        speedup_fp32 = loop_fp32_full / gmm_fp32_full
        print(
            f"    Grouped MM (fp32 full):     {gmm_fp32_full * 1000:8.3f} ms  "
            f"({speedup_fp32:.1f}x — grouped_mm has no TensorCore advantage for fp32)"
        )

    # Memory overhead of the cast
    print()
    print("  Peak Memory (forward, grouped_mm path):")
    if hasattr(F, "grouped_mm"):
        mem_bf16 = measure_memory(benchmark_grouped_mm, stacked_bf16, all_tokens_bf16, offsets)
        mem_fp32_cast = measure_memory(
            benchmark_grouped_mm_fp32_cast,
            stacked_fp32,
            all_tokens_bf16,
            offsets,
        )
        print(f"    bf16 weights (native):      {mem_bf16 * 1000:8.1f} MB")
        print(
            f"    fp32 weights (cast+gmm):    {mem_fp32_cast * 1000:8.1f} MB  "
            f"(+{(mem_fp32_cast - mem_bf16) * 1000:.1f} MB temporary bf16 weight copies)"
        )

    print(f"{'=' * 70}")


if __name__ == "__main__":
    sys.exit(main())
