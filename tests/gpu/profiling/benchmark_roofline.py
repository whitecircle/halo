#!/usr/bin/env python
"""
Benchmark: roofline probes for the GPU Training Theory doc.

Measures the foundational hardware anchors used in
``agent-docs/reference/gpu-training-theory.md`` — on whatever single GPU it runs on,
so the numbers are *achieved*, not spec-sheet peaks:

  1. HBM bandwidth          (large device-to-device copy)
  2. bf16 GEMM peak         (large square matmul -> compute ceiling)
  3. small-M roofline sweep (per-expert M at a fixed K=N expert shape: the
                             memory-bound -> compute-bound crossover)
  4. kernel-launch floor    (latency of a trivial kernel)
  5. elementwise fusion     (SwiGLU traffic; pure-HBM, memory-bound)

The ridge point is ``peak_TFLOPs / peak_TBs`` (FLOP/byte at the knee); any GEMM
whose arithmetic intensity ``AI = 2MNK / bytes`` is below it is bandwidth-bound.

Usage:
    python tests/gpu/profiling/benchmark_roofline.py
    # point the expert sweep at your model's expert shape (K=N=intermediate):
    python tests/gpu/profiling/benchmark_roofline.py --expert_kn 512
    # gpt-oss square expert:
    python tests/gpu/profiling/benchmark_roofline.py --expert_kn 2880
"""

import argparse
import sys
import time
import traceback

import torch


def sync_time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_gemm(M: int, N: int, K: int, w: torch.Tensor, dev) -> tuple[float, float, float]:
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    t = sync_time(lambda: a @ w, iters=30, warmup=10)
    flops = 2 * M * N * K
    bytes_moved = 2 * (M * K + K * N + M * N)  # bf16 read A,W + write C
    return t, flops / t / 1e12, flops / bytes_moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--expert_kn",
        type=int,
        default=2880,
        help="square expert shape K=N for the small-M sweep (default: gpt-oss 2880)",
    )
    parser.add_argument("--peak_kn", type=int, default=8192, help="square dim for the compute-peak GEMM")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required")
        return 1

    try:
        _run_roofline(args)
    except Exception as exc:
        print(f"ERROR: roofline benchmark failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


def _run_roofline(args) -> None:
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    print(
        f"# GPU: {torch.cuda.get_device_name(dev)}  SMs={props.multi_processor_count}  "
        f"mem={props.total_memory / 1e9:.0f}GB  cc={props.major}.{props.minor}"
    )

    # ---- 1. HBM bandwidth ----
    print("\n## HBM bandwidth (device-to-device copy)")
    nbytes = 2 << 30  # 2 GiB
    a = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    t = sync_time(lambda: b.copy_(a), iters=50, warmup=10)
    bw = 2 * nbytes / t / 1e12  # read a + write b
    print(f"  copy {nbytes / 2**30:.0f}GiB: {t * 1e3:.3f} ms  -> {bw:.1f} TB/s")

    # ---- 2. bf16 GEMM compute peak ----
    print(f"\n## bf16 GEMM peak  ({args.peak_kn}^3, compute-bound)")
    P = args.peak_kn
    wp = torch.randn(P, P, dtype=torch.bfloat16, device=dev)
    peak = 0.0
    for M in (P * 2, P, P // 2):
        t, tf, ai = measure_gemm(M, P, P, wp, dev)
        peak = max(peak, tf)
        print(f"  M={M:>6} N={P} K={P}: {t * 1e3:7.3f} ms  {tf:7.0f} TFLOP/s  AI={ai:.0f}")
    print(f"  -> peak ~= {peak:.0f} TFLOP/s ;  ridge AI ~= {peak / bw:.0f} FLOP/byte")

    # ---- 3. small-M roofline sweep at a fixed expert shape ----
    KN = args.expert_kn
    print(f"\n## MoE expert GEMM  K=N={KN}  small-M -> large-M crossover  (ridge ~{peak / bw:.0f})")
    print(f"  {'M(tok/exp)':>10} {'ms':>9} {'TFLOP/s':>9} {'AI':>6} {'regime':>16}")
    we = torch.randn(KN, KN, dtype=torch.bfloat16, device=dev)
    ridge = peak / bw
    for M in (128, 256, 512, 1024, 2048, 4096, 8192):
        t, tf, ai = measure_gemm(M, KN, KN, we, dev)
        regime = "weight-bw-bound" if ai < ridge else "compute-bound"
        print(f"  {M:>10} {t * 1e3:>9.4f} {tf:>9.0f} {ai:>6.0f} {regime:>16}")

    # ---- 4. kernel-launch floor ----
    x = torch.randn(8, device=dev)
    t = sync_time(lambda: x.add_(1.0), iters=2000, warmup=200)
    print(f"\n## kernel-launch floor (tiny add): {t * 1e6:.1f} us/op")

    # ---- 5. elementwise SwiGLU (memory-bound; fusion target) ----
    N = H = 8192
    ea = torch.randn(N, H, dtype=torch.bfloat16, device=dev)
    eb = torch.randn(N, H, dtype=torch.bfloat16, device=dev)
    te = sync_time(lambda: torch.nn.functional.silu(ea) * eb, iters=100, warmup=20)
    traffic = 3 * N * H * 2  # read a, read b, write out (bf16), minimum
    print(
        f"\n## SwiGLU elementwise {N}x{H} bf16: {te * 1e3:.3f} ms  "
        f"-> {traffic / te / 1e12:.1f} TB/s effective (pure HBM, ~0 useful FLOPs)"
    )
    print(f"   (vs HBM peak {bw:.1f} TB/s: sub-peak => extra passes => fusion headroom)")


if __name__ == "__main__":
    sys.exit(main())
