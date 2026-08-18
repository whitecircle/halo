#!/usr/bin/env python
"""
Benchmark attention implementations on Qwen3-8B: forward + backward on 16k sequences.

Compares performance and memory across attention backends:
  - flash_attention_2 (FlashAttention v2)
  - flash_attention_3 (FlashAttention v3, Hopper+, if installed)
  - flex_attention (PyTorch built-in, torch >=2.5)
  - sdpa (PyTorch scaled_dot_product_attention)
  - eager (no attention optimization)

Each backend is loaded fresh, warmed up, then timed over multiple iterations.
Reports: forward time, backward time, peak memory, throughput.

Usage (1 GPU):
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_attention_implementations.py

    # Custom seq length and iterations
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_attention_implementations.py \
        --seq 16384 --iters 10 --warmup 3

    # Only test specific backends
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_attention_implementations.py \
        --backends flash_attention_2 sdpa eager

    # With gradient checkpointing (saves memory, adds recompute overhead)
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_attention_implementations.py \
        --gradient_checkpointing

    # Different model
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_attention_implementations.py \
        --model qwen3-0.6b --seq 32768
"""

import argparse
import contextlib
import sys
import time
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.models import MODEL_CONFIGS
from tests.common.utils import cleanup_memory, gpu_mem_gb, gpu_peak_mem_gb, log

# Configuration

ALL_BACKENDS = [
    "flash_attention_4",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
    "sdpa",
    "eager",
]

DEFAULT_BACKENDS = [
    "flash_attention_4",
    "flash_attention_2",
    "flex_attention",
    "sdpa",
    "eager",
]


# Helpers


def check_backend_available(backend: str) -> bool:
    """Check if an attention backend is available."""
    if backend == "flash_attention_4":
        # FA4 (Blackwell, CuTe DSL) lives in the flash_attn.cute submodule and
        # co-exists with FA2 under the same flash_attn namespace.
        try:
            import flash_attn.cute

            return True
        except ImportError:
            return False
    elif backend == "flash_attention_3":
        try:
            import flash_attn_3  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend == "flex_attention":
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend == "sdpa":
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")
    elif backend == "eager":
        return True
    return False


def can_load_with_backend(model_name: str, backend: str) -> bool:
    """Check if model config accepts this attention implementation."""
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = backend
        return True
    except (ValueError, AttributeError):
        return False


def create_input_batch(tokenizer, seq_len: int, batch_size: int, device: str):
    """Create a fixed input batch for benchmarking."""
    # Generate token IDs directly (faster than encoding text)
    vocab_size = tokenizer.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def benchmark_single_backend(
    model_name: str,
    backend: str,
    seq_len: int,
    batch_size: int,
    num_iters: int,
    num_warmup: int,
    gradient_checkpointing: bool,
    device: str,
) -> dict:
    """Load model with given backend and benchmark forward + backward."""
    result = {
        "backend": backend,
        "status": "ok",
        "mem_after_load_gb": 0.0,
        "peak_mem_gb": 0.0,
        "fwd_ms": 0.0,
        "bwd_ms": 0.0,
        "total_ms": 0.0,
        "fwd_tokens_per_sec": 0.0,
        "total_tokens_per_sec": 0.0,
    }

    # Check availability
    if not check_backend_available(backend):
        result["status"] = "skipped (not installed)"
        return result

    if not can_load_with_backend(model_name, backend):
        result["status"] = f"skipped (model rejects {backend})"
        return result

    log(f"\n  --- {backend} ---")
    cleanup_memory()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Load model
        log(f"    Loading model with attn_implementation={backend}...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=backend,
        ).to(device)

        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

        model.train()
        load_time = time.perf_counter() - t0
        result["mem_after_load_gb"] = gpu_mem_gb()
        log(f"    Loaded in {load_time:.1f}s, GPU mem: {result['mem_after_load_gb']:.2f} GB")

        # Create input
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        input_ids, attention_mask, labels = create_input_batch(tokenizer, seq_len, batch_size, device)
        total_tokens = batch_size * seq_len

        # Warmup
        log(f"    Warmup ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Reset peak memory after warmup
        torch.cuda.reset_peak_memory_stats()

        # Timed iterations
        log(f"    Benchmarking ({num_iters} iterations)...")
        fwd_times = []
        bwd_times = []

        for _ in range(num_iters):
            torch.cuda.synchronize()

            # Forward
            t_fwd_start = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()

            # Backward
            t_bwd_start = time.perf_counter()
            out.loss.backward()
            torch.cuda.synchronize()
            t_bwd_end = time.perf_counter()

            fwd_times.append((t_fwd_end - t_fwd_start) * 1000)
            bwd_times.append((t_bwd_end - t_bwd_start) * 1000)

            model.zero_grad(set_to_none=True)

        # Compute stats (drop first iteration which may have residual warmup effects)
        drop = min(1, num_iters - 1)
        fwd_times = fwd_times[drop:]
        bwd_times = bwd_times[drop:]

        avg_fwd = sum(fwd_times) / len(fwd_times)
        avg_bwd = sum(bwd_times) / len(bwd_times)
        avg_total = avg_fwd + avg_bwd

        result["fwd_ms"] = avg_fwd
        result["bwd_ms"] = avg_bwd
        result["total_ms"] = avg_total
        result["peak_mem_gb"] = gpu_peak_mem_gb()
        result["fwd_tokens_per_sec"] = total_tokens / (avg_fwd / 1000)
        result["total_tokens_per_sec"] = total_tokens / (avg_total / 1000)

        log(f"    Forward:  {avg_fwd:8.1f} ms  (min={min(fwd_times):.1f}, max={max(fwd_times):.1f})")
        log(f"    Backward: {avg_bwd:8.1f} ms  (min={min(bwd_times):.1f}, max={max(bwd_times):.1f})")
        log(f"    Total:    {avg_total:8.1f} ms")
        log(f"    Peak mem: {result['peak_mem_gb']:.2f} GB")
        log(f"    Fwd throughput:   {result['fwd_tokens_per_sec']:,.0f} tokens/sec")
        log(f"    Total throughput: {result['total_tokens_per_sec']:,.0f} tokens/sec")

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        log("    OUT OF MEMORY")
    except Exception as e:
        result["status"] = f"error: {e}"
        log(f"    ERROR: {e}")
        traceback.print_exc()
    finally:
        # Cleanup model
        if "model" in locals():
            del model
        if "out" in locals():
            del out
        cleanup_memory()

    return result


# Main


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention implementations (forward + backward)")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model config key (default: qwen3-8b)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model path",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=16384,
        help="Sequence length (default: 16384 = 16k)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed iterations per backend (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations per backend (default: 3)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=None,
        help=f"Backends to test (default: {DEFAULT_BACKENDS})",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory, adds recompute overhead)",
    )
    args = parser.parse_args()

    # --- Distributed init (required by torchrun) ---
    rank, world_size, local_rank = init_distributed()
    PartialState()
    device = f"cuda:{local_rank}"

    failed = False
    try:
        succeeded = _run_backends(args, rank, world_size, local_rank, device)
        if not succeeded:
            failed = True
    except Exception as exc:
        failed = True
        log(f"\n  FATAL: attention benchmark crashed: {exc}")
        traceback.print_exc()
    finally:
        # Cleanup
        with contextlib.suppress(Exception):
            dist.barrier()
        teardown_distributed()

    return 1 if failed else 0


def _run_backends(args, rank, world_size, local_rank, device) -> bool:
    """Run the configured backends; return True if at least one succeeded."""
    # --- Resolve model ---
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]

    backends = args.backends or DEFAULT_BACKENDS

    log(f"\n{'=' * 70}")
    log("  Attention Implementation Benchmark")
    log(f"{'=' * 70}")
    log(f"  Model: {model_name} ({full_params / 1e9:.1f}B params)")
    log(f"  Sequence length: {args.seq:,} tokens")
    log(f"  Batch size: {args.batch_size}")
    log(f"  Iterations: {args.iters} (warmup: {args.warmup})")
    log(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    log(f"  Backends to test: {backends}")
    log(f"{'=' * 70}")

    # --- Ensure model is downloaded ---
    log("\nDownloading model (if needed)...")
    if rank == 0:
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dist.barrier()

    # --- Run benchmarks ---
    results = []
    for backend in backends:
        r = benchmark_single_backend(
            model_name=model_name,
            backend=backend,
            seq_len=args.seq,
            batch_size=args.batch_size,
            num_iters=args.iters,
            num_warmup=args.warmup,
            gradient_checkpointing=args.gradient_checkpointing,
            device=device,
        )
        results.append(r)

    # --- Summary table ---
    log(f"\n{'=' * 70}")
    log("  RESULTS SUMMARY")
    log(f"  Model: {model_name} | Seq: {args.seq:,} | Batch: {args.batch_size}")
    log(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    log(f"{'=' * 70}")

    # Header
    log(
        f"  {'Backend':<22} {'Status':<12} {'Fwd (ms)':>10} {'Bwd (ms)':>10} "
        f"{'Total (ms)':>11} {'Peak Mem':>10} {'Tok/s (fwd)':>13} {'Tok/s (total)':>14}"
    )
    log(f"  {'-' * 22} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 11} {'-' * 10} {'-' * 13} {'-' * 14}")

    for r in results:
        if r["status"] == "ok":
            log(
                f"  {r['backend']:<22} {'OK':<12} {r['fwd_ms']:>10.1f} {r['bwd_ms']:>10.1f} "
                f"{r['total_ms']:>11.1f} {r['peak_mem_gb']:>9.2f}G "
                f"{r['fwd_tokens_per_sec']:>13,.0f} {r['total_tokens_per_sec']:>14,.0f}"
            )
        else:
            log(f"  {r['backend']:<22} {r['status']:<12}")

    # --- Relative comparison (vs best) ---
    ok_results = [r for r in results if r["status"] == "ok"]
    if len(ok_results) >= 2:
        best_total = min(r["total_ms"] for r in ok_results)
        best_fwd = min(r["fwd_ms"] for r in ok_results)
        lowest_mem = min(r["peak_mem_gb"] for r in ok_results)

        log("\n  Relative Performance (lower is better for time, higher for throughput):")
        log(f"  {'Backend':<22} {'Fwd vs best':>12} {'Total vs best':>14} {'Mem vs lowest':>14}")
        log(f"  {'-' * 22} {'-' * 12} {'-' * 14} {'-' * 14}")
        for r in ok_results:
            fwd_ratio = r["fwd_ms"] / best_fwd
            total_ratio = r["total_ms"] / best_total
            mem_ratio = r["peak_mem_gb"] / lowest_mem
            log(f"  {r['backend']:<22} {fwd_ratio:>11.2f}x {total_ratio:>13.2f}x {mem_ratio:>13.2f}x")

    log(f"\n{'=' * 70}\n")

    # A run is only successful if at least one backend produced timings;
    # if every backend errored/OOMed the process must exit nonzero.
    succeeded = len(ok_results) > 0
    if not succeeded:
        log("  ERROR: no backend completed successfully")

    return succeeded


if __name__ == "__main__":
    sys.exit(main())
