#!/usr/bin/env python
"""
Benchmark: AdamWBF16 (stochastic rounding) vs adamw_torch_fused on gpt-oss-20b.

Measures optimizer step time, peak GPU memory, and stale weight percentage
on the actual 20.7B MoE model (32 experts, 24 layers).

Single GPU can't fit full 20.91B params + optimizer states (~168 GB needed),
so we freeze the first N layers to keep trainable params within memory.
The optimizer processes the same parameter shapes/sizes as real training.

Usage (single GPU, no torchrun needed):
    CUDA_VISIBLE_DEVICES=4 python tests/gpu/optimizers/bench_adamw_bf16.py
    CUDA_VISIBLE_DEVICES=4 python tests/gpu/optimizers/bench_adamw_bf16.py --freeze-layers 12
"""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM

from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.models import GPT_OSS_20B

MODEL_PATH = GPT_OSS_20B
FREEZE_LAYERS = 8  # Freeze first N layers to fit in single-GPU memory


def set_random_gradients(model):
    """Set random gradients on all trainable parameters (in-place reuse)."""
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.randn_like(p)
            else:
                p.grad.normal_()


def count_stale_subset(model, old_data, param_names):
    """Count stale % on a subset of named parameters."""
    stale = total = 0
    for name, p in model.named_parameters():
        if name in param_names and p.requires_grad:
            stale += (p.data == old_data[name]).sum().item()
            total += p.numel()
    return stale / total if total > 0 else 0


def benchmark_step(model, optimizer, num_warmup=3, num_steps=10):
    """Benchmark optimizer step time (ms)."""
    # Create persistent gradients
    set_random_gradients(model)

    # Initialize optimizer states
    optimizer.step()

    # Warmup (reuse same grads — fine for timing, not for correctness)
    for _ in range(num_warmup):
        optimizer.step()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return times


def stale_check(model, optimizer, sample_layer_idx):
    """Check stale % on a single layer's params (memory-safe)."""
    # Pick params from one layer to snapshot (avoids cloning all 14B params)
    prefix = f"model.layers.{sample_layer_idx}."
    target_names = set()
    old_data = {}
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.requires_grad:
            target_names.add(name)
            old_data[name] = p.data.clone()

    set_random_gradients(model)
    optimizer.step()

    stale_pct = count_stale_subset(model, old_data, target_names)
    del old_data
    return stale_pct, len(target_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--freeze-layers", type=int, default=FREEZE_LAYERS, help="Freeze first N layers to fit in GPU memory"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e9:.2f}B")

    # Freeze first N layers to fit optimizer states in memory
    frozen_count = 0
    for name, p in model.named_parameters():
        for i in range(args.freeze_layers):
            if f"model.layers.{i}." in name:
                p.requires_grad = False
                frozen_count += p.numel()
                break

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen first {args.freeze_layers} layers: {frozen_count / 1e9:.2f}B params")
    print(f"Trainable params: {trainable_params / 1e9:.2f}B")

    # Memory estimate: weights (all, bf16) + grads (trainable, bf16) + states (trainable, 2*bf16)
    est_mem = total_params * 2 + trainable_params * 6
    print(
        f"Estimated memory: {est_mem / 1e9:.1f} GB "
        f"(weights={total_params * 2 / 1e9:.1f} + grads={trainable_params * 2 / 1e9:.1f} "
        f"+ states={trainable_params * 4 / 1e9:.1f})"
    )

    mem_after_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU memory after load: {mem_after_load:.1f} GB")

    # Layer to use for stale % check (first trainable layer)
    sample_layer = args.freeze_layers

    # ─── Benchmark 1: adamw_torch_fused ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Benchmark: adamw_torch_fused (baseline)")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()
    opt_fused = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        fused=True,
    )

    # Stale check on a single layer
    stale_fused, stale_nparams = stale_check(model, opt_fused, sample_layer)

    # Benchmark step time
    times_fused = benchmark_step(model, opt_fused, args.warmup, args.steps)
    mem_fused = torch.cuda.max_memory_allocated() / 1e9

    # Check state dtype
    sample_state = next(iter(opt_fused.state.values()))
    state_dtype = sample_state["exp_avg"].dtype

    median_fused = sorted(times_fused)[len(times_fused) // 2]
    mean_fused = sum(times_fused) / len(times_fused)
    print(f"  Step time (median): {median_fused:.1f} ms")
    print(f"  Step time (mean):   {mean_fused:.1f} ms")
    print(f"  Step time (min/max): {min(times_fused):.1f} / {max(times_fused):.1f} ms")
    print(f"  Peak memory:        {mem_fused:.1f} GB")
    print(f"  State dtype:        {state_dtype}")
    print(f"  Stale % (layer {sample_layer}, {stale_nparams} params): {stale_fused * 100:.1f}%")

    # Cleanup
    del opt_fused, sample_state
    for p in model.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Benchmark 2: AdamWBF16 ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Benchmark: AdamWBF16 (stochastic rounding)")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()
    opt_sr = AdamWBF16(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # Stale check on same layer
    stale_sr, _ = stale_check(model, opt_sr, sample_layer)

    # Benchmark step time
    times_sr = benchmark_step(model, opt_sr, args.warmup, args.steps)
    mem_sr = torch.cuda.max_memory_allocated() / 1e9

    # Check state dtype
    sample_state = next(iter(opt_sr.state.values()))
    state_dtype_sr = sample_state["exp_avg"].dtype

    median_sr = sorted(times_sr)[len(times_sr) // 2]
    mean_sr = sum(times_sr) / len(times_sr)
    print(f"  Step time (median): {median_sr:.1f} ms")
    print(f"  Step time (mean):   {mean_sr:.1f} ms")
    print(f"  Step time (min/max): {min(times_sr):.1f} / {max(times_sr):.1f} ms")
    print(f"  Peak memory:        {mem_sr:.1f} GB")
    print(f"  State dtype:        {state_dtype_sr}")
    print(f"  Stale % (layer {sample_layer}): {stale_sr * 100:.1f}%")

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    overhead_pct = (median_sr - median_fused) / median_fused * 100
    mem_diff = mem_sr - mem_fused

    print(f"  Model:              {args.model}")
    print(f"  Total params:       {total_params / 1e9:.2f}B")
    print(f"  Trainable params:   {trainable_params / 1e9:.2f}B")
    print()
    print(f"  {'Metric':<22} {'Fused':>12} {'AdamWBF16':>12} {'Overhead':>12}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Step time (ms)':<22} {median_fused:>12.1f} {median_sr:>12.1f} {overhead_pct:>+11.1f}%")
    print(f"  {'Peak memory (GB)':<22} {mem_fused:>12.1f} {mem_sr:>12.1f} {mem_diff:>+11.1f} GB")
    print(f"  {'Stale % (layer)':<22} {stale_fused * 100:>11.1f}% {stale_sr * 100:>11.1f}%")
    print()

    # Extrapolate to full model
    full_step_fused = median_fused * (total_params / trainable_params)
    full_step_sr = median_sr * (total_params / trainable_params)
    print("  Extrapolated full-model step time:")
    print(f"    Fused:    ~{full_step_fused:.0f} ms")
    print(f"    AdamWBF16: ~{full_step_sr:.0f} ms")
    print()

    if overhead_pct <= 10:
        print("  RESULT: Excellent - overhead within 10%")
    elif overhead_pct <= 30:
        print("  RESULT: Good - overhead within 30%")
    elif overhead_pct <= 50:
        print("  RESULT: Acceptable - overhead within 50%")
    else:
        print(f"  RESULT: High overhead ({overhead_pct:.0f}%) - optimization needed")

    # ─── Per-step timing detail ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Per-step timing (ms)")
    print(f"{'=' * 60}")
    print(f"  {'Step':<6} {'Fused':>10} {'AdamWBF16':>10} {'Ratio':>8}")
    for i, (tf, ts) in enumerate(zip(times_fused, times_sr, strict=False)):
        print(f"  {i + 1:<6} {tf:>10.1f} {ts:>10.1f} {ts / tf:>7.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    main()
