#!/usr/bin/env python
"""
Benchmark: Muon vs AdamW vs AdamWBF16 step time on Qwen3.5-2B.

Loads the real Qwen3.5-2B model in bf16 and compares optimizer step time,
peak memory, and convergence on synthetic data. Reloads from disk per
optimizer to avoid OOM from deep copies.

Usage (single GPU):
    python tests/gpu/optimizers/bench_muon_qwen3_5.py
    python tests/gpu/optimizers/bench_muon_qwen3_5.py --steps 50 --seq-len 512
"""

import argparse
import gc
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.optimizers.adamw_bf16 import AdamWBF16
from src.optimizers.flash_adamw import create_flash_adamw_optimizer
from src.optimizers.muon import create_muon_optimizer
from tests.common.models import QWEN3_5_2B


def load_model(model_name, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="eager",
    )
    model.train()
    return model


def make_batch(batch_size, seq_len, vocab_size, device="cuda"):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return input_ids, labels


def create_optimizer_for(name, model, lr):
    if name == "AdamW (fused)":
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    elif name == "AdamWBF16 (SR)":
        return AdamWBF16(model.parameters(), lr=lr)
    elif name == "Muon (GNS)":
        return create_muon_optimizer(model, lr=lr)
    elif name == "Muon (std NS)":
        return create_muon_optimizer(model, lr=lr, ns_algorithm="standard_newton_schulz")
    elif name == "FlashAdamW":
        return create_flash_adamw_optimizer(model, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def bench_step_time(model, opt, input_ids, labels, warmup, timed):
    """Measure fwd+bwd+step time with GC disabled for stable results.

    Records loss from step 0 (before any warmup) for fair convergence comparison.
    """
    all_losses = []

    # Record initial loss (step 0, no optimizer step yet)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        all_losses.append(out.loss.item())

    # Warmup steps (record losses for convergence curve)
    for _ in range(warmup):
        opt.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        opt.step()
        all_losses.append(out.loss.item())
    torch.cuda.synchronize()

    # Timed steps
    times_full = []
    times_optim = []

    gc.disable()
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        opt.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        times_full.append((t2 - t0) * 1000)
        times_optim.append((t2 - t1) * 1000)
        all_losses.append(out.loss.item())
    gc.enable()

    return {
        "full_ms": sorted(times_full),
        "optim_ms": sorted(times_optim),
        "all_losses": all_losses,
    }


def print_table(headers, rows, col_widths):
    hdr = "".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
    sep = "".join("-" * w for w in col_widths)
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print(f"  {''.join(str(c).ljust(w) for c, w in zip(row, col_widths, strict=False))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=QWEN3_5_2B)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    opt_names = ["AdamW (fused)", "AdamWBF16 (SR)", "Muon (GNS)", "Muon (std NS)", "FlashAdamW"]

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = getattr(config, "vocab_size", None) or config.text_config.vocab_size

    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Model:    {args.model}")
    print(
        f"Config:   batch={args.batch}, seq={args.seq_len}, lr={args.lr}, "
        f"warmup={args.warmup}, timed_steps={args.steps}"
    )
    print()

    results = {}
    memory = {}

    for opt_name in opt_names:
        print(f"--- {opt_name} ---")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Reload model from disk each time to avoid OOM from deep copies
        print("  Loading model...")
        model = load_model(args.model)
        total_params = sum(p.numel() for p in model.parameters())
        sum(p.numel() for p in model.parameters() if p.requires_grad)

        opt = create_optimizer_for(opt_name, model, args.lr)

        torch.manual_seed(42)
        input_ids, labels = make_batch(args.batch, args.seq_len, vocab_size)

        r = bench_step_time(model, opt, input_ids, labels, args.warmup, args.steps)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        results[opt_name] = r
        memory[opt_name] = peak_mb

        med_full = r["full_ms"][len(r["full_ms"]) // 2]
        med_opt = r["optim_ms"][len(r["optim_ms"]) // 2]
        l = r["all_losses"]
        print(
            f"  full step: {med_full:.1f} ms (med)  optim step: {med_opt:.1f} ms (med)  "
            f"peak mem: {peak_mb:.0f} MB  loss: {l[0]:.3f} -> {l[-1]:.3f}"
        )

        del model, opt, input_ids, labels
        gc.collect()
        torch.cuda.empty_cache()

    # ── Tables ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"RESULTS -- {args.model} ({total_params / 1e9:.2f}B params)")
    print("=" * 80)

    # Step time table
    print()
    print("Step Time (ms):")
    rows = []
    for name, r in results.items():
        f = r["full_ms"]
        o = r["optim_ms"]
        rows.append(
            [
                name,
                f"{f[len(f) // 2]:.1f}",
                f"{min(f):.1f}",
                f"{max(f):.1f}",
                f"{o[len(o) // 2]:.1f}",
                f"{min(o):.1f}",
                f"{max(o):.1f}",
            ]
        )
    print_table(
        ["Optimizer", "Full(med)", "Full(min)", "Full(max)", "Opt(med)", "Opt(min)", "Opt(max)"],
        rows,
        [20, 12, 12, 12, 12, 12, 12],
    )

    # Memory table
    print()
    print("Peak Memory:")
    rows = []
    baseline = memory[opt_names[0]]
    for name, peak in memory.items():
        delta = f"{(peak - baseline) / baseline * 100:+.1f}%" if name != opt_names[0] else "-"
        rows.append([name, f"{peak:.0f}", delta])
    print_table(["Optimizer", "Peak MB", "vs AdamW"], rows, [20, 12, 12])

    # Convergence table (from step 0 — same init for all)
    total_steps = args.warmup + args.steps
    print()
    print(f"Loss (from step 0, {total_steps} total steps):")
    rows = []
    for name, r in results.items():
        l = r["all_losses"]
        init = l[0]
        final = l[-1]
        reduction = (init - final) / init * 100 if init > 0 else 0
        rows.append([name, f"{init:.4f}", f"{final:.6f}", f"{reduction:.1f}%"])
    print_table(["Optimizer", "Step 0", "Final", "Reduction"], rows, [20, 12, 12, 12])

    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    main()
