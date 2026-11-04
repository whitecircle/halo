#!/usr/bin/env python
"""Muon vs AdamW vs AdamWBF16 vs FlashAdamW: convergence, step time and peak GPU memory. Runs without torchrun."""

import argparse
import copy
import gc
import time

import torch
import torch.nn as nn

from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.flash_adamw import build_flash_adamw_optimizer
from halo.optimizers.muon import build_muon_optimizer


class TransformerFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden),
                        "gate": nn.Linear(hidden, intermediate, bias=False),
                        "up": nn.Linear(hidden, intermediate, bias=False),
                        "down": nn.Linear(intermediate, hidden, bias=False),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            h = layer["norm"](x)
            x = x + layer["down"](nn.functional.silu(layer["gate"](h)) * layer["up"](h))
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def create_model(hidden, intermediate, num_layers, device="cuda"):
    torch.manual_seed(42)
    model = TransformerFFN(hidden, intermediate, num_layers)
    return model.to(device=device, dtype=torch.bfloat16)


def make_optimizers(model, lr):
    torch.manual_seed(42)
    optimizers = {}

    m = create_model_copy(model)
    optimizers["AdamW (fused)"] = (m, torch.optim.AdamW(m.parameters(), lr=lr, fused=True))

    m = create_model_copy(model)
    optimizers["AdamWBF16 (SR)"] = (m, AdamWBF16(m.named_parameters(), lr=lr))

    m = create_model_copy(model)
    optimizers["Muon (GNS)"] = (m, build_muon_optimizer(m, lr=lr))

    m = create_model_copy(model)
    optimizers["Muon (std NS)"] = (m, build_muon_optimizer(m, lr=lr, ns_algorithm="standard_newton_schulz"))

    # Quantized states + 24-bit master weights.
    m = create_model_copy(model)
    optimizers["FlashAdamW"] = (m, build_flash_adamw_optimizer(m, lr=lr))

    return optimizers


def create_model_copy(model):
    return copy.deepcopy(model)


def bench_convergence(optimizers, hidden, batch, seq_len, num_steps):
    device = "cuda"
    torch.manual_seed(123)
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)
    y = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)

    results = {}
    for name, (model, opt) in optimizers.items():
        losses = []
        for step in range(num_steps):
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.mse_loss(out.float(), y.float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        results[name] = losses
        for p in model.parameters():
            p.grad = None

    return results


def bench_step_time(optimizers, hidden, batch, seq_len, warmup=5, timed=20):
    device = "cuda"

    results = {}
    for name, (model, opt) in optimizers.items():
        torch.manual_seed(0)
        x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)
        y = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)

        for _ in range(warmup):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x).float(), y.float())
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        times_full = []
        times_optim = []
        for _ in range(timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x).float(), y.float())
            loss.backward()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            opt.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            times_full.append((t2 - t0) * 1000)
            times_optim.append((t2 - t1) * 1000)

        results[name] = {
            "full_ms": sorted(times_full),
            "optim_ms": sorted(times_optim),
        }

        for p in model.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()

    return results


def bench_memory(model_template, hidden, intermediate, num_layers, lr, batch, seq_len):
    results = {}

    for opt_name in ["AdamW (fused)", "AdamWBF16 (SR)", "Muon (GNS)", "FlashAdamW"]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = create_model(hidden, intermediate, num_layers)
        if opt_name == "AdamW (fused)":
            opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=True)
        elif opt_name == "AdamWBF16 (SR)":
            opt = AdamWBF16(m.named_parameters(), lr=lr)
        elif opt_name == "FlashAdamW":
            opt = build_flash_adamw_optimizer(m, lr=lr)
        else:
            opt = build_muon_optimizer(m, lr=lr)

        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
        y = torch.randn_like(x)

        # 3 steps to fully initialize states.
        for _ in range(3):
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(x).float(), y.float())
            loss.backward()
            opt.step()

        peak = torch.cuda.max_memory_allocated() / 1e6
        results[opt_name] = peak

        del m, opt, x, y, loss
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths, strict=False))
    sep_line = "".join("-" * w for w in col_widths)

    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in rows:
        line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths, strict=False))
        print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Muon vs AdamW optimizers")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=None, help="FFN intermediate size (default: 4*hidden)")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    args = parser.parse_args()

    if args.intermediate is None:
        args.intermediate = 4 * args.hidden

    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:    {torch.__version__}")
    ref_model = create_model(args.hidden, args.intermediate, args.layers)
    total_params = ref_model.param_count()
    print(f"Model:      {args.layers}L x {args.hidden}h x {args.intermediate}i = {total_params / 1e6:.1f}M params")
    print(f"Data:       batch={args.batch}, seq={args.seq_len}, steps={args.steps}, lr={args.lr}")
    print()

    print("=" * 70)
    print("CONVERGENCE BENCHMARK")
    print("=" * 70)

    optimizers = make_optimizers(ref_model, lr=args.lr)
    convergence = bench_convergence(optimizers, args.hidden, args.batch, args.seq_len, args.steps)

    rows = []
    for name, losses in convergence.items():
        init_loss = losses[0]
        final_loss = losses[-1]
        reduction = (init_loss - final_loss) / init_loss * 100
        min_loss = min(losses)
        min_step = losses.index(min_loss) + 1
        rows.append(
            [
                name,
                f"{init_loss:.4f}",
                f"{final_loss:.4f}",
                f"{reduction:.1f}%",
                f"{min_loss:.4f}",
                f"@{min_step}",
            ]
        )

    print()
    print_table(
        ["Optimizer", "Init Loss", "Final Loss", "Reduction", "Best Loss", "Best Step"],
        rows,
        col_widths=[20, 12, 12, 12, 12, 12],
    )

    print()
    print("=" * 70)
    print("STEP TIME BENCHMARK")
    print("=" * 70)

    # The convergence run consumed them.
    del optimizers
    gc.collect()
    torch.cuda.empty_cache()
    optimizers = make_optimizers(ref_model, lr=args.lr)
    timing = bench_step_time(optimizers, args.hidden, args.batch, args.seq_len, args.warmup, args.timed)

    rows = []
    for name, t in timing.items():
        full = t["full_ms"]
        optim = t["optim_ms"]
        median_full = full[len(full) // 2]
        median_optim = optim[len(optim) // 2]
        rows.append(
            [
                name,
                f"{median_full:.1f}",
                f"{min(full):.1f}",
                f"{max(full):.1f}",
                f"{median_optim:.1f}",
                f"{min(optim):.1f}",
                f"{max(optim):.1f}",
            ]
        )

    print()
    print_table(
        ["Optimizer", "Full(med)", "Full(min)", "Full(max)", "Opt(med)", "Opt(min)", "Opt(max)"],
        rows,
        col_widths=[20, 12, 12, 12, 12, 12, 12],
    )
    print("  (all times in ms)")

    print()
    print("=" * 70)
    print("PEAK MEMORY BENCHMARK")
    print("=" * 70)

    del optimizers
    gc.collect()
    torch.cuda.empty_cache()
    memory = bench_memory(ref_model, args.hidden, args.intermediate, args.layers, args.lr, args.batch, args.seq_len)

    rows = []
    baseline = None
    for name, peak_mb in memory.items():
        if baseline is None:
            baseline = peak_mb
            overhead = "-"
        else:
            overhead = f"{(peak_mb - baseline) / baseline * 100:+.1f}%"
        rows.append([name, f"{peak_mb:.0f}", overhead])

    print()
    print_table(
        ["Optimizer", "Peak MB", "vs AdamW"],
        rows,
        col_widths=[20, 12, 12],
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_name = min(convergence, key=lambda n: convergence[n][-1])
    best_loss = convergence[best_name][-1]
    print(f"  Best final loss:   {best_name} ({best_loss:.4f})")

    fastest_name = min(timing, key=lambda n: timing[n]["full_ms"][len(timing[n]["full_ms"]) // 2])
    fastest_ms = timing[fastest_name]["full_ms"][len(timing[fastest_name]["full_ms"]) // 2]
    print(f"  Fastest full step: {fastest_name} ({fastest_ms:.1f} ms)")

    lowest_mem_name = min(memory, key=lambda n: memory[n])
    print(f"  Lowest memory:     {lowest_mem_name} ({memory[lowest_mem_name]:.0f} MB)")

    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    main()
