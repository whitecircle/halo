#!/usr/bin/env python
"""Plot the Gemma 4 26B-A4B full-SFT framework comparison (2x B300) from its result JSONs.

Reads one JSON per framework (the protocol schema: ``cluster_tokens_per_second``,
``peak_mem_allocated_gib``) plus the per-framework expert-path line counts given below, and writes:

- ``<out>/gemma4_sft_pareto.png``: throughput against peak memory per GPU, with the Pareto frontier.
- ``<out>/gemma4_sft_complexity.png``: throughput against the lines of framework code on the Gemma 4
  expert path.
- ``<out>/gemma4_sft_losses.png``: per-step training loss of every framework on identical batches.

Usage: python scripts/analysis/plot_gemma4_framework_benchmarks.py \
    --runs agent-docs/assets/benchmarks/gemma4-sft-2026-09/results --out agent-docs/assets/benchmarks
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# name in the chart -> (result file under --runs, lines of framework code on the expert path, what they are)
FRAMEWORKS = {
    "Halo": ("halo.json", 470, "EP layer, grouped path, packed GeGLU, fused permute"),
    "Axolotl 0.19.0": ("axolotl.json", 130, "transformers grouped_mm experts forward"),
    "NeMo AutoModel": ("automodel.json", 365, "GroupedExpertsDeepEP + compiled weighted GeGLU"),
    "Unsloth 2026.9.11": ("unsloth.json", 320, "forward_native_grouped_mm"),
    "MS-SWIFT 4.5.3": ("ms_swift.json", 130, "transformers grouped_mm experts forward"),
    "Megatron Bridge 0.6.2": ("megatron_062.json", 3100, "Gemma 4 MoE glue + TEGroupedMLP + TE GroupedLinear"),
}


# name in the loss chart -> result file under --runs (the protocol: identical tokens, labels and order).
LOSS_RUNS = {
    "Halo": "halo.json",
    "Halo, round-to-nearest test": "halo_nearest.json",
    "Megatron Bridge 0.6.2": "megatron_062.json",
    "MS-SWIFT 4.5.3": "ms_swift.json",
    "Axolotl 0.19.0": "axolotl.json",
    "NeMo AutoModel": "automodel.json",
    "Unsloth 2026.9.11": "unsloth.json",
}
HALO_COLOR, OTHER_COLORS = "#c2410c", ["#475569", "#0f766e", "#6d28d9", "#0891b2", "#1d4ed8", "#65a30d"]


def plot_losses(runs: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (name, file) in enumerate(LOSS_RUNS.items()):
        path = runs / file
        if not path.exists():
            continue
        losses = json.loads(path.read_text()).get("losses") or []
        halo = name == "Halo"
        color = HALO_COLOR if name.startswith("Halo") else OTHER_COLORS[i % len(OTHER_COLORS)]
        dash = "--" if "test" in name else "-"
        ax.plot(range(1, len(losses) + 1), losses, dash, color=color, lw=2.2 if halo else 1.3, label=name)
    style(ax, "Optimizer step", "Training loss", "Loss on identical batches (canonical tokens, labels, order)")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gemma4_sft_losses.png", dpi=160)


def load(runs: Path) -> list[dict]:
    rows = []
    for name, (file, loc, _) in FRAMEWORKS.items():
        path = runs / file
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "name": name,
                "tok_s": data["cluster_tokens_per_second"],
                "mem": data["peak_mem_allocated_gib"],
                "loc": loc,
            }
        )
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Rows no other row beats on both throughput (higher) and memory (lower)."""
    front = [
        r
        for r in rows
        if not any(
            o["tok_s"] >= r["tok_s"]
            and o["mem"] <= r["mem"]
            and o is not r
            and (o["tok_s"], o["mem"]) != (r["tok_s"], r["mem"])
            for o in rows
        )
    ]
    return sorted(front, key=lambda r: r["mem"])


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def scatter(ax, rows, x_key):
    for r in rows:
        highlight = r["name"].startswith("Halo (")
        ax.scatter(
            r[x_key], r["tok_s"], s=70 if highlight else 45, color="#c2410c" if highlight else "#475569", zorder=3
        )
        ax.annotate(r["name"], (r[x_key], r["tok_s"]), textcoords="offset points", xytext=(6, 4), fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    front = pareto(rows)
    ax.plot([r["mem"] for r in front], [r["tok_s"] for r in front], color="#c2410c", alpha=0.4, lw=1.5, zorder=2)
    scatter(ax, rows, "mem")
    style(ax, "Peak memory allocated per GPU (GiB)", "Cluster tokens/s", "Gemma 4 26B-A4B full SFT, 2x B300, seq 2048")
    fig.tight_layout()
    fig.savefig(args.out / "gemma4_sft_pareto.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    scatter(ax, rows, "loc")
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 5000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(
        ax, "Lines of framework code on the Gemma 4 expert path (log)", "Cluster tokens/s", "Complexity against speed"
    )
    fig.tight_layout()
    fig.savefig(args.out / "gemma4_sft_complexity.png", dpi=160)
    plot_losses(args.runs, args.out)
    for r in rows:
        print(f"{r['name']:24s} {r['tok_s']:8.0f} tok/s {r['mem']:6.1f} GiB {r['loc']:5d} LOC")


if __name__ == "__main__":
    main()
