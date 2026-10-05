#!/usr/bin/env python
"""Regenerate the Halo-vs-stock-TRL benchmark charts in agent-docs/assets/benchmarks/.

Writes the two charts ``agent-docs/optimization/halo-vs-stock-trl.md`` embeds: ``throughput_4k16k.png`` and
``memory_4k16k.png``.

Numbers mirror that page's 4k-16k table (gpt-oss-20b, 8x B300, GC-on); edit the dicts here and re-run
to keep the images and the table in step. No GPU needed.

    python scripts/diagrams/gen_trl_benchmark_charts.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from _style_base import assets_dir, save_figure

OUT = assets_dir("benchmarks")

C_TRL = "#999999"
C_EP1Z2 = "#ff7f0e"
C_EP1Z3 = "#d62728"
C_EP2 = "#2ca02c"
C_EP8 = "#1f77b4"


def _save(fig, fname):
    """Write `fname` into agent-docs/assets/benchmarks, creating it on a fresh clone."""
    save_figure(fig, os.path.join(OUT, fname))


def _bar_labels(ax, bars, fmt="{:,.0f}", fontsize=7):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def throughput_memory_4k16k():
    """5-bar grouped charts: TRL z3 / EP1 z2 / EP1 z3 / EP2 z2 / EP8 z2."""
    groups = ["4k·b1", "4k·b2", "4k·b4", "16k·b1", "16k·b2"]
    series = [
        ("stock TRL (ZeRO-3)", C_TRL),
        ("Halo EP1 (ZeRO-2)", C_EP1Z2),
        ("Halo EP1 (ZeRO-3)", C_EP1Z3),
        ("Halo EP2 (ZeRO-2)", C_EP2),
        ("Halo EP8 (ZeRO-2)", C_EP8),
    ]
    tput = {
        "stock TRL (ZeRO-3)": [3836, 5474, 6688, 6461, 7445],
        "Halo EP1 (ZeRO-2)": [11236, 17653, 23590, 20690, 23159],
        "Halo EP1 (ZeRO-3)": [9694, 15532, 21678, 19505, 22321],
        "Halo EP2 (ZeRO-2)": [12432, 17862, 20554, 18436, 19698],
        "Halo EP8 (ZeRO-2)": [10905, 12637, 13239, 12382, 12936],
    }
    mem = {
        "stock TRL (ZeRO-3)": [47.6, 48.2, 50.6, 50.6, 55.6],
        "Halo EP1 (ZeRO-2)": [60.3, 65.5, 75.8, 75.8, 96.5],
        "Halo EP1 (ZeRO-3)": [28.7, 29.3, 37.9, 37.9, 58.6],
        "Halo EP2 (ZeRO-2)": [78.9, 80.5, 85.2, 85.0, 112.0],
        "Halo EP8 (ZeRO-2)": [25.5, 33.2, 48.2, 49.0, 80.3],
    }
    for data, ylabel, title, fname, fmt in [
        (
            tput,
            "tokens/s/GPU",
            "Throughput: Halo vs stock TRL (gpt-oss-20b, GC-on)",
            "throughput_4k16k.png",
            "{:,.0f}",
        ),
        (
            mem,
            "peak memory (GiB)",
            "Peak memory: Halo vs stock TRL (gpt-oss-20b, GC-on)",
            "memory_4k16k.png",
            "{:,.0f}",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.8, 4.62), dpi=100)
        x = np.arange(len(groups))
        w = 0.16
        for i, (name, color) in enumerate(series):
            bars = ax.bar(x + (i - 2) * w, data[name], w, label=name, color=color)
            _bar_labels(ax, bars, fmt=fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, ncol=1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, fname)
        plt.close(fig)


if __name__ == "__main__":
    throughput_memory_4k16k()
    print("wrote throughput_4k16k.png, memory_4k16k.png ->", OUT)
