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


def _bar_labels(ax, bars, fontsize=7):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:,.0f}",
            (b.get_x() + b.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def throughput_memory_4k16k():
    """5-bar grouped charts: TRL z3 / EP1 z2 / EP1 z3 / EP2 z2 / EP8 z2."""
    groups = ["4k·b1", "4k·b2", "4k·b4", "16k·b1", "16k·b2"]
    # (name, color, tokens/s/GPU per group, peak memory GiB per group)
    series = [
        ("stock TRL (ZeRO-3)", "#999999", [3836, 5474, 6688, 6461, 7445], [47.6, 48.2, 50.6, 50.6, 55.6]),
        ("Halo EP1 (ZeRO-2)", "#ff7f0e", [11236, 17653, 23590, 20690, 23159], [60.3, 65.5, 75.8, 75.8, 96.5]),
        ("Halo EP1 (ZeRO-3)", "#d62728", [9694, 15532, 21678, 19505, 22321], [28.7, 29.3, 37.9, 37.9, 58.6]),
        ("Halo EP2 (ZeRO-2)", "#2ca02c", [12432, 17862, 20554, 18436, 19698], [78.9, 80.5, 85.2, 85.0, 112.0]),
        ("Halo EP8 (ZeRO-2)", "#1f77b4", [10905, 12637, 13239, 12382, 12936], [25.5, 33.2, 48.2, 49.0, 80.3]),
    ]
    # (column of ``series`` plotted, y label, title, output file)
    charts = [
        (2, "tokens/s/GPU", "Throughput: Halo vs stock TRL (gpt-oss-20b, GC-on)", "throughput_4k16k.png"),
        (3, "peak memory (GiB)", "Peak memory: Halo vs stock TRL (gpt-oss-20b, GC-on)", "memory_4k16k.png"),
    ]
    for column, ylabel, title, fname in charts:
        fig, ax = plt.subplots(figsize=(8.8, 4.62), dpi=100)
        x = np.arange(len(groups))
        w = 0.16
        for i, row in enumerate(series):
            offset = (i - (len(series) - 1) / 2) * w
            bars = ax.bar(x + offset, row[column], w, label=row[0], color=row[1])
            _bar_labels(ax, bars)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, ncol=1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save_figure(fig, os.path.join(OUT, fname))
        plt.close(fig)


if __name__ == "__main__":
    throughput_memory_4k16k()
    print("wrote throughput_4k16k.png, memory_4k16k.png ->", OUT)
