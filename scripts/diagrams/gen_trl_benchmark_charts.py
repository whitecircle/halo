#!/usr/bin/env python
"""Regenerate the Halo-vs-stock-TRL benchmark charts in agent-docs/assets/benchmarks/.

Writes the charts ``agent-docs/optimization/halo-vs-stock-trl.md`` embeds: ``throughput_4k16k.png``,
``memory_4k16k.png`` and ``long_context.png``. Each group labels only its lead against stock TRL, since the
page's tables carry every value.

Numbers mirror that page's tables (gpt-oss-20b, 8x B300, GC-on); edit the rows here and re-run to keep the
images and the tables in step. No GPU needed.

    python scripts/diagrams/gen_trl_benchmark_charts.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _chart_style import *

BASELINE = "stock TRL (ZeRO-3)"

GROUPS_4K16K = ["4k · b1", "4k · b2", "4k · b4", "16k · b1", "16k · b2"]
# layout -> (tokens/s/GPU per group, peak GiB per group)
SERIES_4K16K = {
    BASELINE: ([3836, 5474, 6688, 6461, 7445], [47.6, 48.2, 50.6, 50.6, 55.6]),
    "Halo EP1 (ZeRO-2)": ([11236, 17653, 23590, 20690, 23159], [60.3, 65.5, 75.8, 75.8, 96.5]),
    "Halo EP1 (ZeRO-3)": ([9694, 15532, 21678, 19505, 22321], [28.7, 29.3, 37.9, 37.9, 58.6]),
    "Halo EP2 (ZeRO-2)": ([12432, 17862, 20554, 18436, 19698], [78.9, 80.5, 85.2, 85.0, 112.0]),
    "Halo EP8 (ZeRO-2)": ([10905, 12637, 13239, 12382, 12936], [25.5, 33.2, 48.2, 49.0, 80.3]),
}

GROUPS_LONG = ["64k", "128k", "256k"]
SERIES_LONG = {
    BASELINE: ([6876, 5961, 4463], [66, 93, 148]),
    "Halo EP1 (ZeRO-3, dense)": ([14654, 9566, 5728], [100, 67, 112]),
    "Halo EP8+CP8": ([5272, 3810, 2395], [32, 47, 77]),
    "Halo dense CP-only (ZeRO-3)": ([4007, 3279, 1836], [29, 32, 56]),
}

BAR_SPAN = 0.8  # of a group's slot, shared by its bars
LEGEND_DROP = -0.09


def grouped_bars(ax, groups, series, column, shades, pick_lead, ratio_format, inside):
    """One bar per layout per group, the baseline in slate; each group labels its lead Halo layout ÷ the baseline.

    `inside` turns the label up the bar, for a lead too short to clear the taller bars beside it.
    """
    x = np.arange(len(groups))
    width = BAR_SPAN / len(series)
    baseline = series[BASELINE][column]
    rows = [(name, values[column]) for name, values in series.items()]
    colors = [SLATE, *shades]
    for i, (name, values) in enumerate(rows):
        ax.bar(x + (i - (len(rows) - 1) / 2) * width, values, width, label=name, color=colors[i])
    for g in x:
        lead, i = pick_lead((values[g], i) for i, (name, values) in enumerate(rows) if name != BASELINE)
        ax.text(
            g + (i - (len(rows) - 1) / 2) * width,
            lead - 0.02 * ax.get_ylim()[1] if inside else lead,
            ratio_format.format(lead / baseline[g]),
            ha="center",
            va="top" if inside else "bottom",
            rotation=90 if inside else 0,
            fontsize=SMALL,
            fontweight="bold",
            color=on_fill(colors[i]) if inside else INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    tidy(ax, "y")


def throughput_axes(ax, groups, series):
    grouped_bars(ax, groups, series, 0, TEAL_SHADES, max, "{:.1f}×", inside=False)
    k_ticks(ax.yaxis)
    ax.set_ylabel("tokens/s/GPU", fontsize=SMALL, color=INK2)


def memory_axes(ax, groups, series):
    grouped_bars(ax, groups, series, 1, AMBER_SHADES, min, "{:.2f}×", inside=True)
    ax.set_ylabel("peak GiB", fontsize=SMALL, color=INK2)


def single_chart(draw, title, sub, name):
    """A one-axes chart of the 4k-16k table, its legend in one row under the axis."""
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    fig.subplots_adjust(top=0.8)
    heading(fig, title, sub)
    draw(ax, GROUPS_4K16K, SERIES_4K16K)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_DROP),
        ncol=len(SERIES_4K16K),
        frameon=False,
        fontsize=TINY,
        columnspacing=1.2,
        handlelength=1.2,
    )
    save(fig, name, BENCHMARKS)


def long_context_chart():
    """Throughput and memory of the 64k-256k tables, side by side with one legend each."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.subplots_adjust(top=0.8, wspace=0.16)
    heading(
        fig,
        "Long context: Halo vs stock TRL",
        "gpt-oss-20b · 8 × B300 · b1 · GC on · Halo 1.0.0 · labels: fastest and lowest-memory Halo layout ÷ stock TRL",
    )
    throughput_axes(left, GROUPS_LONG, SERIES_LONG)
    memory_axes(right, GROUPS_LONG, SERIES_LONG)
    for ax in (left, right):
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, LEGEND_DROP),
            ncol=2,
            frameon=False,
            fontsize=TINY,
            columnspacing=1.2,
            handlelength=1.2,
        )
    save(fig, "long_context", BENCHMARKS)


if __name__ == "__main__":
    single_chart(
        throughput_axes,
        "Throughput: Halo vs stock TRL",
        "gpt-oss-20b · 8 × B300 · GC on · Halo 1.1.0 or earlier · labels: fastest Halo layout ÷ stock TRL",
        "throughput_4k16k",
    )
    single_chart(
        memory_axes,
        "Peak memory: Halo vs stock TRL",
        "gpt-oss-20b · 8 × B300 · GC on · Halo 1.1.0 or earlier · labels: lowest-memory Halo layout ÷ stock TRL",
        "memory_4k16k",
    )
    long_context_chart()
