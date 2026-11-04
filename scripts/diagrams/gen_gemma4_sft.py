"""Generate the Gemma 4 framework comparison for optimization/throughput-benchmarks.md.

Reads the per-run summaries the page cites; each framework's point is the mean of its valid runs, as the page's
tables state. Only Halo's point carries a value, its lead over the next framework, since the tables hold the rest.
"""

import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
from _chart_style import *

SUMMARIES = os.path.join(assets_dir(BENCHMARKS), "gemma4-sft-2026-09", "summary_{seq}.tsv")
NAMES = {
    "halo": "Halo",
    "axolotl": "Axolotl",
    "automodel": "NeMo AutoModel",
    "unsloth": "Unsloth",
    "megatron_bridge": "Megatron Bridge",
    "ms_swift": "MS-SWIFT",
}
# Label placement per panel, in points from the marker: framework -> (dx, dy, ha).
LABEL_AT = {
    2048: {
        "automodel": (9, -2, "left"),
        "axolotl": (9, 0, "left"),
        "unsloth": (-9, 6, "right"),
        "megatron_bridge": (0, -15, "center"),
        "ms_swift": (-9, -2, "right"),
    },
    16384: {
        "automodel": (-9, 0, "right"),
        "axolotl": (-9, 0, "right"),
        "unsloth": (-9, 4, "right"),
        "megatron_bridge": (-9, -2, "right"),
        "ms_swift": (-9, 0, "right"),
    },
}
# Frameworks that turned on their own activation checkpointing to fit 16,384 tokens per row, per the page's table.
CHECKPOINTED_AT = {2048: set(), 16384: {"automodel", "unsloth", "megatron_bridge", "ms_swift"}}


def read_points(seq):
    """Framework -> (mean peak GiB, mean tokens/s) over its valid runs."""
    runs = defaultdict(list)
    with open(SUMMARIES.format(seq=seq), newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["valid"] == "yes":
                runs[row["framework"]].append((float(row["peak_alloc_gib"]), float(row["tokens_per_second"])))
    return {name: (mean(r[0] for r in rows), mean(r[1] for r in rows)) for name, rows in runs.items()}


def draw_panel(ax, seq):
    points = read_points(seq)
    halo_rate = points["halo"][1]
    runner_up, runner_rate = max(
        ((name, rate) for name, (_, rate) in points.items() if name != "halo"), key=lambda p: p[1]
    )
    for name, (peak, rate) in points.items():
        if name == "halo":
            ax.plot(peak, rate, "o", ms=11, color=TEAL, zorder=3)
            ax.annotate(
                f"Halo, {halo_rate / runner_rate:.2f}× {NAMES[runner_up]}",
                (peak, rate),
                xytext=(11, 0),
                textcoords="offset points",
                va="center",
                fontsize=SMALL,
                fontweight="bold",
                color=INK,
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.5},
            )
            continue
        hollow = name in CHECKPOINTED_AT[seq]
        ax.plot(peak, rate, "o", ms=8.5, mfc=BG if hollow else SLATE, mec=SLATE, mew=1.5, zorder=3)
        dx, dy, ha = LABEL_AT[seq][name]
        ax.annotate(
            NAMES[name],
            (peak, rate),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=TINY,
            color=INK2,
        )
    ax.set_title(f"{seq:,} tokens per row", loc="left", fontsize=SMALL, color=INK2, pad=6)
    ax.set_xlim(75, 300)
    ax.set_ylim(0, 19000)
    ax.set_xticks(range(100, 301, 50))
    ax.set_yticks(range(0, 15001, 5000))
    k_ticks(ax.yaxis)
    ax.set_xlabel("peak GiB per GPU", fontsize=SMALL, color=INK2)
    tidy(ax, "both")


fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
fig.subplots_adjust(top=0.78, left=0.065, right=0.99, wspace=0.05)
heading(
    fig,
    "Gemma 4 26B-A4B, full SFT",
    "2 × B300 · BF16 · each framework at its fastest tried setup · hollow: it needed checkpointing to fit",
)
draw_panel(left, 2048)
draw_panel(right, 16384)
left.set_ylabel("cluster tokens/s", fontsize=SMALL, color=INK2)

save(fig, "gemma4_sft_pareto", BENCHMARKS)
