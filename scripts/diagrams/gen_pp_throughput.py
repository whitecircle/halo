"""Generate the pipeline-parallel throughput chart for parallelism/pipeline-parallelism.md.

The doc's two measured tables as bars: slate runs without PP, pale teal PP with gradient checkpointing, teal PP
without it. Each GC-off row names its gain and its peak against the row it is compared to; the tables carry the rest.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (row, tokens/s/GPU, allocated peak GiB, the earlier row a label compares it to, that row's name in the label)
ONE_HOST = [
    ("Qwen3-8B dp8, GC on", 17062, 33.0, None, None),
    ("Qwen3-8B pp2 × dp4, GC on", 16291, 27.9, None, None),
    ("Qwen3-8B pp2 × dp4, GC off", 19445, 66.3, 0, "dp8"),
    ("gpt-oss-20b ep8, GC off", 20066, 63.1, None, None),
    ("gpt-oss-20b pp2 × ep4, GC on", 17398, 35.3, None, None),
    ("gpt-oss-20b pp2 × ep4, GC off", 22223, 62.6, 3, "ep8"),
]
TWO_NODES = [
    ("Qwen3-8B pp2, max_length 4096, GC on", 14970, 22.1, None, None),
    ("Qwen3-8B pp2, max_length 4096, GC off", 17740, 41.3, 0, "GC on"),
    ("gpt-oss-20b pp2 × ep8, max_length 8192, GC on", 12755, 24.2, None, None),
    ("gpt-oss-20b pp2 × ep8, max_length 8192, GC off", 15500, 45.2, 2, "GC on"),
]

PANELS = (
    ("One 8-GPU host · max_length 8192 · m = 16 · the stages exchange over host shared memory", ONE_HOST),
    ("Two nodes, 2 × 8 · shipped recipes · m = 8 · the stages exchange over EFA", TWO_NODES),
)


def bar_style(row):
    """Slate without PP, teal with it; the GC-on PP rows are the pale ones."""
    if "pp2" not in row:
        return {"color": SLATE}
    if row.endswith("GC on"):
        return {"color": TEAL_T, "edgecolor": TEAL, "linewidth": 0.9}
    return {"color": TEAL}


def draw_panel(ax, caption, rows):
    ys = list(range(len(rows)))[::-1]
    for y, (row, rate, peak, base, base_name) in zip(ys, rows, strict=True):
        ax.barh(y, rate, height=0.62, zorder=2, **bar_style(row))
        if base is None:
            continue
        base_rate, base_peak = rows[base][1], rows[base][2]
        ax.text(
            rate + 300,
            y,
            f"+{rate / base_rate - 1:.1%} over {base_name}, {peak / base_peak:.1f}× its peak",
            va="center",
            fontsize=SMALL,
            fontweight="bold",
            color=INK,
            zorder=3,
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.5},
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=SMALL)
    ax.set_title(caption, loc="left", fontsize=SMALL, color=INK2, pad=6)
    ax.set_xlim(0, 34000)
    ax.set_xticks(range(0, 25001, 5000))
    k_ticks(ax.xaxis)
    tidy(ax, "x")


fig, axes = plt.subplots(
    2,
    1,
    figsize=(11.2, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [len(ONE_HOST), len(TWO_NODES)], "hspace": 0.2},
)
fig.subplots_adjust(top=0.82, left=0.255, right=0.99)
heading(fig, "Pipeline parallelism", "B300 · packed ultrachat_200k · tokens/s/GPU over the steady steps")
for ax, (caption, rows) in zip(axes, PANELS, strict=True):
    draw_panel(ax, caption, rows)
axes[-1].set_xlabel("tokens/s/GPU", fontsize=SMALL, color=INK2)

save(fig, "pp_throughput", BENCHMARKS)
