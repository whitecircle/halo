"""Generate the pipeline-parallel throughput chart for parallelism/pipeline-parallelism.md.

The doc's like-for-like arms as bars: slate runs without PP, teal with it. Each PP row names its gap to the layouts of
its group, and a layout that did not fit keeps a row with its reason. A PP rate is attended tokens over its step time,
as the doc's note says; the tables carry the gradient-checkpointed arms and the peaks.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (row, tokens/s/GPU or the reason it has none, the earlier and later rows of the group a PP row compares with)
FAST_LINK = [
    ("Qwen3-8B dp16", 18337, ()),
    ("Qwen3-8B pp2 × dp8", 17608, ((0, "dp16"), (2, "dp8 × tp2"))),
    ("Qwen3-8B dp8 × tp2", 10780, ()),
    ("gpt-oss-20b ep8 with DP", 18685, ()),
    ("gpt-oss-20b pp2 × ep8", 15715, ((3, "ep8 with DP"), (5, "ep16"))),
    ("gpt-oss-20b ep16 global", 4115, ()),
]
PLAIN_SOCKETS = [
    ("Qwen3-8B dp16", 1117, ()),
    ("Qwen3-8B pp2 × dp8", 15414, ((0, "dp16"),)),
]
ONE_MACHINE = [
    ("Qwen3-8B dp8", 21359, ()),
    ("Qwen3-8B pp2 × dp4", 19537, ((0, "dp8"),)),
    ("gpt-oss-20b ep8", 19989, ()),
    ("gpt-oss-20b pp2 × ep4", 21836, ((2, "ep8"),)),
]
GLM = [
    ("ep8 with DP", "out of memory in the first optimizer step", ()),
    ("ep16 global", 1504, ()),
    ("pp2 × ep8", 2423, ((1, "ep16"),)),
]

PANELS = (
    ("Two 8-GPU machines, fast link · max_length 4096 and 8192 · m = 8 · no checkpointing", FAST_LINK),
    (
        "The same two machines forced onto plain TCP sockets · max_length 4096 · m = 8 · no checkpointing",
        PLAIN_SOCKETS,
    ),
    (
        "One 8-GPU machine, the stages over host shared memory · max_length 8192 · m = 16 · no checkpointing",
        ONE_MACHINE,
    ),
    ("GLM-5.3-Flash full fine-tune, two 8-GPU machines · max_length 8192 · m = 8 · gradient checkpointing", GLM),
)


def gap(rate, other):
    """A close layout as a percentage, a far one as a multiple."""
    ratio = rate / other
    if ratio >= 1.5:
        return f"{ratio:.3g}×"
    return f"{ratio - 1:+.1%}".replace("-", "−") + " against"


def draw_panel(ax, caption, rows):
    ys = list(range(len(rows)))[::-1]
    for y, (row, rate, compared) in zip(ys, rows, strict=True):
        if isinstance(rate, str):
            ax.text(250, y, rate, va="center", fontsize=SMALL, color=INK2, zorder=3)
            continue
        ax.barh(y, rate, height=0.62, zorder=2, color=TEAL if "pp2" in row else SLATE)
        label = f"{rate:,}"
        if compared:
            label += " · " + ", ".join(f"{gap(rate, rows[i][1])} {name}" for i, name in compared)
        ax.text(
            rate + 300,
            y,
            label,
            va="center",
            fontsize=SMALL,
            fontweight="bold" if compared else "normal",
            color=INK,
            zorder=3,
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.5},
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=SMALL)
    ax.set_title(caption, loc="left", fontsize=SMALL, color=INK2, pad=6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 36000)
    ax.set_xticks(range(0, 25001, 5000))
    k_ticks(ax.xaxis)
    tidy(ax, "x")


fig, axes = plt.subplots(
    len(PANELS),
    1,
    figsize=(11.2, 7.6),
    sharex=True,
    gridspec_kw={"height_ratios": [len(rows) for _, rows in PANELS], "hspace": 0.27},
)
fig.subplots_adjust(top=0.875, left=0.17, right=0.99, bottom=0.07)
heading(fig, "Pipeline parallelism", "B300 · tokens/s/GPU over the steady steps · slate = no PP · teal = PP")
fig.texts[1].set_y(0.945)
for ax, (caption, rows) in zip(axes, PANELS, strict=True):
    draw_panel(ax, caption, rows)
axes[-1].set_xlabel("tokens/s/GPU", fontsize=SMALL, color=INK2)

save(fig, "pp_throughput", BENCHMARKS)
