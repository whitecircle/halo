"""Generate the two Nemotron-H figures for models/nemotron-h.md.

- nemotron_block_time.png: a forward and backward at EP8 and 8,192 tokens per rank, split by block kind. A segment's
  length is the step times the page's share, so the two releases read on one axis.
- nemotron_memory.png: the Memory table's per-rank ranges against the device, the number a run is sized by.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (release, step seconds, tok/s per rank, Mamba %, MoE %, attention %, head %)
BLOCK_ROWS = (
    ("Super VL", 2.54, 3226, 64.8, 32.6, 1.6, 1.0),
    ("Lightning", 0.766, 10694, 58.0, 36.2, 3.2, 2.6),
)
BLOCK_KINDS = (("Mamba", BLUE), ("MoE", TEAL), ("attention", VIOLET), ("head", ROSE))
LABEL_MIN_SECONDS = 0.2  # a segment narrower than this takes its name from the legend alone

DEVICE_GB = 288
# (group, recipe, tokens per rank, allocated low, allocated high, reserved low, reserved high, label)
MEMORY_ROWS = (
    ("Super VL", "LoRA", "16,384", 92, 94, 118, 165, None),
    ("Super VL", "Full fine-tune", "8,192", 169, 171, 186, 202, None),
    ("Super VL", "Full fine-tune", "about 31,700", 227, 235, 268, 283, "top rank 99.4%"),
    ("Super VL", "Frozen experts", "about 48,800", 188, 194, 241, 283, "top rank 99%"),
    ("Super VL", "Full fine-tune on images", "about 6,790", 168, 168, 184, 188, None),
    ("Super VL", "The same, half the rows text", "", 168, 168, 183, 186, None),
    ("Super VL", "SMPO, full pairs", "2 × 4,096", 170, 170, 178, 184, None),
    ("Super VL", "SMPO, its Tulu rows", "up to 2 × 4,096", 157, 161, 165, 170, None),
    ("Super VL", "Offline GRPO", "the recipe's rows", 164, 165, 169, 179, None),
    ("Lightning", "Full fine-tune", "8,192", 51.2, 51.6, 55, 60, None),
)


fig, ax = plt.subplots(figsize=(10.4, 2.75))
fig.subplots_adjust(top=0.62, left=0.1, right=0.985, bottom=0.2)
heading(
    fig,
    "Where a Nemotron step goes",
    "EP8 on 8 × B300 · 8,192 tokens per rank · the tower's forward and backward, no optimizer or reduce",
)
fig.texts[1].set_y(0.84)
ys = [1, 0]
for y, (_, step, rate, *shares) in zip(ys, BLOCK_ROWS, strict=True):
    left = 0.0
    for (kind, hue), share in zip(BLOCK_KINDS, shares, strict=True):
        seconds = step * share / 100
        ax.barh(y, seconds, left=left, height=0.56, color=hue, edgecolor=BG, lw=1.0, zorder=2)
        if seconds >= LABEL_MIN_SECONDS:
            ax.text(
                left + seconds / 2,
                y,
                f"{kind} {share:.1f}%",
                ha="center",
                va="center",
                fontsize=TINY,
                color=on_fill(hue),
            )
        left += seconds
    ax.text(
        left + 0.03,
        y,
        f"{step:g} s, {rate:,} tok/s per rank",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        color=INK,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.5},
    )
ax.set_yticks(ys)
ax.set_yticklabels([row[0] for row in BLOCK_ROWS], fontsize=SMALL)
ax.set_xlim(0, 3.6)
ax.set_xticks((0, 0.5, 1.0, 1.5, 2.0, 2.5))
ax.set_xlabel("seconds per step", fontsize=SMALL, color=INK2)
ax.set_ylim(-0.5, 1.5)
tidy(ax, "x")
handles = [plt.Rectangle((0, 0), 1, 1, color=hue, label=kind) for kind, hue in BLOCK_KINDS]
ax.legend(
    handles=handles, loc="lower right", ncol=4, frameon=False, fontsize=TINY, handlelength=1.1, columnspacing=1.2
)

save(fig, "nemotron_block_time", BENCHMARKS)


fig, ax = plt.subplots(figsize=(10.4, 4.9))
fig.subplots_adjust(top=0.82, left=0.27, right=0.985, bottom=0.12)
heading(
    fig,
    "Nemotron-H memory against the device",
    "EP8 on 8 × B300 · reentrant checkpointing · the shipped recipes · each span is the range across ranks",
)
ys = []
y = 0.0
last_group = None
for row in MEMORY_ROWS:
    if row[0] != last_group:
        y += 0.75 if last_group else 0.0
        last_group = row[0]
        ax.text(
            -0.37,
            y - 0.72,
            last_group,
            transform=ax.get_yaxis_transform(),
            fontsize=SMALL,
            fontweight="bold",
            color=INK2,
        )
    ys.append(y)
    y += 1
for y, (_, _, _, a_low, a_high, r_low, r_high, label) in zip(ys, MEMORY_ROWS, strict=True):
    ax.plot([r_low, r_high], [y, y], color=AMBER, lw=7, solid_capstyle="butt", zorder=2)
    ax.plot([a_low, a_high], [y, y], color=AMBER_SHADES[0], lw=7, solid_capstyle="butt", zorder=3)
    if a_low == a_high:
        ax.plot(a_low, y, "|", ms=11, mew=3, color=AMBER_SHADES[0], zorder=3)
    if label:
        ax.text(
            a_low - 3,
            y,
            label,
            ha="right",
            va="center",
            fontsize=TINY,
            fontweight="bold",
            color=INK,
            zorder=4,
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.2},
        )
ax.axvline(DEVICE_GB, color=ROSE, lw=1.6, zorder=4)
ax.text(
    DEVICE_GB - 3,
    ys[-1],
    f"device, {DEVICE_GB} GB",
    ha="right",
    va="center",
    fontsize=SMALL,
    fontweight="bold",
    color=INK,
)
ax.set_yticks(ys)
ax.set_yticklabels(
    [recipe + (f", {tokens}" if tokens else "") for _, recipe, tokens, *_ in MEMORY_ROWS], fontsize=TINY
)
ax.set_ylim(max(ys) + 0.7, -0.95)
ax.set_xlim(0, 300)
ax.set_xticks(range(0, 301, 50))
ax.set_xlabel(
    "GB per GPU: dark = peak allocated, light = reserved. Row labels hold tokens per rank.", fontsize=SMALL, color=INK2
)
tidy(ax, "x")

save(fig, "nemotron_memory", BENCHMARKS)
