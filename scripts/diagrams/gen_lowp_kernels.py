"""Generate the two low-precision figures for optimization/low-precision-moe-kernels.md.

- lowp_whole_step.png: whole-step tokens/s as a ratio to bf16, one row per shape. The torchao row stands in its own
  group: it ran only on one shape, through a test wrapper, and Halo has not adopted it.
- lowp_native_step.png: the milliseconds native fp8 adds to a gpt-oss-20b ep8 step, as a waterfall that sums to the
  measured step.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (shape, bf16 tok/s/GPU, simulated fp8, simulated mxfp4, native fp8, native mxfp4)
SHAPES = (
    ("gpt-oss-20b ep8 b4", 13399, 0.449, 0.183, 0.539, 0.531),
    ("Qwen3-30B-A3B ep8 b8", 14063, 0.294, 0.098, 0.425, 0.369),
    ("Qwen3.5-35B-A3B ep8 b8", 16536, 0.442, 0.171, 0.533, 0.373),
)
TORCHAO_ROW = ("gpt-oss-20b ep8 b4, torchao MXFP8", 1.133)
# (legend name, marker, hollow) in the order of a SHAPES row's ratios.
ARMS = (
    ("simulated fp8", "o", True),
    ("simulated mxfp4", "D", True),
    ("native fp8", "o", False),
    ("native mxfp4", "D", False),
)

BF16_MS = 1311
NATIVE_MS = 2424
DELTAS = (
    ("activation cast", 392),
    ("weight cast", 63),
    ("padding and un-padding", 59),
    ("expert kernel", -37),
    ("combine wait", 309),
    ("the rest, about 300 outside the EP spans", 327),
)
if BF16_MS + sum(delta for _, delta in DELTAS) != NATIVE_MS:
    raise RuntimeError("the deltas do not sum to the measured step")

fig, ax = plt.subplots(figsize=(10.4, 3.5))
fig.subplots_adjust(top=0.72, left=0.27, right=0.985, bottom=0.27)
heading(
    fig,
    "No low-precision path Halo ships beats bf16",
    "whole steps on 8 × B300 · 4,096-token rows · GC on · uniform token ids · a 1.2.0 development build",
)
fig.texts[1].set_y(0.875)
ys = [0, 1, 2]
for y, (_, _, *ratios) in zip(ys, SHAPES, strict=True):
    ax.axhline(y, color=LINE, lw=0.7, zorder=1)
    for ratio, (_, marker, hollow) in zip(ratios, ARMS, strict=True):
        ax.plot(ratio, y, marker, ms=8, mfc=BG if hollow else BLUE, mec=BLUE, mew=1.5, zorder=3)
    best = max(ratios)
    ax.text(
        best + 0.03,
        y,
        f"{best:.2f}×",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        color=INK,
        zorder=4,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.0},
    )
torchao_y = 3.45
ax.axhline(torchao_y, color=LINE, lw=0.7, zorder=1)
ax.plot(TORCHAO_ROW[1], torchao_y, "s", ms=8, color=TEAL, zorder=3)
ax.text(
    TORCHAO_ROW[1] - 0.03,
    torchao_y,
    f"{TORCHAO_ROW[1]:.2f}× in a test wrapper, quality unmeasured, not adopted",
    ha="right",
    va="center",
    fontsize=SMALL,
    color=INK,
    zorder=4,
    bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.0},
)
ax.axvline(1.0, color=INK, lw=1.2, zorder=2)
ax.text(1.0, -0.72, "bf16", ha="center", va="bottom", fontsize=SMALL, fontweight="bold", color=INK)
ax.set_yticks([*ys, torchao_y])
ax.set_yticklabels([row[0] for row in SHAPES] + [TORCHAO_ROW[0]], fontsize=SMALL)
ax.set_ylim(torchao_y + 0.55, -0.75)
ax.set_xlim(0, 1.25)
ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0, 1.25))
ax.set_xlabel("tokens/s per GPU ÷ bf16 on the same shape", fontsize=SMALL, color=INK2)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(length=0, labelsize=SMALL)
handles = [
    plt.Line2D([], [], marker=marker, ls="", ms=7.5, mfc=BG if hollow else BLUE, mec=BLUE, mew=1.4, label=name)
    for name, marker, hollow in ARMS
]
fig.legend(
    handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=TINY, handletextpad=0.3, columnspacing=1.6
)

save(fig, "lowp_whole_step", BENCHMARKS)


fig, ax = plt.subplots(figsize=(10.4, 3.5))
fig.subplots_adjust(top=0.78, left=0.32, right=0.985, bottom=0.15)
heading(
    fig,
    "What native fp8 adds to a step",
    f"gpt-oss-20b ep8 b4 s4096, in a run that syncs the GPU around its timers · {BF16_MS:,} ms on bf16, {NATIVE_MS:,} ms on native fp8",
)
fig.texts[1].set_y(0.895)
rows = [("bf16 step", BF16_MS), *DELTAS, ("native fp8 step", NATIVE_MS)]
ys = list(range(len(rows)))
running = 0
for y, (label, value) in zip(ys, rows, strict=True):
    if label.endswith("step"):
        ax.barh(y, value, height=0.62, color=SLATE, zorder=2)
        running = value if label.startswith("bf16") else running
        ax.text(value + 25, y, f"{value:,}", va="center", fontsize=SMALL, fontweight="bold", color=INK)
        continue
    start = running if value >= 0 else running + value
    ax.barh(y, abs(value), left=start, height=0.62, color=ROSE if value >= 0 else TEAL, zorder=2)
    running += value
    ax.text(start + abs(value) + 25, y, f"{value:+,}".replace("-", "−"), va="center", fontsize=SMALL, color=INK)
ax.set_yticks(ys)
ax.set_yticklabels([label for label, _ in rows], fontsize=SMALL)
ax.set_ylim(len(rows) - 0.4, -0.6)
ax.set_xlim(0, 2800)
ax.set_xticks(range(0, 2501, 500))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
ax.set_xlabel("milliseconds", fontsize=SMALL, color=INK2)
tidy(ax, "x")

save(fig, "lowp_native_step", BENCHMARKS)
