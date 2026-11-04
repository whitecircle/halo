"""Generate the memory lever chart and the EP phase split for optimization/throughput-benchmarks.md.

- memory_levers.png: a lever's measured peak before and after, on one axis. Each row names its model and the release
  it ran on, since the rows come from separate runs and do not compare with each other.
- ep_phase_split.png: the share of dispatch, expert compute and combine in an EP layer's spans.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (group, [(row, before GiB, after GiB)])
LEVERS = (
    (
        "Peak allocated, GiB",
        (
            ("fsdp_shard_ep1_experts, gpt-oss-20b ep1 b1 (1.1.0 or earlier)", 148.3, 60.3),
            ("Length-ordered deal, async GRPO six-GPU row (1.2.0)", 210.25, 183.35),
            ("CP fused head loss, GLM-4.7-Flash ep8 × cp4 s65536 (1.2.0)", 53.30, 46.99),
        ),
    ),
    (
        "Peak reserved, GB",
        (
            ("Expandable segments, dense packed SFT (1.1.0 or earlier)", 41.3, 35.4),
            ("Expandable segments, DPO (1.1.0 or earlier)", 61.6, 52.7),
            ("Expandable segments, SMPO (1.1.0 or earlier)", 41.1, 35.8),
        ),
    ),
)

# (shape, dispatch %, expert compute %, combine with its wait %); the blank between groups splits ep8 from ep2.
PHASE_GROUPS = (
    (
        ("gpt-oss-20b ep8 b1 s4096", 13.3, 47.2, 39.4),
        ("gpt-oss-20b ep8 b4 s4096", 9.0, 45.6, 45.4),
        ("gpt-oss-20b ep8 b1 s16384", 9.0, 45.6, 45.4),
    ),
    (
        ("gpt-oss-20b ep2 b1 s4096", 12.2, 71.8, 15.9),
        ("Qwen3.5-35B-A3B ep2 b4 s4096", 12.2, 74.3, 13.4),
    ),
)
PHASES = (("dispatch", VIOLET), ("expert compute", BLUE), ("combine, wait included", ROSE))


fig, ax = plt.subplots(figsize=(10.8, 3.6))
fig.subplots_adjust(top=0.78, left=0.44, right=0.985, bottom=0.15)
heading(
    fig,
    "Memory before and after a lever",
    "hollow = before · filled = after · each row is its own run, so rows do not compare with each other",
)
fig.texts[1].set_y(0.895)
y = 0.0
ys, labels = [], []
for group, rows in LEVERS:
    ax.text(-0.81, y - 0.66, group, transform=ax.get_yaxis_transform(), fontsize=SMALL, fontweight="bold", color=INK2)
    for row, before, after in rows:
        ax.plot([after, before], [y, y], color=AMBER_T, lw=3.2, solid_capstyle="butt", zorder=2)
        ax.plot(before, y, "o", ms=8, mfc=BG, mec=SLATE, mew=1.6, zorder=3)
        ax.plot(after, y, "o", ms=8.5, color=AMBER, zorder=3)
        ax.text(
            before + 3,
            y,
            f"{after / before - 1:+.0%}".replace("-", "−"),
            va="center",
            fontsize=SMALL,
            fontweight="bold",
            color=INK,
        )
        ys.append(y)
        labels.append(row)
        y += 1
    y += 0.8
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=TINY)
ax.set_ylim(max(ys) + 0.6, -0.95)
ax.set_xlim(0, 235)
ax.set_xticks(range(0, 201, 50))
ax.set_xlabel("per GPU, in each group's unit", fontsize=SMALL, color=INK2)
tidy(ax, "x")

save(fig, "memory_levers", BENCHMARKS)


fig, ax = plt.subplots(figsize=(10.0, 3.3))
fig.subplots_adjust(top=0.73, left=0.25, right=0.985, bottom=0.22)
heading(
    fig,
    "Where an EP layer's time goes",
    "8 × B300 · a 1.2.0 development build · GC on · uniform token ids · shares of the three EP spans, not of the step",
)
fig.texts[1].set_y(0.87)
y = 0.0
ys, labels = [], []
for rows in PHASE_GROUPS:
    for shape, *shares in rows:
        left = 0.0
        for (_, hue), share in zip(PHASES, shares, strict=True):
            ax.barh(y, share, left=left, height=0.68, color=hue, edgecolor=BG, lw=1.0, zorder=2)
            left += share
        ax.text(
            left - 1.2,
            y,
            f"{shares[-1]:.0f}%",
            ha="right",
            va="center",
            fontsize=TINY,
            fontweight="bold",
            color=on_fill(ROSE),
            zorder=3,
        )
        ys.append(y)
        labels.append(shape)
        y += 1
    y += 0.45
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=SMALL)
ax.set_ylim(max(ys) + 0.5, -0.5)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 25))
ax.set_xlabel("share of the EP spans, %", fontsize=SMALL, color=INK2)
tidy(ax, "x")
handles = [plt.Rectangle((0, 0), 1, 1, color=hue, label=name) for name, hue in PHASES]
fig.legend(
    handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=TINY, handlelength=1.1, columnspacing=1.6
)

save(fig, "ep_phase_split", BENCHMARKS)
