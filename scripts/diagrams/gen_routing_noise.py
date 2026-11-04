"""Generate the two trainer-against-engine figures for training-methods/async-grpo/objective.md.

- routing_forced_gap.png: bf16's gap to an fp32 forward with natural routing and with fp32's top-k forced. The rows are
  grouped by what forcing does, so a family whose gap is mostly not routing does not read as a weak row of the others.
- routing_replay_gap.png: the log-ratio over sampled tokens under no replay, R2 and R3, on the families with enough
  scored tokens to quote. One short task at step 1, not a training result.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (group, [(row, natural gap, forced gap)]): mean absolute log-prob gap to the fp32 forward, nats.
FORCED = (
    (
        "Forcing closes most of it",
        (
            ("gpt-oss-20b, GPU, stock transformers", 0.400, 0.079),
            ("Laguna-XS, CPU kernels", 0.679, 0.058),
            ("LFM-2, CPU kernels", 0.375, 0.050),
        ),
    ),
    (
        "Most of it is not routing",
        (
            ("GLM-4.7, CPU kernels", 0.095, 0.072),
            ("Gemma 4, CPU kernels", 0.683, 0.623),
        ),
    ),
)
# Engine and checkpoint, then the log-ratio under none, R2 and R3.
REPLAY = (
    ("vLLM, gpt-oss-20b", 0.0300, 0.0288, 0.0158),
    ("vLLM, Qwen3.6-35B-A3B", 0.0279, 0.0304, 0.0111),
    ("vLLM, GLM-4.7-Flash", 0.0375, 0.0388, 0.0318),
    ("SGLang, gpt-oss-20b", 0.0289, 0.0299, 0.0147),
)

fig, ax = plt.subplots(figsize=(10.4, 3.7))
fig.subplots_adjust(top=0.76, left=0.3, right=0.985, bottom=0.17)
heading(
    fig,
    "What forced routing leaves",
    "bf16 against an fp32 forward on fixed text, no engine · hollow = natural top-k · filled = fp32's top-k forced",
)
fig.texts[1].set_y(0.885)
y = 0.0
ys, labels = [], []
for group, rows in FORCED:
    ax.text(-0.44, y - 0.62, group, transform=ax.get_yaxis_transform(), fontsize=SMALL, fontweight="bold", color=INK2)
    for row, natural, forced in rows:
        ax.axhline(y, color=LINE, lw=0.7, zorder=1)
        ax.plot([forced, natural], [y, y], color=TEAL_T, lw=3.2, solid_capstyle="butt", zorder=2)
        ax.plot(natural, y, "o", ms=8, mfc=BG, mec=SLATE, mew=1.6, zorder=3)
        ax.plot(forced, y, "o", ms=8.5, color=TEAL, zorder=3)
        ys.append(y)
        labels.append(row)
        y += 1
    y += 0.75
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=SMALL)
ax.set_ylim(max(ys) + 0.6, -0.9)
ax.set_xlim(0, 0.75)
ax.set_xticks((0, 0.2, 0.4, 0.6))
ax.set_xlabel("mean absolute log-prob gap, nats", fontsize=SMALL, color=INK2)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(length=0, labelsize=SMALL)
ax.grid(axis="x", color=LINE, lw=0.7)
ax.set_axisbelow(True)

save(fig, "routing_forced_gap", BENCHMARKS)


fig, ax = plt.subplots(figsize=(10.4, 2.95))
fig.subplots_adjust(top=0.7, left=0.21, right=0.985, bottom=0.22)
heading(
    fig,
    "Sampled-token log-ratio by replay mode",
    "one short tool-call task at step 1, before any update · ep1 · one run per arm · R3 is experimental",
)
fig.texts[1].set_y(0.86)
ys = list(range(len(REPLAY)))
for y, (_, none, r2, r3) in zip(ys, REPLAY, strict=True):
    ax.axhline(y, color=LINE, lw=0.7, zorder=1)
    ax.plot(none, y, "o", ms=8.5, mfc=BG, mec=SLATE, mew=1.6, zorder=3)
    ax.plot(r2, y, "s", ms=7, color=SLATE, zorder=3)
    ax.plot(r3, y, "o", ms=9, color=TEAL, zorder=4)
    ax.text(
        r3 - 0.0007,
        y,
        f"{none / r3:.1f}× below none",
        ha="right",
        va="center",
        fontsize=TINY,
        fontweight="bold",
        color=INK,
        zorder=4,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.0},
    )
ax.set_yticks(ys)
ax.set_yticklabels([row[0] for row in REPLAY], fontsize=SMALL)
ax.set_ylim(len(REPLAY) - 0.5, -0.5)
ax.set_xlim(0, 0.042)
ax.set_xticks((0, 0.01, 0.02, 0.03, 0.04))
ax.set_xticklabels(("0", "0.01", "0.02", "0.03", "0.04"))
ax.set_xlabel("mean absolute log-ratio over scored tokens, nats", fontsize=SMALL, color=INK2)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(length=0, labelsize=SMALL)
ax.grid(axis="x", color=LINE, lw=0.7)
ax.set_axisbelow(True)
handles = [
    plt.Line2D([], [], marker="o", ls="", ms=7.5, mfc=BG, mec=SLATE, mew=1.5, label="none"),
    plt.Line2D([], [], marker="s", ls="", ms=6.5, color=SLATE, label="recompute (R2)"),
    plt.Line2D([], [], marker="o", ls="", ms=8, color=TEAL, label="rollout (R3)"),
]
fig.legend(
    handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=TINY, handletextpad=0.3, columnspacing=1.6
)

save(fig, "routing_replay_gap", BENCHMARKS)
