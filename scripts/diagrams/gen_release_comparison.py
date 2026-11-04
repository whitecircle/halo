"""Generate the 1.2.0-against-1.1.0 chart for optimization/throughput-benchmarks.md.

Twelve bench shapes on one 8x B300 host, both releases interleaved on one corpus. Throughput is wall clock
(tokens per step x steps / wall time / 8), since the two releases' logged per-GPU figures differ in clock and in
how a TP or ETP group's shared batch counts. Only the lead and the smallest gain carry a label.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# shape -> (1.1.0 tokens/s/GPU, 1.2.0 tokens/s/GPU, 1.1.0 peak GiB, 1.2.0 peak GiB)
SHAPES = {
    "Qwen3-8B FSDP2": (13935, 15490, 26.02, 26.38),
    "Qwen3-8B cp2": (7635, 8748, 25.07, 24.00),
    "Qwen3-8B tp2": (3861, 7013, 22.53, 22.53),
    "gpt-oss-20b ep1": (11114, 14304, 60.29, 61.83),
    "gpt-oss-20b ep2": (12121, 14107, 78.97, 77.32),
    "gpt-oss-20b ep8 + gathered save": (10349, 11896, 26.87, 26.47),
    "gpt-oss-20b ep8, DeepEP legacy": (8801, 9826, 26.67, 26.24),
    "gpt-oss-20b ep8 × tp2": (4749, 6013, 31.11, 31.16),
    "gpt-oss-20b ep8 × cp2": (7501, 9085, 25.18, 23.58),
    "gpt-oss-20b ep4 × etp2": (4589, 9521, 26.14, 25.75),
    "Qwen3.5-35B-A3B ep8, seq 4096": (7893, 8974, 43.03, 41.63),
    "Qwen3.5-35B-A3B ep8, seq 16384": (11618, 14275, 74.89, 74.96),
}

LEAD_LABELS = 2  # largest gains labeled, beside the smallest


def gain(shape):
    return shape[1] / shape[0]


# Ascending, so the largest gain sits on the top row of a y axis that counts upward.
ROWS = sorted(SHAPES.items(), key=lambda item: gain(item[1]))
Y = range(len(ROWS))

fig, (rate, memory) = plt.subplots(
    1, 2, figsize=(11.4, 5.7), sharey=True, gridspec_kw={"width_ratios": [3.0, 1.25], "wspace": 0.05}
)
fig.subplots_adjust(top=0.83, left=0.215, right=0.99)
heading(
    fig,
    "1.2.0 against 1.1.0",
    "8 × B300 · batch 1, GC on, seq 4096 unless a row says · wall clock, median of 3 runs (Qwen3.5 at seq 4096: 6)",
)

for y, (_, (before, after, _, _)) in zip(Y, ROWS, strict=True):
    rate.plot([before, after], [y, y], color=TEAL_T, lw=3.2, solid_capstyle="butt", zorder=2)
    rate.plot(before, y, "o", ms=7.5, mfc=BG, mec=SLATE, mew=1.6, zorder=3)
    rate.plot(after, y, "o", ms=8, color=TEAL, zorder=3)

labeled = {0, *range(len(ROWS) - LEAD_LABELS, len(ROWS))}
for y in labeled:
    shape = ROWS[y][1]
    rate.text(shape[1] + 330, y, f"{gain(shape):.2f}×", va="center", fontsize=SMALL, fontweight="bold", color=INK)

top = len(ROWS) - 1
first = ROWS[top][1]
rate.text(first[0], top + 0.48, "1.1.0", ha="center", va="bottom", fontsize=SMALL, fontweight="bold", color=SLATE)
rate.text(first[1], top + 0.48, "1.2.0", ha="center", va="bottom", fontsize=SMALL, fontweight="bold", color=TEAL)

rate.set_yticks(list(Y))
rate.set_yticklabels([label for label, _ in ROWS], fontsize=SMALL)
rate.set_xlim(0, 17500)
rate.set_xlabel("tokens/s/GPU", fontsize=SMALL, color=INK2)
k_ticks(rate.xaxis)
tidy(rate, "x")

deltas = [row[3] - row[2] for _, row in ROWS]
memory.barh(list(Y), deltas, height=0.55, color=AMBER, zorder=2)
memory.axvline(0, color=INK3, lw=0.9, zorder=3)
for y in (deltas.index(max(deltas)), deltas.index(min(deltas))):
    delta = deltas[y]
    memory.text(
        delta + (0.12 if delta > 0 else -0.12),
        y,
        f"{delta:+.2f}".replace("-", "−"),
        ha="left" if delta > 0 else "right",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        color=INK,
    )
memory.set_xlim(-3.0, 3.0)
memory.set_xticks([-2, -1, 0, 1, 2])
memory.set_xticklabels(["−2", "−1", "0", "+1", "+2"])
memory.set_xlabel("peak GiB, 1.2.0 − 1.1.0", fontsize=SMALL, color=INK2)
tidy(memory, "x")
memory.spines["left"].set_visible(False)

rate.set_ylim(-0.6, len(ROWS) - 0.05)

save(fig, "release_comparison", BENCHMARKS)
