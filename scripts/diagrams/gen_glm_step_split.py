"""Generate the GLM-5.3-Flash step split for models/glm5-next.md.

One LoRA step at EP8 and 8,192 tokens on a per-part clock, before and after the DSA and hyper-connection kernels, sorted
by what each part takes now. The parts sum to the clock's step, which reads about 2.6 s over an unclocked one.
"""

import matplotlib.pyplot as plt
from _chart_style import *

# (part, seconds on the general kernels, seconds on Halo's)
PARTS = (
    ("DSA attention", 21.91, 4.63),
    ("MoE compute", 4.50, 4.50),
    ("MoE router, dispatch, combine", 3.99, 3.67),
    ("KDA conv, projections, gate, output", 3.28, 3.27),
    ("KDA scan", 3.06, 3.06),
    ("DSA projections, indexer, mask", 1.84, 1.83),
    ("Everything else", 1.97, 1.78),
    ("Hyper-connection weights", 7.17, 1.31),
    ("Hyper-connection mix", 3.39, 0.44),
)
BAR = 0.38

fig, ax = plt.subplots(figsize=(10.8, 4.9))
fig.subplots_adjust(top=0.83, left=0.27, right=0.985, bottom=0.11)
before_total, after_total = (sum(part[i] for part in PARTS) for i in (1, 2))
heading(
    fig,
    f"GLM-5.3-Flash LoRA step: {before_total:.1f} s to {after_total:.1f} s",
    "EP8 · 8,192 tokens · B300 · slate = torch's general kernel and upstream's hyper-connections · teal = Halo's kernels",
)
ys = list(range(len(PARTS)))[::-1]
for y, (_, before, after) in zip(ys, PARTS, strict=True):
    ax.barh(y + BAR / 2, before, height=BAR, color=SLATE, zorder=2)
    ax.barh(y - BAR / 2, after, height=BAR, color=TEAL, zorder=2)
    ax.text(before + 0.15, y + BAR / 2, f"{before:.2f}", va="center", fontsize=SMALL, color=INK2)
    ax.text(after + 0.15, y - BAR / 2, f"{after:.2f}", va="center", fontsize=SMALL, fontweight="bold", color=INK)
ax.set_yticks(ys)
ax.set_yticklabels([part[0] for part in PARTS], fontsize=SMALL)
ax.set_xlabel("seconds per step, mean of the ranks over forward, recompute and backward", fontsize=SMALL, color=INK2)
ax.set_xlim(0, 24)
tidy(ax, "x")

save(fig, "glm_step_split", BENCHMARKS)
