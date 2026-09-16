"""Generate the EP token-routing figure for parallelism/expert-parallelism.md.

Two ranks, two different batches: the router picks top-k experts per token, DeepEP's all-to-all
dispatch moves each token to the rank that owns its expert, the experts run as one grouped GEMM,
and the combine all-to-all returns every token to the rank it came from.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 12.4, 4.9

COL_XS, COL_W = columns(W, 3, 0.3, 1.7)
ROW_H = card_height(2)
ROW_YS = (2.72, 1.25)
HEAD_Y = 4.40

LANES = [
    (
        "Rank 0 · batch A",
        ("Experts 0–15", "Rank 0 · batch A"),
        ["[B·S, hidden] flat rows", "router top-k per token"],
    ),
    (
        "Rank 1 · batch B",
        ("Experts 16–31", "Rank 1 · batch B"),
        ["[B·S, hidden] flat rows", "router top-k per token"],
    ),
]
EXPERT_LINES = ["grouped_mm(x, w, offs)", "tokens sorted by expert"]
BACK_LINES = ["same rows, same order", "weighted by top-k probs"]

fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "EP token routing", "ep 2 · 32 experts → 16 per rank · dp 2 (EP ⊥ DP)")

for y, (head, (expert_head, back_head), lines) in zip(ROW_YS, LANES, strict=True):
    card(ax, COL_XS[0], y, COL_W, ROW_H, head, lines, color=SLATE, mono_lines=True)
    card(ax, COL_XS[1], y, COL_W, ROW_H, expert_head, EXPERT_LINES, color=BLUE, mono_lines=True)
    card(ax, COL_XS[2], y, COL_W, ROW_H, back_head, BACK_LINES, color=SLATE, mono_lines=True)

mids = [y + ROW_H / 2 for y in ROW_YS]
for left, right in ((COL_XS[0], COL_XS[1]), (COL_XS[1], COL_XS[2])):
    for src in mids:
        for dst in mids:
            arrow(ax, left + COL_W, src, right, dst, color=VIOLET, lw=1.3)

for left, right, name in ((COL_XS[0], COL_XS[1], "dispatch"), (COL_XS[1], COL_XS[2], "combine")):
    ax.text(
        (left + COL_W + right) / 2,
        HEAD_Y,
        f"DeepEP {name}\nall-to-all",
        ha="center",
        va="top",
        fontsize=SMALL,
        color=VIOLET,
        fontweight="bold",
        linespacing=1.45,
    )

footnote(
    ax,
    0.3,
    0.3,
    11.8,
    "EP is orthogonal to DP: every token returns to the rank it came from, so rank 0 still trains on batch A.",
)

save(fig, "ep_token_routing")
plt.close(fig)
print("✓ ep_token_routing.png")
