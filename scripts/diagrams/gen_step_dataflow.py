"""Generate the 'one weight through a step' diagram for gpu-training-theory.md §8.

A timeline of the three matmuls a training step runs on one linear layer
(fprop + dgrad + wgrad = 6P FLOPs, P = params) over a proportional view of what
fills HBM: weights, gradients and optimizer state per parameter, plus
activations, which scale with the batch rather than the parameter count.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = plt.subplots(figsize=(11.4, 6.2))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 6.2)
ax.axis("off")

ax.text(0.3, 6.0, "One weight through a step", fontsize=TITLE, fontweight="bold", va="top")
ax.text(
    11.2, 5.92, "Y = X @ W   ·   P = layer parameters", fontsize=SUB, color=INK2, va="top", ha="right", fontfamily=MONO
)

xs = [0.3, 3.18, 6.06, 8.94]
box_w, box_h, box_y = 2.45, 1.15, 3.7
mid_y = box_y + box_h / 2


def stage(x, op, role, flop=None, cheap=False, fill=BLUE_T, edge_c=BLUE):
    patch = FancyBboxPatch(
        (x, box_y), box_w, box_h, boxstyle="round,pad=0.04", facecolor=fill, edgecolor=edge_c, lw=1.4
    )
    if cheap:
        patch.set_linestyle((0, (4, 2)))
    ax.add_patch(patch)
    ax.text(
        x + box_w / 2,
        box_y + box_h * 0.64,
        op,
        ha="center",
        va="center",
        color=INK if not cheap else INK2,
        fontsize=LABEL,
        fontweight="bold",
        fontfamily=MONO,
    )
    ax.text(
        x + box_w / 2,
        box_y + box_h * 0.27,
        role,
        ha="center",
        va="center",
        color=INK2 if not cheap else INK3,
        fontsize=SMALL,
    )
    if flop:
        ax.text(
            x + box_w - 0.16,
            box_y + box_h - 0.16,
            flop,
            ha="right",
            va="center",
            color=INK,
            fontsize=TINY,
            fontweight="bold",
            fontfamily=MONO,
        )


ax.text(xs[0] + box_w / 2, 5.18, "FORWARD", ha="center", fontsize=SECTION, fontweight="bold", color=INK)
ax.text((xs[1] + xs[2] + box_w) / 2, 5.18, "BACKWARD", ha="center", fontsize=SECTION, fontweight="bold", color=INK)
ax.text(xs[3] + box_w / 2, 5.18, "UPDATE", ha="center", fontsize=SECTION, fontweight="bold", color=INK)

stage(xs[0], "Y = X @ W", "saves X", flop="2P", fill=BLUE_T, edge_c=BLUE)
stage(xs[1], "dX = dY @ Wᵀ", "→ previous layer", flop="2P", fill=VIOLET_T, edge_c=VIOLET)
stage(xs[2], "dW = Xᵀ @ dY", "needs saved X", flop="2P", fill=VIOLET_T, edge_c=VIOLET)
stage(xs[3], "W −= lr·m/√v", "m, v · elementwise", cheap=True, fill=SLATE_T, edge_c=SLATE)


def arrow(x0, x1, label=None, color=INK2):
    # inset both ends so the arrow sits in the gap rather than on a border
    ax.annotate(
        "",
        xy=(x1 - 0.08, mid_y),
        xytext=(x0 + 0.08, mid_y),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
    )
    if label:
        ax.text(
            (x0 + x1) / 2, mid_y + 0.18, label, ha="center", va="bottom", fontsize=TINY, color=color, fontweight="bold"
        )


arrow(xs[0] + box_w, xs[1], color=AMBER)
arrow(xs[1] + box_w, xs[2])
arrow(xs[2] + box_w, xs[3], "dW")

brace_y = box_y - 0.28
ax.annotate(
    "",
    xy=(xs[2] + box_w, brace_y),
    xytext=(xs[0], brace_y),
    arrowprops={"arrowstyle": "|-|, widthA=0.5, widthB=0.5", "color": INK, "lw": 1.4},
)
ax.text(
    (xs[0] + xs[2] + box_w) / 2,
    brace_y - 0.22,
    "forward 2P  +  backward 4P  =  6P FLOPs / step",
    ha="center",
    va="top",
    fontsize=SMALL,
    fontweight="bold",
    color=INK,
    fontfamily=MONO,
)

ax.text(0.3, 2.45, "Memory in HBM", fontsize=SECTION, fontweight="bold", color=INK, va="top")
ax.text(0.3, 2.04, "per parameter — 8 B total (bf16 AdamW)", fontsize=SMALL, color=INK2, va="top")

bar_x0, bar_w, bar_y, bar_h = 0.3, 6.3, 1.05, 0.62
scale = bar_w / 8.0  # 8 bytes/param spans the bar
segs = [("Weights", "2 B", 0, 2, BLUE), ("Gradients", "2 B", 2, 4, TEAL), ("Optimizer  m, v", "4 B", 4, 8, VIOLET)]
for name, byts, a, b, color in segs:
    x, w = bar_x0 + a * scale, (b - a) * scale
    ax.add_patch(Rectangle((x, bar_y), w, bar_h, facecolor=color, edgecolor=INK, lw=1.3))
    ax.text(x + w / 2, bar_y + bar_h * 0.64, name, ha="center", va="center", fontsize=SMALL, color="#ffffff")
    ax.text(
        x + w / 2,
        bar_y + bar_h * 0.28,
        byts,
        ha="center",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        color="#ffffff",
        fontfamily=MONO,
    )

# activations: not per parameter, set apart with a dashed card
ax.text(7.05, 2.04, "+ activations — the long-context cost", fontsize=SMALL, color=INK2, va="top")
act = FancyBboxPatch((7.05, bar_y), 4.15, bar_h, boxstyle="round,pad=0.02", facecolor=AMBER_T, edgecolor=AMBER, lw=1.3)
act.set_linestyle((0, (4, 2)))
ax.add_patch(act)
ax.text(7.05 + 4.15 / 2, bar_y + bar_h * 0.64, "Activations", ha="center", va="center", fontsize=SMALL, color=INK)
ax.text(
    7.05 + 4.15 / 2,
    bar_y + bar_h * 0.28,
    "∝ batch × seq, not per-parameter",
    ha="center",
    va="center",
    fontsize=TINY,
    color=INK2,
)

ax.text(
    0.3,
    0.5,
    "The first three shard with FSDP / EP; activations don't — X is saved at forward, freed after wgrad.",
    fontsize=TINY,
    color=INK3,
    va="top",
)

save(plt.gcf(), "step_dataflow")
plt.close()
print("✓ step_dataflow.png")
