"""Generate the four-collectives diagram for gpu-training-theory.md §9.

Four panels (all-gather, reduce-scatter, all-reduce, all-to-all), each a
before → after view over 4 ranks × 4 chunk slots. Notation: a₀ = chunk a's
copy/slice from rank 0; Σa = chunk a summed over ranks. The panel order shows
the two §9 identities: all-reduce is the RS+AG composition, and all-to-all is
a transpose of the rank×chunk grid.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import FancyBboxPatch, Rectangle

R = 4
SUBS = "₀₁₂₃"
LETTERS = "abcd"
CW, CH = 0.58, 0.46  # cell size


def grid(ax, x0, y0, cells, label, stroke, tint):
    """Draw a 4×4 rank-by-chunk grid. cells[r][c] = text or None (absent)."""
    for r in range(R):
        y = y0 - r * CH
        ax.text(x0 - 0.12, y - CH / 2, f"R{r}", ha="right", va="center", fontsize=TINY, color=INK3)
        for c in range(R):
            x = x0 + c * CW
            txt = cells[r][c]
            if txt is None:
                ax.add_patch(Rectangle((x, y - CH), CW, CH, facecolor="#ffffff", edgecolor=LINE, lw=0.9))
            else:
                is_sum = txt.startswith("Σ")
                fill = stroke if is_sum else tint
                ax.add_patch(Rectangle((x, y - CH), CW, CH, facecolor=fill, edgecolor=stroke, lw=1.1))
                ax.text(
                    x + CW / 2,
                    y - CH / 2,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=SMALL,
                    fontfamily=MONO,
                    fontweight="bold" if is_sum else "normal",
                    color="#ffffff" if is_sum else INK,
                )
    ax.text(x0 + R * CW / 2, y0 - R * CH - 0.16, label, ha="center", va="top", fontsize=TINY, color=INK2)


def panel(ax, x0, y0, name, users, before, after, cost, stroke, tint):
    ax.text(x0, y0, name, fontsize=SECTION, fontweight="bold", color=INK, va="top")
    ax.text(x0, y0 - 0.42, users, fontsize=SMALL, color=INK2, va="top")
    gy = y0 - 0.9
    gw = R * CW
    grid(ax, x0 + 0.34, gy, before, "before", stroke, tint)
    ax.annotate(
        "",
        xy=(x0 + 0.34 + gw + 0.72, gy - 2 * CH),
        xytext=(x0 + 0.34 + gw + 0.14, gy - 2 * CH),
        arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.6},
    )
    grid(ax, x0 + 0.34 + gw + 0.86, gy, after, "after", stroke, tint)
    ax.text(
        x0 + 0.34,
        gy - R * CH - 0.52,
        cost,
        fontsize=SMALL,
        color=INK,
        va="top",
        fontweight="bold",
        fontfamily=MONO,
    )


fig, ax = plt.subplots(figsize=(12.8, 9.0))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 9.4)
ax.axis("off")

ax.text(0.3, 9.25, "The four collectives", fontsize=TITLE, fontweight="bold", va="top")
ax.text(
    0.3,
    8.78,
    "4 ranks × 4 chunk slots · aᵢ = chunk a from rank i · Σa = chunk a summed over ranks · D = tensor bytes",
    fontsize=SUB,
    color=INK2,
    va="top",
)

ag_before = [[f"{LETTERS[c]}{SUBS[c]}" if c == r else None for c in range(R)] for r in range(R)]
ag_after = [[f"{LETTERS[c]}{SUBS[c]}" for c in range(R)] for r in range(R)]
rs_before = [[f"{LETTERS[c]}{SUBS[r]}" for c in range(R)] for r in range(R)]
rs_after = [[f"Σ{LETTERS[c]}" if c == r else None for c in range(R)] for r in range(R)]
ar_after = [[f"Σ{LETTERS[c]}" for c in range(R)] for r in range(R)]
a2a_after = [[f"{LETTERS[r]}{SUBS[c]}" for c in range(R)] for r in range(R)]

X0, X1 = 0.3, 7.1
Y0, Y1 = 8.2, 4.35
panel(
    ax,
    X0,
    Y0,
    "all-gather",
    "FSDP forward — everyone gets every shard",
    ag_before,
    ag_after,
    "(R−1)/R × D per GPU",
    BLUE,
    BLUE_T,
)
panel(
    ax,
    X1,
    Y0,
    "reduce-scatter",
    "FSDP backward — sum, keep your shard",
    rs_before,
    rs_after,
    "(R−1)/R × D per GPU  ·  all-gather's dual",
    TEAL,
    TEAL_T,
)
panel(
    ax,
    X0,
    Y1,
    "all-reduce",
    "DDP gradient sync · TP layer outputs",
    rs_before,
    ar_after,
    "= reduce-scatter + all-gather  →  2·(R−1)/R × D",
    VIOLET,
    VIOLET_T,
)
panel(
    ax,
    X1,
    Y1,
    "all-to-all",
    "EP dispatch · CP Ulysses (seq↔head)",
    rs_before,
    a2a_after,
    "a transpose — each chunk has one destination",
    AMBER,
    AMBER_T,
)

foot = FancyBboxPatch((0.3, 0.06), 12.9, 0.5, boxstyle="round,pad=0.06", facecolor=BG, edgecolor=INK2, lw=1.2)
foot.set_linestyle((0, (4, 2)))
ax.add_patch(foot)
ax.text(
    6.75,
    0.31,
    "backward runs the dual:  the gradient of an all-gather is a reduce-scatter (and vice versa) —"
    " an all-to-all's is another all-to-all",
    ha="center",
    va="center",
    fontsize=SMALL,
    color=INK,
)

save(plt.gcf(), "collective_ops")
plt.close()
print("✓ collective_ops.png")
