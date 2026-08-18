"""Generate the GEMM-anatomy diagram for gpu-training-theory.md §2.

Defines M, N, K visually: a layer is Y = X @ W, where M is the row count (the
token count) and K, N are the weight's fixed shape. W is read from HBM
regardless of M, which is the asymmetry the roofline rests on.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9.6, 4.7))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 9.6)
ax.set_ylim(0.42, 4.7)
ax.axis("off")

ax.text(0.2, 4.55, "A layer is a matrix multiply (GEMM)", fontsize=TITLE, fontweight="bold", va="top")
ax.text(0.2, 4.05, "Y = X @ W", fontsize=SUB, color=INK2, va="top", family=MONO)


def box(x, y, w, h, face, edge, label, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=face, edgecolor=edge, lw=1.4))
    ax.text(
        x + w / 2,
        y + h * 0.60,
        label,
        ha="center",
        va="center",
        fontsize=LABEL,
        fontweight="bold",
        color=INK,
        family=MONO,
    )
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=TINY, color=INK2, family=MONO)


def dim(x0, y0, x1, y1, label, color=INK2, weight="normal"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2, "shrinkA": 1, "shrinkB": 1},
    )
    ax.text(
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        label,
        ha="center",
        va="center",
        fontsize=SMALL,
        color=color,
        fontweight=weight,
        family=MONO,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": BG, "edgecolor": "none"},
    )


# geometry: A [M×K], B [K×N], C [M×N]
Ax, Aw, Ah, Ay = 1.55, 1.2, 1.55, 1.72
Bx, Bw, Bh, By = 4.05, 1.55, 1.2, 1.89
Cx, Cw, Ch, Cy = 6.55, 1.55, 1.55, 1.72

box(Ax, Ay, Aw, Ah, BLUE_T, BLUE, "X", "[M × K]")
box(Bx, By, Bw, Bh, AMBER_T, AMBER, "W", "[K × N]")
box(Cx, Cy, Cw, Ch, TEAL_T, TEAL, "Y", "[M × N]")
ax.text(Ax + Aw / 2, Ay - 0.22, "input activations", ha="center", va="top", fontsize=TINY, color=INK3)
ax.text(Bx + Bw / 2, By - 0.39, "weight", ha="center", va="top", fontsize=TINY, color=INK3)
ax.text(Cx + Cw / 2, Cy - 0.22, "output", ha="center", va="top", fontsize=TINY, color=INK3)

ax.text((Ax + Aw + Bx - 0.28) / 2, Ay + Ah / 2, "@", ha="center", va="center", fontsize=18, color=INK2, family=MONO)
ax.text((Bx + Bw + Cx) / 2, Ay + Ah / 2, "=", ha="center", va="center", fontsize=18, color=INK2, family=MONO)

# M — the row dimension (tokens), highlighted on X and Y
dim(Ax - 0.28, Ay, Ax - 0.28, Ay + Ah, "M", color=BLUE, weight="bold")
dim(Cx + Cw + 0.28, Cy, Cx + Cw + 0.28, Cy + Ch, "M", color=BLUE, weight="bold")
# K — the shared/contracted dimension (X columns = W rows)
dim(Ax, Ay + Ah + 0.22, Ax + Aw, Ay + Ah + 0.22, "K")
dim(Bx - 0.28, By, Bx - 0.28, By + Bh, "K")
# N — output width (W columns = Y columns)
dim(Bx, By + Bh + 0.22, Bx + Bw, By + Bh + 0.22, "N")
dim(Cx, Cy + Ch + 0.22, Cx + Cw, Cy + Ch + 0.22, "N")

ax.text(
    0.2,
    1.02,
    "M = rows = tokens  (batch × seq, or routed per expert) — the knob you turn.",
    fontsize=TINY,
    color=INK,
    va="center",
)
ax.text(
    0.2,
    0.66,
    "K, N = the layer's width — fixed by the model.   W (the weight) is read from HBM regardless of M.",
    fontsize=TINY,
    color=INK2,
    va="center",
)

save(plt.gcf(), "gemm_anatomy")
plt.close()
print("✓ gemm_anatomy.png")
