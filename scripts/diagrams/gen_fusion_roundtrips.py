"""Generate the fusion round-trips diagram for gpu-training-theory.md §5.

y = silu(a) · b as three naive elementwise kernels (8 tensor passes over HBM)
versus one fused kernel (3 passes). The op count is identical; only the HBM
traffic changes, which sets the runtime for kernels at AI ~ 1.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = plt.subplots(figsize=(11.4, 5.6))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 5.8)
ax.axis("off")

ax.text(0.3, 5.72, "Fusion = fewer HBM round trips", fontsize=TITLE, fontweight="bold", va="top")
ax.text(
    0.3,
    5.26,
    "y = silu(a) · b, elementwise · AI ≈ 1 → runtime = HBM traffic",
    fontsize=SUB,
    color=INK2,
    va="top",
)

HBM_Y, HBM_H = 3.95, 0.52
BOX_Y, BOX_H = 1.55, 1.0


def hbm_bar(x0, w, label):
    ax.add_patch(Rectangle((x0, HBM_Y), w, HBM_H, facecolor=SLATE_T, edgecolor=SLATE, lw=1.4))
    ax.text(x0 + w / 2, HBM_Y + HBM_H / 2, label, ha="center", va="center", fontsize=LABEL, fontweight="bold")


# Intermediates that exist only to be re-read by the next kernel.
THROWAWAY = ("s", "t")


def kernel(x, w, name, reads, writes, fill, edge):
    ax.add_patch(
        FancyBboxPatch((x, BOX_Y), w, BOX_H, boxstyle="round,pad=0.04", facecolor=fill, edgecolor=edge, lw=1.4)
    )
    ax.text(
        x + w / 2,
        BOX_Y + BOX_H * 0.62,
        name,
        ha="center",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        fontfamily=MONO,
    )
    ax.text(
        x + w / 2,
        BOX_Y + BOX_H * 0.25,
        "in registers",
        ha="center",
        va="center",
        fontsize=TINY,
        color=INK3,
    )
    total = len(reads) + len(writes)
    xs = [x + w * (i + 1) / (total + 1) for i in range(total)]
    for i, t in enumerate(reads):
        tcolor = ROSE if t in THROWAWAY else INK2
        ax.annotate(
            "",
            xy=(xs[i], BOX_Y + BOX_H + 0.06),
            xytext=(xs[i], HBM_Y - 0.06),
            arrowprops={"arrowstyle": "-|>", "color": tcolor, "lw": 1.5},
        )
        ax.text(
            xs[i] + 0.07,
            (HBM_Y + BOX_Y + BOX_H) / 2,
            t,
            fontsize=TINY,
            color=tcolor,
            ha="left",
            va="center",
            fontfamily=MONO,
        )
    for j, t in enumerate(writes):
        i = len(reads) + j
        tcolor = ROSE if t in THROWAWAY else INK
        ax.annotate(
            "",
            xy=(xs[i], HBM_Y - 0.06),
            xytext=(xs[i], BOX_Y + BOX_H + 0.06),
            arrowprops={"arrowstyle": "-|>", "color": tcolor, "lw": 1.5},
        )
        ax.text(
            xs[i] + 0.07,
            (HBM_Y + BOX_Y + BOX_H) / 2,
            t,
            fontsize=TINY,
            color=tcolor,
            ha="left",
            va="center",
            fontweight="bold",
            fontfamily=MONO,
        )


# ── Naive: three kernels, 8 tensor passes ──────────────────────────────
ax.text(0.3, 4.82, "NAIVE — 3 kernels", fontsize=SECTION, fontweight="bold", color=INK, va="top")
hbm_bar(0.3, 6.1, "HBM")
kernel(0.35, 1.9, "s=sigmoid(a)", ["a"], ["s"], AMBER_T, AMBER)
kernel(2.4, 1.9, "t=a·s", ["a", "s"], ["t"], AMBER_T, AMBER)
kernel(4.45, 1.9, "y=t·b", ["t", "b"], ["y"], AMBER_T, AMBER)
ax.text(
    0.3, 0.95, "8 tensor passes over HBM", fontsize=LABEL, fontweight="bold", color=AMBER, fontfamily=MONO, va="top"
)
ax.text(
    0.3,
    0.55,
    "5 reads + 3 writes — s and t exist only\nto be re-read by the next kernel",
    fontsize=TINY,
    color=INK2,
    va="top",
    linespacing=1.3,
)

# ── Fused: one kernel, 3 passes ────────────────────────────────────────
ax.text(
    7.1, 4.82, "FUSED — 1 kernel  (Liger / torch.compile)", fontsize=SECTION, fontweight="bold", color=INK, va="top"
)
hbm_bar(7.1, 4.1, "HBM")
kernel(7.75, 2.8, "y=silu(a)·b", ["a", "b"], ["y"], TEAL_T, TEAL)
ax.text(
    7.1,
    0.95,
    "3 tensor passes — ~2.7× less traffic",
    fontsize=LABEL,
    fontweight="bold",
    color=TEAL,
    fontfamily=MONO,
    va="top",
)
ax.text(
    7.1,
    0.55,
    "same math, same FLOPs; intermediates\nnever leave registers",
    fontsize=TINY,
    color=INK2,
    va="top",
    linespacing=1.3,
)

save(plt.gcf(), "fusion_roundtrips")
plt.close()
print("✓ fusion_roundtrips.png")
