"""Generate the roofline diagram for gpu-training-theory.md §2.

A log-log roofline: the diagonal HBM-bandwidth ceiling rising to the flat
compute ceiling, meeting at the ridge point (AI = peak / bandwidth). Anchors
are the B300 bf16 numbers used throughout the theory page.
"""

import matplotlib.pyplot as plt
import numpy as np
from _theory_style import *

PEAK = 1818.0  # TFLOP/s — best dense bf16 8192^3 GEMM, idle B300 (arch ceiling 2464)
BW = 6.6  # TFLOP/s per FLOP/byte  (6.6 TB/s)
RIDGE = PEAK / BW  # ≈ 275 FLOP/byte

X_MIN, X_MAX = 2.0, 8000.0
Y_MIN, Y_MAX = 10.0, 3200.0

fig, ax = plt.subplots(figsize=(9.2, 6.3))
fig.patch.set_facecolor(BG)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

ax.axvspan(X_MIN, RIDGE, color=AMBER_T, zorder=0)

ai_diag = np.geomspace(X_MIN, RIDGE, 200)
ax.plot(ai_diag, BW * ai_diag, color=INK, lw=2.6, zorder=5, solid_capstyle="round")
ai_flat = np.geomspace(RIDGE, X_MAX, 200)
ax.plot(ai_flat, np.full_like(ai_flat, PEAK), color=INK, lw=2.6, zorder=5, solid_capstyle="round")

ax.plot([RIDGE], [PEAK], "o", ms=8, color=ROSE, zorder=6)
ax.plot([RIDGE, RIDGE], [Y_MIN, PEAK], ls=(0, (4, 3)), color=INK2, lw=1.2, zorder=4)
ax.annotate(
    f"ridge point\nAI ≈ {RIDGE:.0f} FLOP/byte",
    xy=(RIDGE, PEAK),
    xytext=(RIDGE / 5.5, PEAK * 1.12),
    fontsize=SMALL,
    fontweight="bold",
    color=INK,
    ha="left",
    va="bottom",
    arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2, "shrinkA": 2, "shrinkB": 6},
)

ax.text(
    X_MAX / 1.25,
    PEAK * 1.16,
    f"compute ceiling  ·  measured {PEAK:.0f} TFLOP/s (bf16)",
    ha="right",
    va="bottom",
    fontsize=SMALL,
    color=INK2,
)

ax.text(13, 1780, "MEMORY-BOUND", ha="center", va="top", fontsize=SECTION, fontweight="bold", color=AMBER)
ax.text(
    13,
    1330,
    "below the ridge\nmore cores don't help —\nmove fewer bytes, faster HBM",
    ha="center",
    va="top",
    fontsize=SMALL,
    color=INK2,
    linespacing=1.32,
)
ax.text(1500, 132, "COMPUTE-BOUND", ha="center", va="bottom", fontsize=SECTION, fontweight="bold", color=BLUE)
ax.text(
    1500,
    116,
    "above the ridge\nfaster HBM doesn't help —\nbigger dims, lower precision",
    ha="center",
    va="top",
    fontsize=SMALL,
    color=INK2,
    linespacing=1.32,
)

ax.text(
    RIDGE / 1.18,
    15.5,
    "← fine-grained MoE experts (small per-expert M)",
    ha="right",
    va="bottom",
    fontsize=TINY,
    style="italic",
    color=INK3,
)
ax.text(
    RIDGE * 1.18, 15.5, "large-M GEMM · dense MLP →", ha="left", va="bottom", fontsize=TINY, style="italic", color=INK3
)

MEASURED = [(118, 180, "M=128"), (378, 548, "M=512"), (1225, 1375, "M=8192")]  # the §2 sweep table
POINT_COLORS = [AMBER, BLUE, TEAL]  # memory-bound · mid · throughput
for (ai, tf, _label), pc in zip(MEASURED, POINT_COLORS, strict=True):
    ax.plot([ai], [tf], "o", ms=7.5, markerfacecolor=BG, markeredgecolor=pc, markeredgewidth=1.6, zorder=7)
ax.text(
    MEASURED[1][0] * 1.13,
    MEASURED[1][1],
    "M=512",
    fontsize=TINY,
    color=INK2,
    ha="left",
    va="center",
    fontfamily=MONO,
)
ax.text(
    MEASURED[2][0], MEASURED[2][1] / 1.22, "M=8192", fontsize=TINY, color=INK2, ha="center", va="top", fontfamily=MONO
)
ax.annotate(
    "M=128 — below even the memory roof:\ntoo few tiles to fill 148 SMs (§1)",
    xy=(MEASURED[0][0], MEASURED[0][1]),
    xytext=(2.7, 350),
    fontsize=TINY,
    color=INK2,
    ha="left",
    va="center",
    linespacing=1.32,
    arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.1, "shrinkA": 4, "shrinkB": 5},
)
ax.text(
    0.985,
    0.03,
    "○ measured bf16 expert GEMM · K=N=2880 (§2)",
    transform=ax.transAxes,
    fontsize=TINY,
    color=INK3,
    ha="right",
    va="bottom",
    style="italic",
)

ax.set_xlabel("Arithmetic intensity — FLOP / byte  (log scale)", fontsize=11, color=INK, labelpad=8)
ax.set_ylabel("Achievable throughput — TFLOP/s  (log scale)", fontsize=11, color=INK, labelpad=8)
ax.set_title("The Roofline", fontsize=TITLE, fontweight="bold", color=INK, pad=32, loc="left")
ax.text(
    0.0,
    1.045,
    "NVIDIA B300 · bf16 · achieved, not spec peak",
    transform=ax.transAxes,
    fontsize=SUB,
    color=INK2,
    ha="left",
    va="bottom",
)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(colors=INK2, which="both")

plt.tight_layout()

# Axis tick numbers render in mono (matching the code-like point labels).
fig.canvas.draw()
for _lbl in (
    ax.get_xticklabels() + ax.get_yticklabels() + ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True)
):
    _lbl.set_fontfamily(MONO)

# Diagonal label rotated to the line's on-screen angle — computed after layout so the transform is final.
fig.canvas.draw()
a0, a1 = 6.0, 60.0  # two AI points on the diagonal, both below the ridge
(px0, py0) = ax.transData.transform((a0, BW * a0))
(px1, py1) = ax.transData.transform((a1, BW * a1))
diag_angle = np.degrees(np.arctan2(py1 - py0, px1 - px0))
ax.text(
    7.4,
    BW * 7.4 * 0.56,
    "memory ceiling = AI × 6.6 TB/s",
    ha="left",
    va="top",
    fontsize=SMALL,
    color=INK2,
    rotation=diag_angle,
    rotation_mode="anchor",
)

save(plt.gcf(), "roofline")
plt.close()
print("✓ roofline.png")
