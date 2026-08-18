"""Generate the throughput-vs-M diagram for gpu-training-theory.md §4.

MEASURED bf16 GEMM efficiency on a B300, gpt-oss expert (K=N=2880, the §2 shape),
swept over M. Plotted as % of the shape's own peak so the curve is clock-
invariant (numerator and denominator both scale with boost clocks) and carries
no absolute number to clash with the §2 table. The point of §4: the tile/wave
structure makes throughput non-smooth — it RAMPS at small M (few tiles → most of
the 148 SMs idle) then RIPPLES near peak, not the deep sawtooth the geometry
predicts in the abstract (small M is occupancy-limited regardless).
"""

import matplotlib.pyplot as plt
from _theory_style import *

# ── Measured: (M, % of this shape's peak) for bf16 [M,2880]@[2880,2880], B300 ──
DATA = [
    (64, 5),
    (128, 11),
    (192, 17),
    (256, 21),
    (320, 27),
    (384, 31),
    (448, 38),
    (512, 42),
    (576, 47),
    (640, 53),
    (704, 54),
    (768, 64),
    (832, 53),
    (896, 74),
    (960, 80),
    (1024, 84),
    (1088, 90),
    (1152, 94),
    (1216, 82),
    (1280, 87),
    (1344, 85),
    (1408, 94),
    (1472, 89),
    (1536, 90),
    (1600, 74),
    (1664, 75),
    (1728, 79),
    (1792, 67),
    (1856, 84),
    (1920, 86),
    (1984, 69),
    (2048, 73),
    (2112, 80),
    (2176, 85),
    (2240, 88),
    (2304, 87),
    (2368, 72),
    (2496, 81),
    (2560, 81),
    (2624, 82),
    (2752, 78),
    (2880, 84),
    (2944, 86),
    (3008, 85),
    (3072, 86),
    (3136, 80),
    (3264, 87),
    (3328, 85),
    (3392, 88),
    (3456, 88),
    (3520, 73),
    (3584, 78),
    (3712, 80),
    (3840, 77),
    (3968, 80),
    (4096, 82),
    (4224, 76),
    (4288, 84),
    (4416, 84),
    (4480, 84),
    (4608, 86),
    (4736, 81),
    (4800, 84),
    (4928, 74),
    (5056, 80),
    (5184, 79),
    (5312, 79),
    (5440, 78),
    (5568, 77),
    (5696, 80),
    (5824, 79),
    (5952, 80),
    (6080, 82),
    (6144, 83),
    (6272, 79),
    (6400, 82),
    (6528, 80),
    (6656, 78),
    (6784, 77),
    (6912, 78),
    (7040, 81),
    (7168, 78),
    (7296, 81),
    (7424, 81),
    (7552, 82),
    (7680, 83),
    (7808, 78),
    (7936, 79),
    (8064, 78),
    (8192, 79),
]
xs = [m for m, _ in DATA]
ys = [e for _, e in DATA]

fig, ax = plt.subplots(figsize=(8.4, 5.6))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 8300)
ax.set_ylim(0, 106)

ax.axhline(100, ls=(0, (5, 4)), color=INK2, lw=1.2, zorder=2)
ax.text(8250, 101.5, "peak — machine full", ha="right", va="bottom", fontsize=TINY, color=INK2)
ax.axvspan(0, 1150, color=AMBER_T, zorder=0)

ax.plot(xs, ys, color=TEAL, lw=2.2, zorder=5, solid_capstyle="round")

ax.annotate(
    "small M: few tiles, most of the\n148 SMs idle → far below peak",
    xy=(448, 38),
    xytext=(950, 26),
    fontsize=SMALL,
    color=AMBER,
    fontweight="bold",
    ha="left",
    va="center",
    arrowprops={"arrowstyle": "-|>", "color": AMBER, "lw": 1.2, "shrinkA": 2, "shrinkB": 5},
)
ax.annotate(
    "near peak: efficiency ripples as tiles\nrepack across the SMs — not a\ndeep sawtooth",
    xy=(1984, 69),
    xytext=(3300, 44),
    fontsize=SMALL,
    color=INK,
    ha="left",
    va="center",
    arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2, "shrinkA": 2, "shrinkB": 5},
)

ax.set_xlabel("M — output rows (tokens per expert)", fontsize=SMALL, color=INK, labelpad=7)
ax.set_ylabel("Efficiency — % of this shape's peak", fontsize=SMALL, color=INK, labelpad=7)
ax.text(
    0.0,
    1.15,
    "Throughput is not smooth in M",
    transform=ax.transAxes,
    fontsize=TITLE,
    fontweight="bold",
    color=INK,
    ha="left",
    va="bottom",
)
ax.text(
    0.0,
    1.072,
    "Measured bf16 GEMM, B300, gpt-oss expert (K = N = 2880).",
    transform=ax.transAxes,
    fontsize=SUB,
    color=INK2,
    ha="left",
    va="bottom",
)
ax.text(
    0.0,
    1.008,
    "The small-M ramp dominates; tile/wave quantization is the ripple on top.",
    transform=ax.transAxes,
    fontsize=SUB,
    color=INK2,
    ha="left",
    va="bottom",
)

ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontfamily=MONO)
ax.set_xticks([0, 2000, 4000, 6000, 8000])
ax.set_xticklabels(["0", "2000", "4000", "6000", "8000"], fontfamily=MONO)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(colors=INK2)

plt.tight_layout()
save(plt.gcf(), "tile_quantization")
plt.close()
print("✓ tile_quantization.png")
