"""Generate the interconnect bandwidth-ladder diagram for gpu-training-theory.md §9.

Horizontal log-scale bars, one per interconnect tier (HBM, NVLink/NVSwitch,
PCIe, InfiniBand). Filled bar = measured, dashed outline extension = spec,
keeping the page's achieved-not-spec theme visible.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import Rectangle

# (name, role, measured GB/s, spec GB/s or None, value label lines, hue)
TIERS = [
    ("HBM", "GPU ↔ its own memory", 6600, 8000, "6.6 TB/s   (spec 8 TB/s)", TEAL),
    (
        "NVLink / NVSwitch",
        "GPU ↔ GPU, inside the node",
        767,
        1800,
        "spec 1.8 TB/s — ~4× below HBM\n767 GB/s all-reduce bus bw measured (16 GPUs, 2 nodes)",
        BLUE,
    ),
    ("PCIe Gen5 ×16", "CPU ↔ GPU", 64, None, "64 GB/s per direction\n~100× below HBM", AMBER),
    ("InfiniBand NDR", "node ↔ node", 48, None, "~48 GB/s  (400 Gb/s)\n~130× below HBM — the node boundary", ROSE),
]

fig, ax = plt.subplots(figsize=(9.9, 4.4))
fig.patch.set_facecolor(BG)
ax.set_xscale("log")
ax.set_xlim(9, 30000)
ax.set_ylim(-0.52, len(TIERS) - 0.48)
ax.invert_yaxis()

BAR_H = 0.7
for i, (name, role, meas, spec, vlabel, hue) in enumerate(TIERS):
    y = i - BAR_H / 2
    if spec:
        outline = Rectangle((9, y), spec - 9, BAR_H, facecolor="none", edgecolor=hue, lw=1.1)
        outline.set_linestyle((0, (3, 2)))
        ax.add_patch(outline)
    ax.add_patch(Rectangle((9, y), meas - 9, BAR_H, facecolor=hue, edgecolor=INK, lw=1.3))
    anchor = spec if spec else meas
    ax.text(
        anchor * 1.13,
        i,
        vlabel,
        va="center",
        ha="left",
        fontsize=SMALL,
        fontweight="bold",
        color=INK,
        linespacing=1.34,
        fontfamily=MONO,
    )
    label_c = on_fill(hue)  # white or dark ink, whichever reads on this bar
    ax.text(10.4, i - 0.16, name, va="center", ha="left", fontsize=LABEL, fontweight="bold", color=label_c)
    ax.text(10.4, i + 0.19, role, va="center", ha="left", fontsize=TINY, color=label_c, alpha=0.82)

ax.set_yticks([])
ax.set_xlabel("bandwidth per GPU — GB/s  (log scale)", fontsize=11, color=INK, labelpad=8)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#d9dee5")
ax.tick_params(colors=INK2, which="both")

ax.set_title("The bandwidth ladder", fontsize=TITLE, fontweight="bold", color=INK, pad=32, loc="left")
ax.text(
    0.0,
    1.06,
    "filled = measured · dashed = spec — a collective is priced by the slowest tier it touches",
    transform=ax.transAxes,
    fontsize=SUB,
    color=INK2,
    ha="left",
    va="bottom",
)

plt.tight_layout()
save(plt.gcf(), "interconnect_tiers")
plt.close()
print("✓ interconnect_tiers.png")
