"""Generate the interconnect bandwidth ladder for gpu-training-theory.md §9.

One row per link on a log axis, split at the node boundary: a filled dot is a measured reading, a hollow dot a spec.
Dots rather than bars, since a log axis has no zero for a bar to grow from. The 767 GB/s all-reduce sits with the
cross-node rows because it ran across two nodes, so it does not read as NVLink alone.
"""

import matplotlib.pyplot as plt
from _chart_style import *
from matplotlib.ticker import FixedLocator, NullLocator

# (section, [(link, what it joins, measured GB/s or None, spec GB/s or None, reading)])
SECTIONS = [
    (
        "INSIDE A NODE",
        [
            ("HBM", "a GPU and its own memory", 6600, 8000, "6.6 TB/s measured, spec 8 TB/s"),
            ("NVLink 5 / NVSwitch", "GPU to GPU", None, 1800, "spec 1.8 TB/s, about 4× below HBM"),
            ("PCIe Gen5 ×16", "CPU to GPU", None, 64, "spec 64 GB/s per direction, about 100× below HBM"),
        ],
    ),
    (
        "ACROSS NODES",
        [
            ("NCCL all-reduce", "16 GPUs on 2 nodes", 767, None, "767 GB/s bus bandwidth, measured"),
            ("EFA on B300", "node to node", 93, 100, "93 GB/s NCCL broadcast measured, spec about 100"),
            ("InfiniBand NDR", "node to node", None, 50, "spec about 50 GB/s, about 130× below HBM"),
        ],
    ),
]
X_MIN, X_MAX = 20, 20000
SECTION_GAP = 0.75
HBM_ROW = "HBM"

fig, ax = plt.subplots(figsize=(11.0, 4.5))
fig.subplots_adjust(left=0.205, right=0.61, top=0.79, bottom=0.13)
heading(
    fig,
    "The bandwidth ladder",
    "filled = measured · hollow = spec · a collective is priced by the slowest tier it touches",
)

ax.set_xscale("log")
ax.set_xlim(X_MIN, X_MAX)

y = 0.0
ticks, tick_labels = [], []
for section, rows in SECTIONS:
    ax.text(
        -0.455, y - 0.72, section, transform=ax.get_yaxis_transform(), fontsize=SMALL, fontweight="bold", color=INK2
    )
    for link, joins, measured, spec, reading in rows:
        hue = AMBER if link == HBM_ROW else VIOLET
        ax.axhline(y, color=LINE, lw=0.7, zorder=1)
        if measured and spec:
            ax.plot([measured, spec], [y, y], color=hue, lw=2.0, zorder=2, solid_capstyle="butt")
        if spec:
            ax.plot(spec, y, "o", ms=8.5, mfc=BG, mec=hue, mew=1.8, zorder=3)
        if measured:
            ax.plot(measured, y, "o", ms=9, color=hue, zorder=4)
        ax.text(
            -0.455, y - 0.13, link, transform=ax.get_yaxis_transform(), fontsize=LABEL, fontweight="bold", va="center"
        )
        ax.text(-0.455, y + 0.21, joins, transform=ax.get_yaxis_transform(), fontsize=TINY, color=INK3, va="center")
        ax.text(1.03, y, reading, transform=ax.get_yaxis_transform(), fontsize=SMALL, color=INK, va="center")
        y += 1
    y += SECTION_GAP
ax.set_ylim(y - SECTION_GAP - 0.45, -1.0)

ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"{value / 1000:g} TB/s" if value >= 1000 else f"{value:g} GB/s")
)
ax.set_yticks([])
ax.tick_params(length=0, labelsize=SMALL)
ax.grid(axis="x", color=LINE, lw=0.7)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.set_xlabel("bandwidth per GPU, log scale", fontsize=SMALL, color=INK2)

save(fig, "interconnect_tiers")
