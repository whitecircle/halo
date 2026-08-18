"""Generate EP Group Tree + Multi-Node EP+TP diagrams — individual images."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.size"] = 10.5

GROUP_BG = "#f1f5f9"
GPU_BG = ACCENT_LIGHT


def tree_node(ax, cx, cy, w, h, text, bg=CARD, border=CARD_BORDER, fs=11):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.05", facecolor=bg, edgecolor=border, linewidth=1.1
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=TEXT, linespacing=1.3)


def tree_edge(ax, x1, y1, x2, y2):
    mid_y = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, mid_y], color=BORDER_SOFT, lw=1.3, solid_capstyle="round")
    ax.plot([x1, x2], [mid_y, mid_y], color=BORDER_SOFT, lw=1.3, solid_capstyle="round")
    ax.plot([x2, x2], [mid_y, y2], color=BORDER_SOFT, lw=1.3, solid_capstyle="round")


# Image 1: EP Group Hierarchy
fig, ax = plt.subplots(figsize=(11, 11))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 9.7, "EP Group Hierarchy", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
ax.text(5, 9.0, "4 GPUs  \u00b7  EP = 2  \u00b7  DP = 4", ha="center", va="top", fontsize=12, color=TEXT_SEC)

tree_node(ax, 5, 8.0, 5.5, 0.9, "WORLD", bg=OP_BG, border=OP_BG, fs=14)
ax.texts[-1].set_color(OP_TEXT)

tree_edge(ax, 5, 7.55, 2.7, 6.4)
tree_edge(ax, 5, 7.55, 7.3, 6.4)
tree_node(ax, 2.7, 6.0, 3.4, 0.85, "EP Group 0\nranks 0, 1", bg=GROUP_BG, border=CARD_BORDER, fs=10.5)
tree_node(ax, 7.3, 6.0, 3.4, 0.85, "EP Group 1\nranks 2, 3", bg=GROUP_BG, border=CARD_BORDER, fs=10.5)

# GPUs (width 1.7, spaced so the inner two never collide)
for gx, parent_cx, label, exp in [
    (1.4, 2.7, "GPU 0", "exp 0\u201315"),
    (4.0, 2.7, "GPU 1", "exp 16\u201331"),
    (6.0, 7.3, "GPU 2", "exp 0\u201315"),
    (8.6, 7.3, "GPU 3", "exp 16\u201331"),
]:
    tree_edge(ax, parent_cx, 5.57, gx, 4.45)
    tree_node(ax, gx, 4.0, 1.7, 0.8, label, bg=GPU_BG, border=ACCENT, fs=10.5)
    ax.text(gx, 3.2, exp, ha="center", va="top", fontsize=9.5, color=TEXT_SEC)

for x1, x2, cx_label in [(1.4, 4.0, 2.7), (6.0, 8.6, 7.3)]:
    ax.annotate(
        "",
        xy=(x2 - 0.15, 2.6),
        xytext=(x1 + 0.15, 2.6),
        arrowprops={"arrowstyle": "<->", "color": ACCENT, "lw": 1.4},
        zorder=2,
    )
    badge_w = 1.8
    badge = FancyBboxPatch(
        (cx_label - badge_w / 2, 1.95),
        badge_w,
        0.45,
        boxstyle="round,pad=0.04",
        facecolor=OP_BG,
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(badge)
    ax.text(
        cx_label,
        2.175,
        "All-to-All",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=OP_TEXT,
        zorder=5,
    )
    ax.text(cx_label, 1.45, "DeepEP within group", ha="center", va="top", fontsize=9, color=TEXT_TERT, style="italic")

save(plt.gcf(), "ep_group_hierarchy")
plt.close()
print("\u2713 ep_group_hierarchy.png")


# Image 2: Multi-Node EP+TP
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

ax.text(6, 9.8, "Multi-Node EP + TP", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
# The EP+TP shape ParallelismConfig accepts across NVLink domains: one global EP group spanning the
# job. Node-local EP groups under TP (ep8/tp8 here) are rejected by ``_validate_tp``.
ax.text(
    6,
    9.25,
    "2 Nodes  \u00d7  8 GPUs  \u00b7  EP = 16 (global)  \u00b7  TP = 8  \u00b7  DP = 2",
    ha="center",
    va="top",
    fontsize=12,
    color=TEXT_SEC,
)


def draw_node_panel(ax, x, y, w, h, title, tp_label, batch_label):
    node = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06", facecolor=CARD, edgecolor=CARD_BORDER, linewidth=1.2
    )
    ax.add_patch(node)
    ax.text(x + w / 2, y + h + 0.15, title, ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=TEXT)

    inner_pad = 0.25
    inner_w = w - 2 * inner_pad
    block_h = 1.6

    tp_y = y + h - inner_pad - block_h
    tp = FancyBboxPatch(
        (x + inner_pad, tp_y),
        inner_w,
        block_h,
        boxstyle="round,pad=0.04",
        facecolor=GROUP_BG,
        edgecolor=BORDER_SOFT,
        linewidth=1.0,
    )
    ax.add_patch(tp)
    ax.text(
        x + w / 2,
        tp_y + block_h / 2 + 0.2,
        tp_label,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        x + w / 2,
        tp_y + block_h / 2 - 0.25,
        "Attention sharded via\nDTensor (NVLink)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_SEC,
        linespacing=1.3,
    )

    ax.text(
        x + w / 2,
        y + 0.35,
        batch_label,
        ha="center",
        va="center",
        fontsize=12.5,
        color=ACCENT,
        fontweight="bold",
        style="italic",
    )


nw, nh = 5.2, 3.1
draw_node_panel(ax, 0.2, 5.3, nw, nh, "Node 0  (ranks 0\u20137)", "TP Group  (all 8)", "Batch A")
draw_node_panel(ax, 6.6, 5.3, nw, nh, "Node 1  (ranks 8\u201315)", "TP Group  (all 8)", "Batch B")

# One EP group spanning both nodes; two per-node groups is the shape ParallelismConfig rejects.
ep_x, ep_y, ep_w, ep_h = 0.2, 3.3, 11.6, 1.5
ep = FancyBboxPatch(
    (ep_x, ep_y), ep_w, ep_h, boxstyle="round,pad=0.06", facecolor=ACCENT_LIGHT, edgecolor=ACCENT, linewidth=1.2
)
ax.add_patch(ep)
ax.text(
    6,
    ep_y + ep_h / 2 + 0.24,
    "EP Group  (one global group  \u00b7  all 16 ranks)",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=TEXT,
)
ax.text(
    6,
    ep_y + ep_h / 2 - 0.3,
    "Experts distributed via DeepEP all-to-all \u2014 spans both nodes over InfiniBand",
    ha="center",
    va="center",
    fontsize=9.5,
    color=TEXT_SEC,
)
for cx in (2.8, 9.2):
    ax.annotate(
        "",
        xy=(cx, ep_y + ep_h + 0.02),
        xytext=(cx, 5.28),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.4},
        zorder=3,
    )

mid_y = 2.4
ax.plot([0.5, 11.5], [mid_y, mid_y], color=BORDER_SOFT, lw=1.0, ls="--", zorder=1)
for dx in [0.5, 11.5]:
    ax.plot(dx, mid_y, "o", color=ACCENT, markersize=4.5, zorder=5)

badge_text = "DP = 2  \u00b7  Gradient Sync"
badge_w = 3.6
badge = FancyBboxPatch(
    (6 - badge_w / 2, mid_y - 0.22),
    badge_w,
    0.44,
    boxstyle="round,pad=0.05",
    facecolor=OP_BG,
    edgecolor="none",
    zorder=4,
)
ax.add_patch(badge)
ax.text(6, mid_y, badge_text, ha="center", va="center", fontsize=10, fontweight="bold", color=OP_TEXT, zorder=5)
ax.text(
    6,
    mid_y - 0.5,
    "FSDP2 full-shard via InfiniBand",
    ha="center",
    va="top",
    fontsize=9.5,
    color=TEXT_TERT,
    style="italic",
)

save(plt.gcf(), "ep_multi_node_layout")
plt.close()
print("\u2713 ep_multi_node_layout.png")
