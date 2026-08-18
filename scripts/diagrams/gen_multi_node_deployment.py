"""Generate Multi-Node Deployment diagrams.

Split into two individual images:
- multi_node_separate_inference.png — Scenario 1: Separate Inference Node
- multi_node_dedicated_rollout.png — Scenario 2: Dedicated Rollout Nodes
"""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.size"] = 12

PURPLE_M = "#a78bfa"  # NCCL
ORANGE_M = "#fb923c"  # HTTP
GREEN_M = "#86efac"  # Ray

NODE_TRAIN = "#d1fae5"
NODE_INF = "#dbeafe"
NODE_ROLL = "#ffedd5"


def draw_box(ax, x, y, w, h, title="", lines=None, bg=CARD, border=CARD_BORDER, title_size=14):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    if title:
        ax.text(
            x + w / 2,
            y + h - 0.22,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=TEXT,
        )
    if lines:
        sep_y = y + h - 0.58
        ax.plot([x + 0.18, x + w - 0.18], [sep_y, sep_y], color=BORDER_SOFT, lw=0.7, zorder=2)
        for i, line in enumerate(lines):
            ly = y + h - 0.90 - i * 0.44
            ax.plot(x + 0.25, ly, "o", color=ACCENT, markersize=3, zorder=3)
            ax.text(x + 0.42, ly, line, ha="left", va="center", fontsize=11, color=TEXT_SEC, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=BORDER_SOFT, lw=1.2, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": style, "color": color, "lw": lw, "shrinkA": 4, "shrinkB": 4},
        zorder=3,
    )


def arrow_label(ax, x, y, text, color=TEXT_SEC, size=10, rotation=0):
    """Inline label near an arrow, with no background box."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, rotation=rotation, zorder=4)


# Image 1 — Scenario 1: Separate Inference Node
fig, ax = plt.subplots(figsize=(13, 11.5))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 12)
ax.set_ylim(0, 11.5)
ax.set_aspect("equal")
ax.axis("off")

ax.text(
    6, 11.3, "Scenario 1: Separate Inference Node", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT
)
ax.text(
    6,
    10.75,
    "Training + rollout actors on Node 1  \u00b7  vLLM on Node 2",
    ha="center",
    va="top",
    fontsize=12,
    color=TEXT_TERT,
)

draw_box(ax, 0.3, 3.8, 5.2, 6.5, bg=NODE_TRAIN, border=BORDER_SOFT)
ax.text(2.9, 10.05, "Node 1  (Training)", ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)

draw_box(
    ax,
    0.6,
    7.1,
    4.6,
    2.6,
    "Trainer (GPU 0)",
    [
        "Model + Optimizer",
        "VLLMClient",
        "RolloutManager",
    ],
    bg=CARD,
    border=CARD_BORDER,
    title_size=13,
)

draw_box(ax, 0.6, 4.1, 4.6, 2.6, "Ray Actors (CPU)", [], bg=CARD, border=CARD_BORDER, title_size=13)
for i in range(4):
    ax_x = 1.0 + i * 1.05
    actor = FancyBboxPatch(
        (ax_x, 4.35), 0.8, 1.5, boxstyle="round,pad=0.04", facecolor=ACCENT_LIGHT, edgecolor=CARD_BORDER, linewidth=0.5
    )
    ax.add_patch(actor)
    ax.text(ax_x + 0.4, 5.4, f"A{i}", ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT)
    ax.text(ax_x + 0.4, 4.9, "Env", ha="center", va="center", fontsize=10, color=TEXT_SEC)

draw_box(ax, 6.5, 3.8, 5.2, 6.5, bg=NODE_INF, border=BORDER_SOFT)
ax.text(9.1, 10.05, "Node 2  (Inference)", ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)

draw_box(
    ax,
    6.8,
    5.6,
    4.6,
    3.5,
    "vLLM Server (GPU 0)",
    [
        "Model (inference)",
        ":8000 HTTP generation",
        ":51216 NCCL group port",
        "Weight sync receiver",
    ],
    bg=CARD,
    border=CARD_BORDER,
    title_size=13,
)

draw_arrow(ax, 5.2, 8.4, 6.8, 8.4, PURPLE_M, lw=1.4)
arrow_label(ax, 6.0, 8.7, "NCCL :51216", color="#7c3aed", size=10)

draw_arrow(ax, 5.2, 5.3, 6.8, 6.6, ORANGE_M, lw=1.2)
arrow_label(ax, 6.0, 5.55, "HTTP :8000", color="#c2410c", size=10)

draw_box(ax, 0.5, 0.3, 11.0, 3.0, bg=CARD, border=CARD_BORDER)
ax.text(6.0, 3.05, "Network Requirements", ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT)
ax.plot([0.7, 11.3], [2.55, 2.55], color=BORDER_SOFT, lw=0.7)

reqs = [
    ("HTTP :8000", "Actors \u2192 vLLM", "#c2410c", 2.0),
    ("NCCL :51216", "Trainer \u2192 vLLM", "#7c3aed", 4.6),
    ("Ray :6379+", "All nodes \u2192 Head", "#16a34a", 7.6),
    ("Low latency", "Same VPC/DC", TEXT_SEC, 10.2),
]
for label, desc, color, sx in reqs:
    ax.plot(sx, 1.95, "s", color=color, markersize=7, zorder=4)
    ax.text(sx, 1.5, label, ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    ax.text(sx, 1.0, desc, ha="center", va="center", fontsize=10.5, color=TEXT_SEC)

save(plt.gcf(), "multi_node_separate_inference")
plt.close()
print("\u2713 multi_node_separate_inference.png")


# Image 2 — Scenario 2: Dedicated Rollout Nodes
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.set_aspect("equal")
ax.axis("off")

ax.text(
    7, 10.8, "Scenario 2: Dedicated Rollout Nodes", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT
)
ax.text(
    7, 10.25, "Separate training, rollout, and inference nodes", ha="center", va="top", fontsize=12, color=TEXT_TERT
)

draw_box(ax, 0.3, 6.2, 4.4, 3.6, bg=NODE_TRAIN, border=BORDER_SOFT)
ax.text(2.5, 9.55, "Training Node", ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT)

draw_box(
    ax,
    0.55,
    6.45,
    3.9,
    2.7,
    "Trainer (GPU)",
    [
        "Model + Optimizer",
        "InferenceClientManager",
        "Weight sync sender",
    ],
    bg=CARD,
    border=CARD_BORDER,
    title_size=12,
)

draw_box(ax, 8.8, 6.2, 4.9, 3.6, bg=NODE_INF, border=BORDER_SOFT)
ax.text(11.25, 9.55, "Inference Nodes", ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT)

draw_box(
    ax,
    9.05,
    8.0,
    4.4,
    1.5,
    "vLLM #1 (GPU 0)",
    [
        ":8000  \u00b7  :51216",
    ],
    bg=CARD,
    border=CARD_BORDER,
    title_size=12,
)

draw_box(
    ax,
    9.05,
    6.35,
    4.4,
    1.5,
    "vLLM #2 (GPU 0)",
    [
        ":8001  \u00b7  :51217",
    ],
    bg=CARD,
    border=CARD_BORDER,
    title_size=12,
)

draw_box(ax, 0.3, 2.9, 13.4, 2.7, bg=NODE_ROLL, border=BORDER_SOFT)
ax.text(7.0, 5.35, "Rollout Nodes", ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT)

node_w, node_h = 4.0, 1.6
for node_idx, (nx, name) in enumerate([(0.55, "Node 2"), (5.0, "Node 3"), (9.45, "Node 4")]):
    draw_box(ax, nx, 3.05, node_w, node_h, name + " (CPU)", [], bg=CARD, border=CARD_BORDER, title_size=11)
    for i in range(4):
        ax_x = nx + 0.3 + i * 0.9
        actor = FancyBboxPatch(
            (ax_x, 3.15),
            0.7,
            0.85,
            boxstyle="round,pad=0.04",
            facecolor=ACCENT_LIGHT,
            edgecolor=CARD_BORDER,
            linewidth=0.5,
        )
        ax.add_patch(actor)
        ax.text(
            ax_x + 0.35,
            3.6,
            f"A{node_idx * 4 + i}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
        )

draw_arrow(ax, 4.7, 8.8, 9.05, 8.8, PURPLE_M, lw=1.3)
arrow_label(ax, 6.85, 9.1, "NCCL :51216", color="#7c3aed", size=9.5)

draw_arrow(ax, 4.7, 7.2, 9.05, 7.2, PURPLE_M, lw=1.3)
arrow_label(ax, 6.85, 7.5, "NCCL :51217", color="#7c3aed", size=9.5)

draw_arrow(ax, 2.5, 6.2, 2.5, 5.6, GREEN_M, lw=1.0)
arrow_label(ax, 3.1, 5.9, "Ray", color="#16a34a", size=9.5)

draw_arrow(ax, 11.45, 5.6, 11.45, 6.35, ORANGE_M, lw=1.1)
arrow_label(ax, 12.5, 5.95, "HTTP round-robin", color="#c2410c", size=9)

draw_box(ax, 0.4, 0.3, 13.2, 2.3, bg=CARD, border=CARD_BORDER)
ax.text(7.0, 2.35, "Configuration", ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT)
ax.plot([0.6, 13.4], [1.95, 1.95], color=BORDER_SOFT, lw=0.7)

config_lines = [
    'ray_address: "ray-head:6379"',
    "num_rollout_workers: 128",
    "rollout_server_configs:",
    '  - url: "http://inf-1:8000"    # group_port: 51216',
    '  - url: "http://inf-2:8001"    # group_port: 51217',
]
for i, line in enumerate(config_lines):
    ax.text(0.9, 1.6 - i * 0.28, line, ha="left", va="center", fontsize=10, color=TEXT_SEC, family="monospace")

save(plt.gcf(), "multi_node_dedicated_rollout")
plt.close()
print("\u2713 multi_node_dedicated_rollout.png")
