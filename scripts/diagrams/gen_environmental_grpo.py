"""Generate Environmental GRPO Architecture diagrams — individual images."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

TRAIN_BG = CARD
VLLM_BG = ACCENT_LIGHT
RAY_BG = "#f8fafc"
ACTOR_BG = "#f1f5f9"


def draw_comp(ax, x, y, w, h, title, lines, bg=CARD, border=CARD_BORDER, title_size=12):
    """Draw a component box with title and bullet-point lines."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=TEXT)
    if lines:
        sep_y = y + h - 0.55
        ax.plot([x + 0.2, x + w - 0.2], [sep_y, sep_y], color=BORDER_SOFT, lw=0.6, zorder=2)
    for i, line in enumerate(lines):
        ly = y + h - 0.80 - i * 0.42
        ax.plot(x + 0.25, ly, "o", color=ACCENT, markersize=2.5, zorder=3)
        ax.text(x + 0.4, ly, line, ha="left", va="center", fontsize=10, color=TEXT_SEC, zorder=3)


def draw_conn_h(ax, x1, y1, x2, y2, label, color=ACCENT, above=True):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "shrinkA": 4, "shrinkB": 4},
        zorder=2,
    )
    mx = (x1 + x2) / 2
    gap_w = abs(x2 - x1)
    offset = 0.25 if above else -0.25
    badge_w = min(len(label) * 0.085 + 0.3, gap_w - 0.2)
    badge_w = max(badge_w, 0.6)
    badge = FancyBboxPatch(
        (mx - badge_w / 2, y1 + offset - 0.14),
        badge_w,
        0.28,
        boxstyle="round,pad=0.04",
        facecolor=OP_BG,
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(badge)
    ax.text(mx, y1 + offset, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=OP_TEXT, zorder=5)


def draw_conn_v(ax, x, y1, y2, label=""):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.5, "shrinkA": 4, "shrinkB": 4},
        zorder=2,
    )
    if label:
        my = (y1 + y2) / 2
        ax.text(x + 0.15, my, label, ha="left", va="center", fontsize=9, color=TEXT_SEC, style="italic")


# Image 1: Single-Server Mode
fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 9.85, "Single-Server Mode", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
ax.text(
    5,
    9.1,
    "Prefetch auto-disabled \u2014 vLLM blocks during NCCL sync",
    ha="center",
    va="top",
    fontsize=10,
    color=TEXT_TERT,
)

draw_comp(
    ax,
    0.2,
    5.0,
    3.5,
    3.8,
    "Training Node\n(GPU 1)",
    [
        "AsyncEnvironmentalGRPO",
        "Model + Optimizer",
        "VLLMClient",
        "RolloutManager",
    ],
    bg=TRAIN_BG,
)

draw_comp(
    ax,
    6.2,
    6.0,
    3.5,
    2.8,
    "vLLM Server\n(GPU 0)",
    [
        "Model (inference)",
        "Weight sync receiver",
        "port 8000",
    ],
    bg=VLLM_BG,
    border=ACCENT,
)

draw_conn_h(ax, 3.7, 7.8, 6.2, 7.8, "NCCL weights", above=True)
draw_conn_h(ax, 3.7, 6.8, 6.2, 6.8, "HTTP /generate", above=False)

draw_comp(ax, 0.6, 0.4, 8.8, 3.6, "Ray Environment Actors", [], bg=RAY_BG)
for j in range(5):
    ax_x = 1.0 + j * 1.65
    actor = FancyBboxPatch(
        (ax_x, 0.7), 1.35, 2.4, boxstyle="round,pad=0.04", facecolor=ACTOR_BG, edgecolor=BORDER_SOFT, linewidth=0.7
    )
    ax.add_patch(actor)
    ax.text(ax_x + 0.675, 2.75, f"Actor {j}", ha="center", va="top", fontsize=9.5, fontweight="bold", color=TEXT)
    ax.text(
        ax_x + 0.675,
        1.65,
        "Environment\nTokenizer",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_SEC,
        linespacing=1.3,
    )

draw_conn_v(ax, 2.0, 5.0, 4.0, "Ray")

save(plt.gcf(), "environmental_grpo_single_server")
plt.close()
print("\u2713 environmental_grpo_single_server.png")


# Image 2: Multi-Server Mode
fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 9.85, "Multi-Server Mode", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
ax.text(
    5,
    9.1,
    "Prefetch + streamed weight sync \u2014 every server quiesces from the first chunk",
    ha="center",
    va="top",
    fontsize=10,
    color=TEXT_TERT,
)

draw_comp(
    ax,
    0.2,
    4.8,
    3.5,
    4.0,
    "Training Node",
    [
        "AsyncEnvironmentalGRPO",
        "InferenceClientManager",
        "\u251c Client 1  (51216)",
        "\u2514 Client 2  (51217)",
        "Streamed weight sync",
    ],
    bg=TRAIN_BG,
)

draw_comp(
    ax,
    6.2,
    7.2,
    3.5,
    1.6,
    "Server 1  (GPU 0)",
    [
        "port 8000 \u00b7 group 51216",
    ],
    bg=VLLM_BG,
    border=ACCENT,
    title_size=10.5,
)

draw_comp(
    ax,
    6.2,
    5.0,
    3.5,
    1.6,
    "Server 2  (GPU 2)",
    [
        "port 8001 \u00b7 group 51217",
    ],
    bg=VLLM_BG,
    border=ACCENT,
    title_size=10.5,
)

draw_conn_h(ax, 3.7, 7.9, 6.2, 7.9, "NCCL", above=True)
draw_conn_h(ax, 3.7, 5.7, 6.2, 5.7, "NCCL", above=True)

ax.text(
    7.95,
    4.6,
    "HTTP round-robin\nto all servers",
    ha="center",
    va="top",
    fontsize=9,
    color=TEXT_SEC,
    style="italic",
    bbox={"boxstyle": "round,pad=0.15", "facecolor": CARD, "edgecolor": CARD_BORDER, "lw": 0.6},
)

draw_comp(ax, 0.6, 0.4, 8.8, 3.6, "Ray Environment Actors", [], bg=RAY_BG)
for j in range(5):
    ax_x = 1.0 + j * 1.65
    actor = FancyBboxPatch(
        (ax_x, 0.7), 1.35, 2.4, boxstyle="round,pad=0.04", facecolor=ACTOR_BG, edgecolor=BORDER_SOFT, linewidth=0.7
    )
    ax.add_patch(actor)
    ax.text(ax_x + 0.675, 2.75, f"Actor {j}", ha="center", va="top", fontsize=9.5, fontweight="bold", color=TEXT)
    ax.text(
        ax_x + 0.675,
        1.65,
        "Environment\nTokenizer",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_SEC,
        linespacing=1.3,
    )

draw_conn_v(ax, 2.0, 4.8, 4.0, "Ray")

save(plt.gcf(), "environmental_grpo_multi_server")
plt.close()
print("\u2713 environmental_grpo_multi_server.png")
