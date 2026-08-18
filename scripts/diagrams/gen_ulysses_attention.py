"""Generate the Ulysses Attention Data Flow diagram."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 15.5))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 9)
ax.set_ylim(0, 15.5)
ax.set_aspect("equal")
ax.axis("off")

ax.text(4.5, 15.2, "Ulysses Attention Data Flow", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(
    4.5,
    14.6,
    "Context Parallelism  \u00b7  CP = 8  \u00b7  64 attention heads",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_SEC,
)


def draw_data_block(y, title, description, shape_text, accent_left=False):
    """Wide card for data state."""
    w, h = 7.4, 1.5
    x = (9 - w) / 2
    card = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08", facecolor=CARD, edgecolor=CARD_BORDER, linewidth=1.0
    )
    ax.add_patch(card)
    if accent_left:
        accent_bar = FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0", facecolor=ACCENT, edgecolor="none")
        ax.add_patch(accent_bar)

    ax.text(x + 0.35, y + h - 0.3, title, ha="left", va="top", fontsize=12, fontweight="bold", color=TEXT)
    ax.text(x + 0.35, y + 0.35, description, ha="left", va="bottom", fontsize=9.5, color=TEXT_SEC)
    ax.text(
        x + w - 0.35,
        y + h / 2,
        shape_text,
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": ACCENT_LIGHT, "edgecolor": ACCENT, "linewidth": 0.7},
    )
    return y + h / 2  # center y for arrows


def draw_op_block(y, title, subtitle):
    """Narrow dark block for operations."""
    w, h = 5.6, 1.0
    x = (9 - w) / 2
    op = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=OP_BG, edgecolor="none", linewidth=0)
    ax.add_patch(op)
    ax.text(
        x + w / 2, y + h / 2 + 0.12, title, ha="center", va="center", fontsize=11, fontweight="bold", color=OP_TEXT
    )
    ax.text(x + w / 2, y + h / 2 - 0.2, subtitle, ha="center", va="center", fontsize=8.5, color=TEXT_TERT)
    return y + h / 2


def draw_arrow(y_from, y_to):
    ax.annotate(
        "",
        xy=(4.5, y_to),
        xytext=(4.5, y_from),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8, "shrinkA": 4, "shrinkB": 4},
    )


y = 12.5
draw_data_block(
    y, "Input  \u00b7  Per GPU", "Each GPU holds 1/8 of sequence, all 64 heads", "[B, S/8, 64H, D]", accent_left=True
)

y -= 2.0
draw_arrow(y + 1.95, y + 1.06)
draw_op_block(y, "All-to-All  #1   (NCCL)", "scatter heads  \u00b7  gather sequence")

y -= 2.0
draw_arrow(y + 1.95, y + 1.56)
draw_data_block(y, "Redistributed", "Each GPU holds full sequence, 8 of 64 heads", "[B, S, 8H, D]")

y -= 2.0
draw_arrow(y + 1.95, y + 1.06)
draw_op_block(y, "Flash Attention", "full sequence  \u00d7  head subset")

y -= 2.0
draw_arrow(y + 1.95, y + 1.56)
draw_data_block(y, "Attention Output", "Each GPU: attention result for its 8 heads", "[B, S, 8H, D]")

y -= 2.0
draw_arrow(y + 1.95, y + 1.06)
draw_op_block(y, "All-to-All  #2   (NCCL)", "scatter sequence  \u00b7  gather heads")

y -= 2.0
draw_arrow(y + 1.95, y + 1.56)
draw_data_block(
    y,
    "Output  \u00b7  Per GPU",
    "Back to original layout: 1/8 sequence, all 64 heads",
    "[B, S/8, 64H, D]",
    accent_left=True,
)

save(plt.gcf(), "ulysses_attention_flow")
plt.close()
print("\u2713 ulysses_attention_flow.png")
