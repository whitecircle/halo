"""Generate EP Token Routing Flow diagram."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 13))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 12.6, "Expert Parallelism", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(5, 12.0, "Token Routing Flow", ha="center", va="top", fontsize=15, fontweight="normal", color=TEXT_SEC)
ax.text(
    5, 11.45, "dispatch  \u2192  local compute  \u2192  combine", ha="center", va="top", fontsize=10, color=TEXT_TERT
)


def draw_card(x, y, w, h, title, subtitle, bg=CARD, border=CARD_BORDER, accent_left=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    if accent_left:
        bar = FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0", facecolor=ACCENT, edgecolor="none")
        ax.add_patch(bar)
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=11, fontweight="bold", color=TEXT)
    ax.text(x + w / 2, y + 0.3, subtitle, ha="center", va="bottom", fontsize=9, color=TEXT_SEC)


def draw_op(x, y, w, h, title, subtitle):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=OP_BG, edgecolor="none")
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2 + 0.12, title, ha="center", va="center", fontsize=12, fontweight="bold", color=OP_TEXT
    )
    ax.text(x + w / 2, y + h / 2 - 0.2, subtitle, ha="center", va="center", fontsize=8.5, color=TEXT_TERT)


def arrow_down(x, y1, y2):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8, "shrinkA": 4, "shrinkB": 4},
    )


cw, ch = 3.6, 1.3
lx, rx = 0.6, 5.8

y = 9.8
draw_card(lx, y, cw, ch, "Rank 0  \u00b7  Experts 0\u20133", "Batch A:  tokens [t\u2080, t\u2081]", accent_left=True)
draw_card(rx, y, cw, ch, "Rank 1  \u00b7  Experts 4\u20137", "Batch B:  tokens [t\u2082, t\u2083]", accent_left=True)

gap_center_x = (lx + cw + rx) / 2
badge_y = y + ch / 2
badge = FancyBboxPatch(
    (gap_center_x - 0.55, badge_y - 0.22),
    1.1,
    0.44,
    boxstyle="round,pad=0.04",
    facecolor=ACCENT_MID,
    edgecolor=ACCENT,
    linewidth=0.6,
)
ax.add_patch(badge)
ax.text(
    gap_center_x,
    badge_y,
    "different\nbatches",
    ha="center",
    va="center",
    fontsize=7,
    fontweight="bold",
    color=ACCENT,
    linespacing=1.15,
)

y_op1 = 7.8
arrow_down(lx + cw / 2, y, y_op1 + 1.0)
arrow_down(rx + cw / 2, y, y_op1 + 1.0)

ow, oh = 6.4, 1.0
ox = (10 - ow) / 2
draw_op(ox, y_op1, ow, oh, "All-to-All  \u00b7  Dispatch", "route tokens to their assigned experts")

y_proc = 5.9
arrow_down(lx + cw / 2, y_op1, y_proc + ch)
arrow_down(rx + cw / 2, y_op1, y_proc + ch)

draw_card(
    lx,
    y_proc,
    cw,
    ch,
    "Local Expert Compute",
    "Experts 0\u20133 process routed tokens",
    bg=ACCENT_LIGHT,
    border=ACCENT,
)
draw_card(
    rx,
    y_proc,
    cw,
    ch,
    "Local Expert Compute",
    "Experts 4\u20137 process routed tokens",
    bg=ACCENT_LIGHT,
    border=ACCENT,
)

y_op2 = 4.0
arrow_down(lx + cw / 2, y_proc, y_op2 + oh)
arrow_down(rx + cw / 2, y_proc, y_op2 + oh)

draw_op(ox, y_op2, ow, oh, "All-to-All  \u00b7  Combine", "return processed tokens to originating rank")

y_out = 2.1
arrow_down(lx + cw / 2, y_op2, y_out + ch)
arrow_down(rx + cw / 2, y_op2, y_out + ch)

draw_card(lx, y_out, cw, ch, "Rank 0  \u00b7  Output", "Batch A results (back to origin)", accent_left=True)
draw_card(rx, y_out, cw, ch, "Rank 1  \u00b7  Output", "Batch B results (back to origin)", accent_left=True)

insight = FancyBboxPatch(
    (0.8, 0.4), 8.4, 0.9, boxstyle="round,pad=0.06", facecolor=ACCENT_LIGHT, edgecolor=ACCENT, linewidth=0.8
)
ax.add_patch(insight)
ax.text(
    5,
    0.85,
    "Each rank processes different data \u2014 EP just routes tokens to experts and returns them",
    ha="center",
    va="center",
    fontsize=10,
    color=ACCENT,
    fontweight="medium",
)

save(plt.gcf(), "ep_token_routing")
plt.close()
print("\u2713 ep_token_routing.png")
