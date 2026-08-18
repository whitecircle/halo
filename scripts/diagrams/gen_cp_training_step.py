"""Generate CP Training Step Data Flow diagram."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

GREEN_BG = "#f0fdf4"
GREEN_BORDER = "#86efac"
ORANGE_BG = "#fff7ed"
ORANGE_BORDER = "#fdba74"

fig, ax = plt.subplots(figsize=(11, 16))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 11)
ax.set_ylim(0, 16)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5.5, 15.7, "CP Training Step", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(
    5.5, 15.1, "Context Parallelism  \u00b7  Per-layer data flow", ha="center", va="top", fontsize=11, color=TEXT_SEC
)


def draw_block(x, y, w, h, title, subtitle="", bg=CARD, border=CARD_BORDER, title_size=11, shape_badge=""):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    ty = y + h / 2 + (0.12 if subtitle else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=TEXT)
    if subtitle:
        ax.text(x + w / 2, ty - 0.35, subtitle, ha="center", va="center", fontsize=9, color=TEXT_SEC)
    if shape_badge:
        ax.text(
            x + w - 0.3,
            y + h / 2,
            shape_badge,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": ACCENT_LIGHT, "edgecolor": ACCENT, "linewidth": 0.6},
        )


def draw_op(x, y, w, h, title, subtitle=""):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=OP_BG, edgecolor="none")
    ax.add_patch(box)
    ty = y + h / 2 + (0.1 if subtitle else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=11, fontweight="bold", color=OP_TEXT)
    if subtitle:
        ax.text(x + w / 2, ty - 0.28, subtitle, ha="center", va="center", fontsize=8.5, color=TEXT_TERT)


def arrow_down(x, y1, y2):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8, "shrinkA": 6, "shrinkB": 6},
    )


# ── Layout (top to bottom) ──
cw = 8.0
cx = (11 - cw) / 2
center_x = 5.5

y = 13.8
bh = 0.9
draw_block(cx, y, cw, bh, "UlyssesCPModelWrapper receives full batch", shape_badge="[B, S]", title_size=11)

arrow_down(center_x, y, y - 0.6)

y -= 1.6
draw_op(cx, y, cw, 0.9, "Split Sequence for CP Rank", "local_input_ids = input_ids[:, start:end]")

ax.text(
    cx + cw + 0.3,
    y + 0.45,
    "[B, S/CP]",
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
    color=ACCENT,
    family="monospace",
    bbox={"boxstyle": "round,pad=0.12", "facecolor": ACCENT_LIGHT, "edgecolor": ACCENT, "linewidth": 0.6},
)

arrow_down(center_x, y, y - 0.6)

y -= 1.5
draw_block(cx, y, cw, 0.8, "Embeddings", shape_badge="[B, S/CP, hidden]", title_size=11)

arrow_down(center_x, y, y - 0.6)

layer_h = 5.4
layer_top = y - 0.8
layer_bot = layer_top - layer_h
layer_box = FancyBboxPatch(
    (cx - 0.3, layer_bot),
    cw + 0.6,
    layer_h,
    boxstyle="round,pad=0.06",
    facecolor=CARD,
    edgecolor=CARD_BORDER,
    linewidth=1.2,
    linestyle="--",
)
ax.add_patch(layer_box)
ax.text(
    cx,
    layer_top - 0.15,
    "For each Transformer Layer",
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    color=TEXT_SEC,
    style="italic",
)

split_y = layer_top - 0.7

bw = 3.5
bbh = 1.3
att_x = cx + 0.2
att_top = split_y - 0.8
att_y = att_top - bbh
draw_block(
    att_x,
    att_y,
    bw,
    bbh,
    "Ulysses Attention",
    "All-to-All \u2192 FlashAttn \u2192 All-to-All",
    bg=GREEN_BG,
    border=GREEN_BORDER,
    title_size=10.5,
)

moe_x = cx + cw - bw - 0.2
draw_block(
    moe_x,
    att_y,
    bw,
    bbh,
    "MoE Layer",
    "DeepEP expert routing (All-to-All)",
    bg=ORANGE_BG,
    border=ORANGE_BORDER,
    title_size=10.5,
)

arrow_down(att_x + bw / 2, split_y, att_y + bbh)
arrow_down(moe_x + bw / 2, split_y, att_y + bbh)

hidden_y = att_y - 1.2
hidden_h = 0.6
draw_block(cx + 0.5, hidden_y, cw - 1.0, hidden_h, "Hidden state (next layer)", title_size=10)

arrow_down(att_x + bw / 2, att_y, hidden_y + hidden_h)
arrow_down(moe_x + bw / 2, att_y, hidden_y + hidden_h)

ax.text(
    cx + cw - 0.1,
    layer_top - 0.15,
    "\u21bb repeat N layers",
    ha="right",
    va="top",
    fontsize=9.5,
    color=TEXT_TERT,
    fontweight="bold",
)

arrow_down(center_x, layer_bot, layer_bot - 0.6)

out_y = layer_bot - 1.5
draw_block(
    cx,
    out_y,
    cw,
    0.9,
    "Output Logits",
    subtitle="Boundary-aware, globally-normalized loss",
    shape_badge="[B, S/CP, V]",
    title_size=11,
)

save(plt.gcf(), "cp_training_step")
plt.close()
print("\u2713 cp_training_step.png")
