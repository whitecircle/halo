"""Generate SMPO Training Pipeline diagram."""

import functools

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 15))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 14.7, "SMPO Training Pipeline", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(
    5,
    14.1,
    "Reference-free preference optimization  \u00b7  Smooth margin loss",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_SEC,
)

cw = 8.0
cx = (10 - cw) / 2


draw_card, _, draw_arrow_v = bind_drawers(ax, CardLayout(title_dy=0.22, sep_dy=0.55, line_dy=0.35, line_step=0.36))

# Every arrow in this figure runs down the single centered column.
draw_arrow = functools.partial(draw_arrow_v, 5)

# ── Layout (top to bottom with explicit positions) ──
GAP = 0.7  # gap between blocks for arrows

h1 = 1.7
y1 = 12.1
draw_card(
    cx,
    y1,
    cw,
    h1,
    "Dataset",
    [
        "prompt (message list)",
        "chosen (completion)",
        "rejected (completion)",
    ],
    accent_left=True,
)

draw_arrow(y1, y1 - GAP)

h2 = 1.7
y2 = y1 - GAP - h2
draw_card(
    cx,
    y2,
    cw,
    h2,
    "Tokenization & Collation",
    [
        "Tokenize prompt, chosen, rejected",
        "Handle BOS/EOS tokens",
        "Truncate if needed",
    ],
)

draw_arrow(y2, y2 - GAP)

h3 = 2.6
y3 = y2 - GAP - h3
draw_card(
    cx,
    y3,
    cw,
    h3,
    "Concatenated Forward Pass",
    [
        "Input: [chosen | rejected] (2N sequences)",
        "Build full sequences: prompt + completion",
        "Create labels: \u2212100 for prompt tokens",
        "Forward pass \u2192 logits for all 2N sequences",
        "Per-token log probs + token-level clipping",
    ],
    bg=ACCENT_LIGHT,
    border=ACCENT,
)

draw_arrow(y3, y3 - GAP)

h4 = 2.4
y4 = y3 - GAP - h4
draw_card(cx, y4, cw, h4, "Loss Computation", [], bg=CARD, border=CARD_BORDER)

sep_y = y4 + h4 - 0.55
ax.plot([cx + 0.2, cx + cw - 0.2], [sep_y, sep_y], color=BORDER_SOFT, lw=0.6)

formulas = [
    ("L_total", "L_margin + L_sft"),
    ("L_margin", "loss_fn(\u03b2 \u00b7 (log p(chosen) \u2212 log p(rejected) \u2212 margin))"),
    ("L_sft", "\u03b1 \u00b7 CE(chosen) + (1\u2212\u03b1) \u00b7 CE(rejected)"),
]
for i, (lhs, rhs) in enumerate(formulas):
    fy = sep_y - 0.4 - i * 0.48
    bw = len(lhs) * 0.11 + 0.25
    badge = FancyBboxPatch(
        (cx + 0.3, fy - 0.15), bw, 0.3, boxstyle="round,pad=0.03", facecolor=OP_BG, edgecolor="none"
    )
    ax.add_patch(badge)
    ax.text(
        cx + 0.3 + bw / 2,
        fy,
        lhs,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=OP_TEXT,
        family="monospace",
    )
    ax.text(
        cx + 0.3 + bw + 0.15, fy, f"= {rhs}", ha="left", va="center", fontsize=9.5, color=TEXT_SEC, family="monospace"
    )

save(plt.gcf(), "smpo_pipeline")
plt.close()
print("\u2713 smpo_pipeline.png")
