"""Generate Offline GRPO Pipeline diagram."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 8.6))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(2.4, 11)
ax.set_aspect("equal")
ax.axis("off")

ax.text(7, 10.7, "Offline GRPO Pipeline", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(
    7,
    10.15,
    "Pre-computed rewards  \u00b7  No generation at train time",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_SEC,
)


draw_card, draw_arrow_h, draw_arrow_v = bind_drawers(ax, CardLayout(line_dy=0.32, line_size=9))

# ── Row 1: Dataset → Advantage Calc → Per-Group Expansion ──
r1_y = 7.8
cw1 = 3.0
ch1 = 1.7

draw_card(
    0.4,
    r1_y,
    cw1,
    ch1,
    "Dataset",
    [
        "prompt (messages)",
        "completions (list)",
        "rewards (list of floats)",
    ],
    accent_left=True,
)

draw_arrow_h(3.4 + 0.15, r1_y + ch1 / 2, 5.3 - 0.15)

draw_card(
    5.3,
    r1_y,
    cw1,
    ch1,
    "Advantage Calculation",
    [
        "quantile_norm (default)",
        "z_norm, minmax, robust",
        "quantile_uniform",
    ],
)

draw_arrow_h(8.3 + 0.15, r1_y + ch1 / 2, 10.2 - 0.15)

draw_card(
    10.2,
    r1_y,
    3.4,
    ch1,
    "Per-Group Expansion",
    [
        "One example per completion",
        "Preserves group_id",
        "Flattened dataset",
    ],
)

draw_arrow_v(11.9, r1_y, 6.2 + 1.0)

# ── Row 2: MultiGroupSampler ──
r2_y = 6.2
draw_card(
    8.0,
    r2_y,
    5.6,
    1.0,
    "MultiGroupSampler",
    [
        "Flattens group-by-group, splits by DP rank",
    ],
    bg=ACCENT_LIGHT,
    border=ACCENT,
    title_size=11,
)

draw_arrow_v(10.8, r2_y, 2.7 + 2.6)

# ── Row 3: Training Loop (large card) ──
tlx, tly = 0.4, 2.7
tlw, tlh = 13.2, 2.6
draw_card(tlx, tly, tlw, tlh, "Training Loop", [], bg=CARD, border=CARD_BORDER)

steps_left = [
    ("1", "Concatenate prompt + completion"),
    ("2", "Forward pass \u2192 per_token_logps"),
    ("3", "(Optional) Reference model \u2192 per_token_kl"),
    ("4", "Compute per_token_loss = \u2212(weight \u00d7 advantage)"),
]
steps_right = [
    ("5", "Apply group weighting: 1/group_size"),
    ("6", "Aggregate loss (grpo, bnpo, or dr_grpo)"),
    ("7", "Backward + optimize"),
]

sep_y = tly + tlh - 0.5
ax.plot([tlx + 0.3, tlx + tlw - 0.3], [sep_y, sep_y], color=BORDER_SOFT, lw=0.6)

for i, (num, text) in enumerate(steps_left):
    sy = sep_y - 0.42 - i * 0.44
    badge = FancyBboxPatch(
        (tlx + 0.35, sy - 0.13), 0.3, 0.26, boxstyle="round,pad=0.03", facecolor=OP_BG, edgecolor="none"
    )
    ax.add_patch(badge)
    ax.text(tlx + 0.5, sy, num, ha="center", va="center", fontsize=8, fontweight="bold", color=OP_TEXT)
    ax.text(tlx + 0.8, sy, text, ha="left", va="center", fontsize=9.5, color=TEXT_SEC)

for i, (num, text) in enumerate(steps_right):
    sy = sep_y - 0.42 - i * 0.44
    badge = FancyBboxPatch(
        (tlx + 7.0, sy - 0.13), 0.3, 0.26, boxstyle="round,pad=0.03", facecolor=OP_BG, edgecolor="none"
    )
    ax.add_patch(badge)
    ax.text(tlx + 7.15, sy, num, ha="center", va="center", fontsize=8, fontweight="bold", color=OP_TEXT)
    ax.text(tlx + 7.45, sy, text, ha="left", va="center", fontsize=9.5, color=TEXT_SEC)

save(plt.gcf(), "offline_grpo_pipeline")
plt.close()
print("\u2713 offline_grpo_pipeline.png")
