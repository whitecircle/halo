"""Generate the Online GRPO (RLVR) pipeline diagram."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

GREEN_BG = "#f0fdf4"
GREEN_BORDER = "#86efac"

fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

ax.text(7.5, 9.7, "RLVR Online GRPO Pipeline", ha="center", va="top", fontsize=20, fontweight="bold", color=TEXT)
ax.text(
    7.5,
    9.15,
    "Live generation  \u00b7  Rule-based rewards  \u00b7  NCCL weight sync",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_SEC,
)


draw_card, draw_arrow_h, draw_arrow_v = bind_drawers(ax, CardLayout(bullet_dx=0.2))

# ── Main pipeline: 3 boxes in a row ──
r1_y = 5.4
ch = 2.8

d_x, d_w = 0.3, 3.0
draw_card(
    d_x,
    r1_y,
    d_w,
    ch,
    "Dataset",
    [
        "prompt (messages)",
        "answer (ground truth)",
        "Rule-based rewards",
        "(no API calls needed)",
    ],
    accent_left=True,
)

v_x, v_w = 5.3, 3.4
draw_card(
    v_x,
    r1_y,
    v_w,
    ch,
    "vLLM Server",
    [
        "Dedicated inference GPU",
        "Generates completions",
        "NCCL weight sync",
        "Separate process group",
    ],
    bg=ACCENT_LIGHT,
    border=ACCENT,
    title_size=12,
)

t_x, t_w = 10.8, 3.8
draw_card(
    t_x,
    r1_y,
    t_w,
    ch,
    "Training Loop",
    [
        "Compute log probs",
        "Run reward functions",
        "Compute advantages",
        "Policy gradient update",
        "Sync weights to vLLM",
    ],
)

arrow_y = r1_y + ch / 2
draw_arrow_h(d_x + d_w, arrow_y, v_x, "prompts")
draw_arrow_h(v_x + v_w, arrow_y, t_x, "completions")

ax.annotate(
    "",
    xy=(v_x + v_w, r1_y + 0.2),
    xytext=(t_x, r1_y + 0.2),
    arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.3, "shrinkA": 6, "shrinkB": 6},
)

sync_mx = (v_x + v_w + t_x) / 2
ax.text(
    sync_mx, r1_y + 0.2 - 0.2, "NCCL weight sync", ha="center", va="center", fontsize=8.5, color=TEXT_SEC, zorder=5
)

rf_y = 0.4
rf_h = 3.6
rf_box = FancyBboxPatch(
    (0.3, rf_y), 14.4, rf_h, boxstyle="round,pad=0.06", facecolor=CARD, edgecolor=CARD_BORDER, linewidth=0.8
)
ax.add_patch(rf_box)
ax.text(
    7.5,
    rf_y + rf_h - 0.2,
    "Reward Functions  (rule-based, no API calls)",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
    color=TEXT,
)
sep_ry = rf_y + rf_h - 0.55
ax.plot([0.7, 14.3], [sep_ry, sep_ry], color=BORDER_SOFT, lw=0.6)

rw = 6.2
rh = 2.0
draw_card(
    0.8,
    rf_y + 0.3,
    rw,
    rh,
    "accuracy_reward",
    [
        "Extract \\boxed{answer} from completion",
        "Compare to ground truth",
        "Returns 1.0 or 0.0",
    ],
    bg=GREEN_BG,
    border=GREEN_BORDER,
    title_size=11,
)

draw_card(
    7.8,
    rf_y + 0.3,
    rw,
    rh,
    "format_reward",
    [
        "Match regex pattern",
        "e.g., <think>...</think>",
        "Returns 1.0 or 0.0",
    ],
    bg=GREEN_BG,
    border=GREEN_BORDER,
    title_size=11,
)

draw_arrow_v(t_x + t_w / 2, r1_y, rf_y + rf_h)

save(plt.gcf(), "online_grpo_pipeline")
plt.close()
print("\u2713 online_grpo_pipeline.png")
