"""Generate the offline GRPO pipeline for training-methods/grpo/offline-grpo.md.

Two bands: what tokenization does once (group → advantages → one row per completion) and what
every training step does (sampler → per-token loss → normalization). The second band reads
right-to-left, under the first, so the hand-off is a straight drop.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 12.2, 5.83
M = 0.305  # left/right margin, matching `title`'s 2.5% inset

(COL_A, COL_B, COL_C), COL_W = columns(W, 3, M)
ROW1_Y, ROW1_H = 3.49, card_height(3)
ROW2_Y, ROW2_H = 1.17, card_height(4)

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "Offline GRPO — pre-scored data", "per-token loss = −(π · A), every row × 1/group_size")

section(ax, M, H - 0.52, "ONCE, AT TOKENIZATION")

card(
    ax,
    COL_A,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "One row = one group",
    [
        "prompt      list[dict]",
        "completions list[list[dict]]",
        "rewards     list[float]",
    ],
    color=TEAL,
    mono_lines=True,
)

card(
    ax,
    COL_B,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "Group advantages",
    [
        "quantile_norm (default) · z_norm",
        "minmax · quantile_uniform · robust",
        "emphasis, then clip to [−10, 10]",
    ],
    color=AMBER,
)

card(
    ax,
    COL_C,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "One row per completion",
    [
        "carries its advantage + group_size",
        "group_id = the source row index",
        "drop_degenerate_groups: ties, n<2",
    ],
    color=SLATE,
)

section(ax, M, ROW2_Y + ROW2_H + 0.30, "EVERY TRAINING STEP")

card(
    ax,
    COL_C,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "MultiGroupSampler",
    [
        "flattens groups in dataset order",
        "cuts the DP slice positionally",
        "a group may straddle ranks",
        "the 1/group_size weight rides",
    ],
    color=SLATE,
)

card(
    ax,
    COL_B,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "Per-token loss",
    [
        "prob_weighted:  −(π · A)",
        "reinforce:      −(log π · A)",
        "min_log_prob −3.0 on A < 0",
        "kl_beta > 0: + β · k3 KL",
    ],
    color=BLUE,
    mono_lines=True,
)

card(
    ax,
    COL_A,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "Normalize",
    [
        "every row × 1/group_size",
        "bnpo: global weighted mean",
        "grpo: per-sequence, then groups",
        "dr_grpo: ÷ (Σ weights × max_len)",
    ],
    color=BLUE,
)

row1_mid = ROW1_Y + ROW1_H / 2
row2_mid = ROW2_Y + ROW2_H / 2
arrow(ax, COL_A + COL_W, row1_mid, COL_B, row1_mid, "rewards")
arrow(ax, COL_B + COL_W, row1_mid, COL_C, row1_mid, "advantage")
arrow(ax, COL_C + 0.5 * COL_W, ROW1_Y, COL_C + 0.5 * COL_W, ROW2_Y + ROW2_H, "expanded dataset", side="right")
arrow(ax, COL_C, row2_mid, COL_B + COL_W, row2_mid, "micro-batch")
arrow(ax, COL_B, row2_mid, COL_A + COL_W, row2_mid, "[B, T] loss")

footnote(
    ax,
    M,
    0.22,
    W - 2 * M,
    "No generation at train time: the rewards ship with the dataset, and each row carries its group's advantage and weight.",
)

save(plt.gcf(), "offline_grpo_pipeline")
plt.close()
print("✓ offline_grpo_pipeline.png")
