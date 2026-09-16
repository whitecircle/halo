"""Generate the online GRPO (RLVR) pipeline for training-methods/grpo/online-grpo.md.

The step is a cycle: the trainer renders and tokenizes, vLLM generates, the rule-based rewards
score the completions, and the updated weights go back over NCCL before the next generation.
The scoring band reads right-to-left under the generation band, so the loop closes on itself.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 12.2, 6.71
M = 0.305  # left/right margin, matching `title`'s 2.5% inset
LANE = 1.50  # band gap: holds the completions arrow and the weight-sync return path

(COL_A, COL_B, COL_C), COL_W = columns(W, 3, M)
ROW2_Y, ROW2_H = 1.17, card_height(4)
ROW1_Y, ROW1_H = ROW2_Y + ROW2_H + LANE, card_height(3)
SYNC_LANE_Y = ROW2_Y + ROW2_H + 0.45  # horizontal run of the weight-sync return

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "Online GRPO (RLVR)", "A = (r − mean) / std over each group of num_generations")

section(ax, M, H - 0.52, "GENERATE")

card(
    ax,
    COL_A,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "Dataset",
    [
        "prompt: text or messages",
        "answer: the ground truth",
        "each prompt × num_generations",
    ],
    color=TEAL,
)

card(
    ax,
    COL_B,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "Render + tokenize",
    [
        "the trainer's chat template",
        "TRL tokenizes the rendered text",
        "over budget → dropped, not cut",
    ],
    color=TEAL,
)

card(
    ax,
    COL_C,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "vLLM server — GPU 7",
    [
        "separate container, :8000",
        "server mode only, never colocate",
        "applies no template of its own",
    ],
    color=BLUE,
)

section(ax, M, ROW2_Y + ROW2_H + 0.38, "SCORE + UPDATE")

card(
    ax,
    COL_C,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "Rewards",
    [
        r"rewards: accuracy — last \boxed{}",
        "equals answer → 1.0 / 0.0",
        "rewards: format — regex, off",
        "weighted sum → reward_weights",
    ],
    color=TEAL,
)

card(
    ax,
    COL_B,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "Group advantages",
    [
        "A = (r − mean) / std, per group",
        "scale_rewards: group",
        "all-equal group → A = 0",
        "num_generations rows per prompt",
    ],
    color=AMBER,
)

card(
    ax,
    COL_A,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "GRPO loss + step",
    [
        "recipes: loss_type grpo, beta 0",
        "IS ratio from the sampling logps",
        "sequence_mask: ratio > 3.0 → 0",
        "FSDP2 / EP / TP update",
    ],
    color=BLUE,
)

row1_mid = ROW1_Y + ROW1_H / 2
row2_mid = ROW2_Y + ROW2_H / 2
arrow(ax, COL_A + COL_W, row1_mid, COL_B, row1_mid, "prompts")
arrow(ax, COL_B + COL_W, row1_mid, COL_C, row1_mid, "token ids", color=TEAL)
arrow(
    ax,
    COL_C + 0.40 * COL_W,
    ROW1_Y,
    COL_C + 0.40 * COL_W,
    ROW2_Y + ROW2_H,
    "completions + logprobs",
    color=TEAL,
    side="right",
)
arrow(ax, COL_C, row2_mid, COL_B + COL_W, row2_mid, "reward r")
arrow(ax, COL_B, row2_mid, COL_A + COL_W, row2_mid, "advantage")

sync_x0 = COL_A + 0.85 * COL_W
sync_x1 = COL_C + 0.10 * COL_W
polyline_arrow(
    ax,
    [
        (sync_x0, ROW2_Y + ROW2_H),
        (sync_x0, SYNC_LANE_Y),
        (sync_x1, SYNC_LANE_Y),
        (sync_x1, ROW1_Y),
    ],
    "NCCL weight sync — before the next generation",
    ((sync_x0 + sync_x1) / 2, SYNC_LANE_Y + 0.09),
    color=VIOLET,
)

footnote(
    ax,
    M,
    0.22,
    W - 2 * M,
    "vLLM only, server mode only: the engine must own GPUs no trainer rank uses — a rank cannot broadcast to itself.",
)

save(plt.gcf(), "online_grpo_pipeline")
plt.close()
print("✓ online_grpo_pipeline.png")
