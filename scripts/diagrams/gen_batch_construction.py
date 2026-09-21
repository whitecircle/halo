"""Generate the two async-GRPO batch figures.

- batch_rollout_pipeline.png — one training step, from the weight sync to the GRPO loss.
- batch_prompt_expansion.png — how the sampler's repeated prompts become a rank's round and a step.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

# --- Shape drawn throughout: the code-contests stage-1 recipe ---
DP, BS, GA, GEN = 6, 1, 24, 8
ROUND = BS * GA  # rows a rank generates in one round
GROUPS = ROUND // GEN  # prompts a rank draws
TOTAL = DP * ROUND

# ======================================================================================
# batch_rollout_pipeline.png — one training step
# ======================================================================================
fig, ax = plt.subplots(figsize=(12.6, 8.95))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 8.95)
ax.axis("off")

L, R = 0.25, 12.35
W = R - L

title(ax, "One async GRPO step", sub_mono=f"one rank's round: {ROUND} rows = {GROUPS} prompts × {GEN} generations")

# --- Step start: the weight sync ---
card(
    ax,
    L,
    8.00,
    W,
    card_height(1, has_title=False),
    None,
    ["The step starts with an NCCL weight push to every rollout engine, which is paused for the push."],
    color=VIOLET,
)

# --- Band 1: the generation round ---
section(ax, L, 7.72, "GENERATION ROUND")
GAP = 0.72
CW = (W - 3 * GAP) / 4
bx = [L + i * (CW + GAP) for i in range(4)]
by, bh = 5.98, card_height(3)

card_row(
    ax,
    bx,
    by,
    CW,
    bh,
    [
        (
            "Trainer rank",
            [f"{ROUND} rows = {GROUPS} prompts × {GEN}", f"{GEN} consecutive rows each", "groups are rank-local"],
            BLUE,
        ),
        ("RolloutManager", ["one episode per row", "round-robin: actors × URLs", "max_concurrent_rollouts"], TEAL),
        (
            "Ray environment actors",
            ["num_rollout_workers / rank", "one environment each", "tools · sandbox · grader"],
            TEAL,
        ),
        ("vLLM or SGLang", ["POST /v1/chat/completions", "messages + tool schema", "tokenizes server-side"], VIOLET),
    ],
    labels=[f"{ROUND} rows", "", "HTTP"],
)

# --- Band 2: the episode ---
section(ax, L, 5.70, "MULTI-TURN EPISODE  —  one row, one actor")
sy, sh = 4.46, card_height(1)
SGAP, SR = 0.45, 9.15
SW = (SR - L - 3 * SGAP) / 4
sx = [L + i * (SW + SGAP) for i in range(4)]
card_row(
    ax,
    sx,
    sy,
    SW,
    sh,
    [
        ("generate", ["one assistant turn"], TEAL),
        ("parse", ["tool_calls"], TEAL),
        ("execute", ["in the environment"], TEAL),
        ("observe", ["result → history"], TEAL),
    ],
)

card(
    ax,
    9.60,
    sy + sh - card_height(3),
    R - 9.60,
    card_height(3),
    "The episode ends at",
    ["a final answer", "max_turns burned", "spent recoveries"],
    color=SLATE,
)

# Loop back to `generate`: down from `observe`, left under the row, up again (`arrow` is straight only).
lane = 4.12
lx0, lx1 = sx[3] + SW / 2, sx[0] + SW / 2
ax.plot([lx0, lx0], [sy, lane], color=INK2, lw=1.4, solid_capstyle="butt")
ax.plot([lx1, lx1], [lane, sy - 0.02], color=INK2, lw=1.4, solid_capstyle="butt")
arrow(ax, lx0, lane, lx1, lane, "next turn · up to max_turns", side="below", lw=1.4)

card(
    ax,
    L,
    3.06,
    SR - L,
    card_height(1, has_title=False),
    None,
    ["A turn the engine cut at its token cap is nudged and retried: it costs a turn, not the episode."],
    color=ROSE,
    dashed=True,
)

# --- Band 3: the return path ---
section(ax, L, 2.80, "RETURN PATH")
ry, rh = 1.08, card_height(3)
card_row(
    ax,
    bx,
    ry,
    CW,
    rh,
    [
        ("Grade", ["the environment scores", "the whole trajectory", "objective + shaping rungs"], TEAL),
        (
            "Training rows",
            ["one row per assistant turn", "prompt = rendered history", "completion = sampled ids"],
            AMBER,
        ),
        (
            "Group advantages",
            [f"over the group's {GEN} rows", "reward − group baseline", "shared by its turn rows"],
            BLUE,
        ),
        ("GRPO loss → step", ["IS ratio vs sampling logps", "trust-region masks", "then the next sync"], VIOLET),
    ],
)

footnote(
    ax,
    L,
    0.25,
    W,
    "With one server the round and the update are serial; with two or more, a prefetched round overlaps the update.",
)

save(plt.gcf(), "batch_rollout_pipeline")
plt.close()
print("✓ batch_rollout_pipeline.png")


# ======================================================================================
# batch_prompt_expansion.png — the counting
# ======================================================================================
fig, ax = plt.subplots(figsize=(10.2, 6.6))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 6.6)
ax.axis("off")

title(ax, "Batch construction", sub_mono=f"dp {DP} × bs {BS} × steps_per_generation {GA} = {TOTAL} rows")

GX, GW = 0.85, 7.35
GRP_GAP = 0.12
CELL = (GW - (GROUPS - 1) * GRP_GAP) / ROUND


def cell_x(i):
    """Left edge of row `i` of a rank's round — group blocks are set apart by a gap."""
    return GX + i * CELL + (i // GEN) * GRP_GAP


section(ax, 0.3, 6.00, "ONE RANK'S GENERATION ROUND")
ax.text(
    0.3,
    5.72,
    f"TRL's RepeatSampler delivers each prompt's {GEN} rows consecutively, rank-local.",
    fontsize=SMALL,
    color=INK2,
    va="top",
)

for g in range(GROUPS):
    x0 = cell_x(g * GEN)
    chip(ax, x0, 5.00, GEN * CELL, 0.45, f"prompt {g + 1}", color=BLUE, sub=f"group {g + 1}", bold=True)
    arrow(ax, x0 + GEN * CELL / 2, 4.98, x0 + GEN * CELL / 2, 4.71, lw=1.3)

R0_Y, R0_H = 4.22, 0.46
THIN_H, THIN_GAP = 0.22, 0.07
LABX = GX + GW + 0.18

ax.text(GX - 0.12, R0_Y + R0_H / 2, "R0", ha="right", va="center", fontsize=SMALL, color=INK2, fontfamily=MONO)
for i in range(ROUND):
    chip(ax, cell_x(i), R0_Y, CELL, R0_H, color=TEAL)
ax.text(LABX, R0_Y + R0_H / 2, f"{ROUND} rows", ha="left", va="center", fontsize=SMALL, fontweight="bold", color=INK)

for r in range(1, DP):
    ry = R0_Y - 0.12 - r * (THIN_H + THIN_GAP)
    ax.text(GX - 0.12, ry + THIN_H / 2, f"R{r}", ha="right", va="center", fontsize=TINY, color=INK3, fontfamily=MONO)
    for i in range(ROUND):
        chip(ax, cell_x(i), ry, CELL, THIN_H, color=TEAL)

thin_top = R0_Y - 0.12 - (THIN_H + THIN_GAP) + THIN_H
thin_bot = R0_Y - 0.12 - (DP - 1) * (THIN_H + THIN_GAP)
ax.text(
    LABX,
    (thin_top + thin_bot) / 2,
    f"× {DP} ranks",
    ha="left",
    va="center",
    fontsize=SMALL,
    fontweight="bold",
    color=INK,
)

# --- What each unit is ---
KX, KR, KGAP = 0.3, 9.9, 0.30
KW = (KR - KX - 2 * KGAP) / 3
ky, kh = 1.15, card_height(2)
card(ax, KX, ky, KW, kh, "Row", ["one episode, one trajectory", "its turns share one advantage"], color=TEAL)
card(
    ax,
    KX + KW + KGAP,
    ky,
    KW,
    kh,
    "Group",
    [f"num_generations = {GEN} rows", "one prompt, rank-local"],
    color=BLUE,
)
card(
    ax,
    KX + 2 * (KW + KGAP),
    ky,
    KW,
    kh,
    "Optimizer step",
    [f"{TOTAL} rows = {TOTAL // GEN} prompts × {GEN}", f"{GA} micro-batches of bs {BS}"],
    color=VIOLET,
)

footnote(
    ax,
    KX,
    0.30,
    KR - KX,
    "A group never straddles ranks: per_device_train_batch_size × steps_per_generation must divide by num_generations.",
)

save(plt.gcf(), "batch_prompt_expansion")
plt.close()
print("✓ batch_prompt_expansion.png")
