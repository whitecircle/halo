"""Generate the SMPO loss diagram for training-methods/preference/smpo.md.

One forward over the pair gives per-token log-probs, which feed two paths: the margin path, where
the percentile clip trims them before the per-sequence mean, and the SFT anchors, which take the
same log-probs pre-clip. They rejoin in the total, weighted by `chosen_sft_ratio`.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 12.2, 5.83
M = 0.305  # left/right margin, matching `title`'s 2.5% inset

(COL_A, COL_B, COL_C), COL_W = columns(W, 3, M)
ROW1_Y, ROW1_H = 2.97, card_height(5)
ROW2_Y, ROW2_H = 1.17, card_height(2)
TOTAL_X = COL_B
TOTAL_W = W - M - TOTAL_X

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "SMPO — the preference pair", "z = mean logp(chosen) − mean logp(rejected) − margin")

section(ax, M, H - 0.52, "MARGIN TERM")

card(
    ax,
    COL_A,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "One forward — the pair",
    [
        "the pair concatenated, 2N rows",
        "chosen rows first, then rejected",
        "labels −100 over the prompt",
        "shift + mask, then log-softmax",
        "→ per-token log p, [2N, T]",
    ],
    color=BLUE,
)

card(
    ax,
    COL_B,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "Percentile clip — margin path",
    [
        "runs on the per-token log p",
        "rejected tail → 2% token quantile",
        "chosen tail capped (upper, off)",
        "then min_log_prob −2.3, rejected",
        "one detached bound per half",
    ],
    color=AMBER,
)

card(
    ax,
    COL_C,
    ROW1_Y,
    COL_W,
    ROW1_H,
    "Per-sequence mean → margin",
    [
        "logp_c/r = fp32 Σ ÷ tokens",
        "z = logp_c − logp_r − margin",
        "L_margin = relu(−β·z)²",
        "loss_type smooth_lower_bound",
        "β = 1.2 · margin 0.01 → 0.35",
    ],
    color=BLUE,
    mono_lines=True,
)

section(ax, M, ROW2_Y + ROW2_H + 0.30, "SFT ANCHORS")

card(
    ax,
    COL_A,
    ROW2_Y,
    COL_W,
    ROW2_H,
    "Anchors — pre-clip",
    [
        "mean NLL over completion tokens",
        "taken before the clip, both sides",
    ],
    color=TEAL,
)

card(
    ax,
    TOTAL_X,
    ROW2_Y,
    TOTAL_W,
    ROW2_H,
    "Total",
    [
        "L_total = mean_pairs L_margin + α·CE(chosen) + (1−α)·CE(rejected)",
        "α = chosen_sft_ratio 0.8 · no reference model",
    ],
    color=SLATE,
    mono_lines=True,
)

row1_mid = ROW1_Y + ROW1_H / 2
row2_mid = ROW2_Y + ROW2_H / 2
arrow(ax, COL_A + COL_W, row1_mid, COL_B, row1_mid, "token log p")
arrow(ax, COL_B + COL_W, row1_mid, COL_C, row1_mid, "clipped")
arrow(ax, COL_A + 0.72 * COL_W, ROW1_Y, COL_A + 0.72 * COL_W, ROW2_Y + ROW2_H, "per-token NLL", side="right")
arrow(ax, COL_C + 0.5 * COL_W, ROW1_Y, COL_C + 0.5 * COL_W, ROW2_Y + ROW2_H, "L_margin", side="right")
arrow(ax, COL_A + COL_W, row2_mid, TOTAL_X, row2_mid, "CE terms")

footnote(
    ax,
    M,
    0.22,
    W - 2 * M,
    "The margin term is exactly zero once z clears 0 — where a sigmoid loss keeps pushing — while the anchors train both sides.",
)

save(plt.gcf(), "smpo_pipeline")
plt.close()
print("✓ smpo_pipeline.png")
