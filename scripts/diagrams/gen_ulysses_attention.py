"""Generate the Ulysses attention figure for parallelism/context-parallelism.md.

The layer's own flow across the top (RoPE on the local chunk → all-to-all → flash over the full
sequence → all-to-all back), and under it the ownership transpose the two collectives perform:
one sequence chunk with every head, to every token with a slice of the heads.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 12.4, 5.85

CARD_Y, CARD_H = 4.00, card_height(2)
CARD_GAP, CARD_X0 = 0.22, 0.3

CP, HEADS = 4, 64
CELL_W, CELL_H = 0.66, 0.42
GRID_Y, GRID_XS = 3.17, (0.75, 4.69)  # GRID_Y is the top edge of both grids
RANK_LABELS = [f"r{r}" for r in range(CP)]
CHUNK_LABELS = [f"s{c}" for c in range(CP)]
RULES_X, RULES_W = 8.0, 4.1

STAGES = [
    ("Per-rank input", ["[B, S/4, 64, D]", "RoPE on chunk"], TEAL),
    ("all-to-all #1", ["scatter heads", "gather sequence"], VIOLET),
    ("Flash attention", ["[B, S, 16, D]", "full S, 16 heads"], BLUE),
    ("all-to-all #2", ["scatter sequence", "gather heads"], VIOLET),
    ("Per-rank output", ["[B, S/4, 64, D]", "reshape → o_proj"], TEAL),
]
CARD_XS, CARD_W = columns(W, len(STAGES), CARD_X0, CARD_GAP)


fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "Ulysses attention", "cp 4 · 64 Q / 8 KV heads → 16 Q / 2 KV per rank")

card_row(ax, CARD_XS, CARD_Y, CARD_W, CARD_H, STAGES, mono_lines=True)

section(ax, 0.3, GRID_Y + 0.5, "Who holds what — rank × sequence chunk")

per_rank = HEADS // CP
before = [[f"0–{HEADS - 1}" if c == r else "" for c in range(CP)] for r in range(CP)]
after = [[f"{per_rank * r}–{per_rank * r + per_rank - 1}"] * CP for r in range(CP)]
grid(
    ax,
    GRID_XS[0],
    GRID_Y,
    before,
    cell_w=CELL_W,
    cell_h=CELL_H,
    color=TEAL,
    row_labels=RANK_LABELS,
    col_labels=CHUNK_LABELS,
    caption="one chunk × all 64 heads",
)
grid(
    ax,
    GRID_XS[1],
    GRID_Y,
    after,
    cell_w=CELL_W,
    cell_h=CELL_H,
    color=BLUE,
    row_labels=RANK_LABELS,
    col_labels=CHUNK_LABELS,
    caption="all S × this rank's 16 heads",
)

gap_mid = (GRID_XS[0] + CP * CELL_W + GRID_XS[1]) / 2
arrow(ax, GRID_XS[0] + CP * CELL_W + 0.1, GRID_Y - 0.84, GRID_XS[1] - 0.1, GRID_Y - 0.84, color=VIOLET)
ax.text(gap_mid, GRID_Y + 0.06, "all-to-all", ha="center", va="bottom", fontsize=TINY, color=VIOLET, fontweight="bold")

card(
    ax,
    RULES_X,
    GRID_Y - card_height(3),
    RULES_W,
    card_height(3),
    "Requirements",
    ["num_attention_heads % cp == 0", "num_key_value_heads % cp == 0", "seq_len % cp == 0 (collator pads)"],
    color=SLATE,
    mono_lines=True,
)

footnote(
    ax,
    0.3,
    0.3,
    11.8,
    "Exact, not approximate — attention is independent across heads; the layer issues three all-to-alls:"
    " q, the fused k|v, and the output.",
)

save(fig, "ulysses_attention_flow")
plt.close(fig)
print("✓ ulysses_attention_flow.png")
