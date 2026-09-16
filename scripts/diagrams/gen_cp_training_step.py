"""Generate the CP training-step figure for parallelism/context-parallelism.md.

Top to bottom: `UlyssesCPModelWrapper` slices the batch into one contiguous sequence chunk per CP
rank, every layer runs Ulysses attention over the full sequence and its MLP/MoE on the local chunk
only, the logits stay sharded, and the loss is normalized over the CP group's all-reduced token
count.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 11.9, 6.6

ROW_H = card_height(2)
TOP_Y, (TOP_XS, TOP_W) = 5.00, columns(W, 3, 0.3, 0.25)
LAYER_Y, LAYER_H = 2.95, 1.60
INNER_Y, (INNER_XS, INNER_W) = 3.05, columns(W, 2, 0.55, 0.2)
OUT_Y, (OUT_XS, OUT_W) = 1.32, columns(W, 2, 0.3, 0.5)
ENTER_X = TOP_XS[-1] + TOP_W / 2  # the stack is entered under the last top-band card
EXIT_X = OUT_XS[0] + OUT_W / 2  # and left again above the first output card

fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "A training step under CP", "cp 4 · S 32768 → 8192 tokens per rank")

top_cards = [
    ("Full batch", ["input_ids [B, S]", "labels · mask · position_ids"], SLATE),
    ("split_sequence_for_cp", ["narrow(1, rank·S/4, S/4)", "S % cp_size == 0 (collator pads)"], TEAL),
    ("Embeddings on the chunk", ["[B, S/4] → [B, S/4, hidden]", "each rank holds its own chunk"], BLUE),
]
card_row(ax, TOP_XS, TOP_Y, TOP_W, ROW_H, top_cards, mono_lines=True)

arrow(ax, ENTER_X, TOP_Y, ENTER_X, LAYER_Y + LAYER_H, "hidden states", side="left")

frame(ax, 0.3, LAYER_Y, 11.3, LAYER_H)
ax.text(0.46, LAYER_Y + LAYER_H - 0.11, "× N decoder layers", ha="left", va="top", fontsize=SMALL, color=INK2)

inner_cards = [
    ("Ulysses attention", ["all-to-all → flash → all-to-all", "full S, 64/cp_size heads per rank"], VIOLET),
    ("MLP / MoE experts", ["runs on [B, S/4, hidden] only", "CP-unaware: EP dispatches S/4"], BLUE),
]
card_row(ax, INNER_XS, INNER_Y, INNER_W, ROW_H, inner_cards, mono_lines=True)

arrow(ax, EXIT_X, LAYER_Y, EXIT_X, OUT_Y + ROW_H, "final norm → lm_head", side="right")

out_cards = [
    ("Logits, per chunk", ["[B, S/4, V] — never gathered", "boundary label from the next chunk"], BLUE),
    ("Loss", ["cp_size · local_sum / global_tokens", "global_tokens: all_reduce(SUM, cp)"], TEAL),
]
card_row(ax, OUT_XS, OUT_Y, OUT_W, ROW_H, out_cards, mono_lines=True)

footnote(
    ax,
    0.3,
    0.35,
    11.3,
    "Only attention ever sees the whole sequence — every other layer, and the loss, works on this rank's chunk.",
)

save(fig, "cp_training_step")
plt.close(fig)
print("✓ cp_training_step.png")
