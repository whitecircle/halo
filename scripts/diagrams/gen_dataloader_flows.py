"""Generate the two dataloader-path figures for parallelism/data-loading.md.

One figure per path, drawn on the same grid so the pair reads as a before/after: the standard
path shards batches by **global rank** (one distinct batch per rank), the custom path by **DP
rank** (the ranks of a TP/CP/ETP group, and of a pipeline chain, read the same batch). The worked
shape is world 16 with `tp_size=2` → `data_parallel_size = 8`.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 11.6, 5.7

GATE_Y, GATE_H = 4.40, card_height(1)
STAGE_Y, STAGE_H = 2.94, card_height(2)
STAGE_W, STAGE_XS = 2.6, (0.3, 3.15, 6.0, 8.85)
STRIP_X, STRIP_W = 0.3, 10.9
RANK_Y, BATCH_Y, ROW_H = 1.80, 1.18, 0.5
RANKS, CELL_GAP = 16, 0.1
# The strip is its own page: the cells span STRIP_W with no margin, offset by STRIP_X.
CELL_XS, CELL_W = columns(STRIP_W, RANKS, 0.0, CELL_GAP)

GATE_FLAGS = "is_tp_mode · is_cp_mode · is_expert_tp_mode · is_pp_mode · _dataset_presharded"
SAMPLER = ("Sampler (not distributed)", ["_get_train_sampler()", "→ RandomSampler"], TEAL)


def cell_x(i):
    """Left edge of rank `i`'s cell — both figures share one strip geometry."""
    return STRIP_X + CELL_XS[i]


def rank_row(ax):
    """The 16 ranks — identical in both figures."""
    for r in range(RANKS):
        chip(ax, cell_x(r), RANK_Y, CELL_W, ROW_H, f"r{r}", color=TEAL, fontsize=TINY, mono=True)


def standard_batches(ax):
    """16 distinct batches: one batch cell per rank cell."""
    for r in range(RANKS):
        chip(ax, cell_x(r), BATCH_Y, CELL_W, ROW_H, f"b{r}", color=SLATE, fontsize=TINY, mono=True)


def custom_batches(ax):
    """8 distinct batches: one batch cell spans the two ranks of a TP group."""
    for pair in range(RANKS // 2):
        x0, x1 = cell_x(2 * pair), cell_x(2 * pair + 1) + CELL_W
        chip(ax, x0, BATCH_Y, x1 - x0, ROW_H, f"b{pair}", color=SLATE, fontsize=TINY, mono=True)


def panel(name, head, sub, gate_title, gate_note, stages, batches, strip_caption, foot):
    """Gate strip → four stage cards → the rank/batch strip → the takeaway."""
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    title(ax, head, sub)

    card(
        ax,
        STRIP_X,
        GATE_Y,
        STRIP_W,
        GATE_H,
        gate_title,
        [f"{GATE_FLAGS}   {gate_note}"],
        color=VIOLET,
        mono_lines=True,
    )
    arrow(ax, STAGE_XS[0] + STAGE_W / 2, GATE_Y, STAGE_XS[0] + STAGE_W / 2, STAGE_Y + STAGE_H)

    card_row(ax, STAGE_XS, STAGE_Y, STAGE_W, STAGE_H, stages, mono_lines=True)

    section(ax, STRIP_X, RANK_Y + ROW_H + 0.32, strip_caption)
    rank_row(ax)
    batches(ax)
    footnote(ax, STRIP_X, 0.3, STRIP_W, foot)

    save(fig, name)
    plt.close(fig)
    print(f"✓ {name}.png")


panel(
    "dataloader_standard",
    "Standard dataloader path",
    "world 16 · DDP / FSDP / EP-only · dp = world = 16",
    "_needs_custom_dataloader() → False",
    "— none set",
    [
        ("Trainer (HF / TRL)", ["get_train_dataloader()", "accelerator.prepare(dl)"], SLATE),
        SAMPLER,
        ("Accelerate prepare", ["num_processes = 16", "process_index = rank"], VIOLET),
        ("Batch sharding", ["BatchSamplerShard", "idx % 16 == rank"], BLUE),
    ],
    standard_batches,
    "What each rank reads — 16 distinct batches per step",
    "EP stays on this path: DeepEP returns every token to its origin rank, so EP never reduces data_parallel_size.",
)

panel(
    "dataloader_custom",
    "Custom dataloader path",
    "tp 2 · dp = (world / pp) / max(cp, tp, etp) = 16 / 2 = 8",
    "_needs_custom_dataloader() → True",
    "— any one set",
    [
        ("Trainer (toolkit)", ["get_train_dataloader()", "_prepare_dataloader(dl)"], SLATE),
        SAMPLER,
        ("Accelerate prepare", ["num_processes = 8", "process_index = dp_rank"], VIOLET),
        ("Batch sharding", ["BatchSamplerShard", "idx % 8 == dp_rank"], BLUE),
    ],
    custom_batches,
    "What each rank reads — 8 distinct batches, one per TP group",
    "dp_rank = stage_local_rank // max(tp, cp), so a pipeline chain shares it too;"
    " a pre-sharded dataset passes num_processes = 1.",
)
