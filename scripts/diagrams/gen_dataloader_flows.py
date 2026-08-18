"""Generate DataLoader Pipelines diagrams — individual images."""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.size"] = 10.5


def draw_step(ax, x, y, w, h, text, bg=CARD, border=CARD_BORDER, fs=10.5, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=TEXT,
        linespacing=1.3,
    )


def draw_arrow(ax, x, y1, y2):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8, "shrinkA": 6, "shrinkB": 6},
    )


GAP = 0.6
sw = 7.2
sx = 1.4


def flow_panel(ax, title, subtitle, steps):
    """Draw a vertical flow of steps with an in-axes title (avoids the fig.text top gap)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 11.85, title, ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(5, 11.0, subtitle, ha="center", va="top", fontsize=10.5, color=TEXT_SEC)

    y = 10.2
    for i, (h, text, kw) in enumerate(steps):
        block_y = y - h
        draw_step(ax, sx, block_y, sw, h, text, **kw)

        if kw.get("bg") == OP_BG:
            ax.texts[-1].set_color(OP_TEXT)

        if i < len(steps) - 1:
            draw_arrow(ax, 5, block_y, block_y - GAP)
            y = block_y - GAP
        else:
            y = block_y


# Image 1: Standard DataLoader Flow
std_steps = [
    (0.7, "Trainer.get_train_dataloader()", {"fs": 10.5, "bold": True}),
    (0.8, "_get_train_sampler()\n\u2192 RandomSampler", {"fs": 10}),
    (0.7, "DataLoader(dataset, sampler=RandomSampler)", {"fs": 10}),
    (0.7, "accelerator.prepare(dataloader)", {"fs": 10.5, "bold": True}),
    (
        1.1,
        "accelerate.prepare_data_loader()\nnum_processes = world_size (16)\nprocess_index = global rank",
        {"fs": 10},
    ),
    (0.9, "BatchSamplerShard\nbatches[rank :: world_size]", {"bg": OP_BG, "border": OP_BG, "fs": 10.5}),
    (0.7, "16 ranks \u2192 16 distinct batches/step", {"bg": ACCENT_LIGHT, "border": ACCENT}),
]

fig, ax = plt.subplots(figsize=(10, 12))
fig.patch.set_facecolor(BG)
flow_panel(ax, "Standard DataLoader", "DDP / FSDP / EP-only, dataset not pre-sharded", std_steps)

save(plt.gcf(), "dataloader_standard")
plt.close()
print("\u2713 dataloader_standard.png")


# Image 2: Custom DataLoader Flow
custom_steps = [
    (0.7, "DistributedTrainer.get_train_dataloader()", {"fs": 10.5, "bold": True}),
    (
        1.0,
        "_needs_custom_dataloader() \u2192 True\n(TP, CP, ETP or PP enabled,\nor dataset pre-sharded)",
        {"bg": ACCENT_MID, "border": ACCENT, "fs": 10},
    ),
    (0.7, "RandomSampler (NOT DistributedSampler)", {"fs": 10}),
    (0.7, "_prepare_dataloader(dataloader)", {"fs": 10.5, "bold": True}),
    (1.1, "prepare_data_loader()\ndp_size = world / tp_size (e.g., 16/8 = 2)\ndp_rank = rank // tp_size", {"fs": 10}),
    (
        1.1,
        "BatchSamplerShard\ndp_rank=0 \u2192 batches[0,2,4,...]\ndp_rank=1 \u2192 batches[1,3,5,...]",
        {"bg": OP_BG, "border": OP_BG, "fs": 10},
    ),
    (
        0.8,
        "16 ranks \u2192 2 distinct batches/step\n(8 ranks per TP group share batch)",
        {"bg": ACCENT_LIGHT, "border": ACCENT},
    ),
]

fig, ax = plt.subplots(figsize=(10, 12))
fig.patch.set_facecolor(BG)
flow_panel(ax, "Custom DataLoader", "TP / CP / ETP / PP / pre-sharded  (batch sharing required)", custom_steps)

save(plt.gcf(), "dataloader_custom")
plt.close()
print("\u2713 dataloader_custom.png")
