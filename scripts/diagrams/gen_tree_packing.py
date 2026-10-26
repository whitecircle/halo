"""Generate the tree-packing figure for optimization/tree-packing.md.

`tree_packing_layout` — three rows that share a prefix, packed flat and as a tree: the row layout
with its two per-token columns, and the attention each token gets.
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
from _pipeline_style import *
from matplotlib.patches import Rectangle


class Segment(NamedTuple):
    name: str
    short: str
    tokens: int
    color: str


# The example drawn throughout: the shared prefix and each row's own tail.
PREFIX = Segment("shared prefix", "prefix", 6, SLATE)
TAILS = (Segment("row A", "A", 2, BLUE), Segment("row B", "B", 3, TEAL), Segment("row C", "C", 2, AMBER))
SEGMENTS = (PREFIX, *TAILS)

W, H = 12.6, 6.6
TOKEN = 0.2  # width of one token in the strips
STRIP_H = 0.5
LEFT = 1.55  # left edge of the strips, clear of the column names
CELL = 0.27  # one token in the attention grid
GRID_X, GRID_TOP = 8.75, 5.05

fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

title(ax, "Tree packing", sub_mono=f"{len(TAILS)} rows · {PREFIX.tokens}-token shared prefix")


def strip(y, segments):
    """Segments left to right, one token wide per token; returns each chip's center."""
    x, centers = LEFT, []
    for segment in segments:
        width = segment.tokens * TOKEN
        label = segment.name if segment.tokens > 4 else segment.short
        chip(ax, x, y, width, STRIP_H, label, color=segment.color, fontsize=TINY)
        centers.append(x + width / 2)
        x += width + 0.05
    return centers


def under(y, name, centers, values):
    """One mono value under each chip of the tree strip, with the column's name at the left."""
    ax.text(LEFT - 0.15, y, name, ha="right", va="center", fontsize=TINY, color=INK2, fontfamily=MONO)
    for x, value in zip(centers, values, strict=True):
        ax.text(x, y, value, ha="center", va="center", fontsize=TINY, color=INK2, fontfamily=MONO)


def tokens_of(segments):
    return sum(segment.tokens for segment in segments)


# ── Flat: every row repeats the prefix ───────────────────────────────────────
section(ax, 0.3, 5.75, "PACKED FLAT")
flat = [segment for tail in TAILS for segment in (PREFIX, tail)]
strip(4.85, flat)
ax.text(LEFT, 4.67, f"{tokens_of(flat)} tokens: the prefix is computed once per row", va="top", fontsize=SMALL)

# ── Tree: the prefix once, then one branch per row ───────────────────────────
section(ax, 0.3, 3.75, "PACKED AS A TREE")
centers = strip(2.85, SEGMENTS)
first = PREFIX.tokens
under(2.62, "position_ids", centers, [f"0–{first - 1}", *(f"{first}–{first + tail.tokens - 1}" for tail in TAILS)])
under(2.36, "tree_depth", centers, ["0", *("1" for _ in TAILS)])
ax.text(
    LEFT, 2.1, f"{tokens_of(SEGMENTS)} tokens: every branch continues the prefix's positions", va="top", fontsize=SMALL
)

# ── Attention: a token sees its own path, never a sibling ────────────────────
section(ax, GRID_X - 1.3, 5.75, "ATTENTION IN THE TREE ROW")
tokens = [(index, offset) for index, segment in enumerate(SEGMENTS) for offset in range(segment.tokens)]
position = [offset if segment == 0 else first + offset for segment, offset in tokens]
for query, (query_segment, _) in enumerate(tokens):
    for key, (key_segment, _) in enumerate(tokens):
        sees = key_segment in (0, query_segment) and position[key] <= position[query]
        color = SEGMENTS[key_segment].color
        ax.add_patch(
            Rectangle(
                (GRID_X + key * CELL, GRID_TOP - (query + 1) * CELL),
                CELL,
                CELL,
                facecolor=tint(color) if sees else BG,
                edgecolor=color if sees else LINE,
                lw=1.0 if sees else 0.7,
            )
        )
start = 0
for segment in SEGMENTS:
    middle = start + segment.tokens / 2
    label = {"fontsize": TINY, "color": segment.color, "fontweight": "bold"}
    ax.text(GRID_X + middle * CELL, GRID_TOP + 0.08, segment.short, ha="center", va="bottom", **label)
    ax.text(GRID_X - 0.12, GRID_TOP - middle * CELL, segment.name, ha="right", va="center", **label)
    start += segment.tokens
ax.text(
    GRID_X + len(tokens) * CELL / 2,
    GRID_TOP - len(tokens) * CELL - 0.12,
    "a query (row) sees the prefix and its own branch",
    ha="center",
    va="top",
    fontsize=SMALL,
    color=INK2,
)

footnote(
    ax,
    0.3,
    0.3,
    W - 0.6,
    "Each row keeps the tokens, positions and context it has on its own; only the repeated work is gone.",
)

save(fig, "tree_packing_layout")
