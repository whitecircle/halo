"""Generate Batch Construction & Rollout Pipeline diagrams.

Split into two individual images:
- batch_prompt_expansion.png — Prompt Expansion & GRPO Groups
- batch_rollout_pipeline.png — Rollout Collection Pipeline
"""

import matplotlib.pyplot as plt
from _flow_style import *
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.size"] = 11

GREEN_BG = "#f0fdf4"
GREEN_BD = "#86efac"
GREEN = "#16a34a"
ORANGE_BG = "#fff7ed"
ORANGE_BD = "#fdba74"
ORANGE = "#ea580c"


def _box(ax, x, y, w, h, bg=CARD, border=CARD_BORDER, lw=1.0, ls="-"):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls
    )
    ax.add_patch(p)


def draw_block(ax, x, y, w, h, title, subtitle="", bg=CARD, border=CARD_BORDER, title_size=12):
    _box(ax, x, y, w, h, bg, border)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h / 2 + 0.18,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=TEXT,
        )
        ax.text(x + w / 2, y + h / 2 - 0.22, subtitle, ha="center", va="center", fontsize=10, color=TEXT_SEC)
    else:
        ax.text(
            x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=TEXT
        )


def arrow_down(ax, x, y1, y2, color=BORDER_SOFT):
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "shrinkA": 6, "shrinkB": 6},
    )


def note(ax, x, y, text, size=10):
    """Subtle italic annotation between elements."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=TEXT_TERT, style="italic")


def plabel(ax, x, y, text, color=ACCENT, bg=ACCENT_LIGHT, size=14):
    """Prompt label with rounded box."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": bg, "edgecolor": color, "lw": 0.6},
    )


# Image 1 — Prompt Expansion & GRPO Groups
fig, ax = plt.subplots(figsize=(12, 18))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.axis("off")

CX = 1.0  # content left
CW = 10.0  # content width
CC = CX + CW / 2  # center x

ax.text(CC, 17.5, "Prompt Expansion & GRPO Groups", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
ax.text(
    CC,
    17.0,
    "batch_size = 2  \u00b7  num_generations = 4  \u00b7  grad_accum = 2",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_TERT,
)

_box(ax, CX, 15.3, CW, 1.3, CARD, CARD_BORDER)
ax.text(CC, 16.3, "Dataset", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)
for i, p in enumerate(["P1", "P2", "P3", "P4", "P5", "P6", "\u2026"]):
    px = CX + CW * (i + 1) / 8
    if p == "\u2026":
        ax.text(px, 15.7, p, ha="center", va="center", fontsize=14, color=TEXT_TERT)
    else:
        plabel(ax, px, 15.7, p, size=13)

arrow_down(ax, CC, 15.3, 14.6)
note(ax, CC + 2.5, 14.95, "sample batch_size = 2")

MB_W = 4.5
MB_GAP = 1.0
mb1_x = CC - MB_GAP / 2 - MB_W
mb2_x = CC + MB_GAP / 2
mb1_c = mb1_x + MB_W / 2
mb2_c = mb2_x + MB_W / 2

_box(ax, mb1_x, 12.8, MB_W, 1.7, GREEN_BG, GREEN_BD)
ax.text(mb1_c, 14.1, "Micro-batch 1", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)
plabel(ax, mb1_c - 0.75, 13.35, "P1", GREEN, GREEN_BG, size=16)
plabel(ax, mb1_c + 0.75, 13.35, "P2", GREEN, GREEN_BG, size=16)

_box(ax, mb2_x, 12.8, MB_W, 1.7, ORANGE_BG, ORANGE_BD)
ax.text(mb2_c, 14.1, "Micro-batch 2", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)
plabel(ax, mb2_c - 0.75, 13.35, "P3", ORANGE, ORANGE_BG, size=16)
plabel(ax, mb2_c + 0.75, 13.35, "P4", ORANGE, ORANGE_BG, size=16)

note(ax, mb1_c, 12.5, "grad_accum step 1")
note(ax, mb2_c, 12.5, "grad_accum step 2")

arrow_down(ax, mb1_c, 12.5, 11.6)
arrow_down(ax, mb2_c, 12.5, 11.6)
note(ax, CC, 12.0, "\u00d7 num_generations = 4")

EXP_H = 3.2

_box(ax, mb1_x, 8.3, MB_W, EXP_H, GREEN_BG, GREEN_BD)
ax.text(
    mb1_c,
    11.05,
    "Expanded Rollouts (8 sequences)",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT,
)

ax.plot([mb1_x + MB_W / 2, mb1_x + MB_W / 2], [8.5, 10.6], color=GREEN_BD, lw=0.6, linestyle="--")

ax.text(mb1_c - MB_W / 4, 10.3, "GRPO Group 1", ha="center", va="center", fontsize=10, color=TEXT_SEC, style="italic")
plabel(ax, mb1_c - MB_W / 4, 9.5, "P1 \u00d7 4", GREEN, GREEN_BG, size=13)

ax.text(mb1_c + MB_W / 4, 10.3, "GRPO Group 2", ha="center", va="center", fontsize=10, color=TEXT_SEC, style="italic")
plabel(ax, mb1_c + MB_W / 4, 9.5, "P2 \u00d7 4", GREEN, GREEN_BG, size=13)

_box(ax, mb2_x, 8.3, MB_W, EXP_H, ORANGE_BG, ORANGE_BD)
ax.text(
    mb2_c,
    11.05,
    "Expanded Rollouts (8 sequences)",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT,
)

ax.plot([mb2_x + MB_W / 2, mb2_x + MB_W / 2], [8.5, 10.6], color=ORANGE_BD, lw=0.6, linestyle="--")

ax.text(mb2_c - MB_W / 4, 10.3, "GRPO Group 3", ha="center", va="center", fontsize=10, color=TEXT_SEC, style="italic")
plabel(ax, mb2_c - MB_W / 4, 9.5, "P3 \u00d7 4", ORANGE, ORANGE_BG, size=13)

ax.text(mb2_c + MB_W / 4, 10.3, "GRPO Group 4", ha="center", va="center", fontsize=10, color=TEXT_SEC, style="italic")
plabel(ax, mb2_c + MB_W / 4, 9.5, "P4 \u00d7 4", ORANGE, ORANGE_BG, size=13)

ax.text(CC, 7.8, "Optimizer Step Summary", ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)

ax.annotate(
    "",
    xy=(CC - 0.1, 8.05),
    xytext=(mb1_c, 8.2),
    arrowprops={"arrowstyle": "-", "color": BORDER_SOFT, "lw": 1.5, "shrinkA": 0, "shrinkB": 0},
    zorder=5,
)
ax.annotate(
    "",
    xy=(CC + 0.1, 8.05),
    xytext=(mb2_c, 8.2),
    arrowprops={"arrowstyle": "-", "color": BORDER_SOFT, "lw": 1.5, "shrinkA": 0, "shrinkB": 0},
    zorder=5,
)
ax.annotate(
    "",
    xy=(CC, 7.95),
    xytext=(CC, 8.05),
    arrowprops={"arrowstyle": "-|>", "color": BORDER_SOFT, "lw": 1.5, "shrinkA": 0, "shrinkB": 0},
    zorder=5,
)
SUM_H = 2.4
_box(ax, CX, 4.8, CW, SUM_H, CARD, CARD_BORDER)

for i, (label, val) in enumerate(
    [
        ("Unique prompts", "4"),
        ("Total rollouts", "16"),
        ("GRPO groups", "4 (of 4)"),
        ("Grad accum steps", "2"),
    ]
):
    mx = CX + CW / 4 * (i + 0.5)
    ax.text(mx, 6.6, label, ha="center", va="center", fontsize=11, color=TEXT_SEC)
    ax.text(mx, 5.7, val, ha="center", va="center", fontsize=16, fontweight="bold", color=ACCENT)

ax.text(
    CC,
    3.9,
    "total_rollouts = batch_size \u00d7 num_generations \u00d7 grad_accum = 2 \u00d7 4 \u00d7 2 = 16",
    ha="center",
    va="center",
    fontsize=11,
    color=TEXT_SEC,
    bbox={"boxstyle": "round,pad=0.15", "facecolor": CARD, "edgecolor": CARD_BORDER, "lw": 0.6},
)

save(plt.gcf(), "batch_prompt_expansion")
plt.close()
print("\u2713 batch_prompt_expansion.png")


# Image 2 — Rollout Collection Pipeline
fig, ax = plt.subplots(figsize=(12, 18))
fig.patch.set_facecolor(BG)

ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.axis("off")

CX = 1.0
CW = 10.0
CC = CX + CW / 2

ax.text(CC, 17.5, "Rollout Collection Pipeline", ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
ax.text(
    CC,
    17.0,
    "Per micro-batch: expanded prompts dispatched via Ray actors",
    ha="center",
    va="top",
    fontsize=11,
    color=TEXT_TERT,
)

draw_block(
    ax, CX, 15.3, CW, 1.0, "Trainer", "Samples batch_size prompts, expands \u00d7 num_generations", title_size=13
)

arrow_down(ax, CC, 15.3, 14.8)
note(ax, CC + 2.3, 15.05, "8 expanded prompts")

draw_block(
    ax,
    CX,
    13.3,
    CW,
    1.5,
    "RolloutManager",
    "Semaphore(max_concurrent)  \u00b7  asyncio.gather()  \u00b7  round-robin vLLM",
    bg=ACCENT_LIGHT,
    border=ACCENT,
    title_size=13,
)

ax.text(
    CX + CW - 0.2, 14.55, "HTTP \u2192 vLLM (GPU inference)", ha="right", va="center", fontsize=9.5, color=TEXT_TERT
)

arrow_down(ax, CC, 13.3, 12.8)
note(ax, CC + 2.0, 13.05, "dispatch to actors")

ACTORS_H = 4.0
_box(ax, CX - 0.2, 8.8, CW + 0.4, ACTORS_H, CARD, CARD_BORDER, ls="--")
ax.text(
    CX + 0.1,
    12.55,
    "Ray Environment Actors",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    color=TEXT_SEC,
    style="italic",
)

AW = 1.05  # actor width
AH = 2.2  # actor height
AG = 0.12  # gap between actors
total_w = 8 * AW + 7 * AG
a_start = CC - total_w / 2
a_bot = 9.15  # actor box bottom y

for i in range(8):
    ax_x = a_start + i * (AW + AG)
    abg = GREEN_BG if i < 4 else ORANGE_BG
    abd = GREEN_BD if i < 4 else ORANGE_BD
    pc = GREEN if i < 4 else ORANGE
    pbg = GREEN_BG if i < 4 else ORANGE_BG

    _box(ax, ax_x, a_bot, AW, AH, abg, abd, lw=0.7)

    ax.text(
        ax_x + AW / 2, a_bot + AH - 0.35, f"A{i}", ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT
    )

    plabel(ax, ax_x + AW / 2, a_bot + AH / 2, "P1" if i < 4 else "P2", pc, pbg, size=13)

    ax.text(
        ax_x + AW / 2, a_bot + 0.3, "gen\u2192exec\u2192\u2026", ha="center", va="center", fontsize=8, color=TEXT_TERT
    )

ax.text(
    a_start + 2 * (AW + AG),
    8.85,
    "Group 1 (P1 \u00d7 4)",
    ha="center",
    va="center",
    fontsize=10.5,
    color=GREEN,
    fontweight="bold",
)
ax.text(
    a_start + 6 * (AW + AG),
    8.85,
    "Group 2 (P2 \u00d7 4)",
    ha="center",
    va="center",
    fontsize=10.5,
    color=ORANGE,
    fontweight="bold",
)

arrow_down(ax, CC, 8.8, 8.3)

EP_H = 5.8
_box(ax, CX - 0.2, 2.5, CW + 0.4, EP_H, CARD, CARD_BORDER, ls="--")
ax.text(
    CX + 0.1,
    8.05,
    "Multi-Turn Episode (per actor)",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    color=TEXT_SEC,
    style="italic",
)

steps = [
    ("Turn 1", "Generate\nresponse", ACCENT),
    ("", "Parse\ntool call", TEXT_SEC),
    ("", "Execute\ntool", GREEN),
    ("Turn 2", "Generate with\nobservation", ACCENT),
    ("", "Parse &\nexecute tool", GREEN),
    ("Done", "Final answer\nor max turns", TEXT_SEC),
]
n = len(steps)
tl_l = CX + 0.8
tl_r = CX + CW - 0.8
tl_y = 5.8  # timeline dot y
sp = (tl_r - tl_l) / (n - 1)

for i, (label, desc, color) in enumerate(steps):
    sx = tl_l + i * sp
    ax.plot(sx, tl_y, "o", color=color, markersize=9, zorder=4)
    if label:
        ax.text(sx, tl_y + 0.75, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
    ax.text(sx, tl_y - 1.0, desc, ha="center", va="center", fontsize=9.5, color=TEXT_SEC, linespacing=1.2)
    if i < n - 1:
        nx = tl_l + (i + 1) * sp
        ax.annotate(
            "",
            xy=(nx - 0.15, tl_y),
            xytext=(sx + 0.15, tl_y),
            arrowprops={"arrowstyle": "-|>", "color": BORDER_SOFT, "lw": 1.0},
            zorder=2,
        )

note(ax, CC, 3.6, "latency \u2248 max_turns \u00d7 (generation + tool_execution)")
ax.text(
    CC,
    3.0,
    "reward = env._compute_reward(trajectory)",
    ha="center",
    va="center",
    fontsize=10.5,
    color=TEXT_SEC,
    bbox={"boxstyle": "round,pad=0.12", "facecolor": GREEN_BG, "edgecolor": GREEN_BD, "lw": 0.6},
)

save(plt.gcf(), "batch_rollout_pipeline")
plt.close()
print("\u2713 batch_rollout_pipeline.png")
