"""Shared visual style for the pipeline / flow diagrams.

One palette, one rcParams block and one set of card/arrow primitives across the
flow figures (GRPO/SMPO pipelines, dataloader flows, EP/CP/Ulysses data-flow,
multi-node deployment, …), so they read as a single set. Import with
`from _flow_style import *` at the top of each `gen_*.py`.

Deliberately distinct from its sibling `_theory_style.py`: the theory figures
use the IBM Plex family and a slate accent palette, the flow figures use
matplotlib's bundled DejaVu Sans and a lighter "card" palette (slate ink on
white, blue accent). The font is pinned to the bundled face so the PNG bytes
are the same on every host — CI checks the committed figures by hash.

Each flow script still sets any base font size other than 10 after the import
with `plt.rcParams["font.size"] = <n>`, and defines its one-off tokens
(green/orange callouts, node tints, …) locally.
"""

import functools
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from _style_base import assets_dir, save_figure
from matplotlib.patches import FancyBboxPatch

ASSETS = assets_dir("diagrams")

BG = "#ffffff"
CARD = "#f8fafc"
CARD_BORDER = "#e2e8f0"
ACCENT = "#3b82f6"
ACCENT_LIGHT = "#eff6ff"
ACCENT_MID = "#dbeafe"
OP_BG = "#1e293b"
OP_TEXT = "#ffffff"
TEXT = "#0f172a"
TEXT_SEC = "#64748b"
TEXT_TERT = "#94a3b8"
BORDER_SOFT = "#cbd5e1"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": BG,
        "text.color": TEXT,
    }
)


ARROW_PROPS = {"arrowstyle": "-|>", "color": ACCENT, "lw": 1.8, "shrinkA": 6, "shrinkB": 6}


@dataclass(frozen=True)
class CardLayout:
    """Inner spacing of a :func:`draw_card`, in axes units.

    A figure-level knob rather than a constant: the pipeline figures were laid out at slightly
    different scales, and CI byte-compares the committed PNGs — so each caller keeps the metrics its
    figure was drawn with instead of every card being nudged to one set. The bullet text always
    starts 0.15 right of its marker.
    """

    title_dy: float = 0.2
    sep_dy: float = 0.5
    line_dy: float = 0.3
    line_step: float = 0.34
    bullet_dx: float = 0.25
    line_size: float = 9.5


CARD_LAYOUT = CardLayout()


def draw_card(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines,
    bg=CARD,
    border=CARD_BORDER,
    accent_left=False,
    title_size=12,
    layout=CARD_LAYOUT,
):
    """A rounded card: title, a rule under it, and one accent-bulleted line per entry in `lines`.

    `accent_left` adds the accent spine that marks a pipeline's entry point.
    """
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=bg, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    if accent_left:
        bar = FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0", facecolor=ACCENT, edgecolor="none")
        ax.add_patch(bar)
    ax.text(
        x + w / 2,
        y + h - layout.title_dy,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
    )
    if not lines:
        return
    sep_y = y + h - layout.sep_dy
    ax.plot([x + 0.2, x + w - 0.2], [sep_y, sep_y], color=BORDER_SOFT, lw=0.6)
    for i, line in enumerate(lines):
        ly = sep_y - layout.line_dy - i * layout.line_step
        ax.plot(x + layout.bullet_dx, ly, "o", color=ACCENT, markersize=2.5)
        ax.text(
            x + layout.bullet_dx + 0.15, ly, line, ha="left", va="center", fontsize=layout.line_size, color=TEXT_SEC
        )


def draw_arrow_h(ax, x1, y, x2, label=""):
    """A left-to-right accent arrow, optionally labelled above its midpoint."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=ARROW_PROPS)
    if label:
        ax.text((x1 + x2) / 2, y + 0.18, label, ha="center", va="center", fontsize=8, color=TEXT_SEC, zorder=5)


def draw_arrow_v(ax, x, y1, y2):
    """A top-to-bottom accent arrow at `x`."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=ARROW_PROPS)


def bind_drawers(ax, layout=CARD_LAYOUT):
    """`(draw_card, draw_arrow_h, draw_arrow_v)` bound to one figure's axes and card metrics.

    A flow figure has exactly one axes and one set of card metrics, so binding them once here keeps
    every call site down to the geometry that actually varies.
    """
    return (
        functools.partial(draw_card, ax, layout=layout),
        functools.partial(draw_arrow_h, ax),
        functools.partial(draw_arrow_v, ax),
    )


def save(fig, name, dpi=200):
    """Write `<name>.png` into agent-docs/assets/diagrams with consistent options."""
    save_figure(
        fig,
        os.path.join(ASSETS, f"{name}.png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none",
    )
