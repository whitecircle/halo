"""Card / cell / frame / arrow primitives shared by the pipeline diagrams.

A thin layer over `_theory_style` (palette, fonts, `save`) so every pipeline figure — GRPO, rollout,
parallelism, dataloader — draws the same box, the same labelled arrow and the same takeaway strip.
Import with `from _pipeline_style import *`, which re-exports `_theory_style`.

Geometry convention: **one data unit is one inch** — set `xlim`/`ylim` to the figsize,
and the fixed paddings here stay proportionate to the point sizes in `_theory_style`.
"""

from _theory_style import *
from matplotlib.patches import FancyBboxPatch, Rectangle

CARD_PAD = 0.16  # inset from a card's border to its text
TITLE_DROP = 0.34  # baseline step from a card title to its first line
LINE_H = 0.26  # baseline step between card lines
FOOT_H = 0.5  # height of the takeaway strip
FRAME_PAD = 0.20  # inset from a frame's boundary to the cards inside it
FRAME_LABEL = 0.45  # height a frame's label takes at the top of the frame
COLUMN_GAP = 1.0  # column gap, wide enough to hold an arrow label clear of both borders
DASH = (0, (4, 2))

TINTS = {BLUE: BLUE_T, AMBER: AMBER_T, TEAL: TEAL_T, VIOLET: VIOLET_T, ROSE: ROSE_T, SLATE: SLATE_T}


def tint(color):
    """The pale companion of a role stroke, for box fills."""
    return TINTS.get(color, SLATE_T)


def columns(width, n, margin, gap=COLUMN_GAP):
    """Left edges and common width of `n` equal columns spanning a `width`-inch page inside `margin`."""
    col_w = (width - 2 * margin - (n - 1) * gap) / n
    return [margin + i * (col_w + gap) for i in range(n)], col_w


def card_height(n_lines, has_title=True):
    """Height a `card` needs for `n_lines` lines — use it to give sibling cards one height."""
    return 2 * CARD_PAD + (TITLE_DROP if has_title else 0.0) + n_lines * LINE_H


def card(ax, x, y, w, h, title, lines, *, color=SLATE, mono_lines=False, dashed=False):
    """A rounded tinted box: bold title on the first line, `lines` stacked under it.

    `color` is a role stroke from `_theory_style`; `dashed` marks something optional or off.
    """
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor=tint(color), edgecolor=color, lw=1.3)
    if dashed:
        patch.set_linestyle(DASH)
    ax.add_patch(patch)

    ty = y + h - CARD_PAD
    if title:
        ax.text(x + CARD_PAD, ty, title, ha="left", va="top", fontsize=LABEL, fontweight="bold", color=INK)
        ty -= TITLE_DROP
    for line in lines:
        ax.text(
            x + CARD_PAD,
            ty,
            line,
            ha="left",
            va="top",
            fontsize=SMALL,
            color=INK2,
            fontfamily=MONO if mono_lines else SANS,
        )
        ty -= LINE_H
    return patch


def card_row(ax, xs, y, w, h, cards, *, mono_lines=False, labels=()):
    """A left-to-right chain of equal cards at `xs`, each joined to the next by an arrow at mid-height.

    `cards` holds one `(title, lines, color)` per x; `labels` names the joining arrows, in order.
    """
    mid = y + h / 2
    for i, (x, (head, lines, color)) in enumerate(zip(xs, cards, strict=True)):
        card(ax, x, y, w, h, head, lines, color=color, mono_lines=mono_lines)
        if i:
            arrow(ax, xs[i - 1] + w, mid, x, mid, labels[i - 1] if i - 1 < len(labels) else "")


def chip(ax, x, y, w, h, text="", *, color=SLATE, sub="", fontsize=SMALL, mono=False, bold=False, dashed=False):
    """A small tinted box with a centred label — a strip cell, a GPU square, a span of a timeline.

    `sub` adds a mono caption under the label; `dashed` empties the fill, marking a span that is absent.
    """
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02", facecolor=BG if dashed else tint(color), edgecolor=color, lw=1.2
    )
    if dashed:
        patch.set_linestyle(DASH)
    ax.add_patch(patch)

    if text:
        ax.text(
            x + w / 2,
            y + h * 0.62 if sub else y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=INK2 if dashed else INK,
            fontfamily=MONO if mono else SANS,
        )
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=TINY, color=INK2, fontfamily=MONO)


def grid(ax, x, top, cells, *, cell_w, cell_h, color=SLATE, row_labels=(), col_labels=(), caption=""):
    """A matrix of cells: `cells[r][c]` is one cell's mono label, or "" for an empty outline.

    `top` is the grid's upper edge; `row_labels` sit left of the rows, `col_labels` above the columns.
    """
    for r, row in enumerate(cells):
        y = top - (r + 1) * cell_h
        if r < len(row_labels):
            ax.text(x - 0.12, y + cell_h / 2, row_labels[r], ha="right", va="center", fontsize=TINY, color=INK3)
        for c, text in enumerate(row):
            cx = x + c * cell_w
            ax.add_patch(
                Rectangle(
                    (cx, y),
                    cell_w,
                    cell_h,
                    facecolor=tint(color) if text else BG,
                    edgecolor=color if text else LINE,
                    lw=1.1 if text else 0.9,
                )
            )
            if text:
                ax.text(
                    cx + cell_w / 2, y + cell_h / 2, text, ha="center", va="center", fontsize=TINY, fontfamily=MONO
                )
    for c, label in enumerate(col_labels):
        ax.text(x + (c + 0.5) * cell_w, top + 0.08, label, ha="center", va="bottom", fontsize=TINY, color=INK3)
    if caption:
        ax.text(
            x + len(cells[0]) * cell_w / 2,
            top - len(cells) * cell_h - 0.18,
            caption,
            ha="center",
            va="top",
            fontsize=SMALL,
            color=INK2,
        )


def frame(ax, x, y, w, h, label="", *, color=SLATE, dashed=True):
    """A boundary around a band of cards — a node, an EP group, a repeated layer stack.

    `label` names it at the top-left; a figure wanting another label style draws its own.
    """
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor=BG, edgecolor=color, lw=1.3)
    if dashed:
        patch.set_linestyle(DASH)
    ax.add_patch(patch)
    if label:
        ax.text(
            x + CARD_PAD, y + h - CARD_PAD, label, ha="left", va="top", fontsize=SMALL, fontweight="bold", color=INK2
        )


def arrow(ax, x0, y0, x1, y1, label="", color=INK2, side="above", *, lw=1.6, dashed=False):
    """A short arrow from (x0,y0) to (x1,y1); `label` sits beside it, never across a box.

    `side` places the label: "above"/"below" for a horizontal arrow, "left"/"right" for a vertical one.
    """
    props = {"arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 2, "shrinkB": 2}
    if dashed:
        props["linestyle"] = DASH
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=props)
    if not label:
        return
    mx, my, pad = (x0 + x1) / 2, (y0 + y1) / 2, 0.09
    lx, ly, ha, va = {
        "above": (mx, my + pad, "center", "bottom"),
        "below": (mx, my - pad, "center", "top"),
        "left": (mx - pad, my, "right", "center"),
        "right": (mx + pad, my, "left", "center"),
    }[side]
    ax.text(lx, ly, label, ha=ha, va=va, fontsize=TINY, color=color, fontweight="bold")


def polyline_arrow(ax, points, label, label_xy, color=INK2):
    """An arrow bent through `points`, for a path that cannot run straight; the label sits above `label_xy`."""
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=1.6, zorder=1)
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "shrinkA": 0, "shrinkB": 2},
    )
    ax.text(*label_xy, label, ha="center", va="bottom", fontsize=TINY, color=color, fontweight="bold")


def replica_link(ax, x0, x1, y, color=SLATE):
    """A dashed line joining two peer cards, with a dot at each end."""
    ax.plot([x0, x1], [y, y], color=color, lw=1.2, ls=DASH, solid_capstyle="butt", zorder=1)
    ax.plot([x0, x1], [y, y], "o", color=color, markersize=3.4, zorder=2)


def section(ax, x, y, text):
    """A section heading over a band of cards; `y` is its top."""
    ax.text(x, y, text, ha="left", va="top", fontsize=SECTION, fontweight="bold", color=INK)


def footnote(ax, x, y, w, text):
    """The one-line takeaway: a dashed slate strip across the bottom of the figure."""
    strip = FancyBboxPatch((x, y), w, FOOT_H, boxstyle="round,pad=0.04", facecolor=BG, edgecolor=INK2, lw=1.2)
    strip.set_linestyle(DASH)
    ax.add_patch(strip)
    ax.text(x + w / 2, y + FOOT_H / 2, text, ha="center", va="center", fontsize=SMALL, color=INK)


def title(ax, text, sub_mono=""):
    """Left-aligned title with an optional mono subtitle flushed right on the same line."""
    x0, x1 = ax.get_xlim()
    y1 = ax.get_ylim()[1]
    inset = 0.025 * (x1 - x0)
    ax.text(x0 + inset, y1, text, ha="left", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    if sub_mono:
        ax.text(x1 - inset, y1 - 0.06, sub_mono, ha="right", va="top", fontsize=SUB, color=INK2, fontfamily=MONO)
