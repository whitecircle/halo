"""Shared visual style for the GPU-training-theory diagrams.

One palette and one typeface family across the figures in
`agent-docs/reference/gpu-training-theory.md`, so they read as a single set.
Import with `from _theory_style import *` at the top of each `gen_*.py`.

Fonts are IBM Plex Sans / IBM Plex Mono (SIL OFL, bundled under `fonts/`).
They are registered here by path because the training image ships no system
fonts; without this the diagrams fall back to DejaVu Sans. Weight resolves
within one family: `fontweight="bold"` picks Plex SemiBold, so the per-figure
scripts keep using plain `fontweight=` calls.
"""

import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from _style_base import assets_dir, save_figure

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = assets_dir("diagrams")

_FONTS = os.path.join(_HERE, "fonts")
for _f in (
    "IBMPlexSans-Regular.ttf",
    "IBMPlexSans-SemiBold.ttf",
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-Medium.ttf",
):
    fm.fontManager.addfont(os.path.join(_FONTS, _f))

SANS = "IBM Plex Sans"
# A list rather than a bare name, so matplotlib's per-glyph fallback borrows glyphs Plex Mono lacks
# (such as the transpose superscript) from DejaVu.
MONO = ["IBM Plex Mono", "DejaVu Sans"]

BG = "#ffffff"
INK = "#152232"
INK2 = "#38475b"
INK3 = "#657687"
LINE = "#ccd4e0"

# Roles stay consistent across figures: blue=compute/weights, amber=memory/activations,
# teal=throughput, violet=communication, rose=bottleneck/waste, slate=neutral structure (`_T` = tint).
BLUE, BLUE_T = "#4a76cc", "#dce4f4"
AMBER, AMBER_T = "#c4881f", "#f5e7cc"
TEAL, TEAL_T = "#2f8b80", "#d1e7e2"
VIOLET, VIOLET_T = "#795fc4", "#e5dff3"
ROSE, ROSE_T = "#cf5675", "#f6dae1"
SLATE, SLATE_T = "#5a6a81", "#e4e9f1"


def on_fill(hexcolor):
    """Return INK or white — whichever reads on a solid `hexcolor` fill.

    Uses perceived luminance so labels on a saturated bar or segment stay
    legible (white on teal/blue/rose, dark ink on amber).
    """
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"


TITLE, SUB, SECTION, LABEL, SMALL, TINY = 18, 10.5, 12.5, 10.5, 9.5, 9.0

plt.rcParams.update(
    {
        "font.family": [SANS, "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.facecolor": BG,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlecolor": INK,
    }
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
