"""Axes, tints and headings shared by the benchmark charts under agent-docs/assets/benchmarks.

A thin layer over `_theory_style` (palette, fonts, `save`). Teal carries throughput, amber memory, slate
the baseline a chart compares against. Import with `from _chart_style import *`, which re-exports
`_theory_style`.
"""

from _theory_style import *
from matplotlib.ticker import FuncFormatter

BENCHMARKS = "benchmarks"

# Darkest to palest: one hue per measured quantity, one shade per Halo layout.
TEAL_SHADES = ("#1f5f57", "#2f8b80", "#74b5ab", "#b5d9d3")
AMBER_SHADES = ("#8a5f14", "#c4881f", "#deb566", "#f0dcae")


def tidy(ax, value_axis):
    """Drop the top and right frame and draw a faint grid along `value_axis`, behind the marks."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis=value_axis, color=LINE, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=SMALL)


def k_ticks(axis):
    """Tick labels in thousands: 0, 5k, 10k."""
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))


def heading(fig, title, sub):
    """Figure title and its one-line subtitle, flush left above every axes."""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, sub, ha="left", va="top", fontsize=SUB, color=INK2)
