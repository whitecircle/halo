"""Output destination and save helper shared by the style modules here.

Palettes and rcParams stay per-family (`_flow_style`, `_theory_style`, the benchmark charts); what
they share is the anchored `agent-docs/assets/<kind>` destination and the create-directory-then-save
step. Importing this module also pins matplotlib to the Agg backend, which every generator needs
before it touches `pyplot`.
"""

import os

import matplotlib

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))


def assets_dir(kind: str) -> str:
    """``agent-docs/assets/<kind>``, anchored on this file so the generators run from any working directory."""
    return os.path.join(_HERE, "..", "..", "agent-docs", "assets", kind)


def save_figure(fig, path: str, **savefig_kwargs) -> None:
    """Write ``fig`` to ``path``, creating its directory on a fresh clone."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
