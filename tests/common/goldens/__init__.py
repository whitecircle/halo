"""Golden generators: today's behavior rendered to the committed files the CPU tier compares against,
and the regenerate-or-check command line every golden shares.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path


def regen_main(golden: Path, render: Callable[[], str]) -> None:
    """``--regen`` rewrites ``golden`` with ``render()``; without it, exit non-zero when the committed
    file differs from what today's code renders."""
    parser = argparse.ArgumentParser(description=f"Regenerate or check {golden.name} against today's code.")
    parser.add_argument("--regen", action="store_true", help="rewrite the golden instead of checking it")
    regen = parser.parse_args().regen
    # Host advisories (fabric, topology) fire once per config built and say nothing about the golden.
    logging.disable(logging.WARNING)
    try:
        text = render()
    finally:
        logging.disable(logging.NOTSET)
    if regen:
        golden.write_text(text, encoding="utf-8")
    elif golden.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{golden} is stale against today's code; rerun with --regen for a deliberate change.")
