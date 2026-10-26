"""Sample repository trees for the tests that read source statically (:mod:`scripts.docs.references`).

A sample spells its first-party roots as ``$pkg``, ``$tests``, ``$scripts`` and ``$github``, so the reference
sweep does not read a test's sample as names and paths of the repository's own.
"""

from pathlib import Path
from string import Template
from textwrap import dedent

from scripts.docs.references import PACKAGE


def spelled(text: str) -> str:
    """``text`` with its root placeholders spelled out."""
    return Template(text).substitute(pkg=PACKAGE, tests="tests", scripts="scripts", github=".github")


def write_sample_tree(root: Path, files: dict[str, str]) -> None:
    """Write each ``relative path: source`` of ``files`` under ``root``, both spelled, the source dedented."""
    for relative, text in files.items():
        path = root / spelled(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spelled(dedent(text).lstrip("\n")))
