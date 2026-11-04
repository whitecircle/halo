"""Sample repo trees for static-reading tests; roots spelled ``$pkg``, ``$tests`` and so on keep the reference sweep
from reading them as ours."""

from pathlib import Path
from string import Template
from textwrap import dedent

from scripts.docs.references import PACKAGE


def spelled(text: str) -> str:
    return Template(text).substitute(pkg=PACKAGE, tests="tests", scripts="scripts", github=".github")


def write_sample_tree(root: Path, files: dict[str, str]) -> None:
    """Write each ``relative path: source`` under ``root``, both spelled, the source dedented."""
    for relative, text in files.items():
        path = root / spelled(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spelled(dedent(text).lstrip("\n")))
