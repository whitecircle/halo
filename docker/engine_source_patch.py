"""Rewrite an engine's installed source at image build, failing the build on any upstream change.

Each server image copies this module beside its table of edits and runs the table; the table stays
in the image so ``--verify`` re-checks a running container. Standard library only: it runs in the
engine's interpreter.
"""

import argparse
import dataclasses
import importlib.util
import pathlib
import py_compile
from collections.abc import Callable, Sequence


@dataclasses.dataclass(frozen=True)
class Edit:
    """Replace ``before``, present exactly once, with ``after`` in ``path`` under the engine package."""

    path: str
    before: str
    after: str
    what: str

    def apply(self, text: str) -> str:
        found = text.count(self.before)
        if found != 1:
            raise SystemExit(f"{self.what}: expected the pre-image exactly once, found {found} — upstream changed")
        return text.replace(self.before, self.after)

    def verify(self, text: str) -> None:
        # An insertion keeps its anchor inside the post-image; only a pre-image outside it is unpatched.
        if self.after not in text or text.count(self.before) > self.after.count(self.before):
            raise SystemExit(f"{self.what}: the post-image is not in place in {self.path}")


def _edits_by_file(root: pathlib.Path, edits: Sequence[Edit]) -> dict[pathlib.Path, list[Edit]]:
    by_file: dict[pathlib.Path, list[Edit]] = {}
    for edit in edits:
        by_file.setdefault(root / edit.path, []).append(edit)
    for path in by_file:
        if not path.is_file():
            raise SystemExit(f"{path} is missing — upstream moved the file")
    return by_file


def verify_edits(root: pathlib.Path, edits: Sequence[Edit]) -> None:
    for path, file_edits in _edits_by_file(root, edits).items():
        text = path.read_bytes().decode("utf-8")
        for edit in file_edits:
            edit.verify(text)


def apply_edits(root: pathlib.Path, edits: Sequence[Edit]) -> None:
    """Rewrite each file, drop its stale bytecode and byte-compile it, then verify every edit.

    Every pre-image is matched before any file is written, so a miss (a re-run included) leaves the tree untouched.
    """
    patched: dict[pathlib.Path, str] = {}
    for path, file_edits in _edits_by_file(root, edits).items():
        text = path.read_bytes().decode("utf-8")
        for edit in file_edits:
            text = edit.apply(text)
        patched[path] = text
    for path, text in patched.items():
        path.write_bytes(text.encode("utf-8"))
        for stale in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
            stale.unlink()
        py_compile.compile(str(path), doraise=True)
    verify_edits(root, edits)


def main(package: str, edits: Sequence[Edit], check: Callable[[pathlib.Path], None]) -> None:
    """Patch the installed ``package``, or with ``--verify`` only re-check it; ``check`` runs after either."""
    parser = argparse.ArgumentParser(description=f"Rewrite the installed {package} source in place.")
    parser.add_argument("--verify", action="store_true", help="assert the post-images without rewriting")
    verify_only = parser.parse_args().verify
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{package} is not importable in this interpreter")
    root = pathlib.Path(next(iter(spec.submodule_search_locations)))
    if verify_only:
        verify_edits(root, edits)
    else:
        apply_edits(root, edits)
    check(root)
    print(f"{package}: {len(edits)} source edits {'verified' if verify_only else 'applied'}")  # noqa: T201
