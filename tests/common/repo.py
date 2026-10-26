"""The repository root, by-path import of a repo file, and out-of-process probes run at the root."""

import importlib.util
import pathlib
import subprocess
import sys
from types import ModuleType

# Self-locating rather than cwd-derived: the training images bake a repo copy at /workspace, so a
# relative path can resolve against the wrong tree.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script_module(relative_path: str, name: str | None = None, *, register: bool = False) -> ModuleType:
    """Import a repo file (a ``scripts/`` tool, a ``docker/`` helper) by path, as a module object of its own.

    ``relative_path`` is relative to :data:`REPO_ROOT`; ``name`` defaults to the file stem.
    ``register`` also binds the module in ``sys.modules`` so ``unittest.mock.patch("<name>.x")``
    and pickling resolve it; leave it off for a plain by-path import.
    """
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"no module to import at {path}")
    module_name = name or path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no loadable import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        # Before exec: a module that resolves string annotations or pickles its own classes reads
        # itself back out of sys.modules while its body is still running.
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def probe_findings(script: str, marker: str) -> list[str]:
    """Run ``script`` in a fresh interpreter at the repo root; return the findings it reported.

    A probe whose subject rebinds process-global state (a Liger applier swapping upstream module
    classes, a funnel wrapper installed on ``transformers.dynamic_module_utils``) would leak into
    every later test, so it runs out-of-process and reports through one line,
    ``print(marker + "|".join(findings))``, where an empty list means nothing was wrong. A probe that
    dies, or prints no marker line, raises here rather than reading as a clean run.
    """
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    assert proc.returncode == 0, f"probe subprocess died:\n{proc.stdout}\n{proc.stderr}"

    line = next((line for line in proc.stdout.splitlines() if line.startswith(marker)), None)
    assert line is not None, f"probe printed no {marker!r} line:\n{proc.stdout}\n{proc.stderr}"
    return [finding for finding in line.removeprefix(marker).split("|") if finding]
