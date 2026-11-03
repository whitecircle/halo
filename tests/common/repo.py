"""The repository root, by-path import of a repo file, and out-of-process probes run at the root."""

import importlib.util
import pathlib
import subprocess
import sys
from types import ModuleType

# Self-locating: the training images bake a repo copy at /workspace, so a relative path can resolve against the wrong
# tree.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script_module(relative_path: str, name: str | None = None, *, register: bool = False) -> ModuleType:
    """Import a repo file by path as its own module; ``register`` binds it in ``sys.modules`` so ``patch("<name>.x")``
    and pickling resolve it."""
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"no module to import at {path}")
    module_name = name or path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no loadable import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        # Before exec: a module resolving string annotations or pickling its own classes reads itself out of
        # sys.modules mid-body.
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def probe_findings(script: str, marker: str) -> list[str]:
    """Run ``script`` in a fresh interpreter at the repo root; return the findings its ``print(marker +
    "|".join(findings))`` line reports. For a subject rebinding process-global state, which would leak into later
    tests. A probe that dies or prints no marker raises."""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    assert proc.returncode == 0, f"probe subprocess died:\n{proc.stdout}\n{proc.stderr}"

    line = next((line for line in proc.stdout.splitlines() if line.startswith(marker)), None)
    assert line is not None, f"probe printed no {marker!r} line:\n{proc.stdout}\n{proc.stderr}"
    return [finding for finding in line.removeprefix(marker).split("|") if finding]
