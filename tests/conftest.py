"""Root pytest configuration shared by the CPU and GPU tiers: markers from the GPU manifest, the import path, and a
stale-scratch sweep."""

import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Below the insert it relies on.
from tests.common.scratch import SCRATCH_DIR_TAG
from tests.gpu.manifest import ALL_MARKERS

_EXTRA_MARKERS = {
    "cpu": "CPU-only test (no GPU required)",
}


def pytest_configure(config):
    described = {
        "gpu": "launches a torchrun GPU script (see tests/gpu/manifest.py)",
        "core": "small/fast tier (PR gate)",
        "full": "large-model / many-GPU tier (run by hand)",
    }
    for m in ALL_MARKERS:
        config.addinivalue_line("markers", f"{m}: {described.get(m, m + ' test')}")
    for name, desc in _EXTRA_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``cpu`` marker to everything under ``tests/cpu/``, so ``-m cpu`` selects them with no per-file
    marker."""
    cpu_root = _PROJECT_ROOT / "tests" / "cpu"
    for item in items:
        path = str(item.fspath).replace(os.sep, "/")
        try:
            in_cpu = cpu_root in Path(str(item.fspath)).resolve().parents
        except OSError:
            in_cpu = "/tests/cpu/" in path
        if in_cpu:
            item.add_marker(pytest.mark.cpu)


def pytest_sessionstart(session):
    """Sweep stale test scratch dirs under the launcher's ``TMPDIR``, which a crash leaves behind; only
    ``SCRATCH_DIR_TAG`` names, since it may be shared."""
    cutoff = time.time() - 6 * 3600
    for path in glob.glob(os.path.join(tempfile.gettempdir(), f"{SCRATCH_DIR_TAG}*")):
        try:
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


@pytest.fixture(autouse=True, scope="session")
def _run_from_the_project_root():
    """Run from the project root, as ``halo launch`` does, so a recipe's relative paths resolve wherever pytest
    started."""
    started_in = Path.cwd()
    os.chdir(_PROJECT_ROOT)
    yield
    os.chdir(started_in)


@pytest.fixture(autouse=True)
def _reset_shared_filesystem_consensus():
    """Drop the process-wide shared-filesystem memo after every test, since a later test varying the env would assert
    against a stale scope. Reset only when the filesystem module was imported, so tests that never touched it import no
    torch."""
    yield
    filesystem = sys.modules.get("halo.runtime.filesystem")
    if filesystem is not None:
        filesystem._shared_filesystem_consensus = None
