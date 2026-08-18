"""Root pytest configuration shared by the CPU and GPU tiers.

- Registers every marker the GPU manifest uses (single source of truth:
  ``tests.gpu.manifest.ALL_MARKERS``) so ``--strict-markers`` rejects typos without the
  list being duplicated in ``pyproject.toml``.
- Ensures the project root is importable (CPU tests do ``from src. ...``).
- Sweeps stale scratch dirs left by crashed GPU runs out of the launcher's ``TMPDIR``
  (they sit on the volume the training data does, so leaks eat it).

CPU tests run inside the image (they import torch); GPU nodes are launched as torchrun
subprocesses by ``tests/gpu/conftest.py``.
"""

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

# Extra markers beyond the GPU manifest's set.
_EXTRA_MARKERS = {
    "cpu": "CPU-only test (no GPU required)",
}


def pytest_configure(config):
    from tests.gpu.manifest import ALL_MARKERS

    described = {
        "gpu": "launches a torchrun GPU script (see tests/gpu/manifest.py)",
        "core": "small/fast tier (PR gate)",
        "full": "large-model / many-GPU tier (nightly)",
    }
    for m in ALL_MARKERS:
        config.addinivalue_line("markers", f"{m}: {described.get(m, m + ' test')}")
    for name, desc in _EXTRA_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``cpu`` marker to everything under ``tests/cpu/``.

    The CPU tests are plain pytest modules with no per-file marker, so ``-m cpu``
    (and ``make test-cpu``) would otherwise select nothing. Marking by path keeps the
    selection in one place instead of decorating ~40 files. GPU tests get their markers
    from the manifest in ``tests/gpu/conftest.py``.
    """
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
    """Sweep stale GPU-test scratch dirs (a crash skips the in-test cleanup).

    ``tests.common.distributed.setup_cache_dirs`` builds them with ``mkdtemp``, so they land under
    the launcher's ``TMPDIR`` — read it rather than name a host path, which is wrong on every box
    whose large volume is not ``/mnt``.
    """
    cutoff = time.time() - 6 * 3600
    for pat in ("*_cache_r*", "*_out_*"):
        for path in glob.glob(os.path.join(tempfile.gettempdir(), pat)):
            try:
                if os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


@pytest.fixture(autouse=True)
def _reset_shared_filesystem_consensus():
    """Drop the process-wide shared-filesystem memo after every test.

    ``hub_metadata_main_first`` agrees the ``DIST_*_SHARED_FILESYSTEM`` flags the first time any test
    probes a real checkpoint config and memoizes them for the process; a later test that varies the
    env (order- and xdist-dependent) would otherwise assert against that stale scope. Reset only when
    the runtime module was actually imported — no torch import for tests that never touched it.
    """
    yield
    runtime = sys.modules.get("src.distributed.runtime")
    if runtime is not None:
        runtime.reset_shared_filesystem_consensus()
