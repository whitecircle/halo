"""CPU-tier setup: each test process copies remote code into a dynamic-module cache of its own, since
``$HF_HOME/modules`` is shared across xdist workers and unwritable on a read-only mount. transformers reads it at
import, so it is set here; ``SCRATCH_DIR_TAG`` lets the root conftest reclaim it."""

import os
import shutil
import sys
import tempfile

from tests.common.scratch import SCRATCH_DIR_TAG

# Per process: an xdist worker inherits the controller's value and replaces it with its own. A test's
# spawned children share their worker's cache, and their copies land whole (model_loading.remote_code).
_MODULES_CACHE = None if "transformers" in sys.modules else tempfile.mkdtemp(prefix=f"{SCRATCH_DIR_TAG}hf_modules_")
if _MODULES_CACHE is not None:
    os.environ["HF_MODULES_CACHE"] = _MODULES_CACHE


def pytest_unconfigure(config):
    if _MODULES_CACHE is not None:
        shutil.rmtree(_MODULES_CACHE, ignore_errors=True)
