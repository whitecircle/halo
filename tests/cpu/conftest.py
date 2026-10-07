"""CPU-tier setup: each test process copies remote code into a dynamic-module cache of its own.

transformers copies every ``trust_remote_code`` module into ``HF_MODULES_CACHE``, by default
``$HF_HOME/modules``: shared with every xdist worker and every session mounting the same HF home,
unwritable when that home is mounted read-only, and left holding a copy of each temp checkpoint a
test loads. transformers reads the variable when it is imported, so it is set here, ahead of every
test module. A standalone ``python tests/cpu/<file>.py`` of a file that imports transformers keeps the
default cache.

The directory carries ``SCRATCH_DIR_TAG``, so the root conftest's sweep reclaims it if a crash skips
the removal below.
"""

import os
import shutil
import sys
import tempfile

from tests.common.scratch import SCRATCH_DIR_TAG

# Per process: an xdist worker inherits the controller's value and replaces it with its own. A test's
# spawned children share their worker's cache, and their copies land whole (remote_code_hooks).
_MODULES_CACHE = None if "transformers" in sys.modules else tempfile.mkdtemp(prefix=f"{SCRATCH_DIR_TAG}hf_modules_")
if _MODULES_CACHE is not None:
    os.environ["HF_MODULES_CACHE"] = _MODULES_CACHE


def pytest_unconfigure(config):
    if _MODULES_CACHE is not None:
        shutil.rmtree(_MODULES_CACHE, ignore_errors=True)
