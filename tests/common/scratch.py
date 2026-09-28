"""The spelling shared by the GPU-test scratch dirs and the sweep that reclaims leaked ones.

Import-free: the root ``tests/conftest.py`` reads it in every session, the GPU launcher's included,
and that session must not import torch or ``src``.
"""

# Leading tag on every per-rank scratch dir ``tests.common.distributed.setup_cache_dirs`` allocates.
# The root conftest sweeps leaked dirs by this spelling alone, so it never matches another program's
# dirs in a shared TMPDIR.
SCRATCH_DIR_TAG = "halo-test-"
