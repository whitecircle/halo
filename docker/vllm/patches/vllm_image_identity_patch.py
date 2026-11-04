"""Name this image's patches on ``/server_info``, so the trainer can tell it from upstream vLLM.

Upstream 0.32.0 passes the client's version check and takes the same routes, but with no layerwise-reload patch every
sync reverts the experts behind a ``200 OK``. ``/server_info`` lists the variables ``vllm.envs`` declares, so the row
goes in that table. It lands only where ``sitecustomize`` ran, so a ``PYTHONPATH`` override loses it with the patches.
"""

from __future__ import annotations

import sys

IDENTITY_KEY = "VLLM_HALO_PATCHES"


def apply() -> None:
    """Declare the row on the table every ``vllm.envs`` read consults; idempotent."""
    import vllm.envs as envs  # noqa: PLC0415 - vLLM-only lazy import

    patches = ",".join(sorted(sys.modules["sitecustomize"]._PATCHES.values()))
    envs.environment_variables[IDENTITY_KEY] = lambda: patches
