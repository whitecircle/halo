"""Name this image's engine build and patches on ``/server_info``, so the trainer can tell it from upstream vLLM and
from the toolkit's other vLLM tag.

Upstream vLLM takes the same routes, but with no layerwise-reload patch every sync reverts the experts behind a
``200 OK``. ``/server_info`` lists the variables ``vllm.envs`` declares, so the row goes in that table, as
``<vllm.__version__>|<patch>,<patch>,...``: the client takes a server as the toolkit's image only where the build is
the one it pins for the tag. The row lands only where ``sitecustomize`` ran, so a ``PYTHONPATH`` override loses it
with the patches.
"""

from __future__ import annotations

import sys

IDENTITY_KEY = "VLLM_HALO_PATCHES"


def apply() -> None:
    """Declare the row on the table every ``vllm.envs`` read consults; idempotent."""
    import vllm  # noqa: PLC0415 - vLLM-only lazy import
    import vllm.envs as envs  # noqa: PLC0415 - vLLM-only lazy import

    patches = ",".join(sorted(sys.modules["sitecustomize"]._PATCHES.values()))
    identity = f"{vllm.__version__}|{patches}"
    envs.environment_variables[IDENTITY_KEY] = lambda: identity
