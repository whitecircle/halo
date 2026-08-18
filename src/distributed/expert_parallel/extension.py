"""Deferred import seam for the optional native ``deep_ep`` extension.

``import deep_ep`` loads a CUDA extension and runs ``cuInit`` (which latches
``CUDA_DEVICE_MAX_CONNECTIONS``). That must not happen merely because a dense/DDP/CPU job imported an
EP module to read a class-declared contract — every EP layer class is reachable from
:mod:`src.distributed.expert_parallel.base_layer`, and the expert-weight/checkpoint helpers union over
them. Routing every use through :func:`deep_ep` defers the extension to EP buffer construction, the
first point that genuinely needs it.
"""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=1)
def deep_ep() -> ModuleType:
    """The native ``deep_ep`` module, imported on first use. Raises ``ImportError`` when absent."""
    import deep_ep  # noqa: PLC0415 — optional native extension, deferred on purpose (see module docstring)

    return deep_ep


def deep_ep_available() -> bool:
    """Whether the native extension imports on this host (pays the same deferred import once)."""
    try:
        deep_ep()
    except (ImportError, OSError):
        return False
    return True
