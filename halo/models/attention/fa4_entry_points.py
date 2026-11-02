"""Wrapping flash-attn-4's two ``flash_attn.cute`` entry points together with transformers' cached references."""

from __future__ import annotations

import logging
from collections.abc import Callable

from transformers import modeling_flash_attention_utils as flash_utils

from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Each entry point, the module attribute transformers caches it under, and whether it is the varlen one.
_ENTRY_POINTS = (("flash_attn_varlen_func", "_flash_varlen_fn", True), ("flash_attn_func", "_flash_fn", False))


def install_fa4_entry_point_wrapper(marker: str, wrap: Callable[[Callable, bool], Callable], description: str) -> bool:
    """Replace both entry points with ``wrap(entry_point, varlen)``, and the references transformers already
    cached to them, once per ``marker`` (idempotent). False when ``flash_attn.cute`` is not importable.

    A wrapper must keep its entry point's signature (``functools.wraps``): transformers decides off it which
    optional kwargs (window, softcap, sink) it forwards at all. The ``__dict__`` that copies carries an inner
    wrapper's marker outward, so installs stack in either order.
    """
    ensure_fa4_kernel_cache_env()
    try:
        import flash_attn.cute as cute  # noqa: PLC0415 — optional, Blackwell-only dependency
    except ImportError:
        return False
    if getattr(cute.flash_attn_varlen_func, marker, False):
        return True
    for name, cached, varlen in _ENTRY_POINTS:
        original = getattr(cute, name)
        wrapped = wrap(original, varlen)
        setattr(wrapped, marker, True)
        setattr(cute, name, wrapped)
        if getattr(flash_utils, cached) is original:
            setattr(flash_utils, cached, wrapped)
    if is_global_main_process():
        logger.info(f"Installed {description} on FA4's entry points")
    return True
