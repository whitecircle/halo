"""JIT kernel caches (FA4's CuTe DSL, Triton) anchored on a persistent volume, and the CuTe DSL warning filter."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings

logger = logging.getLogger(__name__)


def silence_cute_dsl_deprecations() -> None:
    """Mute the CuTe DSL's ``DeprecationWarning``, which each FA4 compile re-emits under a fresh warning
    registry; scoped to that package and category, so every other warning still surfaces."""
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cutlass(\.|$)")


def _accepts_writes(path: str) -> bool:
    """Whether ``path`` is a writable directory, or can be created under its nearest existing ancestor."""
    path = os.path.abspath(path)
    while not os.path.lexists(path):
        path = os.path.dirname(path)
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def anchor_jit_cache_dir(var: str, subdir: str) -> None:
    """Point JIT cache variable ``var`` at ``HF_HOME/subdir`` (else the temp dir) unless already set.

    The library defaults are ephemeral in a ``--rm`` container. An unwritable location is skipped with a
    warning, and with none writable the variable stays unset for the library's own default.
    """
    if var in os.environ:
        return
    hf_home = os.environ.get("HF_HOME")
    anchor = os.path.join(hf_home, subdir) if hf_home else None
    if anchor is not None and _accepts_writes(anchor):
        os.environ[var] = anchor
        return
    try:
        fallback = os.path.join(tempfile.gettempdir(), subdir)
    except FileNotFoundError:  # no usable temp dir at all
        fallback = None
    if fallback is not None and _accepts_writes(fallback):
        if anchor is not None:
            logger.warning("%s is not writable, so %s falls back to %s.", anchor, var, fallback)
        os.environ[var] = fallback
        return
    tried = [anchor] if anchor is not None else []
    tried.append(fallback if fallback is not None else "no usable temp dir")
    logger.warning(
        "No writable location for %s (tried %s), so it is left to the library default.", var, ", ".join(tried)
    )


def ensure_fa4_kernel_cache_env() -> None:
    """Enable FA4's persistent kernel cache and TVM-FFI and mute the DSL's deprecations. Must run before
    the first ``import flash_attn.cute``, which reads the environment. Idempotent."""
    silence_cute_dsl_deprecations()
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
    anchor_jit_cache_dir("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "flash_attn_cute_dsl_cache")
