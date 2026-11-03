"""JIT kernel caches (FA4's CuTe DSL, Triton) anchored on a persistent volume, the CuTe DSL warning filter, and the
Liger implementations pinned because of the CuTe DSL calling convention the toolkit turns on."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings

logger = logging.getLogger(__name__)

# Liger's CuTe DSL RoPE, auto-selected at production shapes, caches one TMA handle per shape and swaps its data pointer
# per call; under CUTE_DSL_ENABLE_TVM_FFI the launch passes the tensor the handle was built from, so from the second
# same-shape launch at the latest, backward included, it rotates a stale buffer.
_PINNED_LIGER_IMPLS = {"rope": "nvidia-triton"}


def silence_cute_dsl_deprecations() -> None:
    """Each FA4 compile re-emits the ``DeprecationWarning`` under a fresh warning registry."""
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cutlass(\.|$)")


def _accepts_writes(path: str) -> bool:
    path = os.path.abspath(path)
    while not os.path.lexists(path):
        path = os.path.dirname(path)
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def anchor_jit_cache_dir(var: str, subdir: str) -> None:
    """Point ``var`` at ``HF_HOME/subdir`` (else the temp dir) unless set; defaults die with a ``--rm`` container."""
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


def pin_liger_impls() -> None:
    """Through ``LIGER_KERNEL_IMPL_<OP>``, read ahead of every global setting. A value naming that implementation is
    kept; any other, in that variable or its legacy ``LIGER_KERNEL_BACKEND_<OP>`` spelling, raises."""
    for op, impl in _PINNED_LIGER_IMPLS.items():
        accepted = (impl, impl.split("-", 1)[1])
        pinned_var = f"LIGER_KERNEL_IMPL_{op.upper()}"
        for var in (pinned_var, f"LIGER_KERNEL_BACKEND_{op.upper()}"):
            chosen = os.environ.get(var)
            if chosen and chosen not in accepted:
                raise ValueError(
                    f"{var}={chosen} selects a Liger {op} implementation the toolkit does not run: it pins {op} "
                    f"to {impl}, since the CuTe DSL RoPE returns wrong results under CUTE_DSL_ENABLE_TVM_FFI. "
                    f"Unset {var}."
                )
        if not os.environ.get(pinned_var):
            os.environ[pinned_var] = impl


def ensure_fa4_kernel_cache_env() -> None:
    """Must run before the first ``import flash_attn.cute``, which reads the environment. Idempotent."""
    silence_cute_dsl_deprecations()
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
    pin_liger_impls()
    anchor_jit_cache_dir("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "flash_attn_cute_dsl_cache")
