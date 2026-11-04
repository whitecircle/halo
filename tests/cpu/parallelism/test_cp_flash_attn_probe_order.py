#!/usr/bin/env python
"""CP reaches its architecture's own FlashAttention kernel: :func:`_flash_attn_func` is the whole decision. Both images
ship FA2, so an FA2-first probe silently runs a slower kernel; these pin the order with mocked arch predicates."""

import importlib.util
from unittest.mock import patch

import pytest

import halo.distributed.context_parallel.ulysses_attention as base_layer
import halo.models.attention.selection as selection


def _importable(name: str) -> bool:
    """``find_spec`` is not enough: the Hopper image ships the ``flash_attn.cute`` directory while the import
    raises."""
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any import failure means the probe will skip this backend too
        return False
    return True


HAS_FA2 = _importable("flash_attn")
HAS_FA3 = _importable("flash_attn_interface")
HAS_FA4 = _importable("flash_attn.cute")


def _resolve(*, blackwell: bool, hopper: bool, allow_fa4: bool = True):
    """Bypasses the lru_cache."""
    base_layer._flash_attn_func.cache_clear()
    try:
        with (
            patch.object(selection, "is_blackwell_gpu", return_value=blackwell),
            patch.object(selection, "is_hopper_gpu", return_value=hopper),
        ):
            return base_layer._flash_attn_func(allow_fa4)
    finally:
        base_layer._flash_attn_func.cache_clear()


def _module_of(func) -> str:
    """Seen through ``functools.wraps``."""
    return getattr(func, "__module__", "")


@pytest.mark.skipif(not (HAS_FA2 and HAS_FA4), reason="needs both flash_attn and flash_attn.cute installed")
def test_blackwell_reaches_fa4_even_though_fa2_is_also_installed():
    """FA2 imports fine here, so an FA2-first probe would win silently."""
    func = _resolve(blackwell=True, hopper=False)
    assert "cute" in _module_of(func), (
        f"CP resolved {_module_of(func)!r} on Blackwell while flash_attn.cute is installed — "
        f"an FA2-first probe order makes the FA4 branch dead code"
    )


@pytest.mark.skipif(not HAS_FA2, reason="flash_attn (FA2) not installed")
def test_off_blackwell_and_hopper_it_still_falls_back_to_fa2():
    """Anti-vacuity: the FA4 preference must be arch-gated."""
    func = _resolve(blackwell=False, hopper=False)
    assert _module_of(func).startswith("flash_attn") and "cute" not in _module_of(func), (
        f"expected the plain flash_attn (FA2) entry point off Blackwell/Hopper, got {_module_of(func)!r}"
    )


@pytest.mark.skipif(not (HAS_FA2 and HAS_FA4), reason="needs both flash_attn and flash_attn.cute installed")
def test_a_refused_fa4_gets_fa2():
    """CP rejects SDPA and this probe overrides the label, so ``allow_fa4=False`` alone keeps a run off an
    unvalidated deterministic backward."""
    func = _resolve(blackwell=True, hopper=False, allow_fa4=False)
    assert "cute" not in _module_of(func), (
        f"a refused FA4 resolved {_module_of(func)!r} on Blackwell; allow_fa4=False did not hold"
    )
    assert _module_of(func).startswith("flash_attn"), f"expected the FA2 entry point, got {_module_of(func)!r}"


@pytest.mark.skipif(not HAS_FA3, reason="flash_attn_interface (FA3) not installed in this image")
def test_hopper_prefers_fa3_over_everything_else():
    """Hopper runs its own FA3 for the non-varlen forward CP calls, the path FA2 serves worst."""
    func = _resolve(blackwell=False, hopper=True)
    assert _module_of(func).startswith("flash_attn_interface"), f"expected FA3 on Hopper, got {_module_of(func)!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
