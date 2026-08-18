#!/usr/bin/env python
"""Context Parallelism must reach its architecture's own FlashAttention kernel.

CP does not call attention through the model's module, so ``_attn_implementation`` decides nothing
here: :func:`get_flash_attn_func` is the whole decision. Both training images ship ``flash_attn``
(FA2), so a probe order that tries FA2 before the architecture-specific kernel resolves to FA2 on
every device and makes the branch below it unreachable — a CP run would then execute a different,
slower kernel than every other code path on the same GPU, with nothing logged.

These tests pin the ORDER rather than the outcome: they mock the arch predicates and assert which
module the returned callable came from, so they fail if the probes are reordered. No GPU is needed —
the probe only imports, and the predicates are mocked.

Run: python tests/cpu/parallelism/test_cp_flash_attn_probe_order.py  (or pytest)
"""

import importlib.util
import sys
from unittest.mock import patch

import pytest

import src.distributed.context_parallel.base_layer as base_layer


def _importable(name: str) -> bool:
    """Whether the module actually imports — which is what the probe under test requires.

    ``find_spec`` is not enough: the Hopper image ships the ``flash_attn.cute`` *directory* while the
    import raises ``ModuleNotFoundError: No module named 'cutlass.utils.ampere_helpers'``. The probe
    catches that (a subclass of ImportError) and falls through to FA2, so a spec-based check would
    make these tests demand an FA4 result on an image where FA4 cannot run.
    """
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any import failure means the probe will skip this backend too
        return False
    return True


HAS_FA2 = _importable("flash_attn")
HAS_FA3 = _importable("flash_attn_interface")
HAS_FA4 = _importable("flash_attn.cute")


def _resolve(*, blackwell: bool, hopper: bool, allow_fa4: bool = True):
    """Resolve the CP attention callable under a mocked architecture, bypassing the lru_cache."""
    base_layer.get_flash_attn_func.cache_clear()
    try:
        with (
            patch.object(base_layer, "is_blackwell_gpu", return_value=blackwell),
            patch.object(base_layer, "is_hopper_gpu", return_value=hopper),
        ):
            return base_layer.get_flash_attn_func(allow_fa4)
    finally:
        base_layer.get_flash_attn_func.cache_clear()


def _module_of(func) -> str:
    """Defining module of the callable, seen through ``functools.wraps``."""
    return getattr(func, "__module__", "")


@pytest.mark.skipif(not (HAS_FA2 and HAS_FA4), reason="needs both flash_attn and flash_attn.cute installed")
def test_blackwell_reaches_fa4_even_though_fa2_is_also_installed():
    """The trap, stated as its own precondition: FA2 imports fine here, so an FA2-first probe would
    win silently and the FA4 branch below it would never run."""
    func = _resolve(blackwell=True, hopper=False)
    assert "cute" in _module_of(func), (
        f"CP resolved {_module_of(func)!r} on Blackwell while flash_attn.cute is installed — "
        f"an FA2-first probe order makes the FA4 branch dead code"
    )


@pytest.mark.skipif(not HAS_FA2, reason="flash_attn (FA2) not installed")
def test_off_blackwell_and_hopper_it_still_falls_back_to_fa2():
    """Anti-vacuity control: the FA4 preference must be arch-gated, not unconditional. Without this
    the test above would also pass if the probe simply always returned FA4."""
    func = _resolve(blackwell=False, hopper=False)
    assert _module_of(func).startswith("flash_attn") and "cute" not in _module_of(func), (
        f"expected the plain flash_attn (FA2) entry point off Blackwell/Hopper, got {_module_of(func)!r}"
    )


@pytest.mark.skipif(not (HAS_FA2 and HAS_FA4), reason="needs both flash_attn and flash_attn.cute installed")
def test_a_nan_prone_family_vetoes_fa4_and_gets_fa2():
    """Qwen3.5 / Qwen3-Next / GLM-4-Lite must not reach FA4 through CP.

    Their FA4 backward emits NaN gradients, which `resolve_attn_implementation` avoids by demoting
    the request to SDPA. CP cannot use that escape: it rejects SDPA outright, and this probe
    overrides the configured label anyway — so a user who sets `attn_implementation:
    flash_attention_2` to make CP accept the model (a pattern several shipped configs use) would
    otherwise get FA4 regardless, and NaN out. The veto is the only thing standing there.
    """
    func = _resolve(blackwell=True, hopper=False, allow_fa4=False)
    assert "cute" not in _module_of(func), (
        f"a NaN-prone family resolved {_module_of(func)!r} on Blackwell; the FA4 veto did not hold"
    )
    assert _module_of(func).startswith("flash_attn"), f"expected the FA2 entry point, got {_module_of(func)!r}"


@pytest.mark.skipif(not HAS_FA3, reason="flash_attn_interface (FA3) not installed in this image")
def test_hopper_prefers_fa3_over_everything_else():
    """Hopper must not take the FA2 path: the image stubs out the split-K kernels FA2's non-varlen
    forward — the one CP calls — selects by occupancy heuristic."""
    func = _resolve(blackwell=False, hopper=True)
    assert _module_of(func).startswith("flash_attn_interface"), f"expected FA3 on Hopper, got {_module_of(func)!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
