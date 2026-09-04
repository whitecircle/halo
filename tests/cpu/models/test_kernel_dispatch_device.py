"""Device-aware kernel dispatch: CPU calls must take the torch path, CUDA-shaped dispatch intact.

transformers 5.16's ``use_kernel_func_from_hub_with_fallback`` captures the CUDA-only original
package (causal_conv1d/fla) at decoration time and calls it for every input. These tests fail if
the toolkit shim stops routing CPU tensors to the decorated torch function — the failure is a
crash (``Expected x.is_cuda()``) in every CPU forward of a conv/linear-attention family.
"""

import os
import subprocess
import sys

import pytest
import torch
from transformers.integrations import hub_kernels

from src.models.patches import kernel_dispatch
from src.models.patches.kernel_dispatch import ensure_device_aware_kernel_dispatch
from tests.common.utils import REPO_ROOT


def _decorate_probe():
    calls = []

    @hub_kernels.use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
    def probe(x, weight=None):
        calls.append("torch_path")
        return x + 1

    return probe, calls


def test_cpu_tensor_takes_torch_path_even_with_cuda_only_package_installed():
    pytest.importorskip("causal_conv1d")
    ensure_device_aware_kernel_dispatch()
    probe, calls = _decorate_probe()
    out = probe(torch.zeros(2, 3), weight=torch.ones(3))
    assert calls == ["torch_path"], "CPU call dispatched to the CUDA-only package kernel"
    assert torch.equal(out, torch.ones(2, 3))


def test_patch_is_idempotent():
    ensure_device_aware_kernel_dispatch()
    factory = hub_kernels.use_kernel_func_from_hub_with_fallback
    ensure_device_aware_kernel_dispatch()
    assert hub_kernels.use_kernel_func_from_hub_with_fallback is factory


def test_tensorless_call_reaches_the_captured_implementation():
    """No tensor argument → no device verdict; the shim must defer to the stock dispatch."""
    ensure_device_aware_kernel_dispatch()

    @hub_kernels.use_kernel_func_from_hub_with_fallback(
        "definitely_not_a_real_symbol", "definitely_not_a_real_package"
    )
    def probe(n):
        return n * 2

    assert probe(21) == 42


def test_cpu_delta_rule_restarts_recurrent_state_per_document():
    """The torch fallback ignores ``cu_seqlens`` and runs one recurrence over the packed row; the
    shim must slice per document. A cumsum stands in for the recurrence: without segmentation
    document B starts from document A's running state."""
    pytest.importorskip("fla")
    ensure_device_aware_kernel_dispatch()

    @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
    def probe(query, key, value, g=None, beta=None, **kwargs):
        return value.cumsum(dim=1), value.new_zeros(1)

    v = torch.ones(1, 6, 2)
    out, _ = probe(v, v, v, g=v, beta=v, cu_seqlens=torch.tensor([0, 3, 6]))
    assert out[0, :, 0].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], "state leaked across the boundary"
    leaked, _ = probe(v, v, v, g=v, beta=v)
    assert leaked[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _closure_functions(fn, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(fn) in seen or depth > 6 or not callable(fn):
        return
    seen.add(id(fn))
    yield fn
    for cell in getattr(fn, "__closure__", None) or []:
        value = getattr(cell, "cell_contents", None)
        if callable(value):
            yield from _closure_functions(value, depth + 1, seen)


def test_cuda_capture_resolves_a_submodule_only_package_kernel():
    """``fla`` exports nothing at root, so upstream's getattr walk finds no kernel on a bare package
    import and silently captures the torch body — every CUDA forward then runs the naive scan at
    ~10x the step time. The shim pre-imports the mapped chain; the decoration must capture fla's
    real kernel."""
    pytest.importorskip("fla")
    ensure_device_aware_kernel_dispatch()

    @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
    def probe(query, key, value, **kwargs):
        return value, None

    captured = {
        f"{fn.__module__}.{fn.__qualname__}"
        for fn in _closure_functions(probe)
        if getattr(fn, "__module__", "").startswith("fla.")
    }
    assert any("chunk_gated_delta_rule" in name for name in captured), (
        f"decoration captured no fla kernel (closure held {captured or 'only torch functions'}) — "
        "CUDA forwards would silently run the torch fallback"
    )


def test_torch_capture_with_the_package_installed_warns(caplog, monkeypatch):
    """A capture that lands on torch while the package is importable is a silent 10x — it must warn."""
    pytest.importorskip("fla")
    ensure_device_aware_kernel_dispatch()
    monkeypatch.setitem(hub_kernels._KERNELS_INTERNAL_PATH_MAPPINGS, "chunk_gated_delta_rule", "ops.no_such_submodule")
    monkeypatch.setattr(kernel_dispatch, "_WARNED_TORCH_CAPTURES", set())

    with caplog.at_level("WARNING"):

        @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
        def probe(query, key, value, **kwargs):
            return value, None

    assert any("captured the torch fallback" in r.message for r in caplog.records), (
        "no warning for a torch capture with the package installed — the fallback is silent again"
    )


def test_importing_src_installs_the_shim_before_anything_binds_the_factory():
    """The install has to happen on ``src`` import, not on first use.

    Modeling modules capture the fallback factory AT IMPORT, so a shim installed later decorates
    nothing they already bound. Every test above calls ``ensure_device_aware_kernel_dispatch()``
    itself and then decorates its own probe, which can only ever observe install-then-bind — this
    pins the other order. A fresh interpreter is required: this process installed the shim long ago.
    """
    probe = (
        "import src;"
        "from transformers.integrations import hub_kernels;"
        "print('INSTALLED', getattr(hub_kernels, '_halo_device_aware_kernel_dispatch', False))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": str(REPO_ROOT), "HF_HUB_OFFLINE": "1"},
        timeout=600,
    )
    assert result.returncode == 0, f"importing src failed:\n{result.stderr[-2000:]}"
    assert "INSTALLED True" in result.stdout, (
        "importing `src` no longer installs the device-aware kernel dispatch — every modeling module "
        f"imported afterwards binds the CUDA-only kernel. stdout={result.stdout[-500:]!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
