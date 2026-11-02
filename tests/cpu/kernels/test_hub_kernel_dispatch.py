"""Device-aware kernel dispatch: CPU calls must take the torch path, CUDA-shaped dispatch intact.

transformers 5.20's ``use_kernel_func_from_hub_with_fallback`` captures the CUDA-only original
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

from halo.kernels import hub_kernel_dispatch
from halo.kernels.hub_kernel_dispatch import ensure_device_aware_kernel_dispatch
from tests.common.repo import REPO_ROOT, probe_findings


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
    ensure_device_aware_kernel_dispatch()

    @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
    def probe(query, key, value, g=None, beta=None, **kwargs):
        return value.cumsum(dim=1), value.new_zeros(1)

    v = torch.ones(1, 6, 2)
    out, _ = probe(v, v, v, g=v, beta=v, cu_seqlens=torch.tensor([0, 3, 6]))
    assert out[0, :, 0].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], "state leaked across the boundary"
    leaked, _ = probe(v, v, v, g=v, beta=v)
    assert leaked[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_cpu_delta_rule_segments_by_the_host_bounds_and_consumes_them():
    """The tree recurrence hands fla's ``cu_seqlens_cpu`` along with the device bounds. The fallback reads
    the segments off that host copy (the device copy is on the meta device here, which a read would fault)
    and passes it no further: the torch body takes no such argument."""
    ensure_device_aware_kernel_dispatch()

    @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
    def probe(query, key, value, g=None, beta=None, initial_state=None):
        return value.cumsum(dim=1), value.new_zeros(1)

    v = torch.ones(1, 6, 2)
    device_bounds = torch.tensor([0, 2, 6], device="meta")
    out, _ = probe(v, v, v, g=v, beta=v, cu_seqlens=device_bounds, cu_seqlens_cpu=torch.tensor([0, 2, 6]))
    assert out[0, :, 0].tolist() == [1.0, 2.0, 1.0, 2.0, 3.0, 4.0]


def test_cuda_capture_resolves_a_submodule_only_package_kernel():
    """``fla`` exports nothing at root, so a decoration that resolved only off a bare package import
    would silently capture the torch body — every CUDA forward then running the naive scan at ~10x
    the step time. The decoration must capture fla's real kernel.

    A fresh interpreter is required: once anything in the process has imported the submodule (any
    ``fla.ops`` import loads it), it is an attribute of the ``fla`` root and a root-only resolve finds
    the kernel by walking attributes, so the check would pass against the very regression it pins.
    """
    script = """
import sys

import fla
import halo
from transformers.integrations import hub_kernels

SUBMODULE = "fla.ops.gated_delta_rule"
findings = []
if SUBMODULE in sys.modules:
    findings.append(f"importing halo or fla loaded {SUBMODULE}, so a root-only resolve would pass unseen")


@hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def probe(query, key, value, **kwargs):
    return value, None


def closure_functions(fn, seen):
    if callable(fn) and id(fn) not in seen:
        seen.add(id(fn))
        yield fn
        for cell in getattr(fn, "__closure__", None) or ():
            yield from closure_functions(getattr(cell, "cell_contents", None), seen)


closure = list(closure_functions(probe, set()))
kernel = getattr(sys.modules.get(SUBMODULE), "chunk_gated_delta_rule", None)
if kernel is None or not any(fn is kernel for fn in closure):
    fla_functions = sorted(
        f"{fn.__module__}.{fn.__qualname__}" for fn in closure if getattr(fn, "__module__", "").startswith("fla.")
    )
    findings.append(
        f"decoration did not capture {SUBMODULE}.chunk_gated_delta_rule, so CUDA forwards would silently run "
        f"the torch fallback (fla functions in its closure: {fla_functions or 'none'})"
    )
print("FINDINGS:" + "|".join(findings))
"""
    findings = probe_findings(script, "FINDINGS:")
    assert not findings, "\n".join(findings)


def test_torch_capture_with_the_package_installed_warns(caplog, monkeypatch):
    """A capture that lands on torch while the package is importable is a silent 10x — it must warn."""
    ensure_device_aware_kernel_dispatch()
    monkeypatch.setitem(hub_kernels._KERNELS_INTERNAL_PATH_MAPPINGS, "chunk_gated_delta_rule", "ops.no_such_submodule")
    monkeypatch.setattr(hub_kernel_dispatch, "_warned_torch_captures", set())

    with caplog.at_level("WARNING"):

        @hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
        def probe(query, key, value, **kwargs):
            return value, None

    assert any("captured the torch fallback" in r.message for r in caplog.records), (
        "no warning for a torch capture with the package installed — the fallback is silent again"
    )


def test_a_kernel_package_that_fails_its_own_import_raises(tmp_path, monkeypatch):
    """Only a missing module means "package absent or chain renamed". A package that is present but
    breaks inside its kernel import (a Triton or ABI fault) must stop the load: swallowed, upstream
    captures the torch body and every CUDA forward runs the slow scan."""
    ensure_device_aware_kernel_dispatch()
    ops = tmp_path / "halo_broken_kernel_pkg" / "ops"
    ops.mkdir(parents=True)
    (ops.parent / "__init__.py").write_text("")
    (ops / "__init__.py").write_text("")
    (ops / "scan.py").write_text("raise RuntimeError('simulated triton ABI fault')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match="simulated triton ABI fault"):
        hub_kernels.use_kernel_func_from_hub_with_fallback("halo_probe_scan", "halo_broken_kernel_pkg", "ops.scan")


def test_importing_src_installs_the_shim_before_anything_binds_the_factory():
    """The install happens on ``halo`` import, not on first use.

    Every test above calls ``ensure_device_aware_kernel_dispatch()`` itself and then decorates its own
    probe, which can only ever observe install-then-bind; this pins the install to the import. A fresh
    interpreter is required: this process installed the shim long ago.
    """
    probe = (
        "import halo;"
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
    assert result.returncode == 0, f"importing halo failed:\n{result.stderr[-2000:]}"
    assert "INSTALLED True" in result.stdout, (
        "importing `halo` no longer installs the device-aware kernel dispatch — every modeling module "
        f"imported afterwards binds the CUDA-only kernel. stdout={result.stdout[-500:]!r}"
    )


def test_importing_halo_redecorates_what_a_module_imported_earlier_bound():
    """A modeling module decorates at its own import, so one imported ahead of ``halo`` holds upstream's capture,
    the CUDA-only package included, and every later CPU forward of that family dies with ``Expected x.is_cuda()``
    in whatever test happens to run after the import. The premise check is the negative control: before ``halo``
    the capture fails on CPU. A fresh interpreter is required: this process installed the shim long ago."""
    pytest.importorskip("causal_conv1d")
    script = """
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

x, weight = torch.randn(1, 4, 6), torch.randn(4, 3)
reference = F.silu(F.conv1d(x, weight.unsqueeze(1), padding=2, groups=4)[..., :6])


def cpu_error(fn):
    try:
        out = fn(x, weight, None, activation="silu")
    except RuntimeError as exc:
        return str(exc)
    return None if torch.allclose(out, reference) else "a CPU result off the torch conv"


findings = []
if cpu_error(modeling_qwen3_5.causal_conv1d_fn) is None:
    findings.append("premise: the capture bound ahead of halo ran on CPU, so a missed re-decoration would pass unseen")

# After the modeling module, as a test module whose third-party imports sort above its first-party ones.
import halo

for where, fn in (("the modeling module", modeling_qwen3_5.causal_conv1d_fn), ("a by-name import", causal_conv1d_fn)):
    if error := cpu_error(fn):
        findings.append(f"{where} still holds upstream's capture after importing halo: {error}")
if causal_conv1d_fn is not modeling_qwen3_5.causal_conv1d_fn:
    findings.append("the by-name import was re-decorated apart from its source")
print("FINDINGS:" + "|".join(findings))
"""
    findings = probe_findings(script, "FINDINGS:")
    assert not findings, "\n".join(findings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
