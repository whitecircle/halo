#!/usr/bin/env python
"""Expert-LoRA adapter GEMMs stay bf16 under low-precision expert compute.

``EPMoELayerBase._expert_proj`` runs the BASE expert weight at the layer's compute precision but
passes ``lowp=False`` for the two adapter GEMMs: block-scale quantization along the rank axis is
unrepresentable for r < 32 (the mx block size), and a divisible rank would silently train adapters
in a numerics regime no serving stack has. Pinned at the seam, on the real unbound methods over a
minimal stub (the ``test_ep_sort_sync_free`` idiom — no dispatcher, no model):

- ``fake_quant`` still refuses a sub-block axis (r=16 with mxfp8) — the quantizer gate is unchanged;
- ``_expert_proj`` with ``_lowp_precision=MXFP8`` and r=16 adapters raises nothing, routes the base
  GEMM through MXFP8 and BOTH adapter GEMMs through BF16 (asserted via a recording ``grouped_gemm``),
  and the adapter delta actually lands in the output;
- forcing the adapter GEMMs back through the layer precision raises the r-not-divisible ValueError
  — the in-file non-vacuity control.

Run: python tests/gpu/kernels/test_lowp_expert_lora.py  (1 GPU)
"""

import sys
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

import src.distributed.expert_parallel.base_layer as base_layer_mod
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.grouped_gemm import GroupedGemmPrecision
from src.kernels.lowp.quantization import fake_quant

DEV = "cuda"
E, K, N, R, T_PER = 2, 64, 48, 16, 16  # r=16: NOT divisible by mxfp8's 32-element block


def make_layer():
    """Minimal stand-in carrying only what ``_expert_proj`` / ``_grouped_mm`` read off ``self``."""
    torch.manual_seed(0)
    stub = SimpleNamespace(
        _lowp_precision=GroupedGemmPrecision.MXFP8,
        _lowp_weight_cacheable=True,
        _expert_adapters_enabled=True,
        _expert_lora_attrs=frozenset({"gate_proj"}),
        expert_lora_scaling=2.0,
        expert_lora_dropout=nn.Identity(),
        gate_proj=torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.05,
        # Nonzero B (the real init zeroes it) so the adapter delta is visible in the output.
        gate_proj_lora_A=torch.randn(E, K, R, device=DEV, dtype=torch.bfloat16) * 0.05,
        gate_proj_lora_B=torch.randn(E, R, N, device=DEV, dtype=torch.bfloat16) * 0.05,
    )
    stub._grouped_mm = MethodType(EPMoELayerBase._grouped_mm, stub)
    stub._expert_proj = MethodType(EPMoELayerBase._expert_proj, stub)
    return stub


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(E * T_PER, K, device=DEV, dtype=torch.bfloat16) * 0.3
    offs = torch.arange(T_PER, E * T_PER + 1, T_PER, device=DEV, dtype=torch.int32)
    return x, offs


def test_fake_quant_refuses_sub_block_axis():
    """The quantizer gate is unchanged: a 16-wide axis cannot be mxfp8-quantized (block size 32)."""
    hidden = torch.randn(8, R, device=DEV, dtype=torch.bfloat16)
    try:
        fake_quant(hidden, "mxfp8", axis=-1)
    except ValueError as exc:
        assert "block_size" in str(exc), f"unexpected refusal message: {exc}"
    else:
        raise AssertionError("fake_quant must refuse a 16-wide axis for mxfp8 (32-element blocks)")
    print("  fake_quant refuses a sub-block axis (r=16, mxfp8) PASS")


def test_adapter_gemms_stay_bf16():
    """r=16 adapters under MXFP8 expert compute: no raise, base GEMM MXFP8, adapter GEMMs BF16,
    and the adapter delta lands in the output."""
    layer = make_layer()
    x, offs = make_inputs()

    recorded = []
    real_grouped_gemm = base_layer_mod.grouped_gemm

    def recording(mat_a, mat_b, **kwargs):
        recorded.append(kwargs["precision"])
        return real_grouped_gemm(mat_a, mat_b, **kwargs)

    base_layer_mod.grouped_gemm = recording
    try:
        out = layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    finally:
        base_layer_mod.grouped_gemm = real_grouped_gemm

    expected = [GroupedGemmPrecision.MXFP8, GroupedGemmPrecision.BF16, GroupedGemmPrecision.BF16]
    assert recorded == expected, f"GEMM precisions {recorded} != {expected} (base lowp, adapters bf16)"
    assert out.shape == (E * T_PER, N) and torch.isfinite(out).all()

    layer._expert_adapters_enabled = False
    base_only = layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    layer._expert_adapters_enabled = True
    assert not torch.equal(out, base_only), "adapter delta missing from the output"
    print("  _expert_proj: base MXFP8, adapters BF16, delta present, r=16 accepted PASS")


def test_forced_lowp_adapter_gemm_raises():
    """Non-vacuity control: routing the r=16 adapter GEMMs back through the layer precision must
    raise — proving the lowp=False routing is what makes r=16 work."""
    layer = make_layer()
    x, offs = make_inputs()
    plain_grouped_mm = layer._grouped_mm
    layer._grouped_mm = lambda mat_a, mat_b, **kw: plain_grouped_mm(mat_a, mat_b, **{**kw, "lowp": True})
    try:
        layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    except ValueError as exc:
        assert "block_size" in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError("adapter GEMM through mxfp8 at r=16 must raise")
    print("  forced-lowp adapter GEMM raises the r%32 ValueError (control) PASS")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")  # the sentinel the launcher skips on; a bare exit 0 reads as a PASS
        return 0
    print(f"lowp expert-LoRA tests on {torch.cuda.get_device_name()}")
    failures = []
    for test in (
        test_fake_quant_refuses_sub_block_axis,
        test_adapter_gemms_stay_bf16,
        test_forced_lowp_adapter_gemm_raises,
    ):
        try:
            test()
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  FAIL {test.__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        return 1
    print("\nAll lowp expert-LoRA tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
