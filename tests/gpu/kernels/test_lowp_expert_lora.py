#!/usr/bin/env python
"""Expert-LoRA adapter GEMMs stay bf16 under low-precision expert compute.

Block-scale quantization along the rank axis is unrepresentable for r < 32 (the mx block size), and a divisible rank
would train adapters in a regime no serving stack has.
"""

from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

import halo.distributed.expert_parallel.moe_layer.compute as compute_mod
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.quantization import fake_quant
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import fro_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint

DEV = "cuda"
E, K, N, R, T_PER = 2, 64, 48, 16, 16  # r=16: not divisible by mxfp8's 32-element block.
SIMULATED_FP8 = LowpSpec(precision="fp8")


def make_layer(lowp: LowpSpec | None = SIMULATED_FP8):
    torch.manual_seed(0)
    stub = SimpleNamespace(
        _lowp=lowp,
        _lowp_weight_cacheable=True,
        _expert_adapters_enabled=True,
        _expert_lora_attrs=frozenset({"gate_proj"}),
        expert_lora_scaling=2.0,
        expert_lora_dropout=nn.Identity(),
        gate_proj=torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.05,
        # Nonzero B (the real init zeroes it) so the adapter delta shows.
        gate_proj_lora_A=torch.randn(E, K, R, device=DEV, dtype=torch.bfloat16) * 0.05,
        gate_proj_lora_B=torch.randn(E, R, N, device=DEV, dtype=torch.bfloat16) * 0.05,
    )
    stub._grouped_mm = MethodType(EPMoELayerBase._grouped_mm, stub)
    stub._expert_base_proj = MethodType(EPMoELayerBase._expert_base_proj, stub)
    stub._expert_proj = MethodType(EPMoELayerBase._expert_proj, stub)
    return stub


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(E * T_PER, K, device=DEV, dtype=torch.bfloat16) * 0.3
    offs = torch.arange(T_PER, E * T_PER + 1, T_PER, device=DEV, dtype=torch.int32)
    return x, offs


def test_fake_quant_refuses_sub_block_axis():
    """A 16-wide axis cannot be mxfp8-quantized (block size 32)."""
    hidden = torch.randn(8, R, device=DEV, dtype=torch.bfloat16)
    try:
        fake_quant(hidden, "mxfp8", axis=-1)
    except ValueError as exc:
        assert "block_size" in str(exc), f"unexpected refusal message: {exc}"
    else:
        raise AssertionError("fake_quant must refuse a 16-wide axis for mxfp8 (32-element blocks)")
    print("  fake_quant refuses a sub-block axis (r=16, mxfp8) PASS")


def test_adapter_gemms_stay_bf16():
    layer = make_layer()
    x, offs = make_inputs()

    recorded = []
    real_expert_gemm = compute_mod.expert_gemm

    def recording(mat_a, mat_b, **kwargs):
        recorded.append(kwargs["lowp"] and kwargs["lowp"].precision)
        return real_expert_gemm(mat_a, mat_b, **kwargs)

    compute_mod.expert_gemm = recording
    try:
        out = layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    finally:
        compute_mod.expert_gemm = real_expert_gemm

    expected = ["fp8", None, None]
    assert recorded == expected, f"GEMM precisions {recorded} != {expected} (base lowp, adapters bf16)"
    assert out.shape == (E * T_PER, N) and torch.isfinite(out).all()

    layer._expert_adapters_enabled = False
    base_only = layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    layer._expert_adapters_enabled = True
    assert not torch.equal(out, base_only), "adapter delta missing from the output"
    print("  _expert_proj: base MXFP8, adapters BF16, delta present, r=16 accepted PASS")


def test_forced_lowp_adapter_gemm_raises():
    """Non-vacuity control: the lowp=False routing is what makes r=16 work."""
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


def test_native_base_trains_bf16_adapters():
    """Under ``lowp_backend: native`` the frozen base GEMM runs on DeepGEMM, forward and dgrad with no wgrad, and
    the adapters stay bf16."""
    layer = make_layer(LowpSpec(precision="fp8", backend="native", backward_precision="fp8"))
    x, offs = make_inputs()
    params = [layer.gate_proj_lora_A, layer.gate_proj_lora_B]
    for tensor in (x, *params):
        tensor.requires_grad_(True)
    wgrads, real_wgrad = [], deepgemm._fp8_wgrad
    deepgemm._fp8_wgrad = lambda *args: wgrads.append(args) or real_wgrad(*args)
    out = layer._expert_proj(x, "gate_proj", offs, torch.bfloat16)
    out.float().pow(2).sum().backward()
    reference = make_layer(None)._expert_proj(x.detach(), "gate_proj", offs, torch.bfloat16)
    rel = fro_rel_err(out.detach(), reference)
    assert 0.005 < rel < 0.10, f"native base + bf16 adapters: rel {rel:.4f} vs bf16 outside fp8's band"
    deepgemm._fp8_wgrad = real_wgrad
    for tensor in (x, *params):
        assert tensor.grad.isfinite().all() and tensor.grad.abs().sum() > 0
    assert not wgrads, "the frozen base took an fp8 weight gradient"
    print(
        f"  native fp8 frozen base (fp8 dgrad) with bf16 adapters: rel {rel:.4f}, every gradient finite and nonzero PASS"
    )


def test_attention_lora_reads_the_non_finite_mxfp4_flag(ctx):
    """PEFT calls the base model's ``forward`` past the model's own hooks, so the flag's one read sits on a layer: a
    NaN expert weight, which the MXFP4 cast gives a finite code, must still raise under attention LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(ctx, TINY_MOE_FAMILIES["qwen3_moe"], "lowp_expert_lora", tokenizer, 0)
    lowp = LowpSpec(precision="mxfp4", backend="native", apply_dense_mlp=False)
    model, _ = load_distributed_model(
        checkpoint,
        ParallelismConfig(ep_size=1, lowp=lowp),
        torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="eager",
        use_liger_kernel=False,
    )
    name, weight = ep_layers(model)[0].expert_named_params()[0]
    model = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    input_ids = torch.arange(32, device=DEV)[None]
    model(input_ids=input_ids)
    with torch.no_grad():
        weight.view(-1)[0] = float("nan")
    try:
        model(input_ids=input_ids)
    except ValueError as exc:
        assert "Non-finite expert weight" in str(exc) and name in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError("a NaN expert weight passed a forward under attention LoRA")
    print(f"  attention LoRA over native mxfp4: the NaN in {name} raised at the forward PASS")


@gpu_test_main(exact_world_size=1, prefix="lowp_expert_lora", partial_state=False)
def run(ctx) -> dict:
    log(f"lowp expert-LoRA tests on {torch.cuda.get_device_name()}")
    checks: dict[str, bool] = {}
    record_check(checks, "fake_quant_refuses_sub_block_axis", test_fake_quant_refuses_sub_block_axis)
    record_check(checks, "adapter_gemms_stay_bf16", test_adapter_gemms_stay_bf16)
    record_check(checks, "forced_lowp_adapter_gemm_raises", test_forced_lowp_adapter_gemm_raises)
    if deepgemm.deepgemm_available():  # the Hopper image ships no deep_gemm
        record_check(checks, "native_base_trains_bf16_adapters", test_native_base_trains_bf16_adapters)
        record_check(
            checks,
            "attention_lora_reads_the_non_finite_mxfp4_flag",
            lambda: test_attention_lora_reads_the_non_finite_mxfp4_flag(ctx),
        )
    return {"checks": checks}


if __name__ == "__main__":
    run()
