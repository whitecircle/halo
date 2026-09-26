"""Torch's fused RMSNorm computes each family's eager norm under that family's casting mode.

Gemma mode (Gemma 4, GptOss): the fused kernel normalizes and multiplies the weight in fp32 and casts
once, which is those families' own order. The cases cover a scaled norm, a weightless one
(``with_scale=False``, which LigerRMSNorm cannot express) and an fp32 weight under bf16 activations,
which the fused kernel cannot take in one call.

Llama mode (Qwen3 MoE, GLM-4 MoE Lite): the normalized activations are cast back before the weight
multiply, which then runs in the activation dtype (``weight * x.to(input_dtype)``). A native norm left in
the gemma mode would skip that rounding, which the bf16 case detects.

Outputs agree to bf16 rounding in both modes: the fused reduction sums in another order than the eager
``pow(2).mean``.
"""

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteRMSNorm
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from src.kernels.liger.builder import _native_rms_norm_class

NativeNorm = _native_rms_norm_class(Gemma4RMSNorm, casting_mode="gemma")


@pytest.mark.parametrize("with_scale", [True, False])
@pytest.mark.parametrize(
    ("act_dtype", "weight_dtype"),
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, torch.float32)],
)
def test_native_norm_matches_eager(with_scale, act_dtype, weight_dtype):
    torch.manual_seed(0)
    eager = Gemma4RMSNorm(96, eps=1e-6, with_scale=with_scale).to(weight_dtype)
    native = NativeNorm(96, eps=1e-6, with_scale=with_scale).to(weight_dtype)
    if with_scale:
        with torch.no_grad():
            eager.weight.normal_(1.0, 0.1)
            native.weight.copy_(eager.weight)
    x = (torch.randn(5, 7, 96) * 3).to(act_dtype)
    x_eager, x_native = x.clone().requires_grad_(), x.clone().requires_grad_()
    grad = torch.randn(5, 7, 96).to(act_dtype)
    out_eager, out_native = eager(x_eager), native(x_native)
    out_eager.backward(grad)
    out_native.backward(grad)
    assert out_native.dtype == out_eager.dtype == act_dtype
    tol = {"rtol": 1e-5, "atol": 1e-5} if act_dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(out_native, out_eager, **tol)
    torch.testing.assert_close(x_native.grad, x_eager.grad, **tol)
    if with_scale:
        torch.testing.assert_close(native.weight.grad, eager.weight.grad, **tol)


def _compare(eager, native, act_dtype):
    with torch.no_grad():
        eager.weight.normal_(1.0, 0.1)
        native.weight.copy_(eager.weight)
    x = (torch.randn(5, 7, 96) * 3).to(act_dtype)
    x_eager, x_native = x.clone().requires_grad_(), x.clone().requires_grad_()
    grad = torch.randn(5, 7, 96).to(act_dtype)
    out_eager, out_native = eager(x_eager), native(x_native)
    out_eager.backward(grad.to(out_eager.dtype))
    out_native.backward(grad.to(out_native.dtype))
    assert out_native.dtype == out_eager.dtype
    tol = {"rtol": 1e-5, "atol": 1e-5} if out_eager.dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(out_native, out_eager, **tol)
    torch.testing.assert_close(x_native.grad, x_eager.grad, **tol)
    torch.testing.assert_close(native.weight.grad, eager.weight.grad, **tol)
    return out_eager, out_native


@pytest.mark.parametrize("norm_cls", [Qwen3MoeRMSNorm, Glm4MoeLiteRMSNorm])
@pytest.mark.parametrize(
    ("act_dtype", "weight_dtype"),
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, torch.float32)],
)
def test_llama_mode_matches_the_llama_family_norms(norm_cls, act_dtype, weight_dtype):
    """Including the llama norms' dtype promotion: an fp32 weight times the bf16-cast activations
    returns fp32, and the native norm must return the same dtype."""
    torch.manual_seed(0)
    eager = norm_cls(96, eps=1e-6).to(weight_dtype)
    native = _native_rms_norm_class(norm_cls, casting_mode="llama")(96, eps=1e-6).to(weight_dtype)
    _compare(eager, native, act_dtype)


def test_gemma_mode_matches_gptoss_norm():
    torch.manual_seed(0)
    eager = GptOssRMSNorm(96, eps=1e-5).to(torch.bfloat16)
    native = _native_rms_norm_class(GptOssRMSNorm, casting_mode="gemma")(96, eps=1e-5).to(torch.bfloat16)
    _compare(eager, native, torch.bfloat16)


def test_the_casting_mode_is_observable_in_bf16():
    """The two modes differ by one bf16 rounding before the weight multiply; with a weight far from 1 that
    rounding shows, so a llama-family norm given the gemma mode would not match its eager module."""
    torch.manual_seed(0)
    eager = Qwen3MoeRMSNorm(96, eps=1e-6).to(torch.bfloat16)
    llama = _native_rms_norm_class(Qwen3MoeRMSNorm, casting_mode="llama")(96, eps=1e-6).to(torch.bfloat16)
    gemma = _native_rms_norm_class(Qwen3MoeRMSNorm, casting_mode="gemma")(96, eps=1e-6).to(torch.bfloat16)
    with torch.no_grad():
        eager.weight.uniform_(0.3, 3.0)
        llama.weight.copy_(eager.weight)
        gemma.weight.copy_(eager.weight)
    x = (torch.randn(64, 96) * 3).to(torch.bfloat16)
    reference = eager(x)
    llama_mismatch = (llama(x) != reference).float().mean().item()
    gemma_mismatch = (gemma(x) != reference).float().mean().item()
    assert llama_mismatch < gemma_mismatch, (llama_mismatch, gemma_mismatch)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
