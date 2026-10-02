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

The applier's instance patch binds forwards on a built module — this norm's, LigerRMSNorm's, fla's gated
norm's and the fused GLU MLP's — and a module so patched must survive ``torch.save`` and ``torch.load``.
The fused-loss head's binding cannot be pickled by reference, so loading a model patched with it must
raise rather than come back running the unfused head.
"""

import functools
import importlib
import io
from types import SimpleNamespace

import pytest
import torch
from liger_kernel.transformers import LigerRMSNorm
from transformers import LagunaConfig, LagunaForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteRMSNorm
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from transformers.models.laguna.modeling_laguna import LagunaRMSNorm
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from src.kernels.liger.builder import (
    _fused_glu_forward,
    _native_rms_norm_class,
    _patch_instance,
    _patch_module,
    fused_gated_rms_norm_forward,
    native_rms_norm_forward,
)
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from tests.common.models import TINY_LAGUNA_CONFIG, TINY_QWEN35_MOE_CONFIG

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


@pytest.mark.parametrize(
    "spec", [s for s in LIGER_FAMILY_SPECS if s.rms_norm_kernel == "native"], ids=lambda s: s.model_types[0]
)
def test_the_applier_installs_each_familys_casting_mode(spec):
    """The norm class the family's applier installs follows the family's eager order in bf16 more closely
    than the same kernel under the other casting mode, so a spec with the wrong mode fails here.

    The applier's class swap runs on a stand-in for the modeling module, which leaves transformers'
    classes untouched for the rest of the session.
    """
    module = importlib.import_module(spec.modeling_module)
    staged = SimpleNamespace(__name__=module.__name__, **{name: getattr(module, name) for name in spec.rms_norm})
    _patch_module(staged, spec, {"rms_norm": True})
    other_mode = "llama" if spec.rms_norm_casting_mode == "gemma" else "gemma"
    torch.manual_seed(0)
    x = (torch.randn(64, 96) * 3).to(torch.bfloat16)
    for name in spec.rms_norm:
        eager_cls = getattr(module, name)
        eager = eager_cls(96, 1e-6).to(torch.bfloat16)
        installed = getattr(staged, name)(96, 1e-6).to(torch.bfloat16)
        other = _native_rms_norm_class(eager_cls, casting_mode=other_mode)(96, 1e-6).to(torch.bfloat16)
        assert type(installed) is not eager_cls, f"{name}: the applier installed no native norm"
        with torch.no_grad():
            eager.weight.uniform_(0.3, 3.0)
            installed.weight.copy_(eager.weight)
            other.weight.copy_(eager.weight)
        reference = eager(x)
        installed_mismatch = (installed(x) != reference).float().mean().item()
        other_mismatch = (other(x) != reference).float().mean().item()
        assert installed_mismatch < other_mismatch, (
            f"{name}: the installed {spec.rms_norm_casting_mode!r} mode mismatches the eager norm on "
            f"{installed_mismatch:.1%} of outputs, the {other_mode!r} mode on {other_mismatch:.1%}"
        )


def _spec(model_type: str):
    return next(spec for spec in LIGER_FAMILY_SPECS if model_type in spec.model_types)


def _plain_state(module: torch.nn.Module) -> dict:
    """The instance attributes a patched forward reads (eps, casting mode, offset, gate, ...)."""
    return {name: value for name, value in vars(module).items() if name != "forward" and not name.startswith("_")}


@pytest.mark.parametrize(
    ("model_type", "flags", "build", "forward", "hidden"),
    [
        ("qwen3_moe", {"rms_norm": True}, lambda: Qwen3MoeRMSNorm(96, eps=1e-6), native_rms_norm_forward, 96),
        ("laguna", {"rms_norm": True}, lambda: LagunaRMSNorm(96, eps=1e-6), LigerRMSNorm.forward, None),
        (
            "glm5_next",
            {"rms_norm": True},
            lambda: Glm5NextTextRMSNormGated(96, eps=1e-6),
            fused_gated_rms_norm_forward,
            None,
        ),
        (
            "qwen3_5_moe_text",
            {"swiglu": True},
            lambda: Qwen3_5MoeMLP(Qwen3_5MoeTextConfig(**TINY_QWEN35_MOE_CONFIG), intermediate_size=64),
            _fused_glu_forward,
            TINY_QWEN35_MOE_CONFIG["hidden_size"],
        ),
    ],
    ids=("native_norm", "liger_norm", "gated_norm", "fused_glu_mlp"),
)
def test_an_instance_patched_module_survives_torch_save_and_load(model_type, flags, build, forward, hidden):
    """The instance patch (an applier given ``model=``) binds the toolkit's forward on a built module. A
    bound method would pickle as a lookup of the function's own name on the module: ``torch.load`` then
    fails (a name the module does not carry) or silently binds the class's own forward (a function named
    ``forward``). The loaded module must run the toolkit's forward, bound to itself, over the attributes
    the patch set.

    The Liger and fla kernels are Triton, so their modules are checked for the binding and those
    attributes; the native norm and the GLU MLP also recompute their output. The modules are built from
    transformers' own classes and patched per instance only, which leaves the classes untouched for the
    rest of the session.
    """
    torch.manual_seed(0)
    module = build()
    _patch_instance(module, _spec(model_type), flags)
    assert isinstance(module.forward, functools.partial) and module.forward.func is forward, (
        "the instance patch bound no toolkit forward"
    )
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert isinstance(loaded.forward, functools.partial)
    assert loaded.forward.func is forward and loaded.forward.args == (loaded,)
    assert _plain_state(loaded) == _plain_state(module)
    if hidden is not None:
        x = torch.randn(3, 5, hidden)
        assert torch.equal(loaded(x), module(x))


@pytest.mark.parametrize(
    "spec",
    [s for s in LIGER_FAMILY_SPECS if s.modeling_module and s.rms_norm and s.rms_norm_kernel == "liger"],
    ids=lambda s: s.model_types[0],
)
def test_an_instance_patched_liger_norm_runs_the_specs_variant_at_its_own_epsilon(spec):
    """The instance patch must hand LigerRMSNorm's forward what the class swap does: the spec's offset and
    casting mode, which select the kernel's variant, and the module's own epsilon."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.rms_norm:
        norm = getattr(module, name)(16, 3e-6)
        _patch_instance(norm, spec, {"rms_norm": True})
        assert norm.forward.func is LigerRMSNorm.forward
        configured = {attr: getattr(norm, attr) for attr in ("variance_epsilon", "offset", "casting_mode", "in_place")}
        assert configured == {
            "variance_epsilon": 3e-6,
            "offset": spec.rms_norm_offset,
            "casting_mode": spec.rms_norm_casting_mode,
            "in_place": spec.rms_norm_casting_mode != "gemma",
        }, f"{name}: {configured}"


def test_an_instance_patched_fused_loss_head_refuses_torch_load():
    """The fused-loss head keeps a bound method: its forward is a per-family closure no pickle can
    reference, and the balancing resolver detects the head by that function's module. Loading a model so
    patched must raise, not come back running the family's unfused head."""
    model = LagunaForCausalLM(LagunaConfig(**TINY_LAGUNA_CONFIG))
    _patch_instance(model, _spec("laguna"), {"fused_linear_cross_entropy": True})
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    with pytest.raises(AttributeError, match="lce_forward"):
        torch.load(buffer, weights_only=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
