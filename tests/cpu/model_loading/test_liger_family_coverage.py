#!/usr/bin/env python
"""Per-family Liger coverage: the roster is covered and every declared role is exactly equivalent. Compares each
family's own forward to pure-torch Liger semantics, so it runs on CPU; the GPU suite checks the kernels."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import types
from pathlib import Path

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger.cross_entropy import build_lce_forward
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.model_loading.liger.fused_modules import _bridge_gated_norm
from halo.models.head_transform import resolve_head_transform_spec
from halo.models.moe.family_specs import moe_model_types
from halo.models.moe.router_logits import honors_router_logits_flag
from tests.common.liger_appliers import liger_applier
from tests.common.repo import probe_findings

NATIVE_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.modeling_module]
REMOTE_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.remote_classes]

# Read from huggingface_hub so an HF_HOME redirect is followed; the CPU tier gives each test process an empty
# dynamic-module cache.
_HUB_MODELING_FILES = "models--*/snapshots/*/modeling_*.py"

_LLAMA_RMS_NORM_BODY = (
    "hidden_states.to(torch.float32)",
    "hidden_states.pow(2).mean(-1, keepdim=True)",
    "torch.rsqrt(variance + self.variance_epsilon)",
    "return self.weight * hidden_states.to(input_dtype)",
)

# Wrappers carry no decoder classes; their text tower may be either sibling, or the wrapper runs its own head
# (Qwen3.8-Flash-Next), so pinning one to a spec would be wrong.
RESOLVED_THROUGH_TEXT_CONFIG = {
    "gemma4": "gemma4_text",
    "lfm2_vl": "lfm2",
    "cohere2_vision": "cohere2",
    "mistral3": "mistral4",
    "qwen4_exp": "qwen4_exp_text",
}

# Norm classes that compute Liger's function yet must stay undeclared; the sweep would otherwise report each as a gap.
UNDECLARABLE_NORMS = {
    # No parity test covers the swap. The loaders cast the fp32-pinned weight to the run dtype, where the eager norm
    # returns the input dtype as Liger's kernel does.
    "deepseek_v4": {"DeepseekV4RMSNorm"},
}

# A clamp, trained output scale or differently named projection means the fused kernel would not compute what the
# family computes.
CANONICAL_GLU_ATTRIBUTES = {"down_proj", "act_fn", "gate_proj", "up_proj"}


def _roster_model_types() -> set[str]:
    return set(moe_model_types())


def _self_attributes(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _forward_ast(cls: type) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(cls.forward))).body[0]


def _cached_remote_sources(spec) -> list[str]:
    """These modules relative-import their sibling configuration, so they cannot be imported standalone and their
    checks read the source."""
    sources = []
    for path in Path(HF_HUB_CACHE).glob(_HUB_MODELING_FILES):
        source = path.read_text(encoding="utf-8")
        if all(f"class {name}(" in source for name in spec.remote_classes):
            sources.append(source)
    if not sources:
        pytest.skip(f"no cached modeling module defines {spec.remote_classes}")
    return sources


def _remote_forward_ast(source: str, class_name: str) -> ast.FunctionDef:
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")


def _attributes_between_head_and_loss(forward: ast.FunctionDef, name: str) -> set[str]:
    body = forward.body
    head = next(i for i, stmt in enumerate(body) if "lm_head" in _self_attributes(stmt))
    loss = next(i for i, stmt in enumerate(body) if "loss_function" in _self_attributes(stmt))
    assert head < loss, f"{name}.forward calls loss_function before lm_head"
    return set().union(*(_self_attributes(stmt) for stmt in body[head + 1 : loss])) - {"config", "vocab_size"}


def _liger_rms_norm_reference(x, weight, eps, offset, casting_mode):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if casting_mode == "gemma":
        return (normed * (offset + weight.float())).type_as(x)
    return (offset + weight) * normed.to(x.dtype)


@pytest.mark.parametrize("model_type", sorted(_roster_model_types()))
def test_every_roster_family_resolves_an_applier(model_type):
    if model_type in RESOLVED_THROUGH_TEXT_CONFIG:
        text_type = RESOLVED_THROUGH_TEXT_CONFIG[model_type]
        assert liger_applier(text_type) is not None, (
            f"{model_type} resolves Liger through its text sub-config {text_type}, which now has no applier"
        )
        return
    assert liger_applier(model_type) is not None, (
        f"no Liger applier covers {model_type}; add a LigerFamilySpec to LIGER_FAMILY_SPECS"
    )


@pytest.mark.parametrize("spec", LIGER_FAMILY_SPECS, ids=lambda s: s.model_types[0])
def test_a_listed_wrapper_can_only_wrap_the_family_it_is_listed_under(spec):
    """A wrapper pinned to the wrong sibling patches classes the model never instantiates and reports every kernel
    as applied."""
    for model_type in spec.model_types:
        config_cls = CONFIG_MAPPING.get(model_type)
        if config_cls is None or "text_config" not in (getattr(config_cls, "sub_configs", None) or {}):
            continue
        tower = config_cls().text_config.model_type
        assert tower in spec.model_types, (
            f"{model_type} is a composite wrapper whose default text tower is {tower!r}, outside this "
            f"spec's {spec.model_types}; drop it and let dispatch resolve it through text_config"
        )


@pytest.mark.parametrize(
    "spec", [s for s in NATIVE_SPECS if not s.delegates_to_upstream], ids=lambda s: s.model_types[0]
)
def test_no_liger_expressible_norm_is_left_undeclared(spec):
    """A class that is the llama- or Gemma-style body but is missing from the spec trains unfused with no warning."""
    module = importlib.import_module(spec.modeling_module)
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    expressible = set()
    for name, cls in vars(module).items():
        if not (isinstance(cls, type) and cls.__module__ == module.__name__ and name.endswith("RMSNorm")):
            continue
        try:
            norm = cls(16, 1e-5).to(torch.bfloat16)
        except (TypeError, ValueError):
            continue
        weight = getattr(norm, "weight", None)
        if weight is None or "gate" in inspect.signature(norm.forward).parameters:
            continue
        weight.data.normal_()
        for offset in (0.0, 1.0):
            for casting_mode in ("llama", "gemma"):
                if torch.allclose(
                    norm(x), _liger_rms_norm_reference(x, weight, 1e-5, offset, casting_mode), atol=1e-2
                ):
                    expressible.add(name)
    undeclared = expressible - set(spec.rms_norm) - UNDECLARABLE_NORMS.get(spec.model_types[0], set())
    assert not undeclared, (
        f"{spec.model_types[0]}: {sorted(undeclared)} compute a norm LigerRMSNorm expresses but are not "
        f"declared — they train unfused. Declare them, or record why not in UNDECLARABLE_NORMS."
    )


@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_every_declared_class_exists(spec):
    """A transformers bump that renames a class must fail here, before model load on a GPU node."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.rms_norm + spec.gated_rms_norm + spec.glu_mlp + spec.causal_lm:
        assert isinstance(getattr(module, name, None), type), f"{spec.modeling_module} defines no class {name!r}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_declared_norms_are_what_the_chosen_liger_variant_computes(spec, dtype):
    """bf16 pins ``casting_mode``: the modes differ only in whether the multiply precedes the cast back, so in fp32
    they are bit-identical."""
    module = importlib.import_module(spec.modeling_module)
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16, dtype=dtype)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2
    for name in spec.rms_norm:
        norm = getattr(module, name)(16, 1e-5).to(dtype)
        norm.weight.data.normal_()
        reference = _liger_rms_norm_reference(x, norm.weight, 1e-5, spec.rms_norm_offset, spec.rms_norm_casting_mode)
        assert torch.allclose(norm(x), reference, atol=tolerance), (
            f"{name} is not what LigerRMSNorm(offset={spec.rms_norm_offset}, "
            f"casting_mode={spec.rms_norm_casting_mode!r}) computes in {dtype} — the swap would "
            f"change the model"
        )


@pytest.mark.parametrize("spec", REMOTE_SPECS, ids=lambda s: s.model_types[0])
def test_declared_remote_norms_are_llama_style(spec):
    """Checks the source for the fp32-upcast, weight-multiply body; the grouped variant sits next to it in the same
    file."""
    sources = _cached_remote_sources(spec)
    assert spec.rms_norm_casting_mode == "llama" and spec.rms_norm_offset == 0.0
    for source in sources:
        for name in spec.rms_norm:
            body = source.split(f"class {name}(")[1].split("\nclass ")[0]
            for required in _LLAMA_RMS_NORM_BODY:
                assert required in body, f"{name} no longer computes {required!r}; the Liger swap would change it"


def test_the_norm_check_rejects_the_wrong_liger_variant():
    """Anti-vacuity: the parameters have to matter, or the test above passes on any norm."""
    step = importlib.import_module("transformers.models.step3p7.modeling_step3p7").Step3p7RMSNorm(16, 1e-5)
    laguna = importlib.import_module("transformers.models.laguna.modeling_laguna").LagunaRMSNorm(16, 1e-5)
    step.weight.data.normal_()
    laguna.weight.data.normal_()
    x = torch.randn(3, 5, 16)
    assert not torch.allclose(step(x), _liger_rms_norm_reference(x, step.weight, 1e-5, 0.0, "llama"))
    assert not torch.allclose(laguna(x), _liger_rms_norm_reference(x, laguna.weight, 1e-5, 1.0, "gemma"))

    # Only bf16 shows ``casting_mode`` is discriminating; otherwise the fp32 case pins offset and nothing else.
    step_bf16 = importlib.import_module("transformers.models.step3p7.modeling_step3p7").Step3p7RMSNorm(16, 1e-5)
    step_bf16.weight.data.normal_()
    step_bf16 = step_bf16.to(torch.bfloat16)
    xb = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    gemma = _liger_rms_norm_reference(xb, step_bf16.weight, 1e-5, 1.0, "gemma")
    llama = _liger_rms_norm_reference(xb, step_bf16.weight, 1e-5, 1.0, "llama")
    assert not torch.equal(gemma, llama), "casting_mode makes no difference here — the check is blind to it"


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.rms_norm], ids=lambda s: s.model_types[0])
def test_no_norm_output_is_a_direct_operand_of_an_addition(spec):
    """Liger's llama-mode backward writes dX into the incoming dY buffer, and ``AddBackward`` hands one gradient
    object to both inputs, so a post-norm silently overwrites its sibling's gradient.
    A transformers bump adding a post-norm must fail here and switch that family to ``in_place=False``."""
    if spec.rms_norm_casting_mode == "gemma":
        pytest.skip("the gemma casting mode already uses in_place=False")
    module = importlib.import_module(spec.modeling_module)
    for name, cls in vars(module).items():
        if not isinstance(cls, type) or getattr(cls, "__module__", None) != module.__name__:
            continue
        forward = getattr(cls, "forward", None)
        if forward is None or not inspect.isfunction(forward):
            continue
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(forward)))):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
                continue
            for side in (node.left, node.right):
                is_norm_call = (
                    isinstance(side, ast.Call)
                    and isinstance(side.func, ast.Attribute)
                    and "norm" in side.func.attr.lower()
                )
                assert not is_norm_call, (
                    f"{name}.forward adds a norm output directly (`{ast.unparse(node)[:80]}`); Liger's "
                    f"in-place RMSNorm backward would overwrite the sibling branch's gradient"
                )


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.gated_rms_norm], ids=lambda s: s.model_types[0])
def test_declared_gated_norms_expose_what_the_fused_kernel_reads(spec):
    """The bridge reads ``weight``, ``bias``, ``activation`` and ``eps``; a class missing one is a load-time crash."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.gated_rms_norm:
        norm = getattr(module, name)(16, 1e-5)
        assert "gate" in inspect.signature(norm.forward).parameters, f"{name}.forward takes no gate"
        assert getattr(norm, "weight", None) is not None, f"{name} has no weight to scale by"
        assert getattr(norm, "bias", None) is None, f"{name} carries a bias the eager norm never applies"
        assert getattr(norm, "activation", None), f"{name} declares no activation for the fused gate"
        assert getattr(norm, "variance_epsilon", getattr(norm, "eps", None)) is not None
        # The real bridge also refuses a gate `fla`'s kernel does not implement, which that kernel answers by applying
        # no gate at all.
        _bridge_gated_norm(norm)


def test_a_grouped_gated_norm_is_never_declared():
    """Bailing's ``*GroupRMSNorm`` reduces over ``hidden_size // group_norm_size``, so a last-dim kernel reduces the
    wrong axis and returns the right shape."""
    declared = {name for spec in LIGER_FAMILY_SPECS for name in spec.gated_rms_norm}
    assert declared, "no family declares a gated norm — this check would pass on an empty set"
    assert not [name for name in declared if "Group" in name], f"a grouped gated norm is declared: {declared}"


@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_declared_glu_mlps_are_the_canonical_gated_body(spec):
    """Inkling's ``global_scale``, the ``swiglu_limit`` clamps and LFM-2's ``w1``/``w3``/``w2`` would be dropped or
    missed by the fused forward."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.glu_mlp:
        attributes = _self_attributes(_forward_ast(getattr(module, name)))
        assert attributes == CANONICAL_GLU_ATTRIBUTES, (
            f"{name}.forward touches {sorted(attributes)}, not the canonical "
            f"{sorted(CANONICAL_GLU_ATTRIBUTES)} — the fused GLU would compute a different function"
        )


@pytest.mark.parametrize("spec", [s for s in REMOTE_SPECS if s.glu_mlp], ids=lambda s: s.model_types[0])
def test_declared_remote_glu_mlps_are_the_canonical_gated_body(spec):
    for source in _cached_remote_sources(spec):
        for name in spec.glu_mlp:
            attributes = _self_attributes(_remote_forward_ast(source, name))
            assert attributes == CANONICAL_GLU_ATTRIBUTES, (
                f"remote {name}.forward touches {sorted(attributes)}, not the canonical "
                f"{sorted(CANONICAL_GLU_ATTRIBUTES)} — the fused GLU would compute a different function"
            )


def test_the_glu_check_rejects_a_scaled_or_clamped_body():
    """Anti-vacuity: the three MLPs left undeclared must all fail the same check."""
    undeclared = {
        "transformers.models.inkling.modeling_inkling": "InklingMLP",
        "transformers.models.glm5_next.modeling_glm5_next": "Glm5NextTextMLP",
        "transformers.models.lfm2.modeling_lfm2": "Lfm2MLP",
    }
    for module_path, name in undeclared.items():
        cls = getattr(importlib.import_module(module_path), name)
        assert _self_attributes(_forward_ast(cls)) != CANONICAL_GLU_ATTRIBUTES, (
            f"{name} now matches the canonical GLU body; if that is real, declare it — but the check "
            f"above is no longer proving anything about the ones that do not"
        )


def _declared_head_attributes(spec, head_class_name: str) -> set[str]:
    """Specs resolve by class name, so a bare class of that name stands in for the native or remote head."""
    head_class = type(head_class_name, (), {})
    transform = resolve_head_transform_spec(head_class).transform(CONFIG_MAPPING[spec.model_types[0]]())
    return {"logit_scale"} if transform.logit_scale is not None else set()


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.causal_lm], ids=lambda s: s.model_types[0])
def test_declared_heads_do_nothing_the_fused_loss_would_drop(spec):
    """The fused loss replaces both calls, so anything applied in between vanishes."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.causal_lm:
        allowed = _declared_head_attributes(spec, name)
        between = _attributes_between_head_and_loss(_forward_ast(getattr(module, name)), name)
        assert between <= allowed, (
            f"{name}.forward applies {sorted(between - allowed)} between lm_head and the loss; the "
            f"fused loss replaces both and would drop it. Declare it on the spec or drop causal_lm."
        )


@pytest.mark.parametrize("spec", [s for s in REMOTE_SPECS if s.causal_lm], ids=lambda s: s.model_types[0])
def test_declared_remote_heads_do_nothing_the_fused_loss_would_drop(spec):
    for source in _cached_remote_sources(spec):
        for name in spec.causal_lm:
            allowed = _declared_head_attributes(spec, name)
            between = _attributes_between_head_and_loss(_remote_forward_ast(source, name), name)
            assert between <= allowed, (
                f"remote {name}.forward applies {sorted(between - allowed)} between lm_head and the "
                f"loss; the fused loss replaces both and would drop it."
            )


def test_the_fused_head_does_not_declare_output_router_logits():
    """The resolver reads that parameter off ``type(model).forward``; declaring it would route GLM-4.7-Flash to
    ``aux_loss``, which it has no coefficient for, so no balancing at all."""
    parameters = inspect.signature(build_lce_forward()).parameters
    assert "output_router_logits" not in parameters, (
        "the fused forward declares output_router_logits; take it from **kwargs instead, or every "
        "family whose native head lacks the parameter loses its router balancing"
    )
    assert "kwargs" in parameters, "it still has to be reachable — the forward must accept **kwargs"


def test_the_router_logits_probe_really_keys_on_that_parameter_name():
    class _Declares(torch.nn.Module):
        def forward(self, output_router_logits=None, **kwargs):
            """A head that consults the flag."""

    class _DoesNot(torch.nn.Module):
        def forward(self, **kwargs):
            """A head that does not."""

    assert honors_router_logits_flag(_Declares()) is True
    assert honors_router_logits_flag(_DoesNot()) is False


def test_the_head_check_rejects_inklings_rescaled_head():
    cls = importlib.import_module("transformers.models.inkling.modeling_inkling").InklingForCausalLM
    between = _attributes_between_head_and_loss(_forward_ast(cls), cls.__name__)
    assert between, "InklingForCausalLM no longer rescales/truncates its logits — re-check the exclusion"
    assert not any(spec.causal_lm for spec in LIGER_FAMILY_SPECS if "inkling_text" in spec.model_types)


def test_the_fused_head_reproduces_the_family_head_bit_for_bit():
    """``skip_logits=False`` needs no Triton and shares the fused branch's wiring, so any drift shows as a non-zero
    delta. Subprocess: installing the forward rebinds the upstream classes for the rest of the process."""
    script = """
import inspect, sys, torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.common.liger_appliers import liger_applier
from tests.common.models import TINY_COHERE2_MOE_CONFIG, TINY_GLM4_MOE_LITE_CONFIG, TINY_QWEN4_EXP_CONFIG

failures = []
for model_type, tiny in (
    ("cohere2_moe", TINY_COHERE2_MOE_CONFIG),
    ("glm4_moe_lite", TINY_GLM4_MOE_LITE_CONFIG),
    ("qwen4_exp_text", TINY_QWEN4_EXP_CONFIG),
):
    config = CONFIG_MAPPING[model_type](**{**tiny, "attn_implementation": "eager"})
    torch.manual_seed(0)
    reference = AutoModelForCausalLM.from_config(config).float()
    reference.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    reference_out = reference(input_ids=input_ids, labels=input_ids.clone())
    reference_out.loss.backward()
    reference_grads = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}

    applier = liger_applier(model_type)
    names = set(inspect.signature(applier).parameters) - {"model"}
    applier(**{name: name == "fused_linear_cross_entropy" for name in names})

    torch.manual_seed(0)
    patched = AutoModelForCausalLM.from_config(config).float()
    patched.load_state_dict(reference.state_dict())
    patched.train()
    out = patched(input_ids=input_ids, labels=input_ids.clone(), skip_logits=False)
    out.loss.backward()

    if out.loss.item() != reference_out.loss.item():
        failures.append(f"{model_type}: loss {out.loss.item()} != {reference_out.loss.item()}")
    if not torch.equal(out.logits, reference_out.logits):
        failures.append(f"{model_type}: logits differ (a dropped logit_scale looks exactly like this)")
    for name, p in patched.named_parameters():
        if p.grad is not None and not torch.equal(p.grad, reference_grads[name]):
            failures.append(f"{model_type}: gradient {name} differs")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "the fused head does not reproduce the family head:\n" + "\n".join(failures)


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.rope], ids=lambda s: s.model_types[0])
def test_rope_is_only_declared_where_the_rotary_is_ligers(spec):
    """Liger's rotary rotates the full head_dim; partial, GPT-J interleaved and per-layer-type rotaries compute a
    different function."""
    family_rope = importlib.import_module(spec.modeling_module).apply_rotary_pos_emb
    source = inspect.getsource(family_rope)
    assert "rotary_dim" not in source, f"{spec.model_types[0]} gained a partial rotary; rope must go back off"
    assert "::2" not in source, f"{spec.model_types[0]} gained an interleaved rotary; rope must go back off"

    # Gemma 4's rotary takes one tensor at a time; substituting Liger's would be a TypeError at the first forward.
    parameters = list(inspect.signature(family_rope).parameters)
    assert (
        parameters[:4] == list(inspect.signature(liger_rotary_pos_emb).parameters)[:4] == ["q", "k", "cos", "sin"]
    ), (
        f"{spec.model_types[0]}'s apply_rotary_pos_emb takes {parameters}, which liger_rotary_pos_emb "
        f"cannot stand in for"
    )


@pytest.mark.parametrize("spec", LIGER_FAMILY_SPECS, ids=lambda s: s.model_types[0])
def test_the_applier_offers_only_the_roles_the_spec_fills(spec):
    """An offered role the spec cannot serve would be accepted and do nothing; a delegating spec offers upstream's
    roles instead."""
    parameters = set(inspect.signature(liger_applier(spec.model_types[0])).parameters)
    if spec.delegates_to_upstream:
        upstream = set(inspect.signature(MODEL_TYPE_TO_APPLY_LIGER_FN[spec.model_types[0]]).parameters)
        assert parameters == upstream, f"{spec.model_types[0]} no longer offers upstream's kernel set"
        return
    assert ("rms_norm" in parameters) == bool(spec.rms_norm or spec.gated_rms_norm)
    assert ("swiglu" in parameters) == bool(spec.glu_mlp)
    assert ("fused_linear_cross_entropy" in parameters) == bool(spec.causal_lm)


def test_the_fused_head_forwards_the_router_flag_where_the_head_adds_no_aux_loss():
    """Zaya's head adds no aux loss, so refusing the flag would fail the run when router-load metrics ask for the
    logits."""
    calls = []

    class _Backbone:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                last_hidden_state=torch.zeros(1, 2, 4), past_key_values=None, hidden_states=None, attentions=None
            )

    head = types.SimpleNamespace(
        config=types.SimpleNamespace(output_router_logits=False),
        model=_Backbone(),
        lm_head=torch.nn.Linear(4, 3, bias=False),
        training=False,
        loss_function=lambda **kwargs: None,
    )
    build_lce_forward(router_aux_loss_in_head=False)(
        head, input_ids=torch.zeros(1, 2, dtype=torch.long), output_router_logits=True
    )
    assert calls and calls[-1].get("output_router_logits") is True, "the flag never reached the backbone"

    with pytest.raises(ValueError, match="output_router_logits=False"):
        build_lce_forward(router_aux_loss_in_head=True)(
            head, input_ids=torch.zeros(1, 2, dtype=torch.long), output_router_logits=True
        )
    assert {spec.model_types[0] for spec in LIGER_FAMILY_SPECS if spec.router_aux_loss_in_head} == {
        "deepseek_v4",
        "laguna",
        "qwen4_exp_text",
    }

    # The backbone would otherwise resolve the omitted flag from the config.
    head.config.output_router_logits = True
    build_lce_forward(router_aux_loss_in_head=True)(
        head, input_ids=torch.zeros(1, 2, dtype=torch.long), output_router_logits=False
    )
    assert calls[-1].get("output_router_logits") is False, "the explicit False never reached the backbone"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
