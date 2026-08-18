#!/usr/bin/env python
"""Per-family Liger coverage — the roster is covered, and every declared role is EXACTLY equivalent.

Three failures this file exists to catch, none of which shows up in a loss curve:

* a family silently losing its applier (a spec deleted, a model_type renamed) — the run then trains
  unfused, which only the orchestrator's warning would say;
* a family's own module drifting away from what the declared kernel computes — a clamp, a trained
  output scale, a gated or grouped norm, a partial rotary. Liger would still "apply", and the model
  would compute something else;
* a spec picking the wrong Liger variant — ``offset``/``casting_mode`` are what make LigerRMSNorm
  reproduce a llama-style norm or a Gemma-style one, and the wrong pick is a silent numerics change.

The numeric checks compare each family's OWN forward against a pure-torch statement of Liger's
parameterized semantics, so they run on CPU (Liger's kernels are Triton-only) and never patch
anything — the GPU suite checks the kernels themselves.

    pytest -m cpu tests/cpu/kernels/test_liger_family_coverage.py
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.utils import HF_MODULES_CACHE

from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.kernels.liger.builder import _bridge_gated_norm
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from src.kernels.liger.lce_forward import build_lce_forward
from src.kernels.liger.orchestrator import resolve_liger_applier
from src.models.moe_balancing import honors_output_router_logits_config
from tests.common.utils import probe_findings

PartialState()  # the orchestrator logs through accelerate's rank-aware logger


NATIVE_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.modeling_module]
REMOTE_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.remote_classes]

# Where transformers copies a `trust_remote_code` repo's modeling files — read from transformers so
# an HF_HOME redirect (every run in this repo sets one) is followed rather than guessed.
_HF_MODULES_ROOT = Path(HF_MODULES_CACHE)

# The llama-style RMSNorm body `casting_mode="llama", offset=0.0` reproduces exactly.
_LLAMA_RMS_NORM_BODY = (
    "hidden_states.to(torch.float32)",
    "hidden_states.pow(2).mean(-1, keepdim=True)",
    "torch.rsqrt(variance + self.variance_epsilon)",
    "return self.weight * hidden_states.to(input_dtype)",
)

# Multimodal wrappers carry no decoder classes of their own; the orchestrator resolves them through
# `text_config.model_type`, and the wrapper's text tower may be either sibling (LFM-2 VL and
# Cohere2-Vision both default to the DENSE one), so pinning the wrapper to a spec would be wrong.
RESOLVED_THROUGH_TEXT_CONFIG = {
    "gemma4": "gemma4_text",
    "lfm2_vl": "lfm2",
    "cohere2_vision": "cohere2",
}

# The canonical GLU body a `glu_mlp` entry promises: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`.
# Anything else in the forward — a clamp bound, a trained output scale, differently named
# projections — means the fused kernel would not compute what the family computes.
CANONICAL_GLU_ATTRIBUTES = {"down_proj", "act_fn", "gate_proj", "up_proj"}


def _roster_model_types() -> set[str]:
    """Every ``model_type`` the toolkit's EP layer registry claims — the MoE half of the roster."""
    return {model_type for cls in ep_layer_classes() for model_type in cls.HF_MODEL_TYPES}


def _self_attributes(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _forward_ast(cls: type) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(cls.forward))).body[0]


def _liger_rms_norm_reference(x, weight, eps, offset, casting_mode):
    """LigerRMSNorm's semantics in plain torch, selected by the two parameters a spec chooses."""
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if casting_mode == "gemma":
        return (normed * (offset + weight.float())).type_as(x)
    return (offset + weight) * normed.to(x.dtype)


@pytest.mark.parametrize("model_type", sorted(_roster_model_types()))
def test_every_roster_family_resolves_an_applier(model_type):
    """A supported family with no applier trains fully unfused — the regression this pins."""
    if model_type in RESOLVED_THROUGH_TEXT_CONFIG:
        text_type = RESOLVED_THROUGH_TEXT_CONFIG[model_type]
        assert resolve_liger_applier(text_type) is not None, (
            f"{model_type} resolves Liger through its text sub-config {text_type}, which now has no applier"
        )
        return
    assert resolve_liger_applier(model_type) is not None, (
        f"no Liger applier covers {model_type}; add a LigerFamilySpec in src/kernels/liger/families.py"
    )


@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_every_declared_class_exists(spec):
    """A transformers bump that renames a class must fail here, not at model load on a GPU node."""
    module = importlib.import_module(spec.modeling_module)
    for name in spec.rms_norm + spec.gated_rms_norm + spec.glu_mlp + spec.causal_lm:
        assert isinstance(getattr(module, name, None), type), f"{spec.modeling_module} defines no class {name!r}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_declared_norms_are_what_the_chosen_liger_variant_computes(spec, dtype):
    """The family's own forward must equal LigerRMSNorm under the spec's ``offset``/``casting_mode``.

    This is the assertion that separates a llama-style norm from a Gemma-style ``(1 + w)`` one, and
    that rejects a weightless, grouped or gated norm listed by mistake. **bf16 is what pins
    ``casting_mode``**: the two modes differ only in whether the weight multiply happens before or
    after the cast back, so in fp32 they are bit-identical and the parameter would go unchecked.
    """
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
    """Same assertion for the ``trust_remote_code`` families, read off the cached modeling file.

    Their modules cannot be imported standalone (they relative-import their sibling configuration),
    and a revision bump is the one thing that can move them under a pinned toolkit, so the check is
    on the source: the fp32-upcast, weight-multiply body that ``casting_mode="llama", offset=0.0``
    reproduces — and NOT the grouped variant sitting next to it in the same file.
    """
    sources = [
        path.read_text(encoding="utf-8")
        for path in _HF_MODULES_ROOT.rglob("modeling_*.py")
        if all(f"class {name}(" in path.read_text(encoding="utf-8") for name in spec.remote_classes)
    ]
    if not sources:
        pytest.skip(f"no cached modeling module defines {spec.remote_classes}")
    assert spec.rms_norm_casting_mode == "llama" and spec.rms_norm_offset == 0.0
    for source in sources:
        for name in spec.rms_norm:
            body = source.split(f"class {name}(")[1].split("\nclass ")[0]
            for required in _LLAMA_RMS_NORM_BODY:
                assert required in body, f"{name} no longer computes {required!r}; the Liger swap would change it"


def test_the_norm_check_rejects_the_wrong_liger_variant():
    """Anti-vacuity: the parameters have to matter, or the test above passes on any norm.

    Step-3.7's Gemma-style norm scales by ``1 + w``; asserting it against the llama parameterization
    (offset 0) must fail, and Laguna's llama-style norm against Gemma's (offset 1) likewise.
    """
    step = importlib.import_module("transformers.models.step3p7.modeling_step3p7").Step3p7RMSNorm(16, 1e-5)
    laguna = importlib.import_module("transformers.models.laguna.modeling_laguna").LagunaRMSNorm(16, 1e-5)
    step.weight.data.normal_()
    laguna.weight.data.normal_()
    x = torch.randn(3, 5, 16)
    assert not torch.allclose(step(x), _liger_rms_norm_reference(x, step.weight, 1e-5, 0.0, "llama"))
    assert not torch.allclose(laguna(x), _liger_rms_norm_reference(x, laguna.weight, 1e-5, 1.0, "gemma"))

    # ...and that `casting_mode` alone is discriminating, which only bf16 shows: swapping just the
    # mode (same offset) must change the result, or the fp32 case above pins offset and nothing else.
    step_bf16 = importlib.import_module("transformers.models.step3p7.modeling_step3p7").Step3p7RMSNorm(16, 1e-5)
    step_bf16.weight.data.normal_()
    step_bf16 = step_bf16.to(torch.bfloat16)
    xb = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    gemma = _liger_rms_norm_reference(xb, step_bf16.weight, 1e-5, 1.0, "gemma")
    llama = _liger_rms_norm_reference(xb, step_bf16.weight, 1e-5, 1.0, "llama")
    assert not torch.equal(gemma, llama), "casting_mode makes no difference here — the check is blind to it"


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.rms_norm], ids=lambda s: s.model_types[0])
def test_no_norm_output_is_a_direct_operand_of_an_addition(spec):
    """Liger's llama-mode RMSNorm backward writes dX INTO the incoming dY buffer (``in_place=True``).

    ``AddBackward`` hands the SAME gradient object to both of its inputs, so a sandwich norm —
    ``residual + post_norm(x)`` — would have its sibling branch's gradient overwritten in place.
    Silent, and invisible in the loss. Every family here is pre-norm (the norm output feeds a
    sublayer, never an add), which is what makes ``in_place=True`` safe; a transformers bump adding a
    post-norm must fail here and switch that family to ``in_place=False``.

    Deliberately conservative: it flags ANY ``self.*norm*(...)`` call in an addition, declared or not.
    A false positive costs one extra backward buffer, a false negative costs correctness.
    """
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
    """The bridge to ``fla``'s kernel is four attributes; a class missing one is a load-time crash.

    It reads ``weight``, ``bias``, ``activation`` and ``eps``. The family spells the epsilon
    ``variance_epsilon`` and registers no bias, so the patch bridges both — but ``activation`` is
    passed through as the family's OWN, which is what makes the fused gate provably the eager one.
    The numerics live in the GPU suite; this pins the contract that lets it run at all.
    """
    module = importlib.import_module(spec.modeling_module)
    for name in spec.gated_rms_norm:
        norm = getattr(module, name)(16, 1e-5)
        assert "gate" in inspect.signature(norm.forward).parameters, f"{name}.forward takes no gate"
        assert getattr(norm, "weight", None) is not None, f"{name} has no weight to scale by"
        assert getattr(norm, "bias", None) is None, f"{name} carries a bias the eager norm never applies"
        assert getattr(norm, "activation", None), f"{name} declares no activation for the fused gate"
        assert getattr(norm, "variance_epsilon", getattr(norm, "eps", None)) is not None
        # The real bridge, not a restatement of it: it also refuses a gate `fla`'s kernel does not
        # implement, which that kernel would otherwise answer by applying no gate at all.
        _bridge_gated_norm(norm)


def test_a_grouped_gated_norm_is_never_declared():
    """Anti-vacuity for the role: the shape that must stay OFF the list is the grouped one.

    Bailing's ``*GroupRMSNorm`` gates the same way but reduces over ``hidden_size //
    group_norm_size``, so a kernel normalizing the last dim would reduce the wrong axis and still
    return the right shape. It is the reason the role is a separate spec field rather than a flag.
    """
    declared = {name for spec in LIGER_FAMILY_SPECS for name in spec.gated_rms_norm}
    assert declared, "no family declares a gated norm — this check would pass on an empty set"
    assert not [name for name in declared if "Group" in name], f"a grouped gated norm is declared: {declared}"


@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=lambda s: s.model_types[0])
def test_declared_glu_mlps_are_the_canonical_gated_body(spec):
    """A declared GLU MLP must touch nothing but the three projections and its activation.

    The traps on this roster are all extra terms: Inkling multiplies the output by a trained
    ``global_scale``, GLM-5 Next and Step-3.7 clamp at a ``swiglu_limit``, LFM-2 projects through
    ``w1``/``w3``/``w2`` with no ``act_fn`` at all. Each would be silently dropped or missed by the
    fused forward, which is why none of them is declared.
    """
    module = importlib.import_module(spec.modeling_module)
    for name in spec.glu_mlp:
        attributes = _self_attributes(_forward_ast(getattr(module, name)))
        assert attributes == CANONICAL_GLU_ATTRIBUTES, (
            f"{name}.forward touches {sorted(attributes)}, not the canonical "
            f"{sorted(CANONICAL_GLU_ATTRIBUTES)} — the fused GLU would compute a different function"
        )


def test_the_glu_check_rejects_a_scaled_or_clamped_body():
    """Anti-vacuity: the three MLPs deliberately left undeclared must all fail the same check."""
    undeclared = {
        "transformers.models.inkling.modeling_inkling": "InklingMLP",  # * global_scale
        "transformers.models.glm5_next.modeling_glm5_next": "Glm5NextTextMLP",  # clamp(swiglu_limit)
        "transformers.models.lfm2.modeling_lfm2": "Lfm2MLP",  # w1 / w3 / w2, hardcoded F.silu
    }
    for module_path, name in undeclared.items():
        cls = getattr(importlib.import_module(module_path), name)
        assert _self_attributes(_forward_ast(cls)) != CANONICAL_GLU_ATTRIBUTES, (
            f"{name} now matches the canonical GLU body; if that is real, declare it — but the check "
            f"above is no longer proving anything about the ones that do not"
        )


@pytest.mark.parametrize("spec", [s for s in NATIVE_SPECS if s.causal_lm], ids=lambda s: s.model_types[0])
def test_declared_heads_do_nothing_the_fused_loss_would_drop(spec):
    """Between ``lm_head`` and the loss, a declared head may touch only its declared logit scale.

    The fused loss replaces both calls, so anything applied in between vanishes: Cohere's
    ``logit_scale`` (declared, and folded onto the hidden states), Inkling's
    ``/logits_mup_width_multiplier`` and ``unpadded_vocab_size`` truncation (undeclared, which is
    why Inkling ships no fused head).
    """
    module = importlib.import_module(spec.modeling_module)
    allowed = {spec.logit_scale_attr} if spec.logit_scale_attr else set()
    for name in spec.causal_lm:
        body = _forward_ast(getattr(module, name)).body
        head = next(i for i, stmt in enumerate(body) if "lm_head" in _self_attributes(stmt))
        loss = next(i for i, stmt in enumerate(body) if "loss_function" in _self_attributes(stmt))
        assert head < loss, f"{name}.forward calls loss_function before lm_head"
        between = set().union(*(_self_attributes(stmt) for stmt in body[head + 1 : loss])) - {"config", "vocab_size"}
        assert between <= allowed, (
            f"{name}.forward applies {sorted(between - allowed)} between lm_head and the loss; the "
            f"fused loss replaces both and would drop it. Declare it on the spec or drop causal_lm."
        )


def test_the_fused_head_does_not_declare_output_router_logits():
    """Declaring it would silently disable router balancing on families whose own head lacks it.

    ``honors_output_router_logits_config`` decides whether a family's aux loss can reach the loss by
    looking for that parameter on ``type(model).forward``. Liger patches the forward at model load,
    balancing resolves later — so a declared parameter makes the resolver believe a family that
    never consults the flag honors it. For GLM-4.7-Flash (fused head on by default) that signal is
    the ONLY one routing ``moe_balancing: auto`` to ``bias_update``; losing it falls through to
    ``aux_loss``, which its config carries no ``router_aux_loss_coef`` for — no balancing at all.
    """
    parameters = inspect.signature(build_lce_forward()).parameters
    assert "output_router_logits" not in parameters, (
        "the fused forward declares output_router_logits; take it from **kwargs instead, or every "
        "family whose native head lacks the parameter loses its router balancing"
    )
    assert "kwargs" in parameters, "it still has to be reachable — the forward must accept **kwargs"


def test_the_router_logits_probe_really_keys_on_that_parameter_name():
    """Anti-vacuity for the check above: the probe must be sensitive to the declaration."""

    class _Declares:
        def forward(self, output_router_logits=None, **kwargs):
            """A head that consults the flag."""

    class _DoesNot:
        def forward(self, **kwargs):
            """A head that does not."""

    assert honors_output_router_logits_config(_Declares()) is True
    assert honors_output_router_logits_config(_DoesNot()) is False


def test_the_head_check_rejects_inklings_rescaled_head():
    """Anti-vacuity: Inkling's head is the shape the check must reject, and it is undeclared."""
    cls = importlib.import_module("transformers.models.inkling.modeling_inkling").InklingForCausalLM
    body = _forward_ast(cls).body
    head = next(i for i, stmt in enumerate(body) if "lm_head" in _self_attributes(stmt))
    loss = next(i for i, stmt in enumerate(body) if "loss_function" in _self_attributes(stmt))
    between = set().union(*(_self_attributes(stmt) for stmt in body[head + 1 : loss])) - {"config", "vocab_size"}
    assert between, "InklingForCausalLM no longer rescales/truncates its logits — re-check the exclusion"
    assert not any(spec.causal_lm for spec in LIGER_FAMILY_SPECS if "inkling_text" in spec.model_types)


def test_the_fused_head_reproduces_the_family_head_bit_for_bit():
    """The generic head, driven down its unfused branch, must equal the family's own head exactly.

    ``skip_logits=False`` takes the branch that needs no Triton, which is the whole wiring the fused
    branch shares: the ``logits_to_keep`` slice, the ``logit_scale`` fold, and the ``loss_function``
    call. Any drift there — a dropped scale, a wrong ``vocab_size``, a mis-shifted label — shows up
    as a non-zero delta on a tiny random model. Cohere2 MoE carries the scale (0.0625); GLM-4.7-Flash
    is the plain shape.

    Subprocess: installing the forward rebinds the upstream classes for the rest of the process, and
    a later test in the session building one of these families would take the fused branch on CPU.
    """
    script = """
import inspect, sys, torch
from accelerate import PartialState

PartialState()
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.kernels.liger.orchestrator import resolve_liger_applier
from tests.common.models import TINY_COHERE2_MOE_CONFIG, TINY_GLM4_MOE_LITE_CONFIG

failures = []
for model_type, tiny in (("cohere2_moe", TINY_COHERE2_MOE_CONFIG), ("glm4_moe_lite", TINY_GLM4_MOE_LITE_CONFIG)):
    config = CONFIG_MAPPING[model_type](**{**tiny, "attn_implementation": "eager"})
    torch.manual_seed(0)
    reference = AutoModelForCausalLM.from_config(config).float()
    reference.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    reference_out = reference(input_ids=input_ids, labels=input_ids.clone())
    reference_out.loss.backward()
    reference_grads = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}

    applier = resolve_liger_applier(model_type)
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
    """Liger's rotary rotates the FULL head_dim over concatenated halves.

    Every other rotary on the roster — partial (``rotary_dim`` split), GPT-J interleaved (``::2``),
    per-layer-type — is a different function that the kernel would compute wrongly and silently.
    """
    family_rope = importlib.import_module(spec.modeling_module).apply_rotary_pos_emb
    source = inspect.getsource(family_rope)
    assert "rotary_dim" not in source, f"{spec.model_types[0]} gained a partial rotary; rope must go back off"
    assert "::2" not in source, f"{spec.model_types[0]} gained an interleaved rotary; rope must go back off"

    # Gemma 4's rotary takes ONE tensor at a time; substituting Liger's would be a TypeError at the
    # first forward, so the leading positional parameters must line up.
    parameters = list(inspect.signature(family_rope).parameters)
    assert (
        parameters[:4] == list(inspect.signature(liger_rotary_pos_emb).parameters)[:4] == ["q", "k", "cos", "sin"]
    ), (
        f"{spec.model_types[0]}'s apply_rotary_pos_emb takes {parameters}, which liger_rotary_pos_emb "
        f"cannot stand in for"
    )


@pytest.mark.parametrize("spec", LIGER_FAMILY_SPECS, ids=lambda s: s.model_types[0])
def test_the_applier_offers_only_the_roles_the_spec_fills(spec):
    """An offered role the spec cannot serve would be accepted and then do nothing at all.

    A delegating spec offers UPSTREAM's roles instead — dropping one would silently refuse a kernel
    the family had before the spec existed, and adding one would offer a knob nothing reads.
    """
    parameters = set(inspect.signature(resolve_liger_applier(spec.model_types[0])).parameters)
    if spec.delegates_to_upstream:
        upstream = set(inspect.signature(MODEL_TYPE_TO_APPLY_LIGER_FN[spec.model_types[0]]).parameters)
        assert parameters == upstream, f"{spec.model_types[0]} no longer offers upstream's kernel set"
        return
    assert ("rms_norm" in parameters) == bool(spec.rms_norm or spec.gated_rms_norm)
    assert ("swiglu" in parameters) == bool(spec.glu_mlp)
    assert ("fused_linear_cross_entropy" in parameters) == bool(spec.causal_lm)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
