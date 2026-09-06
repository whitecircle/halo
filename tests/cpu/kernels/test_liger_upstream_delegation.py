#!/usr/bin/env python
"""Delegating Liger specs: upstream's applier runs, and the toolkit adds one role on top.

Qwen3.5 / Qwen3.6 and Qwen3-Next are covered by upstream liger-kernel except for the gated-delta-net
blocks' gated norm and, on the MoE siblings, the shared-expert MLP. Their specs therefore delegate:
the upstream applier keeps every role it declares, and the spec names only what it leaves eager. What
this file pins is the seam, not the kernel numerics (those are
``tests/gpu/kernels/test_liger_family_kernels.py``):

* the toolkit applier claims the model types AND carries upstream's whole knob surface — a dropped
  knob silently refuses a kernel the family had before the spec existed;
* the delegation is derived from liger-kernel's own registry, so a family it stops covering fails at
  import instead of resolving to a spec that patches one role and reports the rest as applied;
* neither side patches what the other did — not at class level, and not when re-applied on a built
  model — and the roles upstream always swaps are absent from the spec;
* a role the spec takes over (``upstream_off``) is the toolkit's variant, not upstream's — GptOss's
  Gemma-cast norm, Gemma 4's always-on dense MLP;
* cross-entropy never reaches upstream's branch (a process-wide ``F.cross_entropy`` rebind, and broken
  outright in liger-kernel 0.8.0's Qwen3.5 applier): the toolkit's scoped patch serves it;
* the EP fused-GLU decision still fires through the delegation.

    pytest -m cpu tests/cpu/kernels/test_liger_upstream_delegation.py
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
from accelerate import PartialState
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.loss import loss_utils
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

from src.kernels.liger import orchestrator
from src.kernels.liger.builder import LigerApplier, LigerFamilySpec, _fused_gated_rms_norm_class
from src.kernels.liger.cross_entropy import liger_cross_entropy
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from tests.common.utils import probe_findings

PartialState()  # the orchestrator logs through accelerate's rank-aware logger


DELEGATING_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.delegates_to_upstream]

# The families the delegation exists for: upstream covers them, and the toolkit adds their GDN
# blocks' gated norm (the Qwen families) or takes over a role upstream gets wrong for them (GptOss's
# norm casting, Gemma 4's EP-surviving dense MLP). Named here so deleting a spec fails rather than
# shrinking the sweep.
EXPECTED_DELEGATING_TYPES = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
    "gpt_oss",
    "gemma4_text",
}

QWEN3_NEXT_MODELING = "transformers.models.qwen3_next.modeling_qwen3_next"


class _DelegatingProbe:
    """A toolkit applier standing in front of a REAL upstream one, without applying anything.

    Every rule the orchestrator derives from the applier reads the same two things a real
    :class:`LigerApplier` exposes — its signature and its ``upstream`` — so the probe exercises them
    while leaving the process's transformers classes untouched.
    """

    def __init__(self, upstream):
        self.upstream = upstream
        self.calls = []

    def __call__(self, rope=False, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=True, swiglu=True):
        self.calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
            }
        )


def test_the_delegating_families_resolve_on_the_toolkit_branch():
    """Each resolves to a toolkit applier that delegates, with at least one role of its own."""
    delegating = {model_type for spec in DELEGATING_SPECS for model_type in spec.model_types}
    assert delegating == EXPECTED_DELEGATING_TYPES

    for model_type in sorted(delegating):
        applier = orchestrator.resolve_liger_applier(model_type)
        assert applier is orchestrator._TOOLKIT_LIGER_APPLIERS[model_type], (
            f"{model_type} no longer resolves on the toolkit branch, where the added role is applied"
        )
        assert applier.upstream is MODEL_TYPE_TO_APPLY_LIGER_FN[model_type], (
            f"{model_type} does not delegate to the applier liger_kernel registers for it"
        )
        spec = applier.spec
        assert spec.rms_norm or spec.gated_rms_norm or spec.glu_mlp, f"{model_type} delegates but adds no role"


def test_the_taken_over_roles_are_declared_where_upstreams_variant_is_wrong():
    """GptOss's norm and Gemma 4's dense MLP are the two roles the toolkit takes back from upstream.

    Upstream applies the llama-cast ``LigerRMSNorm`` to ``GptOssRMSNorm``, which multiplies its
    weight in fp32 before the cast back (Gemma's mode); and its ``geglu`` swaps ``Gemma4TextMLP``,
    the dense MLP every Gemma-4 decoder layer keeps beside its experts, which the EP force-off would
    otherwise strip on every grouped-GEMM run.
    """
    by_type = {spec.model_types[0]: spec for spec in DELEGATING_SPECS}
    assert by_type["gpt_oss"].upstream_off == ("rms_norm",)
    assert by_type["gpt_oss"].rms_norm == ("GptOssRMSNorm",) and by_type["gpt_oss"].rms_norm_casting_mode == "gemma"
    assert by_type["gemma4_text"].upstream_off == ("geglu",)
    assert by_type["gemma4_text"].glu_mlp == ("Gemma4TextMLP",)
    assert "gemma4_text" in orchestrator._TOOLKIT_GLU_SURVIVES_EP


def test_a_withheld_flag_needs_the_role_it_stands_for():
    """``upstream_off`` without the toolkit filling that role would silently unpatch it."""
    with pytest.raises(ValueError, match="without filling that role"):
        LigerFamilySpec(
            model_types=("qwen3_next",),
            modeling_module=QWEN3_NEXT_MODELING,
            gated_rms_norm=("Qwen3NextRMSNormGated",),
            delegates_to_upstream=True,
            upstream_off=("rms_norm",),
        )
    with pytest.raises(ValueError, match="does not delegate"):
        LigerFamilySpec(
            model_types=("a_native_family",),
            modeling_module=QWEN3_NEXT_MODELING,
            rms_norm=("Qwen3NextRMSNorm",),
            upstream_off=("rms_norm",),
        )


@pytest.mark.parametrize("spec", DELEGATING_SPECS, ids=lambda s: s.model_types[0])
def test_a_delegating_applier_offers_upstreams_knobs_with_the_toolkit_loss_convention(spec):
    """Signature = upstream's names, toolkit's loss defaults.

    The orchestrator reads valid kernels, the ``rope`` auto-off and ``_PER_MODEL_DEFAULTS`` off this
    signature. Narrowing it drops a kernel the family used to get; widening it offers one upstream
    would reject as an unexpected keyword.
    """
    applier = orchestrator._TOOLKIT_LIGER_APPLIERS[spec.model_types[0]]
    parameters = inspect.signature(applier).parameters
    upstream = inspect.signature(applier.upstream).parameters
    assert set(parameters) == set(upstream)
    # Liger's own defaults are FLCE on / CE off; the toolkit's are the reverse, because CE keeps the
    # logits every metric reads. The rest of the knobs stay exactly as upstream declares them.
    assert parameters["cross_entropy"].default is True
    assert parameters["fused_linear_cross_entropy"].default is False
    for name in set(upstream) - {"cross_entropy", "fused_linear_cross_entropy", "model"}:
        assert parameters[name].default == upstream[name].default, f"{name} default drifted from upstream's"


def test_the_moe_families_add_their_shared_expert_mlp():
    """Dropping the shared-expert role would leave it eager on every layer, under EP included.

    Upstream's ``swiglu`` patch sets ``*Experts``, which the EP wrapper replaces wholesale; the
    shared expert it adopts unchanged is fused only because the spec names its MLP class.
    """
    shared_expert_mlps = {
        model_type: spec.glu_mlp
        for spec in DELEGATING_SPECS
        for model_type in spec.model_types
        if spec.model_types[0] in ("qwen3_5_moe", "qwen3_next")
    }
    assert shared_expert_mlps == {
        "qwen3_5_moe": ("Qwen3_5MoeMLP",),
        "qwen3_5_moe_text": ("Qwen3_5MoeMLP",),
        "qwen3_next": ("Qwen3NextMLP",),
    }
    # The dense sibling must NOT declare one: upstream's dense applier class-swaps `Qwen3_5MLP`
    # itself, and the patch-time guard below refuses to stack on top of that.
    dense = next(spec for spec in DELEGATING_SPECS if spec.model_types[0] == "qwen3_5")
    assert not dense.glu_mlp


def test_delegation_to_a_family_upstream_does_not_cover_is_refused():
    """Derived from liger-kernel's registry, so a family it drops fails loudly at build.

    Silently falling back to "patch only the added role" would leave the norms, the fused GLU and the
    head unpatched while the orchestrator reported the whole kernel set as applied.
    """
    spec = LigerFamilySpec(
        model_types=("a_family_liger_does_not_cover",),
        modeling_module=QWEN3_NEXT_MODELING,
        gated_rms_norm=("Qwen3NextRMSNormGated",),
        delegates_to_upstream=True,
    )
    with pytest.raises(ValueError, match="a_family_liger_does_not_cover"):
        LigerApplier(spec)


def test_delegation_across_two_upstream_appliers_is_refused():
    """Aliases of one family share an upstream applier; two different ones is a mis-grouped spec.

    The build would pick one of them and patch the other family's model types with it.
    """
    spec = LigerFamilySpec(
        model_types=("qwen3_next", "qwen3_5_moe"),
        modeling_module=QWEN3_NEXT_MODELING,
        gated_rms_norm=("Qwen3NextRMSNormGated",),
        delegates_to_upstream=True,
    )
    with pytest.raises(ValueError, match="one upstream applier"):
        LigerApplier(spec)


def test_a_delegating_spec_may_still_choose_the_familys_loss_default():
    """The delegating signature carries the toolkit's loss convention, not upstream's own defaults.

    ``flce_default`` is the knob Zaya and DeepSeek-V4 use for a binding logits plane; a delegating
    family reaches the same default through the same field, which is what the branch is for.
    """
    spec = LigerFamilySpec(
        model_types=("qwen3_next",),
        modeling_module=QWEN3_NEXT_MODELING,
        gated_rms_norm=("Qwen3NextRMSNormGated",),
        delegates_to_upstream=True,
        flce_default=True,
    )
    parameters = inspect.signature(LigerApplier(spec)).parameters
    assert parameters["fused_linear_cross_entropy"].default is True
    assert parameters["cross_entropy"].default is False


def test_a_delegating_spec_that_adds_nothing_is_refused():
    """Delegating with no added role is a spec that shadows upstream to do exactly nothing."""
    with pytest.raises(ValueError, match="adds no role"):
        LigerFamilySpec(
            model_types=("qwen3_next",),
            modeling_module=QWEN3_NEXT_MODELING,
            delegates_to_upstream=True,
        )


def test_the_fused_gated_norm_needs_fla_and_says_so(monkeypatch):
    """``fla`` owns the only gated-norm kernel on the roster; its absence must not be a silent skip.

    The import is deferred to the patch (``fla`` costs seconds and probes Triton), so a missing or
    renamed ``FusedRMSNormGated`` surfaces at model load. It has to surface as a failure: swallowing
    it would leave the family running the eager norm while the applier logged the role as patched.
    """
    monkeypatch.setitem(sys.modules, "fla.modules", types.ModuleType("fla.modules"))
    with pytest.raises(ImportError, match="FusedRMSNormGated"):
        _fused_gated_rms_norm_class(Qwen3NextRMSNormGated)


def test_cross_entropy_never_reaches_upstreams_branch():
    """Upstream's CE branch is a process-wide ``F.cross_entropy`` rebind, and liger-kernel 0.8.0's
    Qwen3.5 applier cannot even run it (it imports ``liger_cross_entropy`` from the wrong module).

    Both are moot because a delegating applier withholds the flag from upstream and installs the
    toolkit's scoped patch itself — so the dense Qwen3.5 family keeps a fused loss with CE on.

    Subprocess: applying rebinds the family's classes for the rest of the process.
    """
    script = """
import torch.nn.functional as F
from accelerate import PartialState

PartialState()
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.loss import loss_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.kernels.liger.cross_entropy import _TORCH_CROSS_ENTROPY, liger_cross_entropy
from src.kernels.liger.orchestrator import apply_liger_kernel

seen = []
upstream = MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5"]

def spy(**kwargs):
    seen.append(kwargs.get("cross_entropy"))
    return upstream(**kwargs)

MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5"] = spy
from src.kernels.liger import orchestrator
orchestrator._TOOLKIT_LIGER_APPLIERS["qwen3_5"].upstream = spy

applied = apply_liger_kernel(CONFIG_MAPPING["qwen3_5"]())
failures = []
if applied["cross_entropy"] is not True:
    failures.append(f"cross_entropy defaulted off: {applied}")
if seen != [False]:
    failures.append(f"upstream received cross_entropy={seen}")
if F.cross_entropy is not _TORCH_CROSS_ENTROPY:
    failures.append("torch.nn.functional.cross_entropy was rebound process-wide")
if loss_utils.nn.functional.cross_entropy is not liger_cross_entropy:
    failures.append("the scoped patch did not reach transformers' loss path")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "\n".join(failures)


def test_a_delegating_probe_is_called_with_cross_entropy_withheld(monkeypatch):
    """The orchestrator hands a delegating applier CE on; the applier hands upstream CE off."""
    probe = _DelegatingProbe(MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_next"])
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "qwen3_next", probe)
    original = loss_utils.nn
    try:
        applied = orchestrator._apply_liger_for_standard_models("qwen3_next", {})
    finally:
        loss_utils.nn = original
    assert applied["cross_entropy"] is True
    assert probe.calls == [
        {
            "rope": False,
            "cross_entropy": True,
            "fused_linear_cross_entropy": False,
            "rms_norm": True,
            "swiglu": True,
        }
    ], "a toolkit applier receives the effective config; withholding CE from upstream is its own job"
    assert liger_cross_entropy is not None


def test_re_application_neither_stacks_nor_undoes_the_added_glu():
    """HF Trainer re-applies Liger on the built model — through UPSTREAM's applier, not this one.

    Two failures to rule out on the shared-expert MLP a delegating spec names. Stacking: a second
    pass must not subclass the subclass, or every re-application deepens the MRO. Undoing: upstream's
    instance patch binds its own SwiGLU forward over the toolkit's class swap, and the toolkit's
    ``model=`` path — which runs after it, inside the same call — has to be the final word.

    Subprocess: the swaps are process-global, and this one builds a model on top of them.
    """
    script = """
from accelerate import PartialState

PartialState()
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.kernels.liger.builder import _fused_glu_forward
from src.kernels.liger.orchestrator import resolve_liger_applier
from tests.common.models import TINY_QWEN35_MOE_CONFIG

applier = resolve_liger_applier("qwen3_5_moe_text")
flags = dict(rope=False, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=False, swiglu=True)
applier(**flags)
once = modeling.Qwen3_5MoeMLP
applier(**flags)

config = CONFIG_MAPPING["qwen3_5_moe_text"](
    **{**TINY_QWEN35_MOE_CONFIG, "layer_types": ["full_attention"] * 4, "attn_implementation": "eager"}
)
model = AutoModelForCausalLM.from_config(config)
shared = model.model.layers[0].mlp.shared_expert


def toolkit_owns_forward():
    return getattr(shared.forward, "__func__", None) is _fused_glu_forward or (
        "forward" not in shared.__dict__ and type(shared) is once
    )


failures = []
if modeling.Qwen3_5MoeMLP is not once:
    failures.append("a second application stacked another subclass")
if type(shared) is not once:
    failures.append(f"the shared expert is {type(shared).__name__}, not the patched class")
if not toolkit_owns_forward():
    failures.append("the class swap did not fuse the shared expert")

# What HF Trainer does with `use_liger_kernel` on: upstream's applier, on the built model.
MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5_moe"](model=model, **flags)
if toolkit_owns_forward():
    failures.append("upstream's instance patch did not bind over the class swap — the guard is vacuous")

applier(model=model, **flags)
if not toolkit_owns_forward():
    failures.append("re-applying the toolkit applier left upstream's forward in place")
if modeling.Qwen3_5MoeMLP is not once:
    failures.append("the model= path stacked another subclass")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "re-application is not a no-op:\n" + "\n".join(failures)


def test_the_upstream_applier_actually_runs():
    """Dropping the delegation call must fail here rather than pass as a gated-norm-only patch.

    The proof that upstream ran is a class it swapped that the spec never names: with ``rms_norm`` on
    and every other flag off, upstream replaces the family's plain norm while the toolkit replaces the
    gated one. Read off the module's own class table rather than a named expectation, so it holds for
    whatever upstream patches under that flag.

    Subprocess: applying rebinds the upstream classes for the rest of the process, and a later test
    building one of these families would then see the patched ones.
    """
    script = """
import importlib, inspect
from accelerate import PartialState

PartialState()
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from src.kernels.liger.orchestrator import resolve_liger_applier

failures = []
for spec in [s for s in LIGER_FAMILY_SPECS if s.delegates_to_upstream]:
    module = importlib.import_module(spec.modeling_module)
    before = {name: obj for name, obj in vars(module).items() if isinstance(obj, type)}
    rope_before = getattr(module, "apply_rotary_pos_emb", None)
    applier = resolve_liger_applier(spec.model_types[0])
    names = set(inspect.signature(applier).parameters) - {"model"}
    # The norm flag proves upstream ran unless the spec took that role over, where its rotary does.
    proof = "rope" if "rms_norm" in spec.upstream_off else "rms_norm"
    applier(**{name: name in ("rms_norm", proof) for name in names})
    toolkit_roles = set(spec.gated_rms_norm) | set(spec.rms_norm)
    swapped = {n for n, obj in vars(module).items() if isinstance(obj, type) and before.get(n) is not obj}
    for name in spec.gated_rms_norm:
        role = getattr(getattr(module, name), "_halo_liger_patched_role", None)
        if role != "gated_rms_norm":
            failures.append(f"{spec.model_types[0]}: {name} is not the fla-bound class (role={role})")
    for name in spec.rms_norm:
        cls = getattr(module, name)
        if getattr(cls, "_halo_liger_patched_role", None) != "rms_norm" or cls.__mro__[1] is not before[name]:
            failures.append(f"{spec.model_types[0]}: {name} is not the toolkit's subclass of the stock norm")
    upstream_ran = bool(swapped - toolkit_roles) or (
        proof == "rope" and getattr(module, "apply_rotary_pos_emb", None) is not rope_before
    )
    if not upstream_ran:
        failures.append(f"{spec.model_types[0]}: upstream's applier patched nothing — delegation did not run")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "the delegating appliers did not apply both halves:\n" + "\n".join(failures)


def test_a_role_upstream_also_swaps_is_refused_when_the_patch_runs():
    """The dataclass cannot know which classes upstream claims — the patch site can, and must.

    Upstream's DENSE Qwen3.5 applier class-swaps ``Qwen3_5MLP``. A spec naming it would subclass
    Liger's replacement instead of the family's own MLP: still a fused GLU, but stacked on a class
    the family no longer has, and silently so. The guard reads the class's own ``__module__``, so it
    covers any role and any future upstream swap rather than a list of known collisions.

    Subprocess: proving the guard means letting upstream's applier really swap the class.
    """
    script = """
from accelerate import PartialState

PartialState()
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from src.kernels.liger.builder import LigerApplier, LigerFamilySpec

apply_liger_kernel_to_qwen3_5(rope=False, cross_entropy=False, fused_linear_cross_entropy=False,
                              rms_norm=False, swiglu=True)
spec = LigerFamilySpec(
    model_types=("qwen3_5",),
    modeling_module="transformers.models.qwen3_5.modeling_qwen3_5",
    gated_rms_norm=("Qwen3_5RMSNormGated",),
    glu_mlp=("Qwen3_5MLP",),
    delegates_to_upstream=True,
)
try:
    LigerApplier(spec)(rope=False, cross_entropy=False, fused_linear_cross_entropy=False,
                       rms_norm=True, swiglu=True)
    print("RESULT:no raise")
except ValueError as error:
    print("RESULT:" + ("ok" if "Qwen3_5MLP" in str(error) and "upstream" in str(error) else f"wrong: {error}"))
"""
    assert probe_findings(script, "RESULT:") == ["ok"]


def test_ep_keeps_the_shared_expert_glu_a_delegating_spec_names():
    """The force-off is about WHAT the applier swaps, and a delegating spec now swaps two things.

    Upstream's ``Qwen3_5MoeExperts = LigerExperts`` is replaced by ``patch_moe_model_for_ep`` and goes
    inert; the shared-expert MLP the spec names is adopted by that wrapper unchanged and keeps paying
    on every token. Forcing the flag off would delete the second to save the first.
    """
    moe = types.SimpleNamespace(model_type="qwen3_5_moe", text_config=None, num_experts=64)
    assert orchestrator.liger_ep_disables_fused_glu(True, moe) is False

    # Anti-vacuity, both directions: a non-delegating toolkit family keeps its GLU, and an upstream
    # family — whose only GLU patch IS the routed-expert swap — still loses it.
    laguna = types.SimpleNamespace(model_type="laguna", text_config=None, num_experts=256)
    assert orchestrator.liger_ep_disables_fused_glu(True, laguna) is False
    upstream_only = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=128)
    assert orchestrator.liger_ep_disables_fused_glu(True, upstream_only) is True

    # Gemma 4's dense MLP sits beside the experts in every decoder layer and survives the wrapper,
    # so the delegating spec that names it keeps `geglu` on — through the wrapper's text config too.
    text = types.SimpleNamespace(model_type="gemma4_text", num_experts=128)
    gemma4 = types.SimpleNamespace(
        model_type="gemma4", text_config=text, num_experts=128, get_text_config=lambda: text
    )
    assert orchestrator.liger_ep_disables_fused_glu(True, gemma4) is False
    assert orchestrator.liger_ep_disables_fused_glu(True, text) is False
    # ...while a delegating spec that takes over a different role (GptOss's norm) still loses upstream's
    # routed-expert swap, which is inert under the wrapper anyway.
    gpt_oss = types.SimpleNamespace(model_type="gpt_oss", text_config=None, num_local_experts=32)
    assert orchestrator.liger_ep_disables_fused_glu(True, gpt_oss) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
