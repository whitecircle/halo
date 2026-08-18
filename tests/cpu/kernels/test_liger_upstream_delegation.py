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
* the rules keyed on the applier that RUNS still fire through the delegation — the broken CE branch
  of liger-kernel 0.8.0's Qwen3.5 applier, and the EP fused-GLU decision.

    pytest -m cpu tests/cpu/kernels/test_liger_upstream_delegation.py
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
from accelerate import PartialState
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

from src.kernels.liger import orchestrator
from src.kernels.liger.builder import LigerApplier, LigerFamilySpec, _fused_gated_rms_norm_class
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from tests.common.utils import probe_findings

PartialState()  # the orchestrator logs through accelerate's rank-aware logger


DELEGATING_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.delegates_to_upstream]

# The families the delegation exists for: upstream covers them, their GDN blocks' gated norm is what
# the toolkit adds. Named here so deleting a spec fails rather than shrinking the sweep.
EXPECTED_DELEGATING_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"}

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


def test_the_gated_delta_net_families_delegate():
    """The three families resolve to a toolkit applier that delegates, with the gated norm added."""
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
        assert applier.spec.gated_rms_norm, f"{model_type} delegates but adds no role"


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


def test_the_upstream_broken_cross_entropy_default_survives_delegation(monkeypatch):
    """liger-kernel 0.8.0's Qwen3.5 CE branch raises ImportError; the default-off must follow it.

    The workaround is keyed on the applier's qualified name, and delegation puts a toolkit applier in
    front of it. Reading only the resolved applier's name would default ``cross_entropy`` ON and take
    every Qwen3.5 dense run down at model load.
    """
    assert orchestrator._LIGER_CROSS_ENTROPY_BROKEN_APPLIERS & orchestrator._applier_identities(
        orchestrator._TOOLKIT_LIGER_APPLIERS["qwen3_5"]
    ), "the real qwen3_5 applier no longer reports the broken upstream CE branch it delegates to"

    # Probes rather than the real appliers: resolving CE is all this asserts, and applying for real
    # would rebind the upstream classes for every later test in the session.
    broken = _DelegatingProbe(MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5"])
    healthy = _DelegatingProbe(MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_next"])
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "qwen3_5", broken)
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "qwen3_next", healthy)

    assert orchestrator._apply_liger_for_standard_models("qwen3_5", {})["cross_entropy"] is False
    assert broken.calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,
            "rms_norm": True,
            "swiglu": True,
        }
    ]
    # Anti-vacuity: the default-off is caused by the delegate, not by delegation itself.
    assert orchestrator._apply_liger_for_standard_models("qwen3_next", {})["cross_entropy"] is True


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
    applier = resolve_liger_applier(spec.model_types[0])
    names = set(inspect.signature(applier).parameters) - {"model"}
    applier(**{name: name == "rms_norm" for name in names})
    swapped = {n for n, obj in vars(module).items() if isinstance(obj, type) and before.get(n) is not obj}
    for name in spec.gated_rms_norm:
        role = getattr(getattr(module, name), "_halo_liger_patched_role", None)
        if role != "gated_rms_norm":
            failures.append(f"{spec.model_types[0]}: {name} is not the fla-bound class (role={role})")
    if not swapped - set(spec.gated_rms_norm):
        failures.append(f"{spec.model_types[0]}: upstream's applier swapped nothing — delegation did not run")
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
