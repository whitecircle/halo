"""CPU tests for the Liger effective-config seam.

The orchestrator (``apply_liger_kernel``) layers per-model defaults (e.g. Zaya's FLCE-on) and safety
filters (EP swiglu-off, TP CE/FLCE-off) over the user's ``liger_kernel_config``, then records the
EFFECTIVE applied config on the model config (``_halo_liger_applied_config``).
``finalize_liger_after_direct_load`` must decide TRL's ``use_liger_kernel`` flag from that effective
config — deciding from the raw user dict either flips the flag off while the model IS FLCE-patched
(Zaya default, no user dict → TRL reads 1-token-slice logits) or leaves it on while TP forced FLCE off.

    python tests/cpu/kernels/test_liger_effective_config.py
"""

import logging
import sys
import types
from pathlib import Path

import pytest
from accelerate import PartialState
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN

from src.kernels.liger import orchestrator
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from src.models.loading.model_preparation import finalize_liger_after_direct_load
from src.models.moe_balancing import has_ep_wrapper_class

PartialState()  # the orchestrator's logger needs accelerate state initialized

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _RecordingApplier:
    """Stands in for a per-model Liger applier; records the kwargs it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, rope=True, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=True, swiglu=True):
        self.calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
            }
        )


class _FlceOnlyRecordingApplier:
    """Stands in for an FLCE-only applier (deepseek_v4 / glm4_moe_lite signature shape)."""

    def __init__(self):
        self.calls = []

    def __call__(self, fused_linear_cross_entropy=True, model=None):
        self.calls.append({"fused_linear_cross_entropy": fused_linear_cross_entropy})


def _zaya_like_config():
    return types.SimpleNamespace(model_type="zaya", text_config=None)


def test_per_model_defaults_recorded_as_effective(monkeypatch):
    """Zaya defaults FLCE on with NO user dict — the recorded effective config must say so."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = orchestrator.apply_liger_kernel(config, liger_kernel_config=None)
    assert applied is not None
    assert applied["fused_linear_cross_entropy"] is True
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied
    assert applier.calls == [applied]  # recorded config is exactly what the applier ran with


def test_tp_force_off_recorded_as_effective(monkeypatch):
    """Under TP the orchestrator forces CE/FLCE off even when explicitly enabled — the effective
    config must reflect the force-off, not the user's request."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, tp_size=2)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_cp_force_off_recorded_as_effective(monkeypatch):
    """Under CP the fused path is unreachable, so the effective config must say so.

    The Ulysses wrapper calls the model with ``labels=None`` and computes a boundary-aware loss from
    ``outputs.logits`` itself. Every Liger ``lce_forward`` gates on
    ``skip_logits = self.training and labels is not None``, so FLCE never engages — leaving the flag
    on reports a memory saving that does not exist and makes the trainer pin ``use_liger_kernel``
    on that false basis. ``cross_entropy`` goes too: its patch rebinds
    ``torch.nn.functional.cross_entropy`` process-wide, under CP's own fp32 loss included.
    """
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, cp_size=4)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_pp_force_off_recorded_as_effective(monkeypatch):
    """Under PP the fused path is unreachable for the same reason it is under CP.

    The pipeline drives the stage with ``input_ids``/``attention_mask`` only — labels ride to the
    schedule's ``loss_fn`` as the target — so ``skip_logits = self.training and labels is not None``
    never fires and FLCE reports a memory saving that does not exist. ``cross_entropy`` matters more
    here than being merely inert: its patch rebinds ``F.cross_entropy`` process-wide, and the PP
    last-stage loss (``losses._chunked_ce_sum``) calls exactly that function.
    """
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, pp_size=2)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_pp_off_leaves_the_family_default_alone(monkeypatch):
    """Anti-vacuity: pp_size=1 must NOT disable FLCE, or the test above passes on a constant."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    applied = orchestrator.apply_liger_kernel(_zaya_like_config(), None, pp_size=1)
    assert applied["fused_linear_cross_entropy"] is True, (
        "Zaya's family default is FLCE-on; pp_size=1 must leave it on so the PP force-off is provably caused by PP."
    )


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ({"tp_size": 2}, {"cross_entropy", "fused_linear_cross_entropy"}),
        ({"cp_size": 2}, {"cross_entropy", "fused_linear_cross_entropy"}),
        ({"pp_size": 2}, {"cross_entropy", "fused_linear_cross_entropy"}),
        ({"has_ep_wrapped_experts": True}, {"swiglu", "geglu"}),
        ({}, set()),
    ],
)
def test_parallelism_overrides_are_the_one_home_for_the_decision(axis, expected):
    """Both Liger application sites read this function, so it is where the rules must be right.

    Liger is applied at model load AND re-sanitized by the trainer mixin before TRL can re-apply it
    on the wrapped model. If each site restated the rules, an axis handled at one and not the other
    would be silently undone by whichever runs second. Asserting the axis→kernel mapping here pins
    the shared decision.
    """
    assert set(orchestrator.liger_parallelism_overrides(**axis)) == expected


def test_both_sites_fold_the_overrides_with_the_same_semantics():
    """Sharing the rule TABLE is not enough — both sites must APPLY it the same way.

    The fold exempts an explicitly requested ``swiglu``/``geglu`` (inert under EP, not wrong) and
    forces a wrong-loss kernel off regardless. A blanket ``dict.fromkeys(forced_off, False)`` at
    either site stamps that exemption away with no warning, so an explicit ``swiglu: true`` would
    survive model load and be silently reverted before TRL re-applies Liger.
    """
    forced_off = orchestrator.liger_parallelism_overrides(has_ep_wrapped_experts=True, tp_size=2)

    explicit = orchestrator.apply_liger_parallelism_overrides({"swiglu": True, "cross_entropy": True}, forced_off)
    assert explicit["swiglu"] is True, "an explicitly requested swiglu must survive the EP override"
    assert explicit["cross_entropy"] is False, "TP makes the fused softmax a partial-vocab slice — not negotiable"

    # Anti-vacuity: without the explicit request the same override DOES turn swiglu off.
    implicit = orchestrator.apply_liger_parallelism_overrides({}, forced_off)
    assert implicit["swiglu"] is False, "unrequested swiglu must still be disabled under EP"


def test_trainer_mixin_folds_through_the_shared_applier():
    """The mixin must fold via ``apply_liger_parallelism_overrides``, not restate the fold itself.

    Source-level because the fold sits mid-``_init_distributed_config``, behind mixed-precision and
    filesystem setup that a unit call would have to fake. The failure it guards is structural, not
    numeric: ``dict.fromkeys(forced_off, False)`` is a *correct-looking* fold that drops the
    explicit-request exemption the load-time site honors.
    """
    mixin_src = (_REPO_ROOT / "src" / "trainers" / "mixins" / "base.py").read_text()
    assert "apply_liger_parallelism_overrides(" in mixin_src, (
        "the trainer mixin must fold the parallelism overrides through the shared applier"
    )
    assert "dict.fromkeys(forced_off" not in mixin_src, (
        "re-stating the fold drops the swiglu/geglu explicit-request exemption applied at model load"
    )


def test_ep_swiglu_force_off_only_applies_to_moe_models(monkeypatch):
    """The EP swiglu/geglu force-off must key on the model being MoE, not on ``needs_ep_wrappers``.

    ``needs_ep_wrappers`` is ``ep_group_size > 1 or use_grouped_gemm``, and ``use_grouped_gemm``
    defaults True — so it is set on essentially every run, dense ones included. The loader installs EP
    wrappers only for MoE models, so gating Liger on the flag alone silently strips fused SwiGLU from
    every dense model in the repo while logging that EP wrappers were the reason.
    """
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", _RecordingApplier())
    moe_config = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    moe_applied = orchestrator.apply_liger_kernel(moe_config, None, needs_ep_wrappers=True)
    assert moe_applied["swiglu"] is False, "MoE + EP wrappers must still disable Liger's fused MLP"

    dense_config = types.SimpleNamespace(model_type="qwen3_moe", text_config=None)  # no expert count → dense
    dense_applied = orchestrator.apply_liger_kernel(dense_config, None, needs_ep_wrappers=True)
    assert dense_applied["swiglu"] is True, "dense model lost Liger's fused SwiGLU to an EP-only filter"


def test_ep_swiglu_force_off_requires_an_ep_wrapper_class_for_the_family(monkeypatch):
    """``needs_ep_wrappers`` is the run's INTENT; only the registry says a wrapper exists for the family.

    Mixtral declares no EP MoE layer, so patching wraps nothing and the stock expert loop runs —
    forcing Liger's swiglu off there removes the only fused expert path the run has, on the intent
    alone. A family that DOES have a wrapper must still lose it.
    """
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "mixtral", _RecordingApplier())
    config = types.SimpleNamespace(model_type="mixtral", text_config=None, num_experts=64)
    applied = orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=True)
    assert applied["swiglu"] is True, "no EP wrapper class exists for mixtral — nothing replaces the FFN"

    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", _RecordingApplier())
    config = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    applied = orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=True)
    assert applied["swiglu"] is False, "EPQwen3MoELayer wraps this family — Liger's fused MLP is inert"


def test_ep_does_not_strip_a_toolkit_appliers_dense_and_shared_expert_glu(monkeypatch):
    """The EP force-off must key on WHAT the applier swaps, not merely on the family being wrapped.

    Upstream's MoE appliers set the ROUTED-experts class, which ``patch_moe_model_for_ep`` replaces —
    inert under EP, hence the force-off. A toolkit spec names the dense and shared-expert MLPs, and
    every EP wrapper adopts the shared expert unchanged, so the same force-off would delete a fusion
    that still runs on every token of every layer — in exactly the EP configuration these families
    are trained in.
    """
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "laguna", applier)
    config = types.SimpleNamespace(model_type="laguna", text_config=None, num_experts=256)
    applied = orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=True)
    assert applied["swiglu"] is True, "EPLagunaMoELayer adopts LagunaMLP as its shared expert unchanged"

    # Anti-vacuity: the exemption belongs to the spec that names the surviving MLPs, not to a blanket
    # EP no-op — an applier whose only GLU patch is the routed-expert swap still loses it.
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", _RecordingApplier())
    upstream = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    assert orchestrator.apply_liger_kernel(upstream, None, needs_ep_wrappers=True)["swiglu"] is False

    # A DELEGATING spec that names its family's shared-expert MLP keeps the flag for the same reason
    # Laguna does: the EP wrapper adopts that module unchanged.
    delegating = types.SimpleNamespace(model_type="qwen3_5_moe", text_config=None, num_experts=64)
    assert orchestrator.liger_ep_disables_fused_glu(True, delegating) is False


def test_has_ep_wrapper_class_reads_the_registry_including_the_text_config():
    """Derived from the EP layer registry, and resolved through a composite wrapper's text sub-config.

    Imported from its home in the EP package rather than through the Liger orchestrator: the
    predicate answers "does this family have an EP wrapper class", which the registry owns.
    """
    assert has_ep_wrapper_class(types.SimpleNamespace(model_type="qwen3_5_moe")) is True
    assert has_ep_wrapper_class(types.SimpleNamespace(model_type="qwen3_next")) is False
    composite = types.SimpleNamespace(
        model_type="some_multimodal_wrapper",
        get_text_config=lambda: types.SimpleNamespace(model_type="qwen3_5_moe_text"),
    )
    assert has_ep_wrapper_class(composite) is True


def test_user_requested_flce_yields_the_defaulted_cross_entropy(monkeypatch):
    """FLCE alone must not reach the applier with the DEFAULTED cross_entropy, which every applier
    asserts against — the config would die inside liger_kernel on a flag the user never set."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": True})
    assert applied["fused_linear_cross_entropy"] is True
    assert applied["cross_entropy"] is False
    assert applier.calls == [applied], "the applier must actually have run with the resolved pair"


def test_cross_entropy_default_stays_on_without_an_flce_request(monkeypatch):
    """Anti-vacuity: the yield is caused by the FLCE request, not by a blanket CE-off."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = orchestrator.apply_liger_kernel(config, None)
    assert applied["cross_entropy"] is True and applied["fused_linear_cross_entropy"] is False


def test_user_setting_both_loss_kernels_is_refused_by_name(monkeypatch):
    """Two EXPLICIT requests are a contradiction the orchestrator cannot resolve — it must name both
    knobs rather than let a bare assert fire from inside liger_kernel."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    with pytest.raises(ValueError, match="cross_entropy"):
        orchestrator.apply_liger_kernel(config, {"cross_entropy": True, "fused_linear_cross_entropy": True})
    assert applier.calls == [], "the contradiction must be caught before the applier runs"


def test_explicit_cross_entropy_false_with_flce_still_applies(monkeypatch):
    """The spelled-out pair (what the shipped YAMLs write) keeps working unchanged."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = orchestrator.apply_liger_kernel(config, {"cross_entropy": False, "fused_linear_cross_entropy": True})
    assert applied["cross_entropy"] is False and applied["fused_linear_cross_entropy"] is True
    assert applier.calls == [applied]


def test_cp_off_leaves_the_family_default_alone(monkeypatch):
    """Anti-vacuity: cp_size=1 must NOT disable FLCE, or the test above passes on a constant."""
    applier = _RecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    applied = orchestrator.apply_liger_kernel(_zaya_like_config(), None, cp_size=1)
    assert applied["fused_linear_cross_entropy"] is True, (
        "Zaya's family default is FLCE-on; cp_size=1 must leave it on so the CP force-off is provably caused by CP."
    )


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm4_moe_lite"])
def test_flce_only_applier_defaults_flce_on(monkeypatch, model_type):
    """FLCE-only appliers must resolve fused_linear_cross_entropy=True with NO user dict — the generic
    flce=False default would make `use_liger_kernel: true` a total no-op for these families."""
    applier = _FlceOnlyRecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, model_type, applier)
    config = types.SimpleNamespace(model_type=model_type, text_config=None)
    applied = orchestrator.apply_liger_kernel(config, liger_kernel_config=None)
    assert applied == {"fused_linear_cross_entropy": True}
    assert config._halo_liger_applied_config == applied
    assert applier.calls == [applied]


def test_flce_only_applier_user_override_and_tp_force_off_win(monkeypatch):
    """User flce=False and the TP force-off must still beat the FLCE-only default."""
    applier = _FlceOnlyRecordingApplier()
    monkeypatch.setitem(orchestrator._TOOLKIT_LIGER_APPLIERS, "deepseek_v4", applier)
    config = types.SimpleNamespace(model_type="deepseek_v4", text_config=None)
    applied = orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": False})
    assert applied == {"fused_linear_cross_entropy": False}

    config = types.SimpleNamespace(model_type="deepseek_v4", text_config=None)
    applied = orchestrator.apply_liger_kernel(config, liger_kernel_config=None, tp_size=2)
    assert applied == {"fused_linear_cross_entropy": False}


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm4_moe_lite", "zaya"])
def test_the_large_vocab_families_still_default_to_flce(model_type):
    """The families whose logits plane is the binding memory limit must keep FLCE as their default.

    Their appliers also carry the other roles (RMSNorm, fused GLU, scoped CE), so an FLCE-only
    *signature* is not what marks the default — ``_PER_MODEL_DEFAULTS`` is. Losing it silently puts
    GLM-4.7-Flash, DeepSeek-V4 and Zaya back on the full ``[seq, vocab]`` logits plane at the context
    lengths they are trained at.
    """
    defaults = orchestrator._PER_MODEL_DEFAULTS[model_type]
    assert defaults == {"cross_entropy": False, "fused_linear_cross_entropy": True}


def test_every_toolkit_applier_is_registered_in_the_toolkit_registry():
    """One registry, not two: a family self-registering into liger_kernel's own dict would resolve on
    the upstream branch, where ``_PER_MODEL_DEFAULTS`` and the FLCE-only rule never look.

    A DELEGATING spec is the one shape that must appear in both — it exists to run upstream's applier
    — and it must resolve on the toolkit branch, which is what puts the added role on top.
    """
    declared = {model_type for spec in LIGER_FAMILY_SPECS for model_type in spec.model_types}
    delegating = {
        model_type for spec in LIGER_FAMILY_SPECS if spec.delegates_to_upstream for model_type in spec.model_types
    }
    assert declared == set(orchestrator._TOOLKIT_LIGER_APPLIERS)
    assert delegating <= set(MODEL_TYPE_TO_APPLY_LIGER_FN), (
        f"{sorted(delegating - set(MODEL_TYPE_TO_APPLY_LIGER_FN))} delegate to an upstream applier that "
        f"liger_kernel no longer registers"
    )
    leaked = (declared - delegating) & set(MODEL_TYPE_TO_APPLY_LIGER_FN)
    assert not leaked, f"toolkit families also registered into liger_kernel's registry: {sorted(leaked)}"


def test_no_applier_warns_and_records_nothing(monkeypatch, caplog):
    """``use_liger_kernel`` defaults ON, so an uncovered family must SAY it is training unfused."""
    monkeypatch.setattr(orchestrator, "_TOOLKIT_LIGER_APPLIERS", {})
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    with caplog.at_level(logging.WARNING, logger="src.kernels.liger.orchestrator"):
        applied = orchestrator.apply_liger_kernel(config, None)
    assert applied is None
    assert config._halo_liger_applied_config is None
    assert any(
        record.levelno >= logging.WARNING and "model_type_without_any_applier" in record.getMessage()
        for record in caplog.records
    ), f"uncovered family trained unfused with no warning: {[r.getMessage() for r in caplog.records]}"


def test_no_applier_refuses_an_explicit_kernel_request(monkeypatch):
    """A named kernel nothing can deliver is a config error, not a log line."""
    monkeypatch.setattr(orchestrator, "_TOOLKIT_LIGER_APPLIERS", {})
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    with pytest.raises(ValueError, match="fused_linear_cross_entropy"):
        orchestrator.apply_liger_kernel(config, {"fused_linear_cross_entropy": True})


def test_the_ep_swiglu_force_off_is_not_read_as_a_user_request(monkeypatch):
    """Anti-regression on the raise above: under EP the fold writes ``swiglu: False`` into the config
    even with no user dict. Reading THAT as an explicit request would turn every uncovered family
    into a hard failure the moment EP is enabled.

    ``ep_wraps_experts`` is forced True because an uncovered family has no EP wrapper class either,
    so the real predicate answers False and the fold would never run — the test would then pass on a
    config nothing was folded into, which is exactly the vacuity it is written to avoid.
    """
    monkeypatch.setattr(orchestrator, "_TOOLKIT_LIGER_APPLIERS", {})
    monkeypatch.setattr(orchestrator, "ep_wraps_experts", lambda *a, **k: True)
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None, num_experts=8)

    # Premise: the fold really does write swiglu/geglu False for this call.
    assert orchestrator.apply_liger_parallelism_overrides(
        {}, orchestrator.liger_parallelism_overrides(has_ep_wrapped_experts=True)
    ) == {"swiglu": False, "geglu": False}

    assert orchestrator.apply_liger_kernel(config, None, needs_ep_wrappers=True) is None

    # A kernel the user explicitly turned OFF is not a request either.
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    assert orchestrator.apply_liger_kernel(config, {"rms_norm": False}) is None


def _training_config(user_config):
    return types.SimpleNamespace(use_liger_kernel=True, liger_kernel_config=user_config)


def _model_with_applied(applied):
    return types.SimpleNamespace(config=types.SimpleNamespace(_halo_liger_applied_config=applied))


def test_finalize_keeps_flag_for_effective_flce():
    """Zaya regression: FLCE applied via per-model default, user dict empty → the flag must stay on
    (the model returns no logits; TRL's non-liger branches would slice a 1-token logit tensor)."""
    training_config = _training_config(None)
    applied = {"rope": False, "cross_entropy": False, "fused_linear_cross_entropy": True, "rms_norm": True}
    finalize_liger_after_direct_load(training_config, True, _model_with_applied(applied))
    assert training_config.use_liger_kernel is True
    assert training_config.liger_kernel_config == applied  # pinned to what was actually applied


def test_finalize_disables_flag_when_flce_forced_off():
    """TP regression: user asked for FLCE but the orchestrator forced it off → flag must go off
    (the model returns real logits; keeping the flag would drop TRL onto the no-logits branches)."""
    training_config = _training_config({"fused_linear_cross_entropy": True})
    finalize_liger_after_direct_load(training_config, True, _model_with_applied({"fused_linear_cross_entropy": False}))
    assert training_config.use_liger_kernel is False


def test_finalize_disables_flag_when_nothing_applied():
    # Liger unavailable / no applier for the model type: nothing was patched, flag must go off even
    # though the user dict requested FLCE.
    training_config = _training_config({"fused_linear_cross_entropy": True})
    finalize_liger_after_direct_load(training_config, True, _model_with_applied(None))
    assert training_config.use_liger_kernel is False

    # No recorded attribute at all (config never went through the orchestrator).
    training_config = _training_config({"fused_linear_cross_entropy": True})
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    finalize_liger_after_direct_load(training_config, True, model)
    assert training_config.use_liger_kernel is False


def test_finalize_noop_when_liger_disabled():
    training_config = types.SimpleNamespace(use_liger_kernel=False, liger_kernel_config=None)
    finalize_liger_after_direct_load(training_config, False, _model_with_applied({"fused_linear_cross_entropy": True}))
    assert training_config.use_liger_kernel is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
