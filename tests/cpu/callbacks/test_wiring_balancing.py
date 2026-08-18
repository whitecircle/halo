#!/usr/bin/env python
"""Unit tests for :func:`src.distributed.expert_parallel.balancing_strategy.apply_balancing_strategy` — the
MoE balancing side effects on ``model.config``, with emphasis on the silent-failure guards:

* ``aux_loss`` under a policy-gradient (GRPO) trainer is INERT (the loss never adds the model's
  router aux loss), so it must WARN loudly and leave ``output_router_logits`` off — not silently
  flip it on with a misleading "enabling" INFO.
* ``aux_loss`` under a normal LM-loss trainer with a usable coefficient enables
  ``output_router_logits``.
* ``bias_update`` zeroes the aux coefficient and forces ``output_router_logits`` off.
* an EXPLICIT ``bias_update`` on a MoE model where no EP layer accepts the bias AND no native
  ``balancing_biases`` router exists (Zaya) must RAISE — it would silently balance nothing.
  A supporting EP layer, a native-bias router, or a non-MoE model must not raise.
* the export contract: strict ``bias_update`` refuses a transient side-buffer (the export would
  silently serve without the trained bias) and points at ``bias_update_transient``; the transient
  spelling warns on a side-buffer, raises where every router exports natively, and ``auto`` resolves
  to ``none`` (not a silent transient bias) where no exportable slot exists. A materialized
  config-gated slot mirrors its flag (LFM-2 ``use_expert_bias``) into ``model.config``.

Plus :func:`src.callbacks.wiring._detect_moe`'s discard-slot detection, which is derived
from the router CLASS attribute ``_has_discard_expert_slot`` and from nothing else: any family
declaring it opts in, and the hub-native Zaya modeling gets it from the load-time patch (there is no
model_type list to keep in sync — a family whose router does not declare it must NOT be excluded).

Run::

    python tests/cpu/callbacks/test_wiring_balancing.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.callbacks.wiring as wiring
import src.distributed.expert_parallel.balancing_strategy as balancing_strategy
import src.models.moe_balancing as moe_balancing
from src.callbacks.wiring import _detect_moe
from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.models.loading.config_levels import (
    config_export_ready,
    set_config_field_run_scoped,
)
from src.models.moe_balancing import (
    _WARNED_TRANSIENT_ONLY_AUTO,
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    ROUTER_LOGITS_FORCED_OFF_ATTR,
    is_transient_balancing_router,
)


class _Cfg:
    def __init__(self, output_router_logits=False, router_aux_loss_coef=0.0, model_type="test_moe", **extra):
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_type = model_type
        for name, value in extra.items():  # e.g. num_experts / num_experts_per_tok for _detect_moe
            setattr(self, name, value)


class _Model:
    """Minimal model stand-in: a config plus configurable ``modules()`` (default: no EP layers).

    ``forward`` takes ``output_router_logits`` because that parameter is how the toolkit detects that
    a class honours the config flag at all (``honors_output_router_logits_config``). A stand-in
    without it stands in for a multimodal wrapper, not for the causal-LM families these tests cover.
    """

    def __init__(self, cfg: _Cfg, submodules=()):
        self.config = cfg
        self._submodules = list(submodules)

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")

    def modules(self):
        return iter([self, *self._submodules])

    def named_modules(self):
        return iter([("", self), *((f"sub{i}", m) for i, m in enumerate(self._submodules))])

    def register_forward_hook(self, fn):
        return type("Handle", (), {"remove": lambda self: None})()


class _SupportingEPLayer:
    """EP-layer stand-in whose ``enable_bias_balancing`` accepts the bias (GptOss-like)."""

    def enable_bias_balancing(self):
        return True


class _NonSupportingEPLayer:
    """EP-layer stand-in that declines the bias (aux-loss family, e.g. Qwen3)."""

    def enable_bias_balancing(self):
        return False


class _NativeBiasRouter(torch.nn.Module):
    """Zaya-style router shipping its own ``balancing_biases`` as a REGISTERED buffer + counter slot.

    A registered buffer, not a plain attribute: that is what the hub modeling does, and it is what
    keeps :func:`is_transient_balancing_router` False — the buffer rides ``state_dict()`` and exports.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None


class _TransientEPLayer:
    """EP-layer stand-in whose enable attaches the transient side-buffer (Qwen3/Mistral4-like):
    a plain instance attribute no ``state_dict()``, export, or weight-sync ever carries."""

    _supports_bias_balancing = True

    def enable_bias_balancing(self):
        self.balancing_biases = torch.zeros(4)
        self.expert_load_counter = None
        return True


class _NativeAdoptingEPLayer(torch.nn.Module):
    """EP-layer stand-in that adopts a checkpoint-exported native slot (GptOss/GLM4-like)."""

    _supports_bias_balancing = True
    _NATIVE_BALANCING_CONFIG_FLAG = None

    def can_adopt_native_balancing(self):
        return True

    def enable_bias_balancing(self):
        self._native_balancing_bias_adopted = True
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None
        return True


class _ConfigGatedNativeEPLayer(_NativeAdoptingEPLayer):
    """LFM-2-like: the materialized slot is dormant at serving until the config flag flips."""

    _NATIVE_BALANCING_CONFIG_FLAG = "use_expert_bias"


def _capture_warnings(fn):
    calls: list[str] = []
    orig = balancing_strategy.logger.warning
    balancing_strategy.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        balancing_strategy.logger.warning = orig
    return calls


def _capture_resolver_warnings(fn):
    calls: list[str] = []
    orig = moe_balancing.logger.warning
    moe_balancing.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        moe_balancing.logger.warning = orig
    return calls


def test_aux_loss_under_grpo_warns_and_stays_off() -> None:
    """aux_loss + policy_gradient_loss: loud warning, output_router_logits left OFF (inert under GRPO)."""
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.001))
    warnings = _capture_warnings(lambda: apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=True))
    assert model.config.output_router_logits is False, "GRPO aux_loss must not enable router logits"
    assert any("NO EFFECT" in w and "bias_update" in w for w in warnings), warnings


def test_aux_loss_lm_trainer_enables_router_logits() -> None:
    """aux_loss on a normal LM-loss trainer with a usable coef still enables output_router_logits."""
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.001))
    apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=False)
    assert model.config.output_router_logits is True


def test_aux_loss_zero_coef_stays_off() -> None:
    """aux_loss with no usable coefficient (aux-loss-free router) leaves output_router_logits off."""
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=False)
    assert model.config.output_router_logits is False


def test_bias_update_forces_router_logits_off_and_zeros_coef() -> None:
    """bias_update zeroes the aux coefficient and forces output_router_logits off (EP recorder bypass)."""
    model = _Model(_Cfg(output_router_logits=True, router_aux_loss_coef=0.01), submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    assert model.config.output_router_logits is False
    assert model.config.router_aux_loss_coef == 0.0


def test_bias_update_on_moe_without_any_supporting_layer_raises() -> None:
    """Explicit bias_update on a MoE whose EP layers all decline the bias and with no native-bias
    router: silently balancing NOTHING is the failure mode — must raise, naming the layer family."""
    model = _Model(_Cfg(), submodules=[_NonSupportingEPLayer()])
    with pytest.raises(ValueError, match="_NonSupportingEPLayer"):
        apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_on_moe_without_ep_wrappers_raises() -> None:
    """Explicit bias_update on a MoE with no EP wrappers at all (dense-loaded MoE) — same silent
    no-op, must raise."""
    model = _Model(_Cfg())
    with pytest.raises(ValueError, match="bias_update"):
        apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_with_supporting_ep_layer_passes() -> None:
    model = _Model(_Cfg(), submodules=[_SupportingEPLayer()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_with_native_bias_router_passes() -> None:
    """Zaya: no EP layer accepts the bias but the router ships balancing_biases natively — the
    callback drives it, so bias_update must keep working."""
    model = _Model(_Cfg(model_type="zaya"), submodules=[_NativeBiasRouter()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_on_dense_raises() -> None:
    """Explicit bias_update on a dense model balances nothing on every rank — must raise, not
    train behind a silently inert controller."""
    model = _Model(_Cfg())
    with pytest.raises(ValueError, match="dense"):
        apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=False)


def test_none_mode_is_noop() -> None:
    """none: leave both config fields untouched."""
    model = _Model(_Cfg(output_router_logits=True, router_aux_loss_coef=0.01))
    apply_balancing_strategy(model, "none", policy_gradient_loss=False)
    assert model.config.output_router_logits is True
    assert model.config.router_aux_loss_coef == 0.01


# The export contract: bias_update promises the trained bias reaches every exported checkpoint;
# bias_update_transient is the explicit opt-in where the architecture has no slot for it.


def test_bias_update_refuses_a_transient_side_buffer() -> None:
    """Strict bias_update on a family whose bias only lives in the transient side-buffer: the export
    would silently serve without the trained bias, so it must raise and name the opt-in."""
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    with pytest.raises(ValueError, match="bias_update_transient"):
        apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_transient_warns_and_passes_on_side_buffer() -> None:
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    warnings = _capture_warnings(
        lambda: apply_balancing_strategy(model, "bias_update_transient", policy_gradient_loss=False, is_moe=True)
    )
    assert any("TRAINING-TIME routing only" in w for w in warnings), warnings


def test_bias_update_transient_on_native_model_raises() -> None:
    """The transient spelling on a model whose bias exports natively misdocuments the run."""
    model = _Model(_Cfg(), submodules=[_NativeAdoptingEPLayer()])
    with pytest.raises(ValueError, match="Use moe_balancing=bias_update"):
        apply_balancing_strategy(model, "bias_update_transient", policy_gradient_loss=False, is_moe=True)


def test_bias_update_still_passes_on_native_adoption() -> None:
    """Anti-over-rejection twin of the two raises above — and it asserts the POST-state, since a
    strategy that silently left the router transient would satisfy a bare "did not raise"."""
    layer = _NativeAdoptingEPLayer()
    model = _Model(_Cfg(), submodules=[layer])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert getattr(layer, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) is True, (
        "the layer must be flagged as having adopted its native slot"
    )
    assert not is_transient_balancing_router(layer), (
        "a router the export contract accepted must not hold its bias in the transient side-buffer"
    )


def test_native_config_flag_mirrors_into_model_config() -> None:
    """A materialized config-gated slot (LFM-2 use_expert_bias) must flip the flag on model.config,
    or the exported config.json tells serving engines to skip the trained tensor."""
    cfg = _Cfg(use_expert_bias=False)
    model = _Model(cfg, submodules=[_ConfigGatedNativeEPLayer()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert cfg.use_expert_bias is True


class _BiasCapableEPLayer:
    """EP-layer stand-in declaring the class contract ``accepts_bias_balancing`` reads."""

    _supports_bias_balancing = True

    def enable_bias_balancing(self):
        return True


class _MultimodalModel(_Model):
    """Wrapper stand-in whose forward never declares ``output_router_logits``.

    Qwen3_5MoeForConditionalGeneration reads the flag out of ``**kwargs`` only, so the config flag
    this mode sets never reaches its aux loss — the case the aux_loss raise exists for.
    """

    def forward(self, input_ids=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")


def test_aux_loss_dead_end_names_the_knobs_that_create_a_bias_acceptor() -> None:
    """No wrapper accepts the bias here, so bias_update raises in turn: recommending it is a loop.

    The wrapper-less multimodal MoE (use_grouped_gemm off / accelerate path) is exactly where the two
    raises can point at each other — auto resolves to aux_loss, and aux_loss answering "use
    bias_update" against bias_update answering "prefer auto" closes the loop.
    """
    model = _MultimodalModel(_Cfg(router_aux_loss_coef=0.001))
    with pytest.raises(ValueError) as excinfo:
        apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    message = str(excinfo.value)
    assert "does not take output_router_logits" in message
    assert "use_grouped_gemm" in message and "expert_parallel_size" in message, message
    assert "moe_balancing=none" in message, message


def test_aux_loss_still_recommends_bias_update_where_a_wrapper_accepts_it() -> None:
    """Anti-vacuity: with an acceptor present bias_update IS the remedy, and the knob advice is absent."""
    model = _MultimodalModel(_Cfg(router_aux_loss_coef=0.001), submodules=[_BiasCapableEPLayer()])
    with pytest.raises(ValueError) as excinfo:
        apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    message = str(excinfo.value)
    assert "Use moe_balancing=bias_update" in message, message
    assert "use_grouped_gemm" not in message, message


class _DiscardSlotRouter:
    """Router class declaring the toolkit's trailing-discard-slot contract (vendored Zaya)."""

    _has_discard_expert_slot = True


def test_detect_moe_excludes_discard_slot_from_router_class() -> None:
    """The class attribute is the primary signal — any family declaring it opts in."""
    cfg = _Cfg(model_type="some_new_family", num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg, submodules=[_DiscardSlotRouter()])
    is_moe, top_k, exclude_last_slot = _detect_moe(model)
    assert (is_moe, top_k) == (True, 2)
    assert exclude_last_slot is True


def test_zaya_load_patch_declares_the_discard_slot_end_to_end() -> None:
    """Hub-native Zaya reaches the exclusion through the load patch, not a model_type list.

    Both halves of the derivation are pinned: ``patch_zaya_router_load_recording`` must stamp
    ``_has_discard_expert_slot`` on the hub router class, and ``_detect_moe`` must read it off an
    instance of that class. Break either and the discard slot is bias-balanced like a real expert.
    """
    from accelerate import PartialState
    from transformers.models.zaya import modeling_zaya

    from src.models.patches.zaya import patch_zaya_router_load_recording

    PartialState()  # the patch logs through accelerate's logger, which needs an initialized state
    patch_zaya_router_load_recording()
    assert modeling_zaya.ZayaRouter._has_discard_expert_slot is True, (
        "the Zaya load patch no longer declares the discard slot on the hub router class"
    )

    # __new__, not __init__: only the CLASS attribute is read, and constructing the real router
    # would need a full Zaya config.
    router = modeling_zaya.ZayaRouter.__new__(modeling_zaya.ZayaRouter)
    model = _Model(_Cfg(model_type="zaya", num_experts=8, num_experts_per_tok=2), submodules=[router])
    is_moe, _, exclude_last_slot = _detect_moe(model)
    assert is_moe is True
    assert exclude_last_slot is True


def test_detect_moe_keeps_last_slot_when_no_router_class_declares_it() -> None:
    """The model_type is NOT a signal: a family whose routers declare nothing keeps every slot.

    Guards against re-introducing a hand-maintained model-type list beside the class attribute —
    such a list silently excludes a real expert on any family whose name happens to be on it.
    """
    model = _Model(_Cfg(model_type="zaya", num_experts=8, num_experts_per_tok=2))
    is_moe, _, exclude_last_slot = _detect_moe(model)
    assert is_moe is True
    assert exclude_last_slot is False


def test_detect_moe_keeps_last_slot_for_ordinary_moe() -> None:
    """A normal MoE family must keep every expert slot in the balancing statistics."""
    model = _Model(_Cfg(model_type="qwen3_moe", num_experts=8, num_experts_per_tok=2))
    _, _, exclude_last_slot = _detect_moe(model)
    assert exclude_last_slot is False


# moe_balancing: none must be a genuine no-op, end to end through build_perf_callbacks


def _perf_args(moe_balancing: str):
    return SimpleNamespace(
        moe_balancing=moe_balancing,
        enable_efficiency_metrics=False,
        enable_moe_metrics=True,
        enable_torch_profiler=False,
        router_balancing_rate=1e-3,
        num_full_model_params=None,
        report_mfu_diagnostics=False,
    )


_NO_PARALLELISM = SimpleNamespace(tp_size=1, ep_size=1, cp_size=1, expert_tp_size=1, pp_size=1)


def _build_and_start(moe_balancing: str, model):
    """Run the real wiring path, then the callbacks' ``on_train_begin`` (where the writes happen)."""
    callbacks = wiring.build_perf_callbacks(_perf_args(moe_balancing), SimpleNamespace(), model, _NO_PARALLELISM)
    for cb in callbacks:
        cb.on_train_begin(args=None, state=None, control=None, model=model)
    return callbacks


def test_weight_sync_downgrades_bias_update_to_none() -> None:
    """``syncs_to_external_generator=True`` (every on-policy weight-sync script) must downgrade the
    bias modes to ``none``: the vLLM sync forwards PARAMETERS only — an adopted native slot is a
    buffer — so a trained bias would steer the trainer's routing while the generator keeps the
    pretrained one. Without this downgrade, online/env GRPO silently trains off-policy and nothing
    anywhere fails."""
    layer = _NativeAdoptingEPLayer()
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[layer])
    callbacks = wiring.build_perf_callbacks(
        _perf_args("bias_update"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
    )
    assert not any(type(cb).__name__ == "RouterBiasBalancingCallback" for cb in callbacks), (
        "the weight-sync downgrade did not fire — the trainer would balance on a bias the generator never sees"
    )
    assert not getattr(layer, "_native_balancing_bias_adopted", False), (
        "the native slot was adopted despite the downgrade — the export would carry a never-trained bias"
    )


def test_weight_sync_downgrades_the_transient_mode_too() -> None:
    """``bias_update_transient`` is equally invisible to the sync (a ``__dict__`` side-buffer)."""
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_TransientEPLayer()])
    callbacks = wiring.build_perf_callbacks(
        _perf_args("bias_update_transient"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
    )
    assert not any(type(cb).__name__ == "RouterBiasBalancingCallback" for cb in callbacks)


def test_the_same_model_without_weight_sync_gets_the_balancer() -> None:
    """Anti-vacuity twin of the downgrade tests: the ONLY difference is the sync flag."""
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_NativeAdoptingEPLayer()])
    callbacks = wiring.build_perf_callbacks(
        _perf_args("bias_update"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=False,
    )
    assert any(type(cb).__name__ == "RouterBiasBalancingCallback" for cb in callbacks)


# The engine-side serving gap a layer class declares (Laguna under vLLM 0.26.0)


class _ServedDropEPLayer(_NativeAdoptingEPLayer):
    _SERVED_BALANCING_BIAS_DROPPED_BY = "vLLM 0.26.0"


def test_a_declared_served_bias_drop_is_warned_at_enable(caplog) -> None:
    """A family whose serving engine drops the exported bias key (Laguna) must say so at enable —
    the export is correct and a transformers reload routes as trained, but a vLLM-served copy
    routes on the pretrained bias with no other symptom."""
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_ServedDropEPLayer()])
    with caplog.at_level("WARNING"):
        balancing_strategy.apply_balancing_strategy(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    assert any("DROPPED by ['vLLM 0.26.0']" in message for message in caplog.messages), caplog.messages


def test_families_without_the_declaration_do_not_warn(caplog) -> None:
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_NativeAdoptingEPLayer()])
    with caplog.at_level("WARNING"):
        balancing_strategy.apply_balancing_strategy(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    assert not any("DROPPED by" in message for message in caplog.messages), caplog.messages


def test_laguna_layer_declares_the_served_bias_drop() -> None:
    """The real class carries the declaration the warning path reads — pinned so an engine upgrade
    that closes the hole is a deliberate one-line removal, not silent drift."""
    laguna = ep_layer_class_by_model_type()["laguna"]
    assert laguna._SERVED_BALANCING_BIAS_DROPPED_BY == "vLLM 0.26.0"


def test_moe_balancing_none_leaves_a_yaml_configured_aux_loss_alone() -> None:
    """``none`` means "the toolkit adds no balancing intervention" — not "disable the user's".

    A metrics callback zeroing ``router_aux_loss_coef`` and forcing ``output_router_logits=True``
    whenever the mode is not ``aux_loss`` makes asking for no toolkit balancing silently switch off
    aux balancing configured directly in YAML and pay the router-logit memory (~540 MB at
    GPT-OSS-120B b=1 s=16k).
    """
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    _build_and_start("none", model)
    assert cfg.router_aux_loss_coef == 0.001, "moe_balancing=none disabled the configured aux balancing"
    assert cfg.output_router_logits is False, "moe_balancing=none turned router logits on as a side effect"


def test_moe_balancing_none_respects_router_logits_the_user_enabled() -> None:
    """The opt-in direction: metrics ride along when the model/YAML already asked for router logits."""
    cfg = _Cfg(output_router_logits=True, router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    callbacks = _build_and_start("none", model)
    metrics = next(cb for cb in callbacks if type(cb).__name__ == "MoEMetricsCallback")
    assert metrics._hook_handles, "router logits are on, so load metrics must be collected"
    assert cfg.router_aux_loss_coef == 0.001


def test_moe_balancing_aux_loss_still_enables_router_logits() -> None:
    """The one mode that legitimately owns the flag must still set it."""
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    _build_and_start("aux_loss", model)
    assert cfg.output_router_logits is True
    assert cfg.router_aux_loss_coef == 0.001


def test_auto_resolves_to_none_where_only_a_transient_bias_is_possible() -> None:
    """auto on a Mistral4/multimodal-Qwen3.5 shape (aux loss unreachable, bias accepted but only as
    a side-buffer): never silently defaults into a bias no export carries — resolve to none and name
    the explicit opt-in."""
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _MultimodalModel(cfg, submodules=[_BiasCapableEPLayer()])
    _WARNED_TRANSIENT_ONLY_AUTO.discard(type(model).__name__)  # warn-once state is keyed per model class
    warnings = _capture_resolver_warnings(lambda: _build_and_start("auto", model))
    assert any("bias_update_transient" in w for w in warnings), warnings
    assert cfg.output_router_logits is False
    assert not hasattr(model._submodules[0], "balancing_biases"), "auto must not attach the side-buffer"


def test_auto_prefers_native_bias_update_over_none() -> None:
    """auto keeps resolving to bias_update wherever the bias lands in exported state."""
    cfg = _Cfg(num_experts=8, num_experts_per_tok=2)
    model = _MultimodalModel(cfg, submodules=[_NativeAdoptingEPLayer()])
    callbacks = _build_and_start("auto", model)
    assert any(type(cb).__name__ == "RouterBiasBalancingCallback" for cb in callbacks)


class _AliasedCfg(_Cfg):
    """PreTrainedConfig-style ``attribute_map``: ``__setattr__`` lands aliased fields in the
    MAPPED ``__dict__`` slot, and reads resolve back through the alias."""

    attribute_map = {"router_aux_loss_coef": "aux_coef"}

    def __setattr__(self, key, value):
        object.__setattr__(self, self.attribute_map.get(key, key), value)

    def __getattr__(self, key):
        mapped = type(self).attribute_map.get(key)
        if mapped is not None and mapped in self.__dict__:
            return self.__dict__[mapped]
        raise AttributeError(key)


def test_run_scoped_restore_respects_config_attribute_map() -> None:
    """An aliased router field stores under the MAPPED ``__dict__`` key; the run-scoped registry
    must record and restore that same key, or the export block restores nothing and the run-scoped
    zero leaks into config.json."""
    cfg = _AliasedCfg(router_aux_loss_coef=0.02)
    assert cfg.__dict__.get("aux_coef") == 0.02, "sanity: the alias landed in the mapped slot"
    set_config_field_run_scoped(cfg, "router_aux_loss_coef", 0.0)
    assert cfg.__dict__["aux_coef"] == 0.0
    with config_export_ready(cfg):
        assert cfg.__dict__["aux_coef"] == 0.02, "export must restore the MAPPED slot"
    assert cfg.__dict__["aux_coef"] == 0.0, "run state must come back after the export block"


def test_run_scoped_mutations_are_restored_for_config_export() -> None:
    """The zeroed aux coef, the forced router-logits flag and the forced-off stamp configure THIS
    run — a config serialized inside config_export_ready carries the artifact's original
    values, and the run state comes back afterwards."""
    cfg = _Cfg(output_router_logits=True, router_aux_loss_coef=0.01)
    model = _Model(cfg, submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert cfg.router_aux_loss_coef == 0.0
    assert cfg.output_router_logits is False
    assert getattr(cfg, ROUTER_LOGITS_FORCED_OFF_ATTR, False) is True

    with config_export_ready(cfg):
        assert cfg.router_aux_loss_coef == 0.01, "export must carry the hub coefficient"
        assert cfg.output_router_logits is True, "export must carry the pre-run flag"
        assert ROUTER_LOGITS_FORCED_OFF_ATTR not in cfg.__dict__, "toolkit stamp must not serialize"

    assert cfg.router_aux_loss_coef == 0.0, "run state must come back after the export block"
    assert cfg.output_router_logits is False
    assert getattr(cfg, ROUTER_LOGITS_FORCED_OFF_ATTR, False) is True


# ── Rank uniformity: the two seams that keep one verdict for the whole job ──────
#
# Both are COLLECTIVE and both read the LIVE module tree, which under pipeline parallelism is this
# rank's stage only. Every test above takes the non-distributed path, where ``agree_balancing_mode``
# is the identity and ``reject_across_ranks`` degenerates to a local raise — so none of them can see
# a regression in the agreement itself. These drive the distributed halves with a stubbed group.


class _StubDist:
    """Stands in for ``torch.distributed``, replaying a fixed per-rank gather.

    ``get_world_size`` also stands in for the module's world-size read, which goes through
    :func:`get_global_world_size` rather than this object.
    """

    def __init__(self, peer_values):
        self.peer_values = list(peer_values)
        self.gathers: list[object] = []

    def get_world_size(self):
        return len(self.peer_values)

    def all_gather_object(self, out_list, value):
        self.gathers.append(value)
        out_list[:] = list(self.peer_values)


@pytest.fixture
def stub_dist(monkeypatch):
    def _install(peer_values):
        stub = _StubDist(peer_values)
        monkeypatch.setattr(balancing_strategy, "dist", stub)
        monkeypatch.setattr(balancing_strategy, "get_global_world_size", stub.get_world_size)
        return stub

    return _install


@pytest.mark.parametrize(
    ("gathered", "expected"),
    [
        # A stage holding no MoE layer resolves out of ignorance; the informed stage wins.
        (["none", "bias_update"], "bias_update"),
        (["none", "aux_loss"], "aux_loss"),
        (["aux_loss", "bias_update"], "bias_update"),
        (["bias_update_transient", "aux_loss"], "bias_update_transient"),
        (["bias_update", "bias_update_transient"], "bias_update"),
        # Unanimity is a fixed point, in both directions.
        (["none", "none"], "none"),
        (["bias_update", "bias_update"], "bias_update"),
    ],
)
def test_a_split_balancing_verdict_converges_by_precedence(stub_dist, gathered, expected):
    """Every rank must leave with the SAME mode, or the stages take different branches of
    ``apply_balancing_strategy`` — different collectives, and a raise on some ranks only."""
    stub_dist(gathered)
    for local in gathered:
        assert balancing_strategy.agree_balancing_mode(local) == expected, f"rank resolving {local!r} did not converge"


def test_every_rank_enters_the_agreement_gather(stub_dist):
    """The gather is collective: a rank that already holds the winning mode must still call it."""
    stub = stub_dist(["bias_update", "bias_update"])
    balancing_strategy.agree_balancing_mode("bias_update")
    assert stub.gathers == ["bias_update"], "the agreed-already rank skipped the gather"


def test_the_agreement_is_the_identity_outside_distributed():
    """Anti-vacuity for the fixture: without a group the function must not invent a verdict."""
    for mode in ("none", "aux_loss", "bias_update", "bias_update_transient"):
        assert balancing_strategy.agree_balancing_mode(mode) == mode


def test_a_clean_rank_still_enters_the_export_contract_collective(monkeypatch):
    """``_enforce_bias_export_contract`` must reach ``reject_across_ranks`` with ``reason=None``.

    A stage whose routers all adopt a native slot has nothing to report — and is exactly the rank a
    ``if reason: raise`` rewrite would strand in the gather while a transient-carrying stage raises.
    """
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=ValueError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(local_reason)

    monkeypatch.setattr(balancing_strategy, "reject_across_ranks", _record)
    model = _Model(_Cfg(), submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)

    assert len(calls) == 1, f"the export contract did not reach the consensus seam exactly once: {calls}"
    reason, what = calls[0]
    assert reason is None, f"a fully native model reported a rejection reason: {reason}"
    assert "bias_update export contract" in what


def test_a_transient_rank_reports_its_reason_through_the_same_seam(monkeypatch):
    """The failing half of the pair: the reason must be CARRIED, not raised past the seam, so every
    peer prints the real cause instead of a collective timeout."""
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=ValueError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(local_reason)

    monkeypatch.setattr(balancing_strategy, "reject_across_ranks", _record)
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    with pytest.raises(ValueError, match="NO EXPORT CARRIES"):
        apply_balancing_strategy(model, "bias_update", policy_gradient_loss=False, is_moe=True)

    assert len(calls) == 1 and calls[0][0] is not None
    assert "bias_update_transient" in calls[0][0], "the carried reason must name the opt-in spelling"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
