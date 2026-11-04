#!/usr/bin/env python
"""What :func:`apply_balancing_mode` writes to ``model.config`` and the live routers, its refusals, the cross-rank
agreement, and what :func:`configure_moe_balancing` resolves end to end."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from peft.utils import ModulesToSaveWrapper
from transformers.models.zaya import modeling_zaya

import halo.callbacks.balancing as balancing
import halo.callbacks.factory as factory
import halo.models.moe.balancing as moe_balancing
from halo.callbacks.balancing import apply_balancing_mode
from halo.models.config_access import (
    config_export_ready,
    set_config_field_run_scoped,
)
from halo.models.moe.balancing import (
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    _warned_transient_only_auto,
    is_transient_balancing_router,
)
from halo.models.moe.config_fields import resolve_moe_routing
from halo.models.moe.family_specs import MoEFamilySpec, NativeBalancingSlot
from halo.models.moe.router_logits import ROUTER_LOGITS_FORCED_OFF_ATTR
from halo.models.zaya import patch_zaya_router_load_recording
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient


class _Cfg:
    def __init__(self, output_router_logits=False, router_aux_loss_coef=0.0, model_type="test_moe", **extra):
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_type = model_type
        for name, value in extra.items():
            setattr(self, name, value)


class _Model(torch.nn.Module):
    """``forward`` declares ``output_router_logits`` because ``honors_router_logits_flag`` reads it.
    A stand-in without it is a multimodal wrapper."""

    def __init__(self, cfg: _Cfg, submodules=()):
        super().__init__()
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
    def enable_bias_balancing(self):
        return True


class _NonSupportingEPLayer:
    def enable_bias_balancing(self):
        return False


class _NativeBiasRouter(torch.nn.Module):
    """Zaya-style: the registered buffer rides ``state_dict()``, so :func:`is_transient_balancing_router` is False."""

    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None


class _BiasRouterStandIn:
    """The :class:`BiasBalancingRouter` surface, for a layer that selects its experts but has no exported slot."""

    SUPPORTS_BIAS_BALANCING = True
    SEVERS_AUX_LOSS = False

    def can_adopt_native_balancing(self):
        return False


class _TransientEPLayer(_BiasRouterStandIn):
    """Enable attaches the side-buffer (Qwen3/Mistral4-like), a plain attribute no export or weight sync carries."""

    def enable_bias_balancing(self):
        self.balancing_biases = torch.zeros(4)
        self.expert_load_counter = None
        return True


class _NativeAdoptingEPLayer(_BiasRouterStandIn, torch.nn.Module):
    """Adopts a checkpoint-exported native slot (GptOss/GLM4-like)."""

    FAMILY = MoEFamilySpec(model_types=(), balancing_slot=NativeBalancingSlot(("balancing_biases",)))

    def can_adopt_native_balancing(self):
        return True

    def enable_bias_balancing(self):
        self._native_balancing_bias_adopted = True
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None
        return True


class _ConfigGatedNativeEPLayer(_NativeAdoptingEPLayer):
    """LFM-2-like: the materialized slot is dormant at serving until the config flag flips."""

    FAMILY = MoEFamilySpec(
        model_types=(), balancing_slot=NativeBalancingSlot(("balancing_biases",), config_flag="use_expert_bias")
    )


def _capture_warnings(fn):
    calls: list[str] = []
    orig = balancing.logger.warning
    balancing.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        balancing.logger.warning = orig
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
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.001))
    warnings = _capture_warnings(lambda: apply_balancing_mode(model, "aux_loss", policy_gradient_loss=True))
    assert model.config.output_router_logits is False, "GRPO aux_loss must not enable router logits"
    assert any("NO EFFECT" in w and "bias_update" in w for w in warnings), warnings


def test_aux_loss_lm_trainer_enables_router_logits() -> None:
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.001))
    apply_balancing_mode(model, "aux_loss", policy_gradient_loss=False)
    assert model.config.output_router_logits is True


def test_aux_loss_zero_coef_stays_off() -> None:
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    apply_balancing_mode(model, "aux_loss", policy_gradient_loss=False)
    assert model.config.output_router_logits is False


def test_bias_update_forces_router_logits_off_and_zeros_coef() -> None:
    """Forcing the flag off is the EP recorder bypass."""
    model = _Model(_Cfg(output_router_logits=True, router_aux_loss_coef=0.01), submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    assert model.config.output_router_logits is False
    assert model.config.router_aux_loss_coef == 0.0


def test_bias_update_on_moe_without_any_supporting_layer_raises() -> None:
    """Silently balancing nothing is the failure mode."""
    model = _Model(_Cfg(), submodules=[_NonSupportingEPLayer()])
    with pytest.raises(ValueError, match="_NonSupportingEPLayer"):
        apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_on_moe_without_ep_layers_raises() -> None:
    """A dense-loaded MoE: the same silent no-op."""
    model = _Model(_Cfg())
    with pytest.raises(ValueError, match="bias_update"):
        apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_with_supporting_ep_layer_passes() -> None:
    model = _Model(_Cfg(), submodules=[_SupportingEPLayer()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_with_native_bias_router_passes() -> None:
    """Zaya: no EP layer accepts the bias, but the callback drives the router's native ``balancing_biases``."""
    model = _Model(_Cfg(model_type="zaya"), submodules=[_NativeBiasRouter()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_on_dense_raises() -> None:
    model = _Model(_Cfg())
    with pytest.raises(ValueError, match="dense"):
        apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=False)


def test_none_mode_is_noop() -> None:
    model = _Model(_Cfg(output_router_logits=True, router_aux_loss_coef=0.01))
    apply_balancing_mode(model, "none", policy_gradient_loss=False)
    assert model.config.output_router_logits is True
    assert model.config.router_aux_loss_coef == 0.01


# bias_update promises the trained bias reaches every export; bias_update_transient is the opt-in where no slot exists.


def test_bias_update_refuses_a_transient_side_buffer() -> None:
    """The export would silently serve without the trained bias, so it must raise and name the opt-in."""
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    with pytest.raises(ValueError, match="bias_update_transient"):
        apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)


def test_bias_update_transient_warns_and_passes_on_side_buffer() -> None:
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    warnings = _capture_warnings(
        lambda: apply_balancing_mode(model, "bias_update_transient", policy_gradient_loss=False, is_moe=True)
    )
    assert any("TRAINING-TIME routing only" in w for w in warnings), warnings


def test_bias_update_transient_on_native_model_raises() -> None:
    """The transient spelling on a model whose bias exports natively misdocuments the run."""
    model = _Model(_Cfg(), submodules=[_NativeAdoptingEPLayer()])
    with pytest.raises(ValueError, match="Use moe_balancing=bias_update"):
        apply_balancing_mode(model, "bias_update_transient", policy_gradient_loss=False, is_moe=True)


def test_bias_update_still_passes_on_native_adoption() -> None:
    """Asserts the post-state, since a strategy that silently left the router transient would also not raise."""
    layer = _NativeAdoptingEPLayer()
    model = _Model(_Cfg(), submodules=[layer])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert getattr(layer, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) is True, (
        "the layer must be flagged as having adopted its native slot"
    )
    assert not is_transient_balancing_router(layer), (
        "a router the export contract accepted must not hold its bias in the transient side-buffer"
    )


def test_native_config_flag_mirrors_into_model_config() -> None:
    """Otherwise the exported config.json tells serving engines to skip the trained tensor."""
    cfg = _Cfg(use_expert_bias=False)
    model = _Model(cfg, submodules=[_ConfigGatedNativeEPLayer()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert cfg.use_expert_bias is True


def test_native_config_flag_mirrors_through_a_modules_to_save_wrapper() -> None:
    """The flag comes from the wrapped layers; the wrapper has no family spec and is skipped instead of raising."""
    cfg = _Cfg(use_expert_bias=False)
    wrapper = ModulesToSaveWrapper(_ConfigGatedNativeEPLayer(), "default")
    model = _Model(cfg, submodules=list(wrapper.modules()))
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)
    assert cfg.use_expert_bias is True


class _BiasCapableEPLayer(_BiasRouterStandIn):
    """Declares the class contract ``accepts_bias_balancing`` reads."""

    def enable_bias_balancing(self):
        return True


class _MultimodalModel(_Model):
    """Its forward never declares ``output_router_logits``, so the config flag never reaches its aux loss."""

    def forward(self, input_ids=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")


def test_aux_loss_dead_end_names_the_knobs_that_create_a_bias_acceptor() -> None:
    """On a wrapper-less multimodal MoE bias_update raises too, so recommending it would make the raises loop."""
    model = _MultimodalModel(_Cfg(router_aux_loss_coef=0.001))
    with pytest.raises(ValueError) as excinfo:
        apply_balancing_mode(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    message = str(excinfo.value)
    assert "does not take output_router_logits" in message
    assert "use_grouped_gemm" in message and "expert_parallel_size" in message, message
    assert "moe_balancing=none" in message, message


def test_aux_loss_still_recommends_bias_update_where_a_wrapper_accepts_it() -> None:
    model = _MultimodalModel(_Cfg(router_aux_loss_coef=0.001), submodules=[_BiasCapableEPLayer()])
    with pytest.raises(ValueError) as excinfo:
        apply_balancing_mode(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    message = str(excinfo.value)
    assert "Use moe_balancing=bias_update" in message, message
    assert "use_grouped_gemm" not in message, message


class _DiscardSlotRouter:
    _has_discard_expert_slot = True


def test_moe_routing_excludes_discard_slot_from_router_class() -> None:
    """Any family declaring the class attribute opts in."""
    cfg = _Cfg(model_type="some_new_family", num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg, submodules=[_DiscardSlotRouter()])
    is_moe, top_k, exclude_last_slot = resolve_moe_routing(model)
    assert (is_moe, top_k) == (True, 2)
    assert exclude_last_slot is True


def test_zaya_load_patch_declares_the_discard_slot_end_to_end() -> None:
    """Breaking the load patch's stamp or the class-attribute read bias-balances the discard slot like an expert."""
    patch_zaya_router_load_recording()
    assert modeling_zaya.ZayaRouter._has_discard_expert_slot is True, (
        "the Zaya load patch no longer declares the discard slot on the hub router class"
    )

    # Only the class attribute is read, and ``__init__`` would need a full Zaya config.
    router = modeling_zaya.ZayaRouter.__new__(modeling_zaya.ZayaRouter)
    model = _Model(_Cfg(model_type="zaya", num_experts=8, num_experts_per_tok=2), submodules=[router])
    is_moe, _, exclude_last_slot = resolve_moe_routing(model)
    assert is_moe is True
    assert exclude_last_slot is True


def test_moe_routing_keeps_last_slot_when_no_router_class_declares_it() -> None:
    """A hand-maintained model-type list would silently exclude a real expert on any family named on it."""
    model = _Model(_Cfg(model_type="zaya", num_experts=8, num_experts_per_tok=2))
    is_moe, _, exclude_last_slot = resolve_moe_routing(model)
    assert is_moe is True
    assert exclude_last_slot is False


def test_moe_routing_keeps_last_slot_for_ordinary_moe() -> None:
    model = _Model(_Cfg(model_type="qwen3_moe", num_experts=8, num_experts_per_tok=2))
    _, _, exclude_last_slot = resolve_moe_routing(model)
    assert exclude_last_slot is False


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


def _configure_and_build(args, training_config, model, parallelism_config, **balancing_kwargs):
    mode = balancing.configure_moe_balancing(args, model, parallelism_config, **balancing_kwargs)
    return factory.build_run_callbacks(args, training_config, model, parallelism_config, balancing_mode=mode)


def _build_and_start(moe_balancing: str, model):
    """Configure and build as the scripts do, then run ``on_train_begin``, where the writes happen."""
    callbacks = _configure_and_build(_perf_args(moe_balancing), SimpleNamespace(), model, _NO_PARALLELISM)
    for cb in callbacks:
        if not isinstance(cb, type):
            cb.on_train_begin(args=None, state=None, control=None, model=model)
    return callbacks


def test_weight_sync_downgrades_bias_update_to_none() -> None:
    """The sync carries no bias that moves and an adopted native slot is a buffer. A trained bias would steer the
    trainer's routing while the generator keeps the pretrained one, which trains off-policy with no failure."""
    layer = _NativeAdoptingEPLayer()
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[layer])
    callbacks = _configure_and_build(
        _perf_args("bias_update"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
    )
    assert not any(type(cb).__name__ == "BalancingBiasCallback" for cb in callbacks), (
        "the weight-sync downgrade did not fire — the trainer would balance on a bias the generator never sees"
    )
    assert not getattr(layer, "_native_balancing_bias_adopted", False), (
        "the native slot was adopted despite the downgrade — the export would carry a never-trained bias"
    )


def test_weight_sync_downgrades_the_transient_mode_too() -> None:
    """A ``__dict__`` side-buffer is equally invisible to the sync."""
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_TransientEPLayer()])
    callbacks = _configure_and_build(
        _perf_args("bias_update_transient"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
    )
    assert not any(type(cb).__name__ == "BalancingBiasCallback" for cb in callbacks)


def test_the_same_model_without_weight_sync_gets_the_balancer() -> None:
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_NativeAdoptingEPLayer()])
    callbacks = _configure_and_build(
        _perf_args("bias_update"),
        SimpleNamespace(),
        model,
        _NO_PARALLELISM,
        policy_gradient_loss=True,
        syncs_to_external_generator=False,
    )
    assert any(type(cb).__name__ == "BalancingBiasCallback" for cb in callbacks)


def test_a_served_bias_drop_is_warned_at_enable(caplog) -> None:
    """The export is correct and a transformers reload routes as trained, but a vLLM-served Laguna never routes on the
    trained bias, with no other symptom. The engine is named by its client's release."""
    model = _Model(
        _Cfg(model_type="laguna", num_experts=8, num_experts_per_tok=2), submodules=[_NativeAdoptingEPLayer()]
    )
    with caplog.at_level("WARNING"):
        balancing.apply_balancing_mode(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    expected = f"DROPPED by [{VLLMWeightSyncClient.ENGINE_RELEASE!r}]"
    assert any(expected in message for message in caplog.messages), caplog.messages


def test_families_without_the_declaration_do_not_warn(caplog) -> None:
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_NativeAdoptingEPLayer()])
    with caplog.at_level("WARNING"):
        balancing.apply_balancing_mode(model, "bias_update", policy_gradient_loss=True, is_moe=True)
    assert not any("DROPPED by" in message for message in caplog.messages), caplog.messages


def test_moe_balancing_none_leaves_a_yaml_configured_aux_loss_alone() -> None:
    """``none`` means the toolkit adds no balancing, so it must neither switch off aux balancing set in YAML nor pay
    the router-logit memory (~540 MB at GPT-OSS-120B b=1 s=16k)."""
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    _build_and_start("none", model)
    assert cfg.router_aux_loss_coef == 0.001, "moe_balancing=none disabled the configured aux balancing"
    assert cfg.output_router_logits is False, "moe_balancing=none turned router logits on as a side effect"


def test_moe_balancing_none_respects_router_logits_the_user_enabled() -> None:
    cfg = _Cfg(output_router_logits=True, router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    callbacks = _build_and_start("none", model)
    metrics = next(cb for cb in callbacks if type(cb).__name__ == "MoEMetricsCallback")
    assert metrics._hook_handles, "router logits are on, so load metrics must be collected"
    assert cfg.router_aux_loss_coef == 0.001


def test_moe_balancing_aux_loss_still_enables_router_logits() -> None:
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _Model(cfg)
    _build_and_start("aux_loss", model)
    assert cfg.output_router_logits is True
    assert cfg.router_aux_loss_coef == 0.001


def test_auto_resolves_to_none_where_only_a_transient_bias_is_possible() -> None:
    """Mistral4/multimodal-Qwen3.5 shape: aux loss unreachable, bias accepted only as a side-buffer.
    Never default into a bias no export carries."""
    cfg = _Cfg(router_aux_loss_coef=0.001, num_experts=8, num_experts_per_tok=2)
    model = _MultimodalModel(cfg, submodules=[_BiasCapableEPLayer()])
    _warned_transient_only_auto.discard(type(model).__name__)  # warn-once state is keyed per model class
    warnings = _capture_resolver_warnings(lambda: _build_and_start("auto", model))
    assert any("bias_update_transient" in w for w in warnings), warnings
    assert cfg.output_router_logits is False
    assert not hasattr(model._submodules[0], "balancing_biases"), "auto must not attach the side-buffer"


def test_auto_prefers_native_bias_update_over_none() -> None:
    cfg = _Cfg(num_experts=8, num_experts_per_tok=2)
    model = _MultimodalModel(cfg, submodules=[_NativeAdoptingEPLayer()])
    callbacks = _build_and_start("auto", model)
    assert any(type(cb).__name__ == "BalancingBiasCallback" for cb in callbacks)


class _AliasedCfg(_Cfg):
    """``attribute_map``-style: ``__setattr__`` lands aliased fields in the mapped ``__dict__`` slot."""

    attribute_map = {"router_aux_loss_coef": "aux_coef"}

    def __setattr__(self, key, value):
        object.__setattr__(self, self.attribute_map.get(key, key), value)

    def __getattr__(self, key):
        mapped = type(self).attribute_map.get(key)
        if mapped is not None and mapped in self.__dict__:
            return self.__dict__[mapped]
        raise AttributeError(key)


def test_run_scoped_restore_respects_config_attribute_map() -> None:
    """The registry must restore the mapped ``__dict__`` key, or the run-scoped zero leaks into config.json."""
    cfg = _AliasedCfg(router_aux_loss_coef=0.02)
    assert cfg.__dict__.get("aux_coef") == 0.02, "sanity: the alias landed in the mapped slot"
    set_config_field_run_scoped(cfg, "router_aux_loss_coef", 0.0)
    assert cfg.__dict__["aux_coef"] == 0.0
    with config_export_ready(cfg):
        assert cfg.__dict__["aux_coef"] == 0.02, "export must restore the MAPPED slot"
    assert cfg.__dict__["aux_coef"] == 0.0, "run state must come back after the export block"


def test_run_scoped_mutations_are_restored_for_config_export() -> None:
    """The zeroed aux coef, forced router-logits flag and forced-off stamp configure this run only."""
    cfg = _Cfg(output_router_logits=True, router_aux_loss_coef=0.01)
    model = _Model(cfg, submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)
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


# ``agree_balancing_mode`` and ``reject_across_ranks`` are COLLECTIVE and read the live module tree, under PP this
# rank's stage only. The tests above take the non-distributed path, where they degenerate, so these stub the group.


class _StubDist:
    """Replays a fixed per-rank gather. ``get_world_size`` also stands in for :func:`get_global_world_size`."""

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
        monkeypatch.setattr(balancing, "dist", stub)
        monkeypatch.setattr(balancing, "get_global_world_size", stub.get_world_size)
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
        (["none", "none"], "none"),
        (["bias_update", "bias_update"], "bias_update"),
    ],
)
def test_a_split_balancing_verdict_converges_by_precedence(stub_dist, gathered, expected):
    """Otherwise stages take different branches of ``apply_balancing_mode``, with different collectives and a raise on
    some ranks."""
    stub_dist(gathered)
    for local in gathered:
        assert balancing.agree_balancing_mode(local) == expected, f"rank resolving {local!r} did not converge"


def test_every_rank_enters_the_agreement_gather(stub_dist):
    stub = stub_dist(["bias_update", "bias_update"])
    balancing.agree_balancing_mode("bias_update")
    assert stub.gathers == ["bias_update"], "the agreed-already rank skipped the gather"


def test_the_agreement_is_the_identity_outside_distributed():
    for mode in ("none", "aux_loss", "bias_update", "bias_update_transient"):
        assert balancing.agree_balancing_mode(mode) == mode


def test_a_clean_rank_still_enters_the_export_contract_collective(monkeypatch):
    """A stage with all native slots has nothing to report, and an ``if reason: raise`` rewrite would strand it in the
    gather while a transient-carrying stage raises."""
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=ValueError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(local_reason)

    monkeypatch.setattr(balancing, "reject_across_ranks", _record)
    model = _Model(_Cfg(), submodules=[_NativeAdoptingEPLayer()])
    apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)

    assert len(calls) == 1, f"the export contract did not reach the consensus seam exactly once: {calls}"
    reason, what = calls[0]
    assert reason is None, f"a fully native model reported a rejection reason: {reason}"
    assert "bias_update export contract" in what


def test_a_transient_rank_reports_its_reason_through_the_same_seam(monkeypatch):
    """The seam carries the reason, so every peer prints the real cause instead of a collective timeout."""
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=ValueError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(local_reason)

    monkeypatch.setattr(balancing, "reject_across_ranks", _record)
    model = _Model(_Cfg(), submodules=[_TransientEPLayer()])
    with pytest.raises(ValueError, match="NO EXPORT CARRIES"):
        apply_balancing_mode(model, "bias_update", policy_gradient_loss=False, is_moe=True)

    assert len(calls) == 1 and calls[0][0] is not None
    assert "bias_update_transient" in calls[0][0], "the carried reason must name the opt-in spelling"


def test_bias_update_transient_is_refused_under_pipeline_parallelism(monkeypatch):
    """Refused before any collective. pp_size is rank-uniform, so every stage raises and none is left waiting."""

    def collective(*args, **kwargs):
        raise AssertionError("a balancing collective ran before the pipeline-parallel refusal")

    monkeypatch.setattr(balancing, "rank_consensus", collective)
    monkeypatch.setattr(balancing, "reject_across_ranks", collective)
    layer = _TransientEPLayer()
    model = _Model(_Cfg(), submodules=[layer])
    with pytest.raises(ValueError, match="does not run under pipeline parallelism"):
        apply_balancing_mode(model, "bias_update_transient", is_moe=True, pp_size=2)
    assert not hasattr(layer, "balancing_biases"), "the refusal came after the side-buffer was attached"


def test_configure_passes_the_pipeline_size_to_the_refusal():
    model = _Model(_Cfg(num_experts=8, num_experts_per_tok=2), submodules=[_TransientEPLayer()])
    with pytest.raises(ValueError, match="does not run under pipeline parallelism"):
        balancing.configure_moe_balancing(
            _perf_args("bias_update_transient"), model, SimpleNamespace(pp_size=2), policy_gradient_loss=False
        )


@pytest.mark.parametrize(("mode", "pp_size"), [("bias_update_transient", 1), ("bias_update", 2)])
def test_the_pipeline_refusal_is_scoped_to_the_transient_mode_under_pp(mode, pp_size):
    layer = _TransientEPLayer() if mode == "bias_update_transient" else _NativeAdoptingEPLayer()
    apply_balancing_mode(_Model(_Cfg(), submodules=[layer]), mode, is_moe=True, pp_size=pp_size)
    assert hasattr(layer, "balancing_biases")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
