"""``finalize_liger_after_direct_load`` decides TRL's ``use_liger_kernel`` from the effective config; the raw user
dict flips it the wrong way."""

import inspect
import json
import logging
import types

import pytest
import torch
import torch.nn.functional as F
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers import AutoConfig
from transformers.loss import loss_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import CONFIG_NAME

from halo.checkpoint.config_export import save_model_config
from halo.model_loading.liger import dispatch
from halo.model_loading.liger.applier import LigerApplier
from halo.model_loading.liger.cross_entropy import _TORCH_CROSS_ENTROPY, liger_cross_entropy
from halo.model_loading.liger.dispatch import finalize_liger_after_direct_load
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.models.attention.gpt_oss_sinks import RESOLVED_SINKS_RESET_ATTR, SinksPolicy
from halo.models.attention.selection import select_supported_attn_implementation
from halo.models.config_access import TOOLKIT_STAMP_PREFIX, config_model_types, config_sources, get_config_field
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from tests.common.liger_appliers import liger_applier
from tests.common.prefix_lm import tiny_paligemma_config, tiny_paligemma_processor
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, TINY_QWEN3_VL_FAMILY, tiny_family_model

_PRE_LOAD_STAMPS = (dispatch.LIGER_APPLIED_CONFIG_ATTR, dispatch.LIGER_CP_FUSED_LOSS_ATTR, RESOLVED_SINKS_RESET_ATTR)


class _RecordingApplier:
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


class _SwigluOffApplier(_RecordingApplier):
    """An applier whose own ``swiglu`` default is False, GptOss's and Qwen3-VL's shape upstream."""

    def __call__(self, rope=True, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=True, swiglu=False):
        super().__call__(rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu)


def _stub_qwen3_moe_applier(monkeypatch):
    """Both registries resolve Qwen3 MoE; stubbing only upstream's entry would rebind the HF module classes for the
    session."""
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "qwen3_moe", applier)
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "qwen3_moe", applier)
    return applier


def _zaya_like_config():
    return types.SimpleNamespace(model_type="zaya", text_config=None)


def _wrapper_config(wrapper: str, text: str):
    text_config = types.SimpleNamespace(model_type=text)
    return types.SimpleNamespace(model_type=wrapper, text_config=text_config, get_text_config=lambda: text_config)


def test_an_appliers_own_swiglu_default_off_is_honored(monkeypatch):
    """The shared default-on must not override an applier that declares the kernel cannot serve it."""
    applier = _SwigluOffApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "family_with_swiglu_off", applier)
    config = types.SimpleNamespace(model_type="family_with_swiglu_off", text_config=None)
    applied = dispatch.apply_liger_kernel(config, None)
    assert applied["swiglu"] is False
    assert applier.calls[0]["swiglu"] is False
    config = types.SimpleNamespace(model_type="family_with_swiglu_off", text_config=None)
    assert dispatch.apply_liger_kernel(config, {"swiglu": True})["swiglu"] is True


def test_flce_is_never_applied_through_a_multimodal_wrappers_text_tower(monkeypatch, caplog):
    """The fused loss patches a causal-LM class the wrapper never instantiates, and trades away the CE kernel that
    reaches the decoder."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "lfm2", applier)
    with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
        applied = dispatch.apply_liger_kernel(_wrapper_config("lfm2_vl", "lfm2"), {"fused_linear_cross_entropy": True})
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is True, "the working CE kernel must not be traded for a dead FLCE"
    assert applier.calls == [applied]
    assert any("multimodal wrapper" in record.getMessage() for record in caplog.records)

    # Anti-vacuity: on a text-only checkpoint the same applier still fuses the head.
    applied = dispatch.apply_liger_kernel(
        types.SimpleNamespace(model_type="lfm2", text_config=None), {"fused_linear_cross_entropy": True}
    )
    assert applied["fused_linear_cross_entropy"] is True and applied["cross_entropy"] is False


def test_a_wrapper_upstream_registers_still_resolves_to_its_towers_toolkit_applier(monkeypatch):
    """liger-kernel's `gemma4` applier runs its swaps over classes the `gemma4_text` spec takes over, so it is never
    called."""
    upstream_wrapper = _RecordingApplier()
    tower = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "gemma4", upstream_wrapper)
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "gemma4_text", tower)
    applied = dispatch.apply_liger_kernel(_wrapper_config("gemma4", "gemma4_text"), None)
    assert tower.calls == [applied]
    assert upstream_wrapper.calls == [], "upstream's wrapper applier ran over the toolkit-owned tower"

    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "wrapper_of_an_upstream_tower", upstream_wrapper)
    dispatch.apply_liger_kernel(_wrapper_config("wrapper_of_an_upstream_tower", "llama"), {"cross_entropy": False})
    assert len(upstream_wrapper.calls) == 1


def test_an_upstream_applier_gets_cross_entropy_off_and_the_scoped_patch(monkeypatch):
    """Upstream's CE branch rebinds ``torch.nn.functional.cross_entropy`` process-wide, so dispatch installs its
    scoped patch instead."""
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "family_upstream_covers", applier)
    original = loss_utils.nn
    try:
        applied = dispatch.apply_liger_kernel(
            types.SimpleNamespace(model_type="family_upstream_covers", text_config=None), None
        )
        assert loss_utils.nn.functional.cross_entropy is liger_cross_entropy
    finally:
        loss_utils.nn = original
    assert applied["cross_entropy"] is True
    assert applier.calls[0]["cross_entropy"] is False
    assert F.cross_entropy is _TORCH_CROSS_ENTROPY


def test_a_prefix_lm_never_gets_the_fused_loss(monkeypatch, caplog):
    """Upstream's PaliGemma fused forward replaces the one that builds the prefix-LM mask from ``token_type_ids``."""
    applier = _RecordingApplier()
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "paligemma", applier)
    config = tiny_paligemma_config(tiny_paligemma_processor())
    request = {"fused_linear_cross_entropy": True}
    original = loss_utils.nn
    try:
        with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
            applied = dispatch.apply_liger_kernel(config, request)
    finally:
        loss_utils.nn = original
    assert applier.calls[0]["fused_linear_cross_entropy"] is False
    assert applied["fused_linear_cross_entropy"] is False and applied["cross_entropy"] is True
    assert any("prefix LM" in record.getMessage() for record in caplog.records)
    resanitized = dispatch.sanitize_liger_config(request, needs_ep_layers=False, model_config=config)
    assert resanitized["fused_linear_cross_entropy"] is False

    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "family_upstream_covers", applier)
    applied = dispatch.apply_liger_kernel(
        types.SimpleNamespace(model_type="family_upstream_covers", text_config=None), request
    )
    assert applied["fused_linear_cross_entropy"] is True and applier.calls[1]["fused_linear_cross_entropy"] is True


def test_per_model_defaults_recorded_as_effective(monkeypatch):
    """Zaya defaults FLCE on with no user dict."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = dispatch.apply_liger_kernel(config, liger_kernel_config=None)
    assert applied is not None
    assert applied["fused_linear_cross_entropy"] is True
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied
    assert applier.calls == [applied]


def test_tp_force_off_recorded_as_effective(monkeypatch):
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, tp_size=2)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_cp_force_off_recorded_as_effective(monkeypatch):
    """Every Liger ``lce_forward`` gates on ``labels is not None``; the Ulysses wrapper passes ``labels=None``, so a
    patched head never engages."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, cp_size=4)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_pp_force_off_recorded_as_effective(monkeypatch):
    """The pipeline drives the stage without labels, so FLCE would report a memory saving that does not exist."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    applied = dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, pp_size=2)
    assert applied["fused_linear_cross_entropy"] is False
    assert applied["cross_entropy"] is False
    assert config._halo_liger_applied_config == applied


def test_pp_off_leaves_the_family_default_alone(monkeypatch):
    """Anti-vacuity: pp_size=1 must keep FLCE, or the test above passes on a constant."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    applied = dispatch.apply_liger_kernel(_zaya_like_config(), None, pp_size=1)
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
    """An axis handled at one application site and not the other would be silently undone by whichever runs second."""
    assert set(dispatch.liger_parallelism_overrides(**axis)) == expected


def test_both_sites_fold_the_overrides_with_the_same_semantics():
    """A blanket ``dict.fromkeys(forced_off, False)`` at either site would silently revert an explicit ``swiglu:
    true``."""
    qwen3_moe = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    axes = {"needs_ep_layers": True, "model_config": qwen3_moe, "tp_size": 2}

    explicit = dispatch.sanitize_liger_config({"swiglu": True, "cross_entropy": True}, **axes)
    assert explicit["swiglu"] is True, "an explicitly requested swiglu must survive the EP override"
    assert explicit["cross_entropy"] is False, "TP makes the fused softmax a partial-vocab slice — not negotiable"

    # Anti-vacuity: without the explicit request the same override turns swiglu off.
    implicit = dispatch.sanitize_liger_config({}, **axes)
    assert implicit["swiglu"] is False, "unrequested swiglu must still be disabled under EP"


def test_ep_swiglu_force_off_only_applies_to_moe_models(monkeypatch):
    """``use_grouped_gemm`` defaults True, so gating on ``needs_ep_layers`` would strip fused SwiGLU from every
    dense model."""
    _stub_qwen3_moe_applier(monkeypatch)
    moe_config = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    moe_applied = dispatch.apply_liger_kernel(moe_config, None, needs_ep_layers=True)
    assert moe_applied["swiglu"] is False, "MoE + EP layers must still disable Liger's fused MLP"

    dense_config = types.SimpleNamespace(model_type="qwen3_moe", text_config=None)
    dense_applied = dispatch.apply_liger_kernel(dense_config, None, needs_ep_layers=True)
    assert dense_applied["swiglu"] is True, "dense model lost Liger's fused SwiGLU to an EP-only filter"


def test_ep_swiglu_force_off_requires_an_ep_layer_class_for_the_family(monkeypatch):
    """Mixtral declares no EP MoE layer, so the soft EP gate must not fire on the intent alone; its swiglu goes off
    through the routed-experts rule."""
    mixtral = types.SimpleNamespace(model_type="mixtral", text_config=None, num_experts=64)
    assert dispatch.liger_ep_disables_fused_glu(True, mixtral) is False, "no EP layer class exists for mixtral"
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "mixtral", _RecordingApplier())
    applied = dispatch.apply_liger_kernel(mixtral, {"swiglu": True}, needs_ep_layers=True)
    assert applied["swiglu"] is False, "unwrapped routed experts must never reach upstream's LigerExperts swap"

    qwen3_moe = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    assert dispatch.liger_ep_disables_fused_glu(True, qwen3_moe) is True
    _stub_qwen3_moe_applier(monkeypatch)
    applied = dispatch.apply_liger_kernel(qwen3_moe, None, needs_ep_layers=True)
    assert applied["swiglu"] is False, "EPQwen3MoELayer wraps this family — Liger's fused MLP is inert"
    explicit = dispatch.apply_liger_kernel(qwen3_moe, {"swiglu": True}, needs_ep_layers=True)
    assert explicit["swiglu"] is True, "an explicit request for the inert swap under the wrapper is honored"


def test_ep_does_not_strip_a_toolkit_appliers_dense_and_shared_expert_glu(monkeypatch):
    """Every EP layer adopts the shared expert unchanged, so the force-off would delete a fusion that still runs on
    every token."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "laguna", applier)
    config = types.SimpleNamespace(model_type="laguna", text_config=None, num_experts=256)
    applied = dispatch.apply_liger_kernel(config, None, needs_ep_layers=True)
    assert applied["swiglu"] is True, "EPLagunaMoELayer adopts LagunaMLP as its shared expert unchanged"

    # Anti-vacuity: an applier whose only GLU patch is the routed-expert swap still loses it.
    _stub_qwen3_moe_applier(monkeypatch)
    upstream = types.SimpleNamespace(model_type="qwen3_moe", text_config=None, num_experts=64)
    assert dispatch.apply_liger_kernel(upstream, None, needs_ep_layers=True)["swiglu"] is False

    # A delegating spec naming its shared-expert MLP keeps the flag for the same reason Laguna does.
    delegating = types.SimpleNamespace(model_type="qwen3_5_moe", text_config=None, num_experts=64)
    assert dispatch.liger_ep_disables_fused_glu(True, delegating) is False


def test_the_family_spec_lookup_resolves_through_the_text_config():
    assert find_moe_family_spec_for_config(types.SimpleNamespace(model_type="qwen3_5_moe")) is not None
    assert find_moe_family_spec_for_config(types.SimpleNamespace(model_type="mixtral")) is None
    composite = types.SimpleNamespace(
        model_type="some_multimodal_wrapper",
        get_text_config=lambda: types.SimpleNamespace(model_type="qwen3_5_moe_text"),
    )
    assert find_moe_family_spec_for_config(composite) is not None


def test_user_requested_flce_yields_the_defaulted_cross_entropy(monkeypatch):
    """Every applier asserts against the defaulted cross_entropy, a flag the user never set."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True})
    assert applied["fused_linear_cross_entropy"] is True
    assert applied["cross_entropy"] is False
    assert applier.calls == [applied], "the applier must actually have run with the resolved pair"


def test_cross_entropy_default_stays_on_without_an_flce_request(monkeypatch):
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = dispatch.apply_liger_kernel(config, None)
    assert applied["cross_entropy"] is True and applied["fused_linear_cross_entropy"] is False


def test_user_setting_both_loss_kernels_is_refused_by_name(monkeypatch):
    """Names both knobs rather than letting a bare assert fire from inside liger_kernel."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    with pytest.raises(ValueError, match="cross_entropy"):
        dispatch.apply_liger_kernel(config, {"cross_entropy": True, "fused_linear_cross_entropy": True})
    assert applier.calls == [], "the contradiction must be caught before the applier runs"


def test_explicit_cross_entropy_false_with_flce_still_applies(monkeypatch):
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "mistral4", applier)
    config = types.SimpleNamespace(model_type="mistral4", text_config=None)
    applied = dispatch.apply_liger_kernel(config, {"cross_entropy": False, "fused_linear_cross_entropy": True})
    assert applied["cross_entropy"] is False and applied["fused_linear_cross_entropy"] is True
    assert applier.calls == [applied]


def test_cp_off_leaves_the_family_default_alone(monkeypatch):
    """Anti-vacuity: cp_size=1 must keep FLCE, or the test above passes on a constant."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    applied = dispatch.apply_liger_kernel(_zaya_like_config(), None, cp_size=1)
    assert applied["fused_linear_cross_entropy"] is True, (
        "Zaya's family default is FLCE-on; cp_size=1 must leave it on so the CP force-off is provably caused by CP."
    )


def test_cp_hands_a_requested_fused_loss_to_its_wrapper_without_a_force_off_warning(monkeypatch, caplog):
    """The wrapper scores each chunk through the fused kernel, so the request is served and the head stays
    unpatched."""
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", applier)
    config = _zaya_like_config()
    with caplog.at_level(logging.WARNING):
        applied = dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, cp_size=2)
    assert getattr(config, dispatch.LIGER_CP_FUSED_LOSS_ATTR) is True
    assert applied["fused_linear_cross_entropy"] is False and applier.calls == [applied]
    assert not [r for r in caplog.records if "fused_linear_cross_entropy" in r.getMessage()]


@pytest.mark.parametrize(
    ("model_type", "user_config", "expected"),
    [
        ("zaya", None, True),
        ("zaya", {"fused_linear_cross_entropy": False}, False),
        ("qwen3_moe", None, False),
        ("qwen3_moe", {"fused_linear_cross_entropy": True}, True),
    ],
    ids=["family_default_on", "explicit_off", "family_default_off", "explicit_on"],
)
def test_cp_resolves_the_fused_loss_as_the_family_and_the_user_resolve_flce(
    monkeypatch, model_type, user_config, expected
):
    applier = _RecordingApplier()
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, model_type, applier)
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, applier)
    config = types.SimpleNamespace(model_type=model_type, text_config=None)
    dispatch.apply_liger_kernel(config, user_config, cp_size=2)
    assert getattr(config, dispatch.LIGER_CP_FUSED_LOSS_ATTR) is expected


def test_cp_keeps_the_logits_loss_where_no_applier_offers_flce(monkeypatch, caplog):
    def no_flce_applier(rope=True, cross_entropy=True, rms_norm=True):
        pass

    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", no_flce_applier)
    config = _zaya_like_config()
    with caplog.at_level(logging.WARNING):
        dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, cp_size=2)
    assert getattr(config, dispatch.LIGER_CP_FUSED_LOSS_ATTR) is False
    assert any("fused_linear_cross_entropy" in r.getMessage() and "does not" in r.getMessage() for r in caplog.records)


def test_no_cp_leaves_no_fused_loss_stamp(monkeypatch):
    monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, "zaya", _RecordingApplier())
    config = _zaya_like_config()
    dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True}, cp_size=1)
    assert not hasattr(config, dispatch.LIGER_CP_FUSED_LOSS_ATTR)


def _is_stamp(key: str) -> bool:
    return key.startswith(TOOLKIT_STAMP_PREFIX) or key in _PRE_LOAD_STAMPS


def _stamped_keys(payload: dict, path: str = "") -> list[str]:
    found = [f"{path}{key}" for key in payload if _is_stamp(key)]
    for key, value in payload.items():
        if isinstance(value, dict):
            found += _stamped_keys(value, f"{path}{key}.")
    return found


@pytest.mark.parametrize(
    "family",
    [TINY_DENSE_FAMILY, TINY_QWEN3_VL_FAMILY, TINY_MOE_FAMILIES["gpt_oss"]],
    ids=["qwen3", "qwen3_vl", "gpt_oss"],
)
def test_an_export_carries_no_run_scoped_halo_keys(monkeypatch, tmp_path, family):
    """A later run on the checkpoint must not re-arm the fused loss or inherit this run's sinks verdict."""
    torch.manual_seed(0)
    tiny_family_model(family).save_pretrained(tmp_path / "source")
    config = AutoConfig.from_pretrained(tmp_path / "source")
    applier = _RecordingApplier()
    for model_type in config_model_types(config):
        monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, applier)
        monkeypatch.setitem(dispatch._TOOLKIT_LIGER_APPLIERS, model_type, applier)
    select_supported_attn_implementation(config, "eager", SinksPolicy.NEUTRALIZED)
    # ``use_grouped_gemm``'s default sets ``needs_ep_layers``, so a GPT-OSS run's experts are the EP layer's.
    applied = dispatch.apply_liger_kernel(
        config, {"fused_linear_cross_entropy": True}, needs_ep_layers=True, cp_size=2
    )
    model = family.load_class.from_pretrained(tmp_path / "source", config=config)
    assert model.config is not config

    save_model_config(model, str(tmp_path / "export"))

    written = json.loads((tmp_path / "export" / CONFIG_NAME).read_text())
    assert _stamped_keys(written) == []
    reloaded = AutoConfig.from_pretrained(tmp_path / "export")
    assert [key for level in config_sources(reloaded) for key in vars(level) if _is_stamp(key)] == []
    assert get_config_field(model.config, dispatch.LIGER_CP_FUSED_LOSS_ATTR) is True
    assert get_config_field(model.config, dispatch.LIGER_APPLIED_CONFIG_ATTR) == applied
    assert get_config_field(model.config, RESOLVED_SINKS_RESET_ATTR) is True


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm4_moe_lite", "zaya"])
def test_the_large_vocab_families_still_default_to_flce(model_type):
    """Losing the default silently puts GLM-4.7-Flash, DeepSeek-V4 and Zaya back on the full ``[seq, vocab]`` logits
    plane."""
    defaults = dispatch._PER_MODEL_DEFAULTS[model_type]
    assert defaults == {"cross_entropy": False, "fused_linear_cross_entropy": True}


def test_every_toolkit_applier_is_registered_in_the_toolkit_registry():
    """A family self-registering into liger_kernel's own dict would resolve on the upstream branch, where
    ``_PER_MODEL_DEFAULTS`` never looks."""
    declared = {model_type for spec in LIGER_FAMILY_SPECS for model_type in spec.model_types}
    delegating = {
        model_type for spec in LIGER_FAMILY_SPECS if spec.delegates_to_upstream for model_type in spec.model_types
    }
    assert declared == set(dispatch._TOOLKIT_LIGER_APPLIERS)
    assert delegating <= set(MODEL_TYPE_TO_APPLY_LIGER_FN), (
        f"{sorted(delegating - set(MODEL_TYPE_TO_APPLY_LIGER_FN))} delegate to an upstream applier that "
        f"liger_kernel no longer registers"
    )
    self_registered = sorted(t for t, fn in MODEL_TYPE_TO_APPLY_LIGER_FN.items() if isinstance(fn, LigerApplier))
    assert not self_registered, f"toolkit appliers registered into liger_kernel's registry: {self_registered}"
    toolkit = dispatch._TOOLKIT_LIGER_APPLIERS
    shadowed = sorted(t for t in declared if liger_applier(t) is not toolkit[t])
    assert not shadowed, f"toolkit families resolving to liger_kernel's own applier: {shadowed}"


def test_no_applier_warns_and_records_nothing(monkeypatch, caplog):
    """``use_liger_kernel`` defaults on, so an uncovered family must say it is training unfused."""
    monkeypatch.setattr(dispatch, "_TOOLKIT_LIGER_APPLIERS", {})
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
        applied = dispatch.apply_liger_kernel(config, None)
    assert applied is None
    assert config._halo_liger_applied_config is None
    assert any(
        record.levelno >= logging.WARNING and "model_type_without_any_applier" in record.getMessage()
        for record in caplog.records
    ), f"uncovered family trained unfused with no warning: {[r.getMessage() for r in caplog.records]}"


def test_no_applier_refuses_an_explicit_kernel_request(monkeypatch):
    monkeypatch.setattr(dispatch, "_TOOLKIT_LIGER_APPLIERS", {})
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    with pytest.raises(ValueError, match="fused_linear_cross_entropy"):
        dispatch.apply_liger_kernel(config, {"fused_linear_cross_entropy": True})


def test_the_ep_swiglu_force_off_is_not_read_as_a_user_request(monkeypatch):
    """Under EP the fold writes ``swiglu: False`` with no user dict; read as a request it would fail every uncovered
    family. ``wraps_experts`` is forced True, or the fold never runs and the test passes vacuously."""
    monkeypatch.setattr(dispatch, "_TOOLKIT_LIGER_APPLIERS", {})
    monkeypatch.setattr(dispatch, "wraps_experts", lambda *a, **k: True)
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None, num_experts=8)

    assert dispatch.sanitize_liger_config({}, needs_ep_layers=True, model_config=config) == {
        "swiglu": False,
        "geglu": False,
    }

    assert dispatch.apply_liger_kernel(config, None, needs_ep_layers=True) is None

    # A kernel the user explicitly turned OFF is not a request either.
    config = types.SimpleNamespace(model_type="model_type_without_any_applier", text_config=None)
    assert dispatch.apply_liger_kernel(config, {"rms_norm": False}) is None


def _training_config(user_config):
    return types.SimpleNamespace(use_liger_kernel=True, liger_kernel_config=user_config)


def _model_with_applied(applied):
    return types.SimpleNamespace(config=types.SimpleNamespace(_halo_liger_applied_config=applied))


def test_finalize_keeps_flag_for_effective_flce():
    """Zaya: the model returns no logits, and TRL's non-liger branches would slice a 1-token logit tensor."""
    training_config = _training_config(None)
    applied = {"rope": False, "cross_entropy": False, "fused_linear_cross_entropy": True, "rms_norm": True}
    finalize_liger_after_direct_load(training_config, True, _model_with_applied(applied))
    assert training_config.use_liger_kernel is True
    assert training_config.liger_kernel_config == applied


def test_finalize_turns_the_withheld_roles_off_for_trls_reapplication():
    """Upstream's instance patch would bind the llama-cast norm over GptOss's Gemma-cast one."""
    applied = {
        "rope": True,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
        "swiglu": False,
    }
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="gpt_oss", text_config=None, _halo_liger_applied_config=applied)
    )
    training_config = _training_config(None)
    finalize_liger_after_direct_load(training_config, True, model)
    assert training_config.use_liger_kernel is True
    assert training_config.liger_kernel_config == {**applied, "rms_norm": False}
    assert model.config._halo_liger_applied_config == applied, "the effective record must not change"

    # Through a wrapper's text config too: Gemma 4's dense GeGLU and its norms are the toolkit's.
    wrapper = _wrapper_config("gemma4", "gemma4_text")
    geglu_applied = {"rope": False, "cross_entropy": False, "fused_linear_cross_entropy": True, "geglu": True}
    reapplied = dispatch.trl_reapplication_config(wrapper, geglu_applied)
    assert reapplied["geglu"] is False and reapplied["rms_norm"] is False
    assert {flag for flag, on in reapplied.items() if on} == {"fused_linear_cross_entropy"}
    assert dispatch.trl_reapplication_config(_zaya_like_config(), applied) == applied


def test_reapplication_runs_nothing_upstream_registers_for_a_family_the_toolkit_patches_alone(monkeypatch):
    """DeepSeek-V4's registered applier would swap the norms the spec leaves eager and replace the toolkit's fused
    head, so it receives every flag off."""
    received = []

    def upstream(rope=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True, model=None):
        received.append({"rope": rope, "flce": fused_linear_cross_entropy, "rms_norm": rms_norm, "swiglu": swiglu})

    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "zaya", upstream)
    applied = {"rope": False, "cross_entropy": False, "fused_linear_cross_entropy": True, "rms_norm": True}
    reapplied = dispatch.trl_reapplication_config(_zaya_like_config(), applied)
    _apply_liger_kernel_to_instance(model=types.SimpleNamespace(config=_zaya_like_config()), **reapplied)
    assert received == [{"rope": False, "flce": False, "rms_norm": False, "swiglu": False}]


def test_reapplication_hands_no_installed_upstream_applier_a_role_the_toolkit_owns():
    """Swept over the installed liger-kernel's registry, each type resolved as its checkpoint would be."""
    leaks, owned_types = [], []
    for model_type, reapplier in MODEL_TYPE_TO_APPLY_LIGER_FN.items():
        config = (
            CONFIG_MAPPING[model_type]()
            if model_type in CONFIG_MAPPING
            else types.SimpleNamespace(model_type=model_type, text_config=None)
        )
        toolkit = dispatch._toolkit_applier(*dispatch._resolve_applier(*dispatch._liger_model_types(config)))
        if toolkit is None:
            continue
        owned_types.append(model_type)
        reapplied = dispatch.trl_reapplication_config(
            config, {flag: True for flag in toolkit.defaults if flag != "model"}
        )
        for flag, parameter in inspect.signature(reapplier).parameters.items():
            if flag != "model" and reapplied.get(flag, parameter.default) and not toolkit.hands_upstream(flag):
                leaks.append(f"{model_type}: {flag}")
    assert "gpt_oss" in owned_types, "the sweep reached no toolkit-owned type"
    assert not leaks, f"HF's re-application would run these toolkit-owned roles through upstream: {leaks}"


def test_finalize_disables_flag_when_flce_forced_off():
    """The model returns real logits; keeping the flag would drop TRL onto the no-logits branches."""
    training_config = _training_config({"fused_linear_cross_entropy": True})
    finalize_liger_after_direct_load(training_config, True, _model_with_applied({"fused_linear_cross_entropy": False}))
    assert training_config.use_liger_kernel is False


def test_finalize_disables_flag_when_nothing_applied():
    # Liger unavailable / no applier for the model type: nothing was patched, flag must go off even
    # though the user dict requested FLCE.
    training_config = _training_config({"fused_linear_cross_entropy": True})
    finalize_liger_after_direct_load(training_config, True, _model_with_applied(None))
    assert training_config.use_liger_kernel is False

    training_config = _training_config({"fused_linear_cross_entropy": True})
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    finalize_liger_after_direct_load(training_config, True, model)
    assert training_config.use_liger_kernel is False


def test_finalize_noop_when_liger_disabled():
    training_config = types.SimpleNamespace(use_liger_kernel=False, liger_kernel_config=None)
    finalize_liger_after_direct_load(training_config, False, _model_with_applied({"fused_linear_cross_entropy": True}))
    assert training_config.use_liger_kernel is False


def _remote_only_warnings(config, caplog, monkeypatch) -> list[str]:
    monkeypatch.setattr(dispatch, "_run_resolved_applier", lambda *_args, **_kwargs: None)
    with caplog.at_level(logging.WARNING, logger="halo.model_loading.liger.dispatch"):
        dispatch.apply_liger_kernel(config, None)
    return [record.getMessage() for record in caplog.records if "patches only" in record.getMessage()]


def test_a_remote_only_spec_warns_on_transformers_own_class(caplog, monkeypatch):
    """The applier arms on a module that never loads, and its applied line would be all the run shows."""
    assert _remote_only_warnings(CONFIG_MAPPING["bailing_hybrid"](num_experts=4), caplog, monkeypatch)


def test_a_remote_only_spec_stays_quiet_on_the_remote_class(caplog, monkeypatch):
    native = CONFIG_MAPPING["bailing_hybrid"]
    remote = type(native.__name__, (native,), {"__module__": "transformers_modules.x"})
    assert not _remote_only_warnings(remote(num_experts=4), caplog, monkeypatch)


def test_a_spec_with_a_modeling_module_stays_quiet_on_the_native_class(caplog, monkeypatch):
    """Laguna names remote classes too, and its in-library module patches on the native class."""
    assert not _remote_only_warnings(CONFIG_MAPPING["laguna"](), caplog, monkeypatch)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
