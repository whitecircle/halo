"""CPU tests for SDPA on GptOss (a sinks model) when sinks are reset for fine-tuning.

GptOss sets ``_supports_sdpa = False`` because SDPA computes the softmax internally and can't add the
per-head sink logit. But fine-tuning resets sinks to dtype min (``_reset_gpt_oss_sinks``), so the
dropped sink column contributes ``exp(dtype_min) == 0`` — SDPA then matches eager/flex exactly (the
same reasoning that makes FA2 valid). ``validate_attn_implementation`` keeps an explicit ``sdpa``
request for GptOss ONLY when ``sinks_reset=True``; with live sinks every sink-dropping impl
(sdpa, FA2) raises — silent fallback would still drop the sinks. Non-sinks models are unaffected.

    python tests/cpu/models/test_gptoss_sdpa.py
"""

import sys
import types

import pytest
import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, GptOssConfig, LlamaConfig
from trl import ModelConfig

from src.distributed.loading.frozen_models import load_reference_model_for_preference
from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.liger.orchestrator import LIGER_APPLIED_CONFIG_ATTR
from src.models.loading.config_levels import (
    config_export_ready,
    set_config_field_run_scoped,
)
from src.models.patches.attention import (
    _attn_impl_handles_sinks,
    _enable_sink_model_sdpa,
    validate_attn_implementation,
)

PartialState()  # the loader's logger needs accelerate state initialized


def _gptoss():
    return GptOssConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_local_experts=2,
        num_experts_per_tok=1,
        vocab_size=128,
    )


def _llama():
    return LlamaConfig(
        num_hidden_layers=1, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, vocab_size=128
    )


def test_sdpa_kept_for_gptoss_when_sinks_reset():
    assert validate_attn_implementation(_gptoss(), "sdpa", sinks_reset=True) == "sdpa"
    # The class guard was flipped so transformers' _sdpa_can_dispatch will accept it at model build.
    assert AutoModelForCausalLM._model_mapping[GptOssConfig]._supports_sdpa is True


def test_sink_dropping_impls_rejected_for_gptoss_without_reset():
    """Live (unreset) sinks: sdpa and FA2 silently drop the real sink column — measured ~-3 nats of
    logprob shift on gpt-oss-20b. Must raise, never silently fall back (a fallback to FA2 drops the
    sinks just the same); the sink-carrying impls (eager, flex) stay allowed."""
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(_gptoss(), "sdpa", sinks_reset=False)
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(_gptoss(), "flash_attention_2", sinks_reset=False)
    assert validate_attn_implementation(_gptoss(), "eager", sinks_reset=False) == "eager"
    assert validate_attn_implementation(_gptoss(), "flex_attention", sinks_reset=False) == "flex_attention"


def test_enable_sink_model_sdpa_idempotent():
    cfg = _gptoss()
    _enable_sink_model_sdpa(cfg)  # ensure flipped
    assert _enable_sink_model_sdpa(cfg) is False  # already enabled → no-op
    assert AutoModelForCausalLM._model_mapping[GptOssConfig]._supports_sdpa is True


def test_non_sinks_model_sdpa_passthrough():
    # A non-sinks model keeps sdpa regardless of sinks_reset (the gate only affects sinks models).
    assert validate_attn_implementation(_llama(), "sdpa", sinks_reset=False) == "sdpa"


def test_sinks_reset_state_is_per_config_instance():
    """Sinks state must be per-config, not on the config CLASS: class-level state leaks across loads
    in one process, the last-validated load silently overwriting every other GptOssConfig's — a
    reset-sinks reference load then disarms the live-sinks guard the policy load established."""
    live = _gptoss()
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(live, "sdpa", sinks_reset=False)
    reset = _gptoss()
    assert validate_attn_implementation(reset, "sdpa", sinks_reset=True) == "sdpa"

    # State lives on each instance, never on the shared class.
    assert "_halo_sinks_reset" not in type(live).__dict__
    assert live._halo_sinks_reset is False
    assert reset._halo_sinks_reset is True

    # Re-validation contexts (EP/CP/lazy loaders pass sinks_reset=None) enforce per-instance state:
    # the live-sinks config still raises even though a reset-sinks config validated after it.
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(live, "sdpa")
    assert validate_attn_implementation(reset, "sdpa") == "sdpa"


def test_a_fresh_config_inherits_no_sdpa_approval():
    """The sdpa escape hatch must be recorded where the refusal it overrides is — on the config
    INSTANCE, never on the model CLASS.

    A class-level flag is flipped for the whole process by one reset-sinks load, so any later
    re-validation with nothing of its own to go on (EP/CP/lazy loaders pass ``sinks_reset=None``, and
    a loader that built its own config carries no record) reads that stale approval and accepts sdpa
    for a model whose sinks were never reset. sdpa drops the sink column from the softmax silently.
    """
    approved = _gptoss()
    assert validate_attn_implementation(approved, "sdpa", sinks_reset=True) == "sdpa"

    fresh = _gptoss()
    assert validate_attn_implementation(fresh, "sdpa") != "sdpa"
    assert not hasattr(fresh, "_halo_sinks_reset"), "a re-validation must record nothing of its own"

    model_cls = AutoModelForCausalLM._model_mapping[GptOssConfig]
    assert not hasattr(model_cls, "_halo_sdpa_sinks_reset"), (
        "the sinks approval must live on the config instance the run loaded, never on the class"
    )


def _fake_fa3_module(name: str, *, with_s_aux: bool) -> types.ModuleType:
    """A stand-in FA3 build whose VARLEN entry point does or does not accept the sink argument.

    Only the varlen function is defined: it is the one transformers inspects, so a probe that read
    the dense twin instead would find nothing here and report every build sink-less.
    """
    mod = types.ModuleType(name)
    if with_s_aux:

        def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, s_aux=None):
            raise NotImplementedError

    else:

        def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None):
            raise NotImplementedError

    mod.flash_attn_varlen_func = flash_attn_varlen_func
    return mod


def test_fa3_sink_probe_prefers_flash_attn_interface(monkeypatch):
    """transformers dispatches flash_attention_3 via ``flash_attn_interface``; when both modules are
    installed, the sink capability must be read from the kernel actually dispatched — a coexisting
    sink-capable ``flash_attn_3`` build must not mask a sink-less interface build (or vice versa)."""
    monkeypatch.setitem(
        sys.modules, "flash_attn_interface", _fake_fa3_module("flash_attn_interface", with_s_aux=False)
    )
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is False

    monkeypatch.setitem(sys.modules, "flash_attn_interface", _fake_fa3_module("flash_attn_interface", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is True


def test_fa3_sink_probe_falls_back_to_flash_attn_3(monkeypatch):
    # A None sys.modules entry makes the import raise ImportError → the legacy module is probed.
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is True
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=False))
    assert _attn_impl_handles_sinks("flash_attention_3") is False


@pytest.fixture(scope="module")
def tiny_gptoss_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_gptoss")
    AutoModelForCausalLM.from_config(_gptoss()).save_pretrained(path)
    return str(path)


def _ref_load_stubs(checkpoint: str, attn_implementation: str | None = None):
    args = types.SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )
    model_config = ModelConfig(model_name_or_path=checkpoint, attn_implementation=attn_implementation)
    training_config = types.SimpleNamespace()  # no bf16/fp16 flags → fp32 reference
    # Every ``<special>_token_id`` accompanies its token: ``PreTrainedTokenizerBase.__getattr__``
    # resolves them, so no real tokenizer can carry one without the other, and the setup seam the
    # loader runs records the pad id on the model.
    tokenizer = types.SimpleNamespace(
        eos_token="<e>",
        eos_token_id=1,
        bos_token="<b>",
        bos_token_id=2,
        pad_token="<p>",
        pad_token_id=3,
        chat_template="{{ messages }}",
    )
    return args, model_config, training_config, ParallelismConfig(), tokenizer


def test_reference_model_load_applies_sink_reset(tiny_gptoss_checkpoint):
    """The DPO/KTO full-finetune reference must go through the same attn validation + sink handling as
    the policy: with reset sinks the default sdpa is validated (GptOss natively rejects it) and the
    reference's sinks are neutralized to dtype min, matching the policy forward."""
    args, model_config, training_config, parallelism_config, tokenizer = _ref_load_stubs(tiny_gptoss_checkpoint)
    model_ref = load_reference_model_for_preference(
        args, model_config, training_config, parallelism_config, tokenizer, is_vlm=False, method="DPO"
    )
    sinks = model_ref.model.layers[0].self_attn.sinks
    assert torch.all(sinks == torch.finfo(sinks.dtype).min)


def test_reference_model_load_rejects_sink_dropping_impl_with_live_sinks(tiny_gptoss_checkpoint):
    """An EXPLICIT sink-dropping impl must still raise under ``reset_sinks=False`` (on-policy RL).

    The reference resolves its backend through the same ``resolve_attn_implementation`` as the policy,
    so the *unset* case is not pinned to sdpa — it lands on a sink-carrying impl (covered by
    the test below). What must never happen is a user-requested sdpa/FA2 silently dropping live sink
    mass: every reference logprob would shift by nats against the served policy.
    """
    args, model_config, training_config, parallelism_config, tokenizer = _ref_load_stubs(
        tiny_gptoss_checkpoint, attn_implementation="sdpa"
    )
    with pytest.raises(ValueError, match="sink"):
        load_reference_model_for_preference(
            args,
            model_config,
            training_config,
            parallelism_config,
            tokenizer,
            is_vlm=False,
            method="DPO",
            reset_sinks=False,
        )


def test_reference_model_auto_resolves_to_a_sink_carrying_impl(tiny_gptoss_checkpoint):
    """With no explicit request, live sinks must resolve to an impl that CARRIES them.

    Pinning the reference to ``sdpa`` while the policy auto-detects (FA4 on Blackwell) runs the two
    legs of every logratio on different kernels; both read one resolver instead. The property that
    matters is the outcome: whatever it picks must carry the sink column, since the model is loaded
    with live sinks.
    """
    args, model_config, training_config, parallelism_config, tokenizer = _ref_load_stubs(tiny_gptoss_checkpoint)
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        training_config,
        parallelism_config,
        tokenizer,
        is_vlm=False,
        method="DPO",
        reset_sinks=False,
    )
    resolved = model_ref.config._attn_implementation
    assert _attn_impl_handles_sinks(resolved), (
        f"reference resolved to {resolved!r}, which drops the live attention sinks"
    )


def test_reference_model_load_freezes_live_sinks(tiny_gptoss_checkpoint):
    args, model_config, training_config, parallelism_config, tokenizer = _ref_load_stubs(
        tiny_gptoss_checkpoint, attn_implementation="eager"
    )
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        training_config,
        parallelism_config,
        tokenizer,
        is_vlm=False,
        method="DPO",
        reset_sinks=False,
    )
    sinks = model_ref.model.layers[0].self_attn.sinks
    assert sinks.requires_grad is False  # live sinks kept but frozen, mirroring the policy load


def test_toolkit_run_state_never_reaches_the_exported_config():
    """``_halo_*`` stamps describe THIS RUN, not the artifact, and transformers serializes every
    instance attribute — so an exported ``config.json`` would ship them to the hub and to every
    serving engine. They are written run-scoped, so the export-ready window strips them and the
    run keeps reading them afterwards.
    """
    cfg = _gptoss()
    validate_attn_implementation(cfg, "sdpa", sinks_reset=True)
    set_config_field_run_scoped(cfg, LIGER_APPLIED_CONFIG_ATTR, {"rope": True})
    assert cfg._halo_sinks_reset is True and getattr(cfg, LIGER_APPLIED_CONFIG_ATTR) == {"rope": True}

    with config_export_ready(cfg):
        exported = cfg.to_dict()
        halo_keys = [key for key in exported if key.startswith("_halo_")]
        assert halo_keys == [], f"run state leaked into the exported config: {halo_keys}"

    # The run still decides on its own state after the export window closes.
    assert cfg._halo_sinks_reset is True
    assert validate_attn_implementation(cfg, "sdpa") == "sdpa"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
