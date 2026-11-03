"""SDPA on GptOss when sinks are reset for fine-tuning.
A reset sink column contributes ``exp(dtype_min) == 0``, so SDPA matches eager; over live sinks it must raise."""

import functools
import sys
import types

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    GptOssConfig,
    GptOssForCausalLM,
    GptOssForSequenceClassification,
    LlamaConfig,
    PreTrainedModel,
)

from halo.configs.script_arguments import ModelArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import reward_scoring
from halo.model_loading.liger.dispatch import LIGER_APPLIED_CONFIG_ATTR
from halo.model_loading.model_preparation import apply_run_model_policies
from halo.model_loading.parallel.frozen_models import load_reference_model_for_preference
from halo.models.attention import gpt_oss_sinks
from halo.models.attention.gpt_oss_sinks import SinksPolicy, install_sdpa_for_reset_sinks
from halo.models.attention.selection import (
    _attn_impl_handles_sinks,
    validate_attn_implementation,
    verify_live_sinks_carried,
)
from halo.models.config_access import (
    TOOLKIT_STAMP_PREFIX,
    config_export_ready,
    set_config_field_run_scoped,
)
from tests.common.repo import probe_findings
from tests.common.tiny_models import CONTENT_ONLY_CHAT_TEMPLATE, word_level_tokenizer

_TINY_GPTOSS = {
    "num_hidden_layers": 1,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "num_local_experts": 2,
    "num_experts_per_tok": 1,
    "vocab_size": 128,
}


def _gptoss():
    return GptOssConfig(**_TINY_GPTOSS)


def _llama():
    return LlamaConfig(
        num_hidden_layers=1, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, vocab_size=128
    )


def test_sdpa_kept_for_gptoss_when_sinks_reset():
    assert validate_attn_implementation(_gptoss(), "sdpa", SinksPolicy.NEUTRALIZED) == "sdpa"
    # The class guard was flipped so transformers' _sdpa_can_dispatch will accept it at model build.
    assert AutoModelForCausalLM._model_mapping[GptOssConfig]._supports_sdpa is True


def test_sink_dropping_impls_rejected_for_gptoss_without_reset():
    """sdpa and FA2 drop the real sink column, measured ~-3 nats of logprob shift on gpt-oss-20b.
    Must raise: a fallback to FA2 drops the sinks just the same."""
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(_gptoss(), "sdpa", SinksPolicy.LIVE)
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(_gptoss(), "flash_attention_2", SinksPolicy.LIVE)
    assert validate_attn_implementation(_gptoss(), "eager", SinksPolicy.LIVE) == "eager"
    assert validate_attn_implementation(_gptoss(), "flex_attention", SinksPolicy.LIVE) == "flex_attention"


# A tiny GptOss built in a fresh interpreter after importing ``module`` (empty: transformers alone), then the label
# transformers' own check settles on. The kernel import and the device check are stubbed.
_FLASH_LABEL_PROBE = """
import importlib
if {module!r}:
    importlib.import_module({module!r})
import transformers.modeling_utils as modeling_utils
from transformers import GptOssConfig, GptOssForCausalLM
modeling_utils.lazy_import_flash_attention = lambda *args, **kwargs: None
model = GptOssForCausalLM(GptOssConfig(**{config!r}))
model._flash_attn_can_dispatch = lambda *args, **kwargs: True
labels = [model._check_and_adjust_attn_implementation(label) for label in ("flash_attention_2", "flash_attention_3")]
print("LABELS:" + "|".join(labels))
"""


def _flash_labels(module: str) -> list[str]:
    return probe_findings(_FLASH_LABEL_PROBE.format(module=module, config=_TINY_GPTOSS), "LABELS:")


@pytest.mark.parametrize(
    "module",
    [
        "halo.model_loading.checkpoint_coverage",
        "halo.entrypoints.reward_scoring",
        "halo.checkpoint.tools.adapter_merge",
        "scripts.training.embedding",
    ],
)
def test_a_gptoss_flash_label_stays_on_the_local_build_wherever_the_toolkit_builds_one(module):
    """transformers swaps a GptOss FA2/FA3 label for hub vllm-flash-attn3, whose ``kernels`` package the image lacks.
    These builds take the user's label with no validator, so importing any of them must drop the swap process-wide."""
    assert _flash_labels(module) == ["flash_attention_2", "flash_attention_3"]


def test_transformers_alone_swaps_a_gptoss_flash_label_for_the_hub_kernel():
    assert _flash_labels("") == ["kernels-community/vllm-flash-attn3"] * 2


def _labelled_gptoss(model_class, attn_implementation: str, sinks: str):
    """``neutralized`` sinks hold ``dtype.min``, ``neutralized_bf16`` bf16's minimum (a bf16 save read at fp32),
    and ``removed`` sinks are ``None`` (the FA2 reset). Built eager: no flash kernel runs on CPU."""
    model = model_class(GptOssConfig(**_TINY_GPTOSS, num_labels=1, pad_token_id=0))
    model.config._attn_implementation = attn_implementation
    values = {
        "live": 0.5,
        "neutralized": torch.finfo(torch.float32).min,
        "neutralized_bf16": torch.finfo(torch.bfloat16).min,
    }
    for layer in model.model.layers:
        attn = layer.self_attn
        if sinks == "removed":
            attn.sinks = None
        else:
            attn.sinks.data.fill_(values[sinks])
    return model


@pytest.mark.parametrize(
    ("attn_implementation", "sinks", "refused"),
    [
        ("flash_attention_2", "live", True),
        ("sdpa", "live", True),
        ("flash_attention_2", "removed", False),
        ("flash_attention_2", "neutralized", False),
        ("flash_attention_2", "neutralized_bf16", False),
        ("sdpa", "neutralized", False),
        ("eager", "live", False),
        ("flex_attention", "live", False),
    ],
)
def test_a_built_gptoss_that_drops_its_live_sinks_is_refused(attn_implementation, sinks, refused):
    """A sink-dropping backend is exact over neutralized sinks and shifts every logprob by nats over live ones."""
    model = _labelled_gptoss(GptOssForCausalLM, attn_implementation, sinks)
    if refused:
        with pytest.raises(ValueError, match="drops its attention sinks"):
            verify_live_sinks_carried(model)
    else:
        verify_live_sinks_carried(model)


@pytest.mark.parametrize(("policy", "refused"), [(SinksPolicy.LIVE, True), (SinksPolicy.NEUTRALIZED, False)])
def test_the_run_policies_refuse_a_backend_that_drops_the_sinks_the_policy_keeps(policy, refused):
    """The sentence-transformers backbone on a pinned FA2 has no validator, so these policies must end in the check."""
    model = _labelled_gptoss(GptOssForCausalLM, "flash_attention_2", "live")
    apply = functools.partial(
        apply_run_model_policies, model, model.config, sinks_policy=policy, attn_implementation="flash_attention_2"
    )
    if refused:
        with pytest.raises(ValueError, match="drops its attention sinks"):
            apply()
    else:
        apply()
        assert all(layer.self_attn.sinks is None for layer in model.model.layers)


@pytest.mark.parametrize(("sinks", "refused"), [("live", True), ("neutralized", False)])
def test_the_reward_scorer_refuses_a_flash_label_over_a_checkpoint_with_live_sinks(
    monkeypatch, tmp_path, sinks, refused
):
    """The scorer applies no sinks policy, so ``--rm_attn_implementation flash_attention_2`` would drop them."""
    tokenizer = word_level_tokenizer(("a", "b"))
    tokenizer.chat_template = CONTENT_ONLY_CHAT_TEMPLATE
    _labelled_gptoss(GptOssForSequenceClassification, "eager", sinks).save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    build = reward_scoring.from_pretrained_verified

    def build_labelled(model_class, path, *, attn_implementation, **kwargs):
        model = build(model_class, path, attn_implementation="eager", **kwargs)
        model.config._attn_implementation = attn_implementation
        return model

    monkeypatch.setattr(reward_scoring, "from_pretrained_verified", build_labelled)
    load = functools.partial(
        reward_scoring.load_reward_model,
        str(tmp_path),
        "flash_attention_2",
        64,
        "cpu",
        "float32",
        trust_remote_code=False,
    )
    if refused:
        with pytest.raises(ValueError, match="drops its attention sinks"):
            load()
    else:
        assert load()[1].config._attn_implementation == "flash_attention_2"


def test_install_sdpa_for_reset_sinks_is_idempotent():
    cfg = _gptoss()
    install_sdpa_for_reset_sinks(cfg)
    assert install_sdpa_for_reset_sinks(cfg) is False
    assert AutoModelForCausalLM._model_mapping[GptOssConfig]._supports_sdpa is True


def test_non_sinks_model_sdpa_passthrough():
    assert validate_attn_implementation(_llama(), "sdpa", SinksPolicy.LIVE) == "sdpa"


def test_sinks_reset_state_is_per_config_instance():
    """Class-level state leaks across loads: a reset-sinks reference would disarm the policy's live-sinks guard."""
    live = _gptoss()
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(live, "sdpa", SinksPolicy.LIVE)
    reset = _gptoss()
    assert validate_attn_implementation(reset, "sdpa", SinksPolicy.NEUTRALIZED) == "sdpa"

    assert "_halo_sinks_reset" not in type(live).__dict__
    assert live._halo_sinks_reset is False
    assert reset._halo_sinks_reset is True

    # EP/CP/lazy loaders pass no sinks_policy: the live config still raises after a reset config validated.
    with pytest.raises(ValueError, match="sink"):
        validate_attn_implementation(live, "sdpa")
    assert validate_attn_implementation(reset, "sdpa") == "sdpa"


def test_a_fresh_config_inherits_no_sdpa_approval():
    """A class-level flag flipped by one reset-sinks load would let a later re-validation (EP/CP/lazy loaders pass no
    ``sinks_policy``) accept sdpa for a model whose sinks were never reset, silently dropping the sink column."""
    approved = _gptoss()
    assert validate_attn_implementation(approved, "sdpa", SinksPolicy.NEUTRALIZED) == "sdpa"

    fresh = _gptoss()
    assert validate_attn_implementation(fresh, "sdpa") != "sdpa"
    assert not hasattr(fresh, "_halo_sinks_reset"), "a re-validation must record nothing of its own"

    model_cls = AutoModelForCausalLM._model_mapping[GptOssConfig]
    assert not hasattr(model_cls, "_halo_sdpa_sinks_reset"), (
        "the sinks approval must live on the config instance the run loaded, never on the class"
    )


def _fake_fa3_module(name: str, *, with_s_aux: bool) -> types.ModuleType:
    """Only the varlen function exists: a probe reading the dense twin would report every build sink-less."""
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
    """The capability must be read from the kernel transformers dispatches, not a coexisting ``flash_attn_3`` build."""
    monkeypatch.setitem(
        sys.modules, "flash_attn_interface", _fake_fa3_module("flash_attn_interface", with_s_aux=False)
    )
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is False

    monkeypatch.setitem(sys.modules, "flash_attn_interface", _fake_fa3_module("flash_attn_interface", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is True


def test_fa3_sink_probe_falls_back_to_flash_attn_3(monkeypatch):
    # A None sys.modules entry makes the import raise, so the legacy module is probed.
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=True))
    assert _attn_impl_handles_sinks("flash_attention_3") is True
    monkeypatch.setitem(sys.modules, "flash_attn_3", _fake_fa3_module("flash_attn_3", with_s_aux=False))
    assert _attn_impl_handles_sinks("flash_attention_3") is False


def test_a_live_sinks_request_for_an_uninstalled_flash_build_says_it_is_not_installed(monkeypatch):
    """It must name the missing build: an installed FA3 build exposing ``s_aux`` would carry the sinks."""
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setitem(sys.modules, "flash_attn_3", None)
    with pytest.raises(ValueError, match="'flash_attention_3' is not installed") as refusal:
        validate_attn_implementation(_gptoss(), "flash_attention_3", SinksPolicy.LIVE)
    assert "drops" not in str(refusal.value)

    monkeypatch.setitem(
        sys.modules, "flash_attn_interface", _fake_fa3_module("flash_attn_interface", with_s_aux=False)
    )
    with pytest.raises(ValueError, match="silently drops the learnable attention sinks"):
        validate_attn_implementation(_gptoss(), "flash_attention_3", SinksPolicy.LIVE)


def test_the_sdpa_flip_stops_at_pretrained_model(monkeypatch):
    """Clearing the ``PreTrainedModel`` default would admit sdpa for every architecture loaded later in the process."""

    class FamilyBase(PreTrainedModel):
        _supports_sdpa = False

    class FamilyForCausalLM(FamilyBase):
        pass

    monkeypatch.setattr(
        gpt_oss_sinks, "AutoModelForCausalLM", types.SimpleNamespace(_model_mapping={GptOssConfig: FamilyForCausalLM})
    )
    assert PreTrainedModel.__dict__["_supports_sdpa"] is False  # the base default the flip must keep
    assert install_sdpa_for_reset_sinks(_gptoss()) is True
    assert FamilyBase._supports_sdpa is True
    assert PreTrainedModel.__dict__["_supports_sdpa"] is False


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
    model_config = ModelArguments(model_name_or_path=checkpoint, attn_implementation=attn_implementation)
    training_config = types.SimpleNamespace()  # no bf16/fp16 flags → fp32 reference
    # Every ``<special>_token_id`` accompanies its token, as ``PreTrainedTokenizerBase.__getattr__`` resolves them.
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
    """With reset sinks the default sdpa is validated (GptOss natively rejects it) and sinks go to dtype min."""
    args, model_config, training_config, parallelism_config, tokenizer = _ref_load_stubs(tiny_gptoss_checkpoint)
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        training_config,
        parallelism_config,
        tokenizer,
        is_vlm=False,
        method="DPO",
        reset_sinks=True,
        attn_default=None,
    )
    sinks = model_ref.model.layers[0].self_attn.sinks
    assert torch.all(sinks == torch.finfo(sinks.dtype).min)


def test_reference_model_load_rejects_sink_dropping_impl_with_live_sinks(tiny_gptoss_checkpoint):
    """A requested sdpa or FA2 dropping live sinks would shift every reference logprob by nats against the policy."""
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
            attn_default=None,
        )


def test_reference_model_auto_resolves_to_a_sink_carrying_impl(tiny_gptoss_checkpoint):
    """Pinned to ``sdpa`` while the policy auto-detects, the legs of every logratio would run on different kernels."""
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
        attn_default=None,
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
        attn_default=None,
    )
    sinks = model_ref.model.layers[0].self_attn.sinks
    assert sinks.requires_grad is False  # live sinks kept but frozen, mirroring the policy load


def test_toolkit_run_state_never_reaches_the_exported_config():
    """``_halo_*`` stamps describe the run, but transformers writes every instance attribute to ``config.json``."""
    cfg = _gptoss()
    validate_attn_implementation(cfg, "sdpa", SinksPolicy.NEUTRALIZED)
    set_config_field_run_scoped(cfg, LIGER_APPLIED_CONFIG_ATTR, {"rope": True})
    assert cfg._halo_sinks_reset is True and getattr(cfg, LIGER_APPLIED_CONFIG_ATTR) == {"rope": True}

    with config_export_ready(cfg):
        exported = cfg.to_dict()
        halo_keys = [key for key in exported if key.startswith(TOOLKIT_STAMP_PREFIX)]
        assert halo_keys == [], f"run state leaked into the exported config: {halo_keys}"

    # The run still decides on its own state after the export window closes.
    assert cfg._halo_sinks_reset is True
    assert validate_attn_implementation(cfg, "sdpa") == "sdpa"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
