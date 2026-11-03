"""text_only_model: a multimodal config loads its text-only CausalLM sibling on demand, dropping the tower and
honoring ``config.output_router_logits``."""

from types import SimpleNamespace

import pytest
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from halo.entrypoints import model_setup
from halo.model_loading.model_preparation import resolve_auto_model_class


def test_multimodal_config_defaults_to_image_text_class():
    config = SimpleNamespace(model_type="qwen3_5_moe")
    assert resolve_auto_model_class(config) is AutoModelForImageTextToText


def test_text_only_forces_causal_lm_for_multimodal_config():
    config = SimpleNamespace(model_type="qwen3_5_moe")
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_is_a_noop_for_text_configs():
    config = SimpleNamespace(model_type="gpt_oss")
    assert resolve_auto_model_class(config) is AutoModelForCausalLM
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_refuses_a_family_with_no_causal_lm_sibling():
    """glm5_next has no sibling in the mapping; refused here instead of transformers' generic "Unrecognized
    configuration class"."""
    config = SimpleNamespace(model_type="glm5_next")
    assert resolve_auto_model_class(config) is AutoModelForImageTextToText
    with pytest.raises(ValueError, match="glm5_next.*no.*ForCausalLM"):
        resolve_auto_model_class(config, text_only=True)


def test_text_only_refuses_a_native_causal_lm_entry_naming_the_vlm_class():
    """transformers maps ``gemma4`` to its VLM class in the CausalLM mapping too, so the flag would keep the tower
    while claiming to drop it."""
    with pytest.raises(ValueError, match="gemma4.*CausalLM mapping.*Gemma4ForConditionalGeneration"):
        resolve_auto_model_class(SimpleNamespace(model_type="gemma4"), text_only=True)
    # Fuyu's CausalLM entry is its image-text-to-text class under a *ForCausalLM name.
    with pytest.raises(ValueError, match="fuyu.*FuyuForCausalLM"):
        resolve_auto_model_class(SimpleNamespace(model_type="fuyu"), text_only=True)
    assert resolve_auto_model_class(SimpleNamespace(model_type="gemma4_text"), text_only=True) is AutoModelForCausalLM


def test_text_only_accepts_a_remote_code_causal_lm_sibling():
    config = SimpleNamespace(
        model_type="some_remote_vlm", auto_map={"AutoModelForCausalLM": "modeling.RemoteForCausalLM"}
    )
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_refuses_a_conditional_generation_class_under_the_causal_lm_key():
    """step3p7's hub repo maps ``AutoModelForCausalLM`` to its VLM remote code, so counting it loads the full VLM as
    text-only."""
    config = SimpleNamespace(
        model_type="step3p7",
        auto_map={
            "AutoConfig": "configuration_step3p7.Step3p7Config",
            "AutoModelForCausalLM": "modeling_step3p7.Step3p7ForConditionalGeneration",
        },
    )
    with pytest.raises(ValueError, match="step3p7.*Step3p7ForConditionalGeneration.*not a text-only sibling"):
        resolve_auto_model_class(config, text_only=True)


def test_text_only_refuses_a_causal_lm_entry_aliasing_the_vision_class():
    """The class need not be spelled ``*ForConditionalGeneration`` for the aliasing to give it away."""
    config = SimpleNamespace(
        model_type="some_remote_vlm",
        auto_map={
            "AutoModelForCausalLM": "modeling.RemoteVLModel",
            "AutoModelForImageTextToText": "modeling.RemoteVLModel",
        },
    )
    with pytest.raises(ValueError, match="not a text-only sibling"):
        resolve_auto_model_class(config, text_only=True)


def test_text_only_keeps_a_genuine_sibling_beside_a_vision_entry():
    """The check keys on the target class, not on vision entries being present."""
    config = SimpleNamespace(
        model_type="some_remote_vlm",
        auto_map={
            "AutoModelForCausalLM": "modeling.RemoteForCausalLM",
            "AutoModelForImageTextToText": "modeling.RemoteForConditionalGeneration",
        },
    )
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_skips_the_vlm_branch_in_load_policy_model(monkeypatch):
    """The probe enters a rank-coordinated store phase; sft.py loads through this entry point, so patching the
    resolver is not enough."""
    calls = {}
    monkeypatch.setattr(
        model_setup, "probe_checkpoint", lambda *a, **k: calls.setdefault("probe", True) and (None, True)
    )
    monkeypatch.setattr(
        model_setup,
        "load_model_and_tokenizer",
        lambda *a, **k: calls.setdefault("args", a[3]) and ("model", "tokenizer"),
    )
    model_config = SimpleNamespace(model_name_or_path="fake/vlm", model_revision=None, trust_remote_code=False)
    args = SimpleNamespace(text_only_model=True)
    result = model_setup.load_policy_model(SimpleNamespace(), SimpleNamespace(), model_config, args, vlm_run=False)
    assert "probe" not in calls, "text_only_model=True must not enter the vlm_probe store phase"
    assert calls["args"].text_only_model is True
    assert result == ("model", "tokenizer", "tokenizer", False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
