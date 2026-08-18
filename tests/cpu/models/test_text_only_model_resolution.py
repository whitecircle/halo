"""text_only_model resolution: a multimodal config loads its text-only CausalLM sibling on demand.

The flag exists for text-only training of VLM checkpoints (Qwen3.5/3.6): the CausalLM class drops
the vision tower and honors ``config.output_router_logits`` (``moe_balancing: aux_loss``), which the
multimodal wrapper's forward never consults. These tests fail if the flag stops forcing
``AutoModelForCausalLM`` or starts leaking into non-multimodal resolution.
"""

from types import SimpleNamespace

import pytest
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from src.models.loading.model_preparation import resolve_auto_model_class


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
    """glm5_next ships only ``Glm5NextForConditionalGeneration`` — no CausalLM sibling exists in the
    AutoModelForCausalLM mapping, so the flag must be refused HERE, by name, instead of surfacing
    later as transformers' generic "Unrecognized configuration class"."""
    config = SimpleNamespace(model_type="glm5_next")
    assert resolve_auto_model_class(config) is AutoModelForImageTextToText
    with pytest.raises(ValueError, match="glm5_next.*no.*ForCausalLM"):
        resolve_auto_model_class(config, text_only=True)


def test_text_only_accepts_a_remote_code_causal_lm_sibling():
    """A remote-code family is absent from the static mapping but loads through its ``auto_map``
    entry — the refusal must not close that door."""
    config = SimpleNamespace(
        model_type="some_remote_vlm", auto_map={"AutoModelForCausalLM": "modeling.RemoteForCausalLM"}
    )
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_refuses_a_conditional_generation_class_under_the_causal_lm_key():
    """step3p7's hub repo maps ``AutoModelForCausalLM`` to its own legacy remote-code
    ``Step3p7ForConditionalGeneration`` — the multimodal class masquerading under the CausalLM key.
    Counting it as a text-only sibling would silently load the full VLM remote code as the
    "text-only" model, so the refusal must fire and name the masquerade."""
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
    """The second masquerade shape: the CausalLM entry names the SAME class a vision auto_map key
    maps — the class need not be spelled ``*ForConditionalGeneration`` for the aliasing to give it
    away."""
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
    """A repo declaring BOTH a real CausalLM sibling and a vision class must keep working — the
    masquerade check keys on the target class, not on the mere presence of vision entries."""
    config = SimpleNamespace(
        model_type="some_remote_vlm",
        auto_map={
            "AutoModelForCausalLM": "modeling.RemoteForCausalLM",
            "AutoModelForImageTextToText": "modeling.RemoteForConditionalGeneration",
        },
    )
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM


def test_text_only_skips_the_vlm_branch_in_load_model_for_training(monkeypatch):
    """The flag must divert BEFORE the VLM probe: the probe enters a rank-coordinated store phase,
    and the wrapper class it leads to is exactly what text_only_model exists to avoid — patching
    the resolver alone is not enough, since sft.py loads through this entry point."""
    from src.distributed.loading import vlm_setup

    calls = {}
    monkeypatch.setattr(vlm_setup, "is_vlm_model", lambda *a, **k: calls.setdefault("probe", True) or True)
    monkeypatch.setattr(
        vlm_setup,
        "load_model_consuming_init_kwargs",
        lambda *a, **k: calls.setdefault("loader_kwargs", k) and ("model", "tokenizer"),
    )
    model_config = SimpleNamespace(model_name_or_path="fake/vlm", model_revision=None, trust_remote_code=False)
    result = vlm_setup.load_model_for_training(
        model_config, SimpleNamespace(), SimpleNamespace(), text_only_model=True
    )
    assert "probe" not in calls, "text_only_model=True must not enter the vlm_probe store phase"
    assert calls["loader_kwargs"]["text_only_model"] is True
    assert result == ("model", "tokenizer", "tokenizer", False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
