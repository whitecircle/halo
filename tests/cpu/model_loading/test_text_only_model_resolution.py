"""text_only_model: a multimodal config loads its text-only CausalLM sibling on demand, dropping the tower and
honoring ``config.output_router_logits``."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    MiniMaxM3VLForCausalLM,
    NemotronHForCausalLM,
)

from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import model_setup
from halo.model_loading import model_preparation
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.model_preparation import auto_load_model, resolve_auto_model_class, resolve_text_only_config
from halo.model_loading.parallel import loader
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, CompositeTextLayout, MoEFamilySpec
from halo.models.moe.router_logits import ROUTER_LOGITS_FORCED_OFF_ATTR, router_logits_forced_off
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, TINY_UNWRAPPED_COMPOSITE_FAMILIES, tiny_family_model

_DECLARED_SIBLINGS = {"nemotron_h_omni": NemotronHForCausalLM, "minimax_m3_vl": MiniMaxM3VLForCausalLM}
_COMPOSITES = {**TINY_UNWRAPPED_COMPOSITE_FAMILIES, "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}


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


def _claim(monkeypatch, *model_types: str) -> None:
    """The family's spec for this test alone, declaring a composite with no CausalLM entry of its own."""
    spec = MoEFamilySpec(model_types=model_types, text_only_layout=CompositeTextLayout())
    monkeypatch.setattr(
        model_preparation, "find_moe_family_spec", lambda model_type: spec if model_type in model_types else None
    )


def _undeclare(monkeypatch) -> None:
    monkeypatch.setattr(model_preparation, "find_moe_family_spec", lambda _model_type: None)


@pytest.mark.parametrize("name", sorted(_DECLARED_SIBLINGS))
def test_a_composite_with_no_causal_lm_entry_is_refused_undeclared(name, monkeypatch):
    _undeclare(monkeypatch)
    config = tiny_family_model(_COMPOSITES[name]).config
    assert resolve_text_only_config(config) is config
    with pytest.raises(ValueError, match="no text-only\\b|no text-only sibling"):
        resolve_auto_model_class(config, text_only=True)


def test_the_roster_declares_nemotrons_text_class_alone():
    declared = [spec.model_types for spec in MOE_FAMILY_SPECS if spec.text_only_layout is not None]
    assert declared == [("nemotron_h", "nemotron_h_omni")]


@pytest.mark.parametrize("name", sorted(_DECLARED_SIBLINGS))
def test_a_declared_composite_builds_its_text_sub_configs_class(name, monkeypatch):
    config = tiny_family_model(_COMPOSITES[name]).config
    _claim(monkeypatch, config.model_type, config.text_config.model_type)
    narrowed = resolve_text_only_config(config)
    assert narrowed is config.text_config
    assert resolve_auto_model_class(config, text_only=True) is AutoModelForCausalLM
    assert type(AutoModelForCausalLM.from_config(narrowed)) is _DECLARED_SIBLINGS[name]
    assert resolve_auto_model_class(config) is AutoModelForImageTextToText, "only the flag narrows it"


@pytest.mark.parametrize("name", sorted(set(TINY_MOE_VLM_FAMILIES) - {"nemotron_h_omni"}))
def test_the_rest_of_the_roster_stays_on_the_composite_config(name):
    """Its CausalLM mapping takes the composite, or it refuses the flag; neither reads a declared layout."""
    config = tiny_family_model(TINY_MOE_VLM_FAMILIES[name]).config
    assert resolve_text_only_config(config) is config


def test_the_tool_load_reads_a_composite_checkpoint_through_the_narrowed_config(tmp_path):
    """``auto_load_model`` is what the merge and conversion tools load through."""
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])
    composite.language_model.save_pretrained(tmp_path)
    composite.config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForCausalLM.from_pretrained(tmp_path)
    model = auto_load_model(str(tmp_path), text_only=True, dtype=torch.float32)
    assert type(model) is NemotronHForCausalLM
    assert model.config.model_type == "nemotron_h"
    assert torch.equal(model.lm_head.weight, composite.language_model.lm_head.weight)


def _run_config(path: str, *, text_only_model: bool) -> tuple:
    _, model_config, model_class, _ = loader._resolve_run_config(
        path,
        None,
        ParallelismConfig(),
        torch.bfloat16,
        trust_remote_code=False,
        model_config_overrides=None,
        sinks_policy=SinksPolicy.NEUTRALIZED,
        quantization_config=None,
        init_from_scratch=False,
        model_class=None,
        text_only_model=text_only_model,
    )
    return model_config, model_class


def test_the_distributed_load_builds_through_the_narrowed_config(tmp_path):
    """Every per-mode loader hands ``from_pretrained`` this config, and the CausalLM factory refuses the composite's."""
    tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]).save_pretrained(tmp_path)
    model_config, model_class = _run_config(str(tmp_path), text_only_model=True)
    assert model_class is AutoModelForCausalLM and model_config.model_type == "nemotron_h"
    shell = instantiate_on_meta(str(tmp_path), model_class, model_config, torch.bfloat16, trust_remote_code=False)
    assert type(shell) is NemotronHForCausalLM


def test_the_distributed_load_keeps_the_composite_without_the_flag(tmp_path):
    tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]).save_pretrained(tmp_path)
    model_config, model_class = _run_config(str(tmp_path), text_only_model=False)
    assert model_class is AutoModelForImageTextToText and model_config.model_type == "nemotron_h_omni"


def _remote_config_class(base: type) -> type:
    return type(base.__name__, (base,), {"__module__": "transformers_modules.release"})


def test_the_remote_composite_is_refused_and_its_text_class_is_not(tmp_path, monkeypatch):
    """Its forward drops ``seq_idx``, so packed documents cross the Mamba layers with nothing in the loss to show it."""
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])
    composite.save_pretrained(tmp_path)
    remote = _remote_config_class(type(composite.config)).from_pretrained(tmp_path)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: copy.deepcopy(remote))
    with pytest.raises(ValueError, match="trust_remote_code: false"):
        _run_config(str(tmp_path), text_only_model=False)
    model_config, model_class = _run_config(str(tmp_path), text_only_model=True)
    assert model_class is AutoModelForCausalLM and model_config.model_type == "nemotron_h"


def test_a_remote_composite_of_a_family_without_a_refusal_loads(tmp_path, monkeypatch):
    name = next(name for name in TINY_MOE_VLM_FAMILIES if name != "nemotron_h_omni")
    config = tiny_family_model(TINY_MOE_VLM_FAMILIES[name]).config
    remote = _remote_config_class(type(config)).from_dict(config.to_dict())
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: copy.deepcopy(remote))
    assert _run_config(str(tmp_path), text_only_model=False)[0].model_type == config.model_type


def test_the_tool_load_drops_the_stamps_its_checkpoint_carries(tmp_path):
    """The sibling reaches the verified load as a passed ``config``, whose stamps that load keeps as a caller's."""
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])
    composite.language_model.save_pretrained(tmp_path)
    composite.config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload["text_config"][ROUTER_LOGITS_FORCED_OFF_ATTR] = True
    config_path.write_text(json.dumps(payload))
    model = auto_load_model(str(tmp_path), text_only=True, dtype=torch.float32)
    assert type(model) is NemotronHForCausalLM
    assert not router_logits_forced_off(model.config)


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
