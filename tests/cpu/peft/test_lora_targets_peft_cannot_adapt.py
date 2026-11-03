#!/usr/bin/env python
"""``lora_target_modules`` matches PEFT cannot adapt are excluded, not fatal: Gemma 4's ``Gemma4ClippableLinear`` and
DeepSeek-V4's grouped ``o_a_proj``. A plain ``nn.Linear`` tower or a same-map subclass must stay adapted."""

import datetime
import os
import types

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4GroupedLinear
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

import halo.model_loading.parallel.peft_setup as peft_setup
from halo.configs.script_arguments import ModelArguments
from halo.kernels.lowp.linear import LowPrecisionLinear
from halo.model_loading.parallel.peft_setup import build_peft_config, configure_adapter_training
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_DSV4_CONFIG

ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _PlainAttention(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class _ClippableAttention(nn.Module):
    """The real Gemma 4 wrapper: a container holding ``linear`` plus clip buffers."""

    def __init__(self, width: int = 8, clipped: bool = True):
        super().__init__()
        config = types.SimpleNamespace(use_clipped_linears=clipped)
        self.q_proj = Gemma4ClippableLinear(config, width, width)
        self.o_proj = Gemma4ClippableLinear(config, width, width)


class _Multimodal(nn.Module):
    def __init__(self, tower: nn.Module, width: int = 8):
        super().__init__()
        self.language_model = _PlainAttention(width)
        self.vision_tower = tower
        self.lm_head = nn.Linear(width, 16, bias=False)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # PeftModelForCausalLM latches this at wrap time; these tests never generate.
        raise NotImplementedError


def _model_config(targets=ATTENTION_TARGETS, **kwargs) -> ModelArguments:
    return ModelArguments(model_name_or_path="dummy/vlm", use_peft=True, lora_target_modules=targets, **kwargs)


def _adapted_modules(peft_model) -> set[str]:
    return {name.rsplit(".lora_", 1)[0] for name, _ in peft_model.named_parameters() if ".lora_" in name}


def test_wrapped_projection_is_excluded_and_injection_succeeds():
    """Without the exclusion PEFT raises ``Target module Gemma4ClippableLinear(...) is not supported``."""
    model = _Multimodal(_ClippableAttention())

    peft_config = build_peft_config(model, _model_config())
    adapted = _adapted_modules(get_peft_model(model, peft_config))

    assert adapted == {
        "base_model.model.language_model.q_proj",
        "base_model.model.language_model.o_proj",
    }
    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_plain_linear_tower_keeps_its_adapters():
    """Excluding anything there would make existing VLM LoRA runs silently train fewer modules."""
    model = _Multimodal(_PlainAttention())

    peft_config = build_peft_config(model, _model_config())

    assert peft_config.exclude_modules is None
    assert _adapted_modules(get_peft_model(model, peft_config)) == {
        "base_model.model.language_model.q_proj",
        "base_model.model.language_model.o_proj",
        "base_model.model.vision_tower.q_proj",
        "base_model.model.vision_tower.o_proj",
    }


def test_no_adaptable_target_left_fails_loud():
    """The LoRA optimizer accepts empty parameter groups, so an all-excluded run has no other symptom."""
    model = _ClippableAttention()

    with pytest.raises(ValueError, match="PEFT can adapt none of them"):
        build_peft_config(model, _model_config())


def test_unclipped_wrapper_is_excluded_too():
    """``use_clipped_linears: false`` drops the buffers but keeps the container, so PEFT still raises."""
    model = _Multimodal(_ClippableAttention(clipped=False))

    peft_config = build_peft_config(model, _model_config())

    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_all_linear_sentinel_is_untouched():
    """PEFT's injection resolves ``all-linear`` to each wrapper's inner ``linear``; an exclusion would change that."""
    model = _Multimodal(_ClippableAttention())

    peft_config = build_peft_config(model, _model_config(targets=["all-linear"]))

    assert peft_config.exclude_modules is None
    assert _adapted_modules(get_peft_model(model, peft_config)) >= {
        "base_model.model.language_model.q_proj",
        "base_model.model.vision_tower.q_proj.linear",
    }


def test_regex_target_string_still_selects_by_pattern():
    """A regex target is matched with PEFT's own matcher, so the scan neither widens nor narrows it."""
    model = _Multimodal(_ClippableAttention())

    peft_config = build_peft_config(model, _model_config(targets=r"language_model\.q_proj"))

    assert peft_config.exclude_modules is None
    assert _adapted_modules(get_peft_model(model, peft_config)) == {"base_model.model.language_model.q_proj"}


def test_modules_to_save_copy_of_an_unadaptable_module_still_trains():
    """``modules_to_save`` entries are full-trained copies, so an excluded container still gets its trainable copy."""
    model = _Multimodal(_ClippableAttention())

    peft_config = build_peft_config(model, _model_config(lora_modules_to_save=["vision_tower.o_proj"]))
    peft_model = get_peft_model(model, peft_config)

    copies = [
        name for name, param in peft_model.named_parameters() if "vision_tower.o_proj" in name and param.requires_grad
    ]
    assert copies and all("modules_to_save" in name for name in copies), "the container lost its trainable copy"
    with torch.no_grad():
        peft_model.base_model.model.vision_tower.o_proj(torch.zeros(1, 8))


def test_rejection_names_the_expert_adapters_that_would_train_alone(monkeypatch):
    """With native expert adapters live, the message must say what would train, or the user fixes the wrong half."""
    monkeypatch.setattr(peft_setup, "model_has_expert_lora", lambda model: True)
    model = _ClippableAttention()

    with pytest.raises(ValueError, match="native expert adapters would train alone"):
        build_peft_config(model, _model_config())


class _GroupedPastEight(nn.Linear):
    """One class, two geometries: the affine map up to 8 inputs, a half-wide grouped output beyond."""

    def forward(self, x):
        out = nn.functional.linear(x, self.weight)
        return out if self.in_features <= 8 else out[..., : self.out_features // 2]


@pytest.mark.parametrize("wide_first", [False, True])
def test_probe_verdict_is_per_geometry_not_per_class(wide_first):
    """One verdict per width, whichever instance the scan reaches first."""
    narrow, wide = _GroupedPastEight(8, 8, bias=False), _GroupedPastEight(16, 16, bias=False)
    model = nn.Module()
    model.a_proj, model.b_proj = (wide, narrow) if wide_first else (narrow, wide)

    peft_config = build_peft_config(model, _model_config(targets=["a_proj", "b_proj"]))

    assert peft_config.exclude_modules == ["a_proj" if wide_first else "b_proj"]


class _FaultingLinear(nn.Linear):
    def forward(self, x):
        raise torch.OutOfMemoryError("CUDA out of memory")


def test_a_device_fault_in_the_probe_is_not_a_verdict():
    """Swallowing the fault would exclude the module on the one rank that faulted and hang the others."""
    model = nn.Module()
    model.q_proj = _FaultingLinear(8, 8, bias=False)

    with pytest.raises(torch.OutOfMemoryError):
        build_peft_config(model, _model_config(targets=["q_proj", "o_proj"]))


def test_preset_exclusions_are_kept_alongside_the_found_ones():
    model = _Multimodal(_ClippableAttention())
    peft_config = LoraConfig(r=4, target_modules=ATTENTION_TARGETS, exclude_modules=["language_model.q_proj"])

    peft_setup._exclude_unadaptable_lora_targets(model, peft_config)

    assert peft_config.exclude_modules == ["language_model.q_proj", "vision_tower.o_proj", "vision_tower.q_proj"]
    assert _adapted_modules(get_peft_model(model, peft_config)) == {"base_model.model.language_model.o_proj"}


@pytest.mark.parametrize("spelling", ["all-linear", "All-Linear"])
def test_the_all_linear_sentinel_survives_a_one_entry_list(spelling):
    """PEFT reads its sentinel only as a bare string, case-insensitively; a list entry would name a module."""
    model = _Multimodal(_PlainAttention())
    model_config = _model_config([spelling])
    assert model_config.lora_target_modules == [spelling]

    peft_config = build_peft_config(model, model_config)

    assert peft_config.target_modules == "all-linear"
    assert _adapted_modules(get_peft_model(model, peft_config)) >= {"base_model.model.language_model.q_proj"}


WORLD_SIZE = 2
PG_TIMEOUT_SEC = 30


class _FaultsOnRankOne(nn.Linear):
    rank = 0

    def forward(self, x):
        if self.rank == 1:
            raise OSError(28, "No space left on device")
        return nn.functional.linear(x, self.weight)


def _divergent_worker(rank: int, tmp_dir: str) -> None:
    _FaultsOnRankOne.rank = rank
    model = _Multimodal(_PlainAttention())
    model.language_model.q_proj = _FaultsOnRankOne(8, 8, bias=False)
    try:
        build_peft_config(model, _model_config())
        outcome = "NO RAISE"
    except BaseException as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_an_exclusion_list_that_differs_between_ranks_raises_on_every_rank(tmp_path):
    """A rank excluding what its peers adapt trains a different parameter set and surfaces as a collective hang."""
    run_gloo_ranks(_divergent_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))

    outcomes = [(tmp_path / f"result_{rank}.txt").read_text() for rank in range(WORLD_SIZE)]
    assert all(outcome.startswith("ValueError") and "differs across ranks" in outcome for outcome in outcomes), (
        outcomes
    )


def test_configure_adapter_training_applies_the_exclusion():
    model = _Multimodal(_ClippableAttention())
    args = types.SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)

    peft_config = configure_adapter_training(args, model, _model_config())

    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_wrapper_detection_reads_through_an_existing_adapter():
    """A second adapter targets ``lora.Linear`` layers; PEFT dispatches on their ``base_layer``, so they adapt."""
    model = _Multimodal(_PlainAttention())
    adapted = get_peft_model(model, LoraConfig(r=4, target_modules=ATTENTION_TARGETS))

    peft_config = build_peft_config(adapted, _model_config())

    assert peft_config.exclude_modules is None


def _tiny_dsv4() -> DeepseekV4ForCausalLM:
    torch.manual_seed(1234)
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG)).eval()
    model.config._attn_implementation = "eager"
    return model


def test_grouped_projection_is_excluded_under_all_linear_and_the_model_runs():
    """PEFT wraps ``o_a_proj``, whose forward returns ``out_features // n_groups``; the delta dies at first forward."""
    model = _tiny_dsv4()

    peft_config = build_peft_config(model, _model_config(targets=["all-linear"]))
    peft_model = get_peft_model(model, peft_config)

    with torch.no_grad():
        logits = peft_model(input_ids=torch.randint(4, 100, (1, 8)), use_cache=False).logits

    assert torch.isfinite(logits).all()
    assert all(name.endswith("o_a_proj") for name in peft_config.exclude_modules)
    adapted = _adapted_modules(peft_model)
    assert not [name for name in adapted if name.endswith("o_a_proj")]
    assert [name for name in adapted if name.endswith("o_b_proj")], "the sibling mix-down lost its adapter"


def test_grouped_projection_named_directly_is_excluded():
    model = _tiny_dsv4()

    peft_config = build_peft_config(model, _model_config(targets=["o_a_proj", "o_b_proj"]))

    assert all(name.endswith("o_a_proj") for name in peft_config.exclude_modules)
    assert len(peft_config.exclude_modules) == model.config.num_hidden_layers


def test_a_forward_override_that_keeps_the_affine_map_stays_adapted():
    """The low-precision drop-in overrides ``forward`` but computes the same map; a fp8/fp4 run names these."""
    model = _Multimodal(_PlainAttention(64), width=64)  # the fake-quant block is 32 wide
    LowPrecisionLinear.convert_(model.language_model.q_proj, "mxfp8")
    assert type(model.language_model.q_proj) is LowPrecisionLinear, "the retype is this test's premise"

    peft_config = build_peft_config(model, _model_config())

    assert peft_config.exclude_modules is None
    assert "base_model.model.language_model.q_proj" in _adapted_modules(get_peft_model(model, peft_config))


def test_the_two_reasons_are_reported_separately():
    """A tower wrapper and a grouped projection are different fixes."""
    model = _tiny_dsv4()
    model.model.layers[0].self_attn.q_proj = _ClippableAttention().q_proj

    with pytest.warns(UserWarning) as records:
        peft_config = build_peft_config(model, _model_config(targets=["q_proj", "o_a_proj", "o_b_proj"]))

    message = "\n".join(str(record.message) for record in records)
    assert "hold their weight in a child module" in message
    assert "compute something other than their layer's affine map" in message
    assert any(name.endswith("o_a_proj") for name in peft_config.exclude_modules)
    assert "model.layers.0.self_attn.q_proj" in peft_config.exclude_modules


def test_grouped_linear_is_a_linear_subclass():
    """Grouped projections pass PEFT's ``nn.Linear`` dispatch, so a structural class check alone cannot catch them."""
    assert issubclass(DeepseekV4GroupedLinear, nn.Linear)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
