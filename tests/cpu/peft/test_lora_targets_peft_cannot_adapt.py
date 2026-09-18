#!/usr/bin/env python
"""``lora_target_modules`` name matches PEFT cannot adapt must be excluded, not fatal.

Two shapes are out of stock LoRA's reach, and both are reachable by an ordinary target list or by
``all-linear``:

- a wrapper holding its weight in a child module — Gemma 4's ``Gemma4ClippableLinear``, which the
  vision and audio towers spell ``q_proj``…``o_proj`` exactly as the language model does. PEFT
  decomposes the matched module's OWN weight, so injection raises there and takes the run with it.
- a layer subclass computing something other than its base layer's affine map — DeepSeek-V4's
  ``o_a_proj``, an ``nn.Linear`` subclass whose block-diagonal forward returns one group's width.
  PEFT accepts it and the delta dies on a shape mismatch at the first forward.

The counter-cases matter as much: a tower whose ``q_proj`` is a plain ``nn.Linear`` (CLIP/SigLIP/
Pixtral-style) and a subclass that only changes HOW the affine map is computed (the low-precision
compute drop-in) are adapted, and must stay adapted — silently changing which modules train is not an
acceptable fix.

Run: pytest tests/cpu/peft/test_lora_targets_peft_cannot_adapt.py
"""

import contextlib
import datetime
import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4GroupedLinear
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
from trl import ModelConfig

import src.distributed.loading.peft_setup as peft_setup
from src.distributed.loading.peft_setup import build_peft_config, setup_peft_model
from src.kernels.lowp.linear import LowPrecisionLinear
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.ports import free_port

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


def _model_config(targets=ATTENTION_TARGETS, **kwargs) -> ModelConfig:
    return ModelConfig(model_name_or_path="dummy/vlm", use_peft=True, lora_target_modules=targets, **kwargs)


def _adapted_modules(peft_model) -> set[str]:
    return {name.rsplit(".lora_", 1)[0] for name, _ in peft_model.named_parameters() if ".lora_" in name}


def test_wrapped_projection_is_excluded_and_injection_succeeds():
    """Without the exclusion PEFT raises ``Target module Gemma4ClippableLinear(...) is not supported``
    and the run never starts; the language model's adapters must survive intact."""
    model = _Multimodal(_ClippableAttention())

    peft_config = build_peft_config(model, _model_config())
    adapted = _adapted_modules(get_peft_model(model, peft_config))

    assert adapted == {
        "base_model.model.language_model.q_proj",
        "base_model.model.language_model.o_proj",
    }
    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_plain_linear_tower_keeps_its_adapters():
    """A vision tower whose projections are plain ``nn.Linear`` is adapted today. Nothing may be
    excluded there, or existing VLM LoRA runs would silently train fewer modules."""
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
    """Excluding every match would leave an adapter run with nothing to train — the LoRA optimizer
    accepts empty parameter groups, so this has no other symptom."""
    model = _ClippableAttention()

    with pytest.raises(ValueError, match="PEFT can adapt none of them"):
        build_peft_config(model, _model_config())


def test_unclipped_wrapper_is_excluded_too():
    """``use_clipped_linears: false`` drops the buffers but keeps the container — PEFT still raises."""
    model = _Multimodal(_ClippableAttention(clipped=False))

    peft_config = build_peft_config(model, _model_config())

    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_all_linear_sentinel_is_untouched():
    """``all-linear`` is resolved to ``nn.Linear`` paths inside PEFT's own injection, which reaches
    the inner ``linear`` of each wrapper. Excluding anything here would change what it adapts."""
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

    peft_config = build_peft_config(model, _model_config(targets=[r"language_model\.q_proj"]))

    assert peft_config.exclude_modules is None
    assert _adapted_modules(get_peft_model(model, peft_config)) == {"base_model.model.language_model.q_proj"}


def test_modules_to_save_copy_of_an_unadaptable_module_still_trains():
    """``modules_to_save`` entries are full-trained copies, not adapters: a container the scan excludes
    from LoRA targeting must still get its trainable copy, and the run must still forward."""
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
    """With native expert adapters live, an all-excluded attention list does not mean the run trains
    nothing — the message must say what would train, or the user fixes the wrong half."""
    monkeypatch.setattr(peft_setup, "has_ep_lora", lambda model: True)
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
    """A class grouped at one width and plain at another gets one verdict per width, whichever
    instance the scan reaches first."""
    narrow, wide = _GroupedPastEight(8, 8, bias=False), _GroupedPastEight(16, 16, bias=False)
    model = nn.Module()
    model.a_proj, model.b_proj = (wide, narrow) if wide_first else (narrow, wide)

    peft_config = build_peft_config(model, _model_config(targets=["a_proj", "b_proj"]))

    assert peft_config.exclude_modules == ["a_proj" if wide_first else "b_proj"]


class _FaultingLinear(nn.Linear):
    def forward(self, x):
        raise torch.OutOfMemoryError("CUDA out of memory")


def test_a_device_fault_in_the_probe_is_not_a_verdict():
    """An out-of-memory or accelerator fault says nothing about the layer; swallowing it would exclude
    the module on the one rank that faulted and hang the others."""
    model = nn.Module()
    model.q_proj = _FaultingLinear(8, 8, bias=False)

    with pytest.raises(torch.OutOfMemoryError):
        build_peft_config(model, _model_config(targets=["q_proj", "o_proj"]))


def test_preset_exclusions_are_kept_alongside_the_found_ones():
    """A ``LoraConfig`` that already excludes modules keeps them: the scan adds, never replaces."""
    model = _Multimodal(_ClippableAttention())
    peft_config = LoraConfig(r=4, target_modules=ATTENTION_TARGETS, exclude_modules=["language_model.q_proj"])

    peft_setup._exclude_unadaptable_lora_targets(model, peft_config)

    assert peft_config.exclude_modules == ["language_model.q_proj", "vision_tower.o_proj", "vision_tower.q_proj"]
    assert _adapted_modules(get_peft_model(model, peft_config)) == {"base_model.model.language_model.o_proj"}


@pytest.mark.parametrize("spelling", ["all-linear", "All-Linear"])
def test_the_all_linear_sentinel_survives_a_cli_list_override(spelling):
    """A CLI override lands after ``ModelConfig.__post_init__``, so ``--lora_target_modules=all-linear``
    arrives as a one-entry list; PEFT reads its sentinel only as a bare string, case-insensitively, and
    would otherwise look for a module named ``all-linear``."""
    model = _Multimodal(_PlainAttention())
    model_config = _model_config()
    model_config.lora_target_modules = [spelling]  # after __post_init__, as the parser's override lands

    peft_config = build_peft_config(model, model_config)

    assert peft_config.target_modules == "all-linear"
    assert _adapted_modules(get_peft_model(model, peft_config)) >= {"base_model.model.language_model.q_proj"}


# The verdict is agreed across ranks

WORLD_SIZE = 2
PG_TIMEOUT_SEC = 30


class _FaultsOnRankOne(nn.Linear):
    """A probe that fails on one rank only: an exclusion list that differs between ranks."""

    rank = 0

    def forward(self, x):
        if self.rank == 1:
            raise OSError(28, "No space left on device")
        return nn.functional.linear(x, self.weight)


def _divergent_worker(rank: int, tmp_dir: str, port: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
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
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def test_an_exclusion_list_that_differs_between_ranks_raises_on_every_rank(tmp_path):
    """One rank excluding what its peers adapt would train a different parameter set and surface only
    as a hang in the first collective; the disagreement is caught where it arises, on both ranks."""
    mp.start_processes(
        _divergent_worker, args=(str(tmp_path), str(free_port())), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )

    outcomes = [(tmp_path / f"result_{rank}.txt").read_text() for rank in range(WORLD_SIZE)]
    assert all(outcome.startswith("ValueError") and "differs across ranks" in outcome for outcome in outcomes), (
        outcomes
    )


def test_setup_peft_model_applies_the_exclusion():
    """The production seam every training script reaches, not just the helper under it."""
    model = _Multimodal(_ClippableAttention())
    args = types.SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)

    peft_config = setup_peft_model(args, model, _model_config())

    assert set(peft_config.exclude_modules) == {"vision_tower.q_proj", "vision_tower.o_proj"}


def test_wrapper_detection_reads_through_an_existing_adapter():
    """A second adapter targets ``lora.Linear`` layers, whose weight lives in ``base_layer`` — PEFT
    dispatches on that base layer, so they must not be mistaken for unadaptable containers."""
    model = _Multimodal(_PlainAttention())
    adapted = get_peft_model(model, LoraConfig(r=4, target_modules=ATTENTION_TARGETS))

    peft_config = build_peft_config(adapted, _model_config())

    assert peft_config.exclude_modules is None


# A layer subclass whose forward is not its base layer's affine map


def _tiny_dsv4() -> DeepseekV4ForCausalLM:
    torch.manual_seed(1234)
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG)).eval()
    model.config._attn_implementation = "eager"
    return model


def test_grouped_projection_is_excluded_under_all_linear_and_the_model_runs():
    """``all-linear`` reaches DeepSeek-V4's ``o_a_proj``, an ``nn.Linear`` subclass whose grouped
    forward returns ``out_features // n_groups``. PEFT wraps it happily and the delta then dies on a
    shape mismatch at the first forward — the run must instead train, with that one projection skipped.
    """
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
    """Naming the projection is the other way in; the sibling target keeps the run alive."""
    model = _tiny_dsv4()

    peft_config = build_peft_config(model, _model_config(targets=["o_a_proj", "o_b_proj"]))

    assert all(name.endswith("o_a_proj") for name in peft_config.exclude_modules)
    assert len(peft_config.exclude_modules) == model.config.num_hidden_layers


def test_a_forward_override_that_keeps_the_affine_map_stays_adapted():
    """The low-precision compute drop-in overrides ``nn.Linear.forward`` but computes the same map, so
    excluding it would silently stop training the projections a fp8/fp4 run names."""
    model = _Multimodal(_PlainAttention(64), width=64)  # the fake-quant block is 32 wide
    LowPrecisionLinear.convert_(model.language_model.q_proj, "fp8")
    assert type(model.language_model.q_proj) is LowPrecisionLinear, "the retype is this test's premise"

    peft_config = build_peft_config(model, _model_config())

    assert peft_config.exclude_modules is None
    assert "base_model.model.language_model.q_proj" in _adapted_modules(get_peft_model(model, peft_config))


def test_the_two_reasons_are_reported_separately():
    """One counted message per reason: a tower wrapper and a grouped projection are different fixes."""
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
    """Pins this file's premise: the grouped projection passes PEFT's ``nn.Linear`` dispatch, which is
    why a structural class check alone cannot catch it."""
    assert issubclass(DeepseekV4GroupedLinear, nn.Linear)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
