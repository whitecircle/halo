#!/usr/bin/env python
"""A family's fp32 pins survive every wrapper a writer holds the model through. ``_keep_in_fp32_modules`` is a class
attribute, and bf16 pinned tensors are past the reload's re-pin repairing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedModel

from halo.checkpoint.writing import save_dtype_caster
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.models.param_roles import fp32_pinned_names
from tests.common import tiny_models
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    fill_release_scale_bias,
    tiny_family_model,
)

PP_SIZE = 2
_BIAS_SLOT = "e_score_correction_bias"
_BIAS_FAMILIES = {**TINY_MOE_FAMILIES, "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}


def _model() -> DeepseekV4ForCausalLM:
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG))


def test_pins_are_read_through_a_plain_wrapper():
    model = _model()
    pinned = fp32_pinned_names(model)
    assert pinned and any("hc" in name for name in pinned), sorted(pinned)[:5]

    class _Holder(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

    assert fp32_pinned_names(_Holder(model)) == {f"inner.{name}" for name in pinned}


def test_pipeline_stages_pin_exactly_the_unsplit_models_tensors():
    """In global names; an unpinned tensor goes to the save dtype."""
    pinned = fp32_pinned_names(_model())
    seen: set[str] = set()
    for pp_rank in range(PP_SIZE):
        stage = build_pipeline_stage(_model(), pp_rank, PP_SIZE, moe_balancing="none")
        local = fp32_pinned_names(stage)
        assert local, f"stage {pp_rank} derives no pins"
        seen |= {stage.global_parameter_name(name) for name in local}

        cast = save_dtype_caster(stage)
        params = dict(stage.named_parameters())
        kept = next(iter(local))
        assert cast(kept, params[kept].float()).dtype == torch.float32
        unpinned = next(name for name in params if name not in local and "norm" not in name)
        assert cast(unpinned, params[unpinned].float()).dtype != torch.float32
    assert seen == pinned


def test_an_export_keeps_a_pinned_buffer_at_fp32():
    """No loader casts a buffer, so it is fp32 live."""
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"])
    buffers = {name: b for name, b in model.named_buffers() if name.endswith("e_score_correction_bias")}
    assert buffers and all(b.dtype == torch.float32 for b in buffers.values()), buffers
    cast = save_dtype_caster(model)
    assert {name: cast(name, b).dtype for name, b in buffers.items()} == dict.fromkeys(buffers, torch.float32)
    weight = next(name for name, p in model.named_parameters() if name.endswith("gate.weight"))
    assert cast(weight, model.get_parameter(weight)).dtype == torch.bfloat16, "an unpinned tensor is exported too"


def _bf16_reload(family: TinyFamily, saved: PreTrainedModel, directory) -> PreTrainedModel:
    saved.save_pretrained(directory)
    return family.load_class.from_pretrained(directory, dtype=torch.bfloat16)


@pytest.mark.parametrize("name", sorted(_BIAS_FAMILIES))
def test_a_pinned_bias_keeps_its_values_through_a_bf16_reload(name, tmp_path):
    """At a release's scale, where the cast collapses them, so a lost pin cannot pass."""
    family = _BIAS_FAMILIES[name]
    torch.manual_seed(0)
    model = tiny_family_model(family)
    pinned = fp32_pinned_names(model, include_buffers=True)
    if not [key for key in model.state_dict() if key.endswith(_BIAS_SLOT) and key in pinned]:
        pytest.skip("no class in this tree pins an e_score_correction_bias")
    filled = fill_release_scale_bias(model, _BIAS_SLOT)
    reloaded = dict(_bf16_reload(family, model, tmp_path).named_buffers())
    for key, values in filled.items():
        assert reloaded[key].dtype == torch.float32, key
        assert torch.equal(reloaded[key], values), key


def test_an_unpinned_bias_loses_its_values_through_the_same_reload(tmp_path):
    """What the check above would read without the pin: Step-3.7's classes pin no bias."""
    family = TINY_MOE_FAMILIES["step3p7"]
    torch.manual_seed(0)
    model = tiny_family_model(family)
    assert not {key for key in fp32_pinned_names(model, include_buffers=True) if key.endswith(_BIAS_SLOT)}
    filled = fill_release_scale_bias(model)
    reloaded = dict(_bf16_reload(family, model, tmp_path).named_buffers())
    assert any(not torch.equal(reloaded[key].float(), values) for key, values in filled.items())


def test_the_fixture_refuses_a_bias_a_cast_would_keep(monkeypatch):
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"])
    monkeypatch.setattr(tiny_models, "RELEASE_BIAS_CENTER", 0.0)
    monkeypatch.setattr(tiny_models, "RELEASE_BIAS_SPREAD", 1.0)
    with pytest.raises(ValueError, match="would test nothing"):
        fill_release_scale_bias(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
