#!/usr/bin/env python
"""Liger's fused loss trains the head transform each family declares on its ``HeadTransformSpec``, or refuses it.
The kernel is Triton-only, so torch computes the same contraction here."""

import types

import pytest
import torch
from transformers import Cohere2Config, Cohere2ForCausalLM, NemotronHConfig, NemotronHForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.cohere2_moe import Cohere2MoeConfig, Cohere2MoeForCausalLM
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from halo.model_loading.liger import cross_entropy
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.model_loading.liger.fused_modules import apply_fused_modules_to_model
from halo.models.head_transform import HeadTransform
from halo.models.nemotron_h import apply_nemotron_h_patches
from tests.common.models import (
    TINY_ALTERNATING_SLIDING_CONFIG,
    TINY_COHERE2_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_NEMOTRON_H_CONFIG,
)

# Four times the class default (0.0625), so a stale default would train a different loss.
LOGIT_SCALE = 0.25
FAMILIES = {
    "cohere2": (Cohere2Config, Cohere2ForCausalLM, TINY_ALTERNATING_SLIDING_CONFIG),
    "cohere2_moe": (Cohere2MoeConfig, Cohere2MoeForCausalLM, TINY_COHERE2_MOE_CONFIG),
    "glm4_moe_lite": (Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, TINY_GLM4_MOE_LITE_CONFIG),
    "nemotron_h": (NemotronHConfig, NemotronHForCausalLM, TINY_NEMOTRON_H_CONFIG),
}


def _torch_fused_loss(hidden_states, lm_head_weight, labels, hidden_size, shift_labels=None, **kwargs):
    """``LigerForCausalLMLoss``'s contraction on any device."""
    logits = hidden_states.reshape(-1, hidden_size) @ lm_head_weight.T
    logits = logits.view(*hidden_states.shape[:-1], -1)
    return ForCausalLMLoss(logits, labels, vocab_size=logits.shape[-1], shift_labels=shift_labels, **kwargs)


def _tiny_fused_model(family: str, monkeypatch, **overrides):
    config_cls, model_cls, tiny = FAMILIES[family]
    torch.manual_seed(0)
    model = model_cls(config_cls(**{**tiny, **overrides})).train()
    spec = next(spec for spec in LIGER_FAMILY_SPECS if family in spec.model_types)
    apply_fused_modules_to_model(model, spec, {"fused_linear_cross_entropy": True})
    monkeypatch.setattr(cross_entropy, "LigerForCausalLMLoss", _torch_fused_loss)
    return model


def _batch(model) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 12), generator=generator)
    labels = input_ids.clone()
    labels[:, :3] = -100
    return input_ids, labels


@pytest.mark.parametrize("family", ["cohere2", "cohere2_moe"])
def test_fused_loss_trains_the_configured_logit_scale(family, monkeypatch):
    model = _tiny_fused_model(family, monkeypatch, logit_scale=LOGIT_SCALE)
    config_cls = FAMILIES[family][0]
    assert model.config.logit_scale == model.logit_scale == LOGIT_SCALE != config_cls().logit_scale
    input_ids, labels = _batch(model)
    with torch.no_grad():
        native = type(model).forward(model, input_ids=input_ids, labels=labels)
        fused = model(input_ids=input_ids, labels=labels)
        unfused = model(input_ids=input_ids, labels=labels, skip_logits=False)
    assert fused.logits is None, "the forward did not take the fused path"
    torch.testing.assert_close(fused.loss, native.loss)
    torch.testing.assert_close(unfused.logits, native.logits)
    torch.testing.assert_close(unfused.loss, native.loss)


def test_fused_loss_refuses_a_softcapped_head(monkeypatch):
    model = _tiny_fused_model("glm4_moe_lite", monkeypatch, final_logit_softcapping=30.0)
    input_ids, labels = _batch(model)
    with pytest.raises(ValueError, match="fused_linear_cross_entropy: false"):
        model(input_ids=input_ids, labels=labels)


def test_fused_loss_refuses_a_vocabulary_cut(monkeypatch):
    model = _tiny_fused_model("glm4_moe_lite", monkeypatch)
    truncating = types.SimpleNamespace(transform=lambda config: HeadTransform(vocab_size=config.vocab_size // 2))
    monkeypatch.setattr(cross_entropy, "resolve_head_transform_spec", lambda model_class: truncating)
    input_ids, labels = _batch(model)
    with pytest.raises(ValueError, match="fused_linear_cross_entropy: false"):
        model(input_ids=input_ids, labels=labels)


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_nemotrons_fused_loss_is_its_heads_loss(monkeypatch, packed):
    """Its forward hands every kwarg to the loss too, and the packed row's marker must stop at the backbone."""
    apply_nemotron_h_patches()
    model = _tiny_fused_model("nemotron_h", monkeypatch)
    input_ids, labels = _batch(model)
    kwargs = {"use_cache": False}
    if packed:
        kwargs["seq_idx"] = torch.tensor([[0] * 5 + [1] * 7] * 2, dtype=torch.int32)
    with torch.no_grad():
        native = type(model).forward(model, input_ids=input_ids, labels=labels, **kwargs)
        fused = model(input_ids=input_ids, labels=labels, **kwargs)
    assert fused.logits is None, "the forward did not take the fused path"
    torch.testing.assert_close(fused.loss, native.loss)


def test_nemotrons_unfused_logits_stay_in_the_dtype_its_head_returns(monkeypatch):
    """Its head casts them to fp32, where a log-softmax runs on TRL's fp32 branch; the bf16 branch rounds log-probs."""
    model = _tiny_fused_model("nemotron_h", monkeypatch).to(torch.bfloat16).eval()
    input_ids, _ = _batch(model)
    with torch.no_grad():
        native = type(model).forward(model, input_ids=input_ids)
        unfused = model(input_ids=input_ids)
        model.forward = types.MethodType(cross_entropy.build_lce_forward(), model)
        uncast = model(input_ids=input_ids)
    assert native.logits.dtype == unfused.logits.dtype == torch.float32
    assert torch.equal(unfused.logits, native.logits)
    assert uncast.logits.dtype == torch.bfloat16, "the cast is not what keeps them in fp32"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
