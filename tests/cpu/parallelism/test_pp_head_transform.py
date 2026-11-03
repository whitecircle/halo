#!/usr/bin/env python
"""The last pipeline stage applies the family's head transform (``halo/models/head_transform.py``), on its logits and
in the fused head loss. A stage replaces the ``*ForCausalLM`` forward, so undeclared transforms are refused."""

import pytest
import torch
import transformers

from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.distributed.pipeline_parallel.stage_losses import causal_lm_token_loss, fused_causal_lm_token_loss
from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM, HeadTransform, resolve_head_transform
from tests.common.models import (
    TINY_COHERE2_MOE_CONFIG,
    TINY_GEMMA2_CONFIG,
    TINY_GRANITE_CONFIG,
    TINY_MINICPM3_CONFIG,
)
from tests.common.pp_fakes import registered_pp_backbones

_BATCH, _SEQ = 2, 16

# Untied (PP refuses a tied head) and four layers, so a pp2 cut lands on a whole period of its sliding/full interleave.
_COHERE2_MOE_PP_CONFIG = {
    **TINY_COHERE2_MOE_CONFIG,
    "num_hidden_layers": 4,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
    "mlp_layer_types": ["sparse"] * 4,
    "tie_word_embeddings": False,
}

# One family per transform kind: post-head scale, post-head division, pre-head division, softcap.
_FAMILIES = {
    "cohere2_moe": ("Cohere2MoeForCausalLM", "Cohere2MoeConfig", _COHERE2_MOE_PP_CONFIG),
    "granite": ("GraniteForCausalLM", "GraniteConfig", TINY_GRANITE_CONFIG),
    "minicpm3": ("MiniCPM3ForCausalLM", "MiniCPM3Config", TINY_MINICPM3_CONFIG),
    "gemma2": ("Gemma2ForCausalLM", "Gemma2Config", TINY_GEMMA2_CONFIG),
}


@pytest.fixture(autouse=True)
def _transform_families_registered():
    """The head-path math under test is family-agnostic; HyperCLOVAX stays unregistered, since its refusal must come
    first."""
    with registered_pp_backbones("Cohere2MoeModel", "GraniteModel", "MiniCPM3Model", "Gemma2Model"):
        yield


def _build(family: str) -> torch.nn.Module:
    class_name, config_class_name, config = _FAMILIES[family]
    torch.manual_seed(0)
    config = getattr(transformers, config_class_name)(**config, attn_implementation="eager")
    return getattr(transformers, class_name)(config).float().eval()


def _input_ids(model) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5)
    return torch.randint(0, model.config.vocab_size, (_BATCH, _SEQ), generator=generator)


def _stages(family: str) -> list:
    return [build_pipeline_stage(_build(family), rank, 2, moe_balancing="none") for rank in range(2)]


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_last_stage_logits_are_the_family_forward(family):
    """The bare head does not reproduce these logits."""
    model = _build(family)
    input_ids = _input_ids(model)
    first, last = _stages(family)
    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=False).logits
        boundary = first(input_ids)
        logits = last(boundary)
        bare = last.head(last.model(inputs_embeds=boundary, use_cache=False).last_hidden_state)

    assert first.head_transform == IDENTITY_HEAD_TRANSFORM
    assert last.head_transform == resolve_head_transform(model) != IDENTITY_HEAD_TRANSFORM
    assert (bare - reference).abs().max() > 1e-2, f"{family}'s head transform is not load-bearing here"
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_fused_head_loss_applies_the_transform(family):
    model = _build(family)
    input_ids = _input_ids(model)
    reference = causal_lm_token_loss(model(input_ids=input_ids, use_cache=False).logits, input_ids)
    reference.backward()

    first, last = _stages(family)
    with torch.no_grad():
        boundary = first(input_ids)
    last.fused_loss_fn = fused_causal_lm_token_loss
    loss = last(boundary, labels=input_ids)
    loss.backward()

    torch.testing.assert_close(loss, reference, atol=1e-4, rtol=1e-5)
    head_grad = model.get_output_embeddings().weight.grad
    torch.testing.assert_close(last.head.weight.grad, head_grad, atol=1e-5 * head_grad.abs().max().item(), rtol=1e-4)


def test_cohere_logit_scale_is_applied_rather_than_refused():
    model = _build("cohere2_moe")
    assert model.config.logit_scale != 1.0
    reject_unsupported_pp_model(model, "none")
    assert resolve_head_transform(model) == HeadTransform(logit_scale=model.config.logit_scale)


def test_an_undeclared_head_transform_is_refused_by_the_pp_gate():
    """HyperCLOVAX multiplies by ``logits_scaling`` and declares nothing, so returning unscaled logits is the
    alternative."""
    config = transformers.HyperCLOVAXConfig(**{**TINY_GRANITE_CONFIG, "logits_scaling": 0.5})
    model = transformers.HyperCLOVAXForCausalLM(config)
    with pytest.raises(ValueError, match="HyperCLOVAXForCausalLM's forward transforms its head path"):
        reject_unsupported_pp_model(model, "none")
    with pytest.raises(ValueError, match="HyperCLOVAXForCausalLM's forward transforms its head path"):
        build_pipeline_stage(model, 0, 2, moe_balancing="none")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
