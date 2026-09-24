#!/usr/bin/env python
"""The last pipeline stage applies the family's head transform (``src/models/head_transform.py``).

A stage replaces the ``*ForCausalLM`` forward with its own head call, so whatever that forward
applies around the output embedding — Cohere's ``logit_scale``, Granite's ``logits_scaling``
division, MiniCPM3's pre-head division, Gemma's softcap — must be applied by the stage too, on the
logits it returns and inside the fused head loss. Every family either gets its verified transform
or is refused by the split gate, Cohere's ``logit_scale`` included.

Run: python tests/cpu/parallelism/test_pp_head_transform.py
"""

import pytest
import torch
import transformers

from src.distributed.pipeline_parallel.losses import causal_lm_token_loss, fused_causal_lm_token_loss
from src.distributed.pipeline_parallel.split import validate_model_supports_pp
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.models.head_transform import IDENTITY_HEAD_TRANSFORM, HeadTransform, resolve_head_transform
from tests.common.models import (
    TINY_COHERE2_MOE_CONFIG,
    TINY_GEMMA2_CONFIG,
    TINY_GRANITE_CONFIG,
    TINY_MINICPM3_CONFIG,
)

_BATCH, _SEQ = 2, 16

# Cohere2 MoE untied (PP refuses a tied head) and four layers, so a pp2 cut lands on a whole period of
# its sliding/full interleave.
_COHERE2_MOE_PP_CONFIG = {
    **TINY_COHERE2_MOE_CONFIG,
    "num_hidden_layers": 4,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
    "mlp_layer_types": ["sparse"] * 4,
    "tie_word_embeddings": False,
}

# id -> (model class, config class, config): one family per transform kind — a post-head scale, a
# post-head division, a pre-head division, a softcap.
_FAMILIES = {
    "cohere2_moe": ("Cohere2MoeForCausalLM", "Cohere2MoeConfig", _COHERE2_MOE_PP_CONFIG),
    "granite": ("GraniteForCausalLM", "GraniteConfig", TINY_GRANITE_CONFIG),
    "minicpm3": ("MiniCPM3ForCausalLM", "MiniCPM3Config", TINY_MINICPM3_CONFIG),
    "gemma2": ("Gemma2ForCausalLM", "Gemma2Config", TINY_GEMMA2_CONFIG),
}


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
    """The stage chain reproduces the unsplit forward's logits, which the bare head does not."""
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
    """The fused last-stage CE and its head-weight gradient equal the unsplit model's, whose forward
    applies the family's transform before the CE."""
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
    """Cohere's non-unit ``logit_scale`` passes the PP gate: the stage applies it."""
    model = _build("cohere2_moe")
    assert model.config.logit_scale != 1.0
    validate_model_supports_pp(model, "none")
    assert resolve_head_transform(model) == HeadTransform(logit_scale=model.config.logit_scale)


def test_an_undeclared_head_transform_is_refused_by_the_pp_gate():
    """HyperCLOVAX multiplies its logits by ``logits_scaling`` and declares nothing: the gate refuses it
    on every rank, before a stage is built, instead of returning unscaled logits."""
    config = transformers.HyperCLOVAXConfig(**{**TINY_GRANITE_CONFIG, "logits_scaling": 0.5})
    model = transformers.HyperCLOVAXForCausalLM(config)
    with pytest.raises(ValueError, match="HyperCLOVAXForCausalLM's forward transforms its head path"):
        validate_model_supports_pp(model, "none")
    with pytest.raises(ValueError, match="HyperCLOVAXForCausalLM's forward transforms its head path"):
        build_pipeline_stage(model, 0, 2, moe_balancing="none")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
