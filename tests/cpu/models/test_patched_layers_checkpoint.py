#!/usr/bin/env python
"""A decoder layer whose forward a family patch replaces still checkpoints: transformers checkpoints inside the
wrapper around the class's ``forward``, and only while that wrapper is the outermost one."""

from collections import Counter

import pytest
import torch
import transformers.models.deepseek_v4.modeling_deepseek_v4 as deepseek_v4_modeling
import transformers.models.glm5_next.modeling_glm5_next as glm5_next_modeling
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from transformers.modeling_layers import GradientCheckpointingLayer

from halo.models.families.qwen4_exp.forwards import patch_qwen4_exp_document_forwards
from halo.models.gradient_checkpointing import patch_layer_forward
from halo.models.hyper_connections import _PATCHED_ATTR, apply_hyper_connection_patches
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# Per roster family: the patch, the decoder layer class it replaces the forward of, and a child each layer pass calls
# once.
_PATCHED_LAYERS = {
    "deepseek_v4": (apply_hyper_connection_patches, deepseek_v4_modeling.DeepseekV4DecoderLayer, "attn_hc"),
    "glm5_next": (apply_hyper_connection_patches, glm5_next_modeling.Glm5NextTextDecoderLayer, "attn_hc"),
    "qwen4_exp_text": (
        lambda _config: patch_qwen4_exp_document_forwards(),
        qwen4_exp_modeling.Qwen4ExpTextDecoderLayer,
        "attn_hyper_connection",
    ),
}


def _layer_passes(family: str, *, use_reentrant: bool) -> list[int]:
    """How many times each patched decoder layer's body runs over one checkpointed forward and backward."""
    apply, layer_class, child = _PATCHED_LAYERS[family]
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family]).train()
    apply(model.config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    layers = [module for module in model.modules() if isinstance(module, layer_class)]
    assert layers and all(layer.gradient_checkpointing for layer in layers)
    passes = Counter()
    for index, layer in enumerate(layers):
        getattr(layer, child).register_forward_pre_hook(lambda *_args, index=index: passes.update([index]))
    input_ids = torch.randint(
        0, model.get_input_embeddings().num_embeddings, (2, 8), generator=torch.Generator().manual_seed(1)
    )
    model(input_ids=input_ids).logits.float().square().mean().backward()
    return [passes[index] for index in range(len(layers))]


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non_reentrant"])
@pytest.mark.parametrize("family", _PATCHED_LAYERS)
def test_a_patched_decoder_layer_runs_its_forward_and_its_recompute(family, use_reentrant):
    passes = _layer_passes(family, use_reentrant=use_reentrant)
    assert passes == [2] * len(passes), passes


def test_a_replaced_checkpointing_layer_forward_keeps_the_checkpoint():
    calls = []

    class Layer(GradientCheckpointingLayer):
        def forward(self, hidden_states):
            return hidden_states * 2

    def replacement(self, hidden_states):
        return hidden_states * 3

    patch_layer_forward(Layer, replacement)
    layer = Layer().train()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = lambda function, *args: calls.append(function) or function(*args)

    assert torch.equal(layer(torch.ones(2)), torch.full((2,), 3.0))
    assert len(calls) == 1 and calls[0].func is replacement


def test_a_plain_class_takes_the_replacement_as_is():
    owner = type("Owner", (), {"forward": lambda self: "stock"})

    def replacement(self):
        return "replacement"

    patch_layer_forward(owner, replacement)
    assert owner.forward is replacement


def test_the_hyper_connection_marker_reads_through_the_checkpoint_wrapper():
    """``_install`` reads the marker off the class's forward to install once."""
    apply_hyper_connection_patches(tiny_family_model(TINY_MOE_FAMILIES["deepseek_v4"]).config)
    assert getattr(deepseek_v4_modeling.DeepseekV4DecoderLayer.forward, _PATCHED_ATTR, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
