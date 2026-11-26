"""Prefix-LM decoders (PaliGemma): image and prompt tokens attend bidirectionally, the trained tokens causally.
transformers builds that mask from ``token_type_ids`` and leaves the decoder non-causal wherever it builds none."""

from __future__ import annotations

import torch

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.introspection.config_access import model_type_matches
from halo.models.introspection.module_tree import backbone_with_layers

# transformers sets the text config's ``use_bidirectional_attention`` when a checkpoint leaves it unset (every
# published PaliGemma does), so each decoder attention module is built with ``is_causal=False``.
PREFIX_LM_MODEL_TYPE_PREFIXES = ("paligemma",)


def has_prefix_lm_attention(model_config) -> bool:
    """Its mask reaches only a kernel that takes a materialized mask; no flash kernel reads ``token_type_ids``."""
    return model_type_matches(model_config, *PREFIX_LM_MODEL_TYPE_PREFIXES)


def prefix_lm_token_type_ids(labels: torch.Tensor) -> torch.Tensor:
    """1 from each row's first scored label on, 0 before it. The loss scores ``labels[:, 1:]``, so position 0 never
    opens the suffix, and no prefix position, which sees the whole prefix, attends to a token it is scored on."""
    scored = labels != LABEL_IGNORE_INDEX
    scored[:, 0] = False
    return (scored.cumsum(dim=-1) > 0).to(labels.dtype)


def apply_prefix_lm_causal_decoder(model: torch.nn.Module, model_config) -> None:
    """Make the decoder attention causal, so a forward without ``token_type_ids`` scores each token on its past alone.
    A forward with them still gets the prefix-LM mask, as every non-flash backend reads ``is_causal`` only when no mask
    is passed."""
    if not has_prefix_lm_attention(model_config):
        return
    decoder = backbone_with_layers(model)
    if decoder is None:
        raise ValueError(
            f"{type(model).__name__} is a prefix-LM model with no decoder layer list reachable from it, so its "
            f"attention cannot be made causal for training. Name its layer-list attribute in "
            f"DECODER_LAYER_LIST_ATTRS, or its backbone attribute in backbone_with_layers."
        )
    for module in decoder.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = True
