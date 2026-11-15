"""Output classes upstream Liger's fused forwards import, and the ``rope_deltas`` they read. liger-kernel derives each
from a per-family transformers output class that transformers folded into the generic ``CausalLMOutputWithPast`` /
``MoeCausalLMOutputWithPast``, so its appliers fail on import; bound to Liger's generic outputs plus a ``rope_deltas``
field, they run again. Those multimodal forwards also read ``rope_deltas`` off the model output, which transformers now
caches on the model instead, so the generic model outputs answer ``None`` for it."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from liger_kernel.transformers.model import output_classes
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast, LigerMoeCausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast


@dataclass
class _LigerRopeDeltasCausalLMOutput(LigerCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


@dataclass
class _LigerRopeDeltasMoeCausalLMOutput(LigerMoeCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


# Each name, by the generic output its family's head now returns.
_FOLDED_OUTPUT_CLASSES = {
    "LigerQwen2VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen2_5_VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3_5CausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3VLMoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
    "LigerQwen3_5MoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
    "LigerGlm4vMoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
}
# What those families' decoders return.
_FOLDED_MODEL_OUTPUTS = (BaseModelOutputWithPast, MoeModelOutputWithPast)


def supply_folded_output_classes() -> None:
    """Bind each missing name to its generic output, and a ``None`` ``rope_deltas`` to each model output without one;
    what liger-kernel or transformers still define is left alone."""
    for name, generic in _FOLDED_OUTPUT_CLASSES.items():
        if not hasattr(output_classes, name):
            setattr(output_classes, name, generic)
    for model_output in _FOLDED_MODEL_OUTPUTS:
        if not hasattr(model_output, "rope_deltas"):
            model_output.rope_deltas = None
